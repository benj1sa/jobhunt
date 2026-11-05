"""Discovery → leads → canonical jobs → snapshots.

The three stages are separate functions so each can be re-run on its own and every one of them is idempotent.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import Ats, JobStatus, LeadStatus, RunStatus
from ..models.pipeline import IngestRun, Job, JobSnapshot, Lead
from ..schemas.config import BoardConfig
from ..sources.base import RawLead, Source
from .ats import AtsPosting, fetch_posting
from .http import HttpClient, NotFoundError
from .text import content_hash, html_to_text
from .urls import AtsRef, find_ats_links, is_aggregator, is_blocked, normalize_url, parse_ats, posting_url_for

logger = getLogger(__name__)

# A fetched page with less text than this is almost certainly a JavaScript shell, not a job description.
MIN_DESCRIPTION_CHARS = 400


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _ref_to_extra(ref: AtsRef) -> dict[str, str]:
    return {"ats": ref.ats.value, "board": ref.board, "external_id": ref.external_id}


def _ref_from_extra(extra: dict[str, Any]) -> AtsRef | None:
    ref = extra.get("ats_ref")
    if isinstance(ref, dict):
        return AtsRef(Ats(ref["ats"]), ref["board"], ref["external_id"])
    return None


# --- Ingest -----------------------------------------------------------------------------------------------------


@dataclass
class IngestResult:
    source: str
    status: RunStatus
    seen: int = 0
    new: int = 0
    error: str | None = None


async def upsert_leads(session: AsyncSession, leads: Sequence[RawLead]) -> int:
    """Insert unseen leads and refresh `last_seen_at` on known ones. Returns the number of new leads."""
    now = utcnow()
    created = 0
    batch_keys: set[tuple[str, str]] = set()
    for raw in leads:
        url_normalized = normalize_url(raw.url)
        if (raw.source, url_normalized) in batch_keys:
            continue
        batch_keys.add((raw.source, url_normalized))
        existing = (
            await session.execute(
                select(Lead).where(and_(Lead.source == raw.source, Lead.url_normalized == url_normalized))
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.last_seen_at = now
            if existing.job_id is not None:
                job = await session.get(Job, existing.job_id)
                if job is not None:
                    job.last_seen_at = now
            continue
        extra: dict[str, Any] = dict(raw.extra)
        if raw.ats_ref is not None:
            extra["ats_ref"] = _ref_to_extra(raw.ats_ref)
        if raw.description_text:
            # Held on the lead only until resolution turns it into a snapshot; see `resolve_leads`.
            extra["pending_description"] = raw.description_text
            extra["pending_raw"] = raw.raw
        session.add(
            Lead(
                source=raw.source,
                company=raw.company.strip() or "Unknown",
                title=raw.title.strip() or "Unknown role",
                url=raw.url.strip(),
                url_normalized=url_normalized,
                location=raw.location,
                posted_at=raw.posted_at,
                extra=extra,
            )
        )
        created += 1
    await session.flush()
    return created


async def ingest(session: AsyncSession, client: HttpClient, sources: Sequence[Source]) -> list[IngestResult]:
    """Run every source concurrently, then store results sequentially. One failing source never stops the rest."""
    runs: list[IngestRun] = []
    for source in sources:
        run = IngestRun(source=source.name, status=RunStatus.RUNNING)
        session.add(run)
        runs.append(run)
    await session.commit()

    fetched = await asyncio.gather(*(source.fetch(client) for source in sources), return_exceptions=True)

    results: list[IngestResult] = []
    for source, run, outcome in zip(sources, runs, fetched, strict=True):
        if isinstance(outcome, BaseException):
            logger.error("Source %s failed: %r", source.name, outcome, exc_info=outcome)
            run.status = RunStatus.FAILED
            run.error = f"{type(outcome).__name__}: {outcome}"[:2000]
        else:
            run.leads_seen = len(outcome)
            run.leads_new = await upsert_leads(session, outcome)
            run.status = RunStatus.OK
        run.finished_at = utcnow()
        await session.commit()
        results.append(IngestResult(source.name, run.status, run.leads_seen, run.leads_new, run.error))
    return results


# --- Resolve ----------------------------------------------------------------------------------------------------


@dataclass
class ResolveStats:
    resolved: int = 0
    needs_url: int = 0
    failed: int = 0
    new_jobs: int = 0
    errors: list[str] = field(default_factory=list)


async def add_snapshot(session: AsyncSession, job: Job, text: str, raw: dict[str, Any]) -> JobSnapshot | None:
    """Store a description snapshot unless an identical one exists. Returns the new snapshot or None."""
    digest = content_hash(text)
    duplicate = (
        await session.execute(
            select(JobSnapshot.id).where(and_(JobSnapshot.job_id == job.id, JobSnapshot.content_hash == digest))
        )
    ).scalar_one_or_none()
    job.hydrated_at = utcnow()
    job.hydrate_error = None
    if duplicate is not None:
        return None
    snapshot = JobSnapshot(job_id=job.id, content_hash=digest, description_text=text, raw=raw)
    session.add(snapshot)
    await session.flush()
    return snapshot


async def upsert_job(session: AsyncSession, lead: Lead, ref: AtsRef | None) -> tuple[Job, bool]:
    """Find or create the canonical job for a lead. Returns (job, created)."""
    canonical_key = ref.canonical_key if ref else f"url:{lead.url_normalized}"
    job = (await session.execute(select(Job).where(Job.canonical_key == canonical_key))).scalar_one_or_none()
    now = utcnow()
    if job is not None:
        job.last_seen_at = now
        job.status = JobStatus.OPEN
        job.location = job.location or lead.location
        job.posted_at = job.posted_at or lead.posted_at
        return job, False
    job = Job(
        canonical_key=canonical_key,
        ats=ref.ats if ref else Ats.URL,
        board=ref.board if ref else None,
        external_id=ref.external_id if ref else None,
        company=lead.company,
        title=lead.title,
        location=lead.location,
        is_remote=lead.extra.get("is_remote"),
        posting_url=posting_url_for(ref) if ref else lead.url,
        apply_url=lead.extra.get("apply_url"),
        posted_at=lead.posted_at,
        first_seen_at=now,
        last_seen_at=now,
    )
    session.add(job)
    await session.flush()
    return job, True


async def _unwrap_aggregator(client: HttpClient, url: str, boards: list[BoardConfig]) -> AtsRef | None:
    """Fetch an aggregator page (BuiltIn, Idealist, ...) and return the ATS posting it links to, if exactly one."""
    final_url, body = await client.get_text(url)
    if ref := parse_ats(final_url, boards):
        return ref
    links = find_ats_links(body, boards)
    return next(iter(links.values())) if len(links) == 1 else None


async def resolve_lead(
    session: AsyncSession,
    lead: Lead,
    boards: list[BoardConfig],
    client: HttpClient | None,
    override_url: str | None = None,
) -> tuple[Job | None, bool]:
    """Resolve one lead to a job. `override_url` is a human-supplied ATS link for a `needs_url` lead."""
    url = override_url or lead.url
    ref = None if override_url else _ref_from_extra(lead.extra)
    ref = ref or parse_ats(url, boards)

    if ref is None and is_blocked(url):
        lead.status = LeadStatus.NEEDS_URL
        lead.error = "Board blocks automated access. Run `jobhunt resolve <lead-id> <ats-url>`."
        return None, False
    if ref is None and client is not None and is_aggregator(url) and "pending_description" not in lead.extra:
        try:
            ref = await _unwrap_aggregator(client, url, boards)
        except (httpx.HTTPError, NotFoundError) as exc:
            # Never fail a lead on a network error: fall back to a URL-keyed job that hydration retries later.
            logger.warning("Could not unwrap %s (%s); keeping it as a URL job", url, exc)

    if override_url and ref is None:
        # A non-ATS link chosen by hand still becomes a URL-keyed job.
        lead.extra = {**lead.extra, "override_url": override_url}
        lead.url_normalized = normalize_url(override_url)

    job, created = await upsert_job(session, lead, ref)
    if override_url and ref is None:
        job.posting_url = override_url

    pending = lead.extra.get("pending_description")
    if isinstance(pending, str) and pending.strip():
        await add_snapshot(session, job, pending, lead.extra.get("pending_raw") or {})
        lead.extra = {k: v for k, v in lead.extra.items() if k not in ("pending_description", "pending_raw")}

    lead.job_id = job.id
    lead.status = LeadStatus.RESOLVED
    lead.error = None
    return job, created


async def resolve_leads(
    session: AsyncSession,
    boards: list[BoardConfig],
    client: HttpClient | None,
    limit: int | None = None,
) -> ResolveStats:
    """Resolve every `new` lead. Commits after each lead so a crash loses at most one lead's work."""
    stmt = select(Lead).where(Lead.status == LeadStatus.NEW).order_by(Lead.discovered_at)
    if limit:
        stmt = stmt.limit(limit)
    leads = (await session.execute(stmt)).scalars().all()
    stats = ResolveStats()
    for lead in leads:
        try:
            job, created = await resolve_lead(session, lead, boards, client)
        except Exception as exc:
            logger.exception("Failed to resolve lead %s (%s)", lead.id, lead.url)
            await session.rollback()
            lead = await session.get(Lead, lead.id) or lead
            lead.status = LeadStatus.FAILED
            lead.error = f"{type(exc).__name__}: {exc}"[:2000]
            stats.failed += 1
            stats.errors.append(f"{lead.url}: {lead.error}")
        else:
            if job is None:
                stats.needs_url += 1
            else:
                stats.resolved += 1
                stats.new_jobs += int(created)
        await session.commit()
    return stats


async def attach_url(
    session: AsyncSession, lead_id: UUID, url: str, boards: list[BoardConfig], client: HttpClient | None
) -> Job:
    """Manually resolve a lead (usually `needs_url`) to the given posting link."""
    lead = await session.get(Lead, lead_id)
    if lead is None:
        raise LookupError(f"No lead with id {lead_id}")
    job, _ = await resolve_lead(session, lead, boards, client, override_url=url)
    if job is None:
        raise ValueError(f"{url} is also on a blocked board; paste the company's own ATS link instead.")
    await session.commit()
    return job


# --- Hydrate ----------------------------------------------------------------------------------------------------


@dataclass
class HydrateStats:
    hydrated: int = 0
    unchanged: int = 0
    closed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


async def _fetch_for_job(client: HttpClient, job: Job) -> AtsPosting | tuple[str, str]:
    if job.ats == Ats.URL:
        final_url, body = await client.get_text(job.posting_url)
        return final_url, html_to_text(body)
    ref = AtsRef(job.ats, job.board or "", job.external_id or "")
    return await fetch_posting(client, ref)


async def hydrate_jobs(
    session: AsyncSession,
    client: HttpClient,
    refresh: bool = False,
    limit: int | None = None,
) -> HydrateStats:
    """Fetch full descriptions for open jobs without a snapshot (or all open jobs with `refresh`).

    Fetches run concurrently (bounded by the HTTP client); database writes happen one job at a time.
    A 404 marks the job closed, which is how closed postings drop out of the queue.
    """
    has_snapshot = exists().where(JobSnapshot.job_id == Job.id)
    condition = Job.status == JobStatus.OPEN if refresh else and_(Job.status == JobStatus.OPEN, ~has_snapshot)
    stmt = select(Job).where(condition).order_by(Job.first_seen_at)
    if limit:
        stmt = stmt.limit(limit)
    jobs = list((await session.execute(stmt)).scalars().all())
    outcomes = await asyncio.gather(*(_fetch_for_job(client, job) for job in jobs), return_exceptions=True)

    stats = HydrateStats()
    for job, outcome in zip(jobs, outcomes, strict=True):
        if isinstance(outcome, NotFoundError) and job.hydrated_at is not None:
            job.status = JobStatus.CLOSED
            job.hydrate_error = str(outcome)
            stats.closed += 1
        elif isinstance(outcome, NotFoundError):
            # Never fetched successfully: more likely a wrong board slug than a closed job. Keep it open.
            job.hydrate_error = f"{outcome} (never fetched before: check the board slug or attach the link by hand)"
            stats.failed += 1
            stats.errors.append(f"{job.posting_url}: {job.hydrate_error}")
        elif isinstance(outcome, BaseException):
            logger.warning("Hydration failed for %s: %r", job.posting_url, outcome)
            job.hydrate_error = f"{type(outcome).__name__}: {outcome}"[:2000]
            stats.failed += 1
            stats.errors.append(f"{job.posting_url}: {job.hydrate_error}")
        elif isinstance(outcome, AtsPosting):
            job.title = outcome.title or job.title
            if outcome.company and job.company in ("", "Unknown"):
                job.company = outcome.company
            job.location = outcome.location or job.location
            job.is_remote = outcome.is_remote if outcome.is_remote is not None else job.is_remote
            job.apply_url = outcome.apply_url or job.apply_url
            job.posted_at = outcome.posted_at or job.posted_at
            snapshot = await add_snapshot(session, job, outcome.description_text, outcome.raw)
            stats.hydrated += int(snapshot is not None)
            stats.unchanged += int(snapshot is None)
        else:
            _final_url, text = outcome
            if len(text) < MIN_DESCRIPTION_CHARS:
                job.hydrate_error = "Page has too little text (likely rendered by JavaScript); open it manually."
                stats.failed += 1
            else:
                snapshot = await add_snapshot(session, job, text, {"source_url": job.posting_url})
                stats.hydrated += int(snapshot is not None)
                stats.unchanged += int(snapshot is None)
        await session.commit()
    return stats
