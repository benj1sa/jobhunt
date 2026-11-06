"""LLM extraction of structured requirements from job-description snapshots."""

import asyncio
import re
from dataclasses import dataclass, field
from logging import getLogger
from uuid import UUID

import httpx
from sqlalchemy import and_, delete, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import JobStatus, ScreenOutcome
from ..models.pipeline import Job, JobCapability, JobExtraction, JobScreen, JobSkill, JobSnapshot
from ..questions import SCREEN_VERSION
from ..schemas.config import Capability
from ..schemas.extraction import EXTRACTION_SCHEMA_VERSION, JobRequirements, extraction_json_schema
from .filtering import judge_relevance
from .http import NotFoundError
from .llm import LlmConfigError, LlmError, OpenRouterClient
from .skills import normalize_skills

logger = getLogger(__name__)

EXTRACTION_SYSTEM = """You extract structured hiring requirements from job postings.
Rules:
- Use only what the posting states. Use "unknown" or null when it does not say.
- Skills are concrete technologies, languages, frameworks, tools or techniques (e.g. "Python", "PyTorch",
  "Kubernetes", "distributed systems"). Not soft skills, not degrees, not years.
- Use short canonical names: "Python" not "strong Python programming skills".
- "Required" = listed under requirements/qualifications/must-have. "Preferred" = nice-to-have/bonus/plus.
- new_grad = the posting targets recent or upcoming graduates. entry = 0-2 years without a grad framing.
- requires_clearance only for an active/current clearance requirement, not "ability to obtain".
- Salary is annual base in USD; convert hourly by x2080. null if absent.
- capability_evidence: copy every responsibility or expectation about how the person works, word for word from
  the posting (trim to the relevant clause, at most ~30 words). Include soft skills and ways of working
  ("communicate technical ideas clearly", "turn customer feedback into improvements"). Exclude tool lists,
  degrees, years of experience, benefits, location, legal text and descriptions of the company.
  Tag each phrase with one to three capability ids from the list below; use "other" only when none fits."""


def build_system_prompt(capabilities: list[Capability]) -> str:
    lines = [EXTRACTION_SYSTEM, "", "Capability ids:"]
    for capability in capabilities:
        lines.append(f"- {capability.id}: {capability.description}")
        lines.extend(f'    e.g. "{example}"' for example in capability.examples[:2])
    return "\n".join(lines)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class ExtractStats:
    extracted: int = 0
    relevant: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def build_prompt(job: Job, snapshot: JobSnapshot, max_chars: int) -> str:
    text = snapshot.description_text
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return (
        f"Company: {job.company}\nTitle: {job.title}\nListed location: {job.location or 'not listed'}\n\n"
        f"Job description:\n{text}"
    )


async def latest_snapshots_needing_extraction(
    session: AsyncSession, model: str, limit: int | None, rerun: bool = False
) -> list[tuple[Job, JobSnapshot]]:
    """Open jobs whose newest snapshot has no extraction for this model and schema version (all of them with
    `rerun`, e.g. after editing config/capabilities.toml)."""
    newest = (
        select(JobSnapshot.job_id, func.max(JobSnapshot.fetched_at).label("fetched_at"))
        .group_by(JobSnapshot.job_id)
        .subquery()
    )
    already = exists().where(
        and_(
            JobExtraction.snapshot_id == JobSnapshot.id,
            JobExtraction.model == model,
            JobExtraction.schema_version == EXTRACTION_SCHEMA_VERSION,
        )
    )
    # Snapshots the current Jev screen rejected skip extraction. Unscreened snapshots (no TypeSafe key, or a failed
    # screen call) are extracted: a missing screen is never treated as a rejection.
    rejected = exists().where(
        and_(
            JobScreen.snapshot_id == JobSnapshot.id,
            JobScreen.version == SCREEN_VERSION,
            JobScreen.outcome == ScreenOutcome.REJECT,
        )
    )
    stmt = (
        select(Job, JobSnapshot)
        .join(JobSnapshot, JobSnapshot.job_id == Job.id)
        .join(newest, and_(newest.c.job_id == JobSnapshot.job_id, newest.c.fetched_at == JobSnapshot.fetched_at))
        .where(
            and_(Job.status == JobStatus.OPEN, ~rejected)
            if rerun
            else and_(Job.status == JobStatus.OPEN, ~already, ~rejected)
        )
        .order_by(Job.first_seen_at)
    )
    if limit:
        stmt = stmt.limit(limit)
    return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]


async def store_extraction(
    session: AsyncSession,
    job_id: UUID,
    snapshot_id: UUID,
    model: str,
    requirements: JobRequirements,
    aliases: dict[str, str],
    capability_ids: set[str] | None = None,
    source_text: str = "",
) -> JobExtraction:
    """Store (or replace) the extraction for one snapshot, its normalized skills and its capability evidence."""
    previous = (
        await session.execute(
            select(JobExtraction).where(
                and_(
                    JobExtraction.snapshot_id == snapshot_id,
                    JobExtraction.model == model,
                    JobExtraction.schema_version == EXTRACTION_SCHEMA_VERSION,
                )
            )
        )
    ).scalar_one_or_none()
    if previous is not None:
        # Explicit deletes: SQLite only cascades when foreign keys are switched on for the connection.
        await session.execute(delete(JobSkill).where(JobSkill.extraction_id == previous.id))
        await session.execute(delete(JobCapability).where(JobCapability.extraction_id == previous.id))
        await session.delete(previous)
        await session.flush()
    relevance = judge_relevance(requirements)
    extraction = JobExtraction(
        job_id=job_id,
        snapshot_id=snapshot_id,
        model=model,
        schema_version=EXTRACTION_SCHEMA_VERSION,
        data=requirements.model_dump(mode="json"),
        is_relevant=relevance.relevant,
        relevance_reason=relevance.reason,
    )
    session.add(extraction)
    await session.flush()
    required = normalize_skills(requirements.required_skills, aliases)
    preferred = [skill for skill in normalize_skills(requirements.preferred_skills, aliases) if skill not in required]
    for skill in required:
        session.add(JobSkill(job_id=job_id, extraction_id=extraction.id, skill=skill, required=True))
    for skill in preferred:
        session.add(JobSkill(job_id=job_id, extraction_id=extraction.id, skill=skill, required=False))
    haystack = _squash(source_text)
    for evidence in requirements.capability_evidence:
        phrase = evidence.phrase.strip()
        if not phrase:
            continue
        tags = {tag if capability_ids is None or tag in capability_ids else "other" for tag in evidence.capabilities}
        for tag in sorted(tags or {"other"}):
            session.add(
                JobCapability(
                    job_id=job_id,
                    extraction_id=extraction.id,
                    capability=tag,
                    phrase=phrase,
                    verbatim=bool(haystack) and _squash(phrase) in haystack,
                )
            )
    return extraction


async def extract_jobs(
    session: AsyncSession,
    llm: OpenRouterClient,
    model: str,
    aliases: dict[str, str],
    max_chars: int,
    limit: int | None = None,
    capabilities: list[Capability] | None = None,
    rerun: bool = False,
    concurrency: int = 1,
) -> ExtractStats:
    """Extract requirements for every open job's newest unextracted snapshot. Commits per job.

    Prompts are built before any LLM call, and nothing reads ORM attributes after a rollback (which expires them).
    An `LlmConfigError` stops the run: every remaining job would fail the same way.
    """
    stats = ExtractStats()
    capabilities = capabilities or []
    capability_ids = [capability.id for capability in capabilities]
    system = build_system_prompt(capabilities)
    json_schema = extraction_json_schema(capability_ids)
    pending = [
        (
            job.id,
            snapshot.id,
            f"{job.company} - {job.title}",
            job.posting_url,
            build_prompt(job, snapshot, max_chars),
            snapshot.description_text,
        )
        for job, snapshot in await latest_snapshots_needing_extraction(session, model, limit, rerun)
    ]

    async def ask(prompt: str) -> JobRequirements:
        return await llm.structured(
            model=model, system=system, user=prompt, schema=JobRequirements, json_schema=json_schema
        )

    for start in range(0, len(pending), max(1, concurrency)):
        chunk = pending[start : start + max(1, concurrency)]
        answers = await asyncio.gather(*(ask(item[4]) for item in chunk), return_exceptions=True)
        for (job_id, snapshot_id, label, url, _prompt, source_text), answer in zip(chunk, answers, strict=True):
            if isinstance(answer, LlmConfigError):
                raise answer
            if isinstance(answer, (LlmError, httpx.HTTPError, NotFoundError)):
                logger.warning("Extraction failed for %s (%s): %s", label, url, answer)
                stats.failed += 1
                stats.errors.append(f"{label}: {type(answer).__name__}: {answer}"[:500])
                continue
            if isinstance(answer, BaseException):
                raise answer
            try:
                extraction = await store_extraction(
                    session, job_id, snapshot_id, model, answer, aliases, set(capability_ids), source_text
                )
                await session.commit()
            except Exception as exc:
                logger.exception("Could not store extraction for %s", label)
                await session.rollback()
                stats.failed += 1
                stats.errors.append(f"{label}: {type(exc).__name__}: {exc}"[:500])
                continue
            stats.extracted += 1
            stats.relevant += int(extraction.is_relevant)
    return stats
