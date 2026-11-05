"""Jev screen before LLM extraction ("Screen with a Noul first").

Jev answers a few cheap questions about each new snapshot; `questions.decide_screen` turns them into
pass / unsettled / reject. Only `reject` skips extraction. A failed call is recorded as an error and the job
goes on to extraction: a failure is never read as a "no".
"""

import asyncio
from dataclasses import dataclass, field
from logging import getLogger

import httpx
from sqlalchemy import and_, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import JobStatus, ScreenOutcome
from ..models.pipeline import Job, JobScreen, JobSnapshot
from ..questions import SCREEN_VERSION, decide_screen, screen_questions, screen_state
from ..schemas.typesafe import SystemOneResponse
from .http import NotFoundError
from .typesafe import TypeSafeClient, TypeSafeError

logger = getLogger(__name__)

BATCH_SIZE = 20


@dataclass
class ScreenStats:
    passed: int = 0
    unsettled: int = 0
    rejected: int = 0
    failed: int = 0
    input_tokens: int = 0
    errors: list[str] = field(default_factory=list)


async def snapshots_needing_screen(session: AsyncSession, limit: int | None) -> list[tuple[Job, JobSnapshot]]:
    newest = (
        select(JobSnapshot.job_id, func.max(JobSnapshot.fetched_at).label("fetched_at"))
        .group_by(JobSnapshot.job_id)
        .subquery()
    )
    screened = exists().where(and_(JobScreen.snapshot_id == JobSnapshot.id, JobScreen.version == SCREEN_VERSION))
    stmt = (
        select(Job, JobSnapshot)
        .join(JobSnapshot, JobSnapshot.job_id == Job.id)
        .join(newest, and_(newest.c.job_id == JobSnapshot.job_id, newest.c.fetched_at == JobSnapshot.fetched_at))
        .where(and_(Job.status == JobStatus.OPEN, ~screened))
        .order_by(Job.first_seen_at)
    )
    if limit:
        stmt = stmt.limit(limit)
    return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]


async def _ask(client: TypeSafeClient, job: Job, snapshot: JobSnapshot, max_chars: int) -> SystemOneResponse:
    request = client.build_request(
        state=screen_state(job.title, job.company, job.location, snapshot.description_text[:max_chars]),
        questions=screen_questions(),
    )
    return await client.system_one(request)


async def screen_jobs(
    session: AsyncSession, client: TypeSafeClient, max_chars: int, limit: int | None = None
) -> ScreenStats:
    """Screen each open job's newest unscreened snapshot. Requests run concurrently in batches; writes are serial."""
    stats = ScreenStats()
    pending = await snapshots_needing_screen(session, limit)
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        outcomes = await asyncio.gather(
            *(_ask(client, job, snapshot, max_chars) for job, snapshot in batch), return_exceptions=True
        )
        for (job, snapshot), outcome in zip(batch, outcomes, strict=True):
            if isinstance(outcome, (TypeSafeError, httpx.HTTPError, NotFoundError, KeyError, TypeError)):
                logger.warning("Screen failed for %s (%s): %r", job.id, job.posting_url, outcome)
                stats.failed += 1
                stats.errors.append(f"{job.company} - {job.title}: {type(outcome).__name__}: {outcome}"[:500])
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            try:
                decision = decide_screen(outcome)
            except (KeyError, TypeError) as exc:
                logger.warning("Screen answers unusable for %s: %r", job.id, exc)
                stats.failed += 1
                stats.errors.append(f"{job.company} - {job.title}: {exc}"[:500])
                continue
            session.add(
                JobScreen(
                    job_id=job.id,
                    snapshot_id=snapshot.id,
                    version=SCREEN_VERSION,
                    model=outcome.model,
                    outcome=decision.outcome,
                    reason=decision.reason,
                    signals=decision.signals,
                    answers={name: answer.model_dump() for name, answer in outcome.answers.items()},
                )
            )
            stats.input_tokens += outcome.usage.input_tokens
            stats.passed += int(decision.outcome == ScreenOutcome.PASS)
            stats.unsettled += int(decision.outcome == ScreenOutcome.UNSETTLED)
            stats.rejected += int(decision.outcome == ScreenOutcome.REJECT)
        await session.commit()
    return stats
