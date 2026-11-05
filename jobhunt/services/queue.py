"""Read models for the review queue, reviews, health and the skills report."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import Decision, JobStatus, LeadStatus, ScreenOutcome
from ..models.pipeline import IngestRun, Job, JobExtraction, JobScore, JobScreen, JobSkill, JobSnapshot, Lead, Review
from ..questions import SCREEN_VERSION


@dataclass
class QueueItem:
    job: Job
    score: JobScore | None
    extraction: JobExtraction | None
    sources: list[str]
    decision: Decision | None
    review_reason: str | None
    description: str | None = None
    lead_urls: list[str] = field(default_factory=list)

    @property
    def requirements(self) -> dict[str, Any]:
        return self.extraction.data if self.extraction else {}


async def latest_reviews(session: AsyncSession) -> dict[UUID, Review]:
    newest = select(Review.job_id, func.max(Review.created_at).label("created_at")).group_by(Review.job_id).subquery()
    stmt = select(Review).join(newest, and_(newest.c.job_id == Review.job_id, newest.c.created_at == Review.created_at))
    return {review.job_id: review for review in (await session.execute(stmt)).scalars().all()}


async def build_queue(
    session: AsyncSession,
    profile_hash: str,
    include_reviewed: bool = False,
    include_unscored: bool = False,
) -> list[QueueItem]:
    """Open, relevant jobs ordered for review.

    Unreviewed jobs come first by priority, then `later` jobs. `apply`/`applied`/`skip` jobs are hidden unless
    `include_reviewed`. Jobs that are open but not yet extracted/scored are appended with `include_unscored`.
    """
    scores = {
        score.job_id: score
        for score in (await session.execute(select(JobScore).where(JobScore.profile_hash == profile_hash)))
        .scalars()
        .all()
    }
    newest_extraction = (
        select(JobExtraction.job_id, func.max(JobExtraction.created_at).label("created_at"))
        .group_by(JobExtraction.job_id)
        .subquery()
    )
    extractions = {
        extraction.job_id: extraction
        for extraction in (
            await session.execute(
                select(JobExtraction).join(
                    newest_extraction,
                    and_(
                        newest_extraction.c.job_id == JobExtraction.job_id,
                        newest_extraction.c.created_at == JobExtraction.created_at,
                    ),
                )
            )
        )
        .scalars()
        .all()
    }
    reviews = await latest_reviews(session)
    screened_out = set(
        (
            await session.execute(
                select(JobScreen.job_id).where(
                    and_(JobScreen.version == SCREEN_VERSION, JobScreen.outcome == ScreenOutcome.REJECT)
                )
            )
        ).scalars()
    )
    sources: dict[UUID, list[str]] = defaultdict(list)
    lead_urls: dict[UUID, list[str]] = defaultdict(list)
    for job_id, source, url in (
        await session.execute(select(Lead.job_id, Lead.source, Lead.url).where(Lead.job_id.is_not(None)))
    ).all():
        if job_id is None:
            continue
        if source not in sources[job_id]:
            sources[job_id].append(source)
        lead_urls[job_id].append(url)

    jobs = (await session.execute(select(Job).where(Job.status == JobStatus.OPEN))).scalars().all()
    items: list[QueueItem] = []
    for job in jobs:
        extraction = extractions.get(job.id)
        score = scores.get(job.id)
        review = reviews.get(job.id)
        if extraction is not None and not extraction.is_relevant:
            continue
        if extraction is None and job.id in screened_out:
            continue
        if score is None and not include_unscored:
            continue
        if review and review.decision != Decision.LATER and not include_reviewed:
            continue
        items.append(
            QueueItem(
                job=job,
                score=score,
                extraction=extraction,
                sources=sources[job.id],
                decision=review.decision if review else None,
                review_reason=review.reason if review else None,
                lead_urls=lead_urls[job.id],
            )
        )

    def sort_key(item: QueueItem) -> tuple[int, int, float]:
        later = int(item.decision == Decision.LATER)
        unscored = int(item.score is None)
        return (later, unscored, -(item.score.priority if item.score else 0.0))

    items.sort(key=sort_key)
    return items


async def load_description(session: AsyncSession, job_id: UUID) -> str | None:
    stmt = (
        select(JobSnapshot.description_text)
        .where(JobSnapshot.job_id == job_id)
        .order_by(desc(JobSnapshot.fetched_at))
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def record_review(session: AsyncSession, job_id: UUID, decision: Decision, reason: str | None) -> Review:
    review = Review(job_id=job_id, decision=decision, reason=reason or None)
    session.add(review)
    await session.commit()
    return review


async def unresolved_leads(session: AsyncSession) -> Sequence[Lead]:
    stmt = (
        select(Lead)
        .where(Lead.status.in_([LeadStatus.NEEDS_URL, LeadStatus.FAILED]))
        .order_by(desc(Lead.discovered_at))
    )
    return (await session.execute(stmt)).scalars().all()


@dataclass
class SourceHealth:
    source: str
    last_run: datetime | None
    last_status: str
    last_error: str | None
    last_seen: int
    avg_seen: float
    zero_streak: int

    @property
    def healthy(self) -> bool:
        return self.last_status == "ok" and self.zero_streak < 3


async def source_health(session: AsyncSession, window_days: int = 30) -> list[SourceHealth]:
    """Per-source summary of recent ingest runs. Three zero-lead runs in a row counts as unhealthy."""
    since = datetime.now(tz=UTC) - timedelta(days=window_days)
    runs = (
        (
            await session.execute(
                select(IngestRun).where(IngestRun.started_at >= since).order_by(desc(IngestRun.started_at))
            )
        )
        .scalars()
        .all()
    )
    by_source: dict[str, list[IngestRun]] = defaultdict(list)
    for run in runs:
        by_source[run.source].append(run)
    report: list[SourceHealth] = []
    for source, source_runs in sorted(by_source.items()):
        latest = source_runs[0]
        zero_streak = 0
        for run in source_runs:
            if run.status.value == "ok" and run.leads_seen == 0:
                zero_streak += 1
            else:
                break
        ok_runs = [run for run in source_runs if run.status.value == "ok"]
        report.append(
            SourceHealth(
                source=source,
                last_run=latest.started_at,
                last_status=latest.status.value,
                last_error=latest.error,
                last_seen=latest.leads_seen,
                avg_seen=sum(run.leads_seen for run in ok_runs) / len(ok_runs) if ok_runs else 0.0,
                zero_streak=zero_streak,
            )
        )
    return report


@dataclass
class SkillStat:
    skill: str
    jobs: int
    required_in: int
    weighted: float
    have: bool


async def skill_report(
    session: AsyncSession, profile_hash: str, have: set[str], include_skipped: bool = False
) -> list[SkillStat]:
    """Skill frequency across relevant open jobs (newest extraction per job).

    `weighted` sums each job's value score (0.5 if unscored), so skills from jobs Ben actually wants count more.
    Jobs he skipped are excluded unless `include_skipped`.
    """
    newest = (
        select(JobExtraction.job_id, func.max(JobExtraction.created_at).label("created_at"))
        .group_by(JobExtraction.job_id)
        .subquery()
    )
    relevant = (
        select(JobExtraction.id, JobExtraction.job_id)
        .join(newest, and_(newest.c.job_id == JobExtraction.job_id, newest.c.created_at == JobExtraction.created_at))
        .join(Job, Job.id == JobExtraction.job_id)
        .where(and_(JobExtraction.is_relevant.is_(True), Job.status == JobStatus.OPEN))
        .subquery()
    )
    rows = (
        await session.execute(
            select(JobSkill.skill, JobSkill.required, JobSkill.job_id).join(
                relevant, relevant.c.id == JobSkill.extraction_id
            )
        )
    ).all()
    values = {
        score.job_id: score.value
        for score in (await session.execute(select(JobScore).where(JobScore.profile_hash == profile_hash)))
        .scalars()
        .all()
    }
    skipped = set()
    if not include_skipped:
        skipped = {
            job_id for job_id, review in (await latest_reviews(session)).items() if review.decision == Decision.SKIP
        }

    stats: dict[str, SkillStat] = {}
    for skill, required, job_id in rows:
        if job_id in skipped:
            continue
        stat = stats.setdefault(skill, SkillStat(skill, 0, 0, 0.0, skill.lower() in have))
        stat.jobs += 1
        stat.required_in += int(required)
        stat.weighted += values.get(job_id, 0.5)
    return sorted(stats.values(), key=lambda s: (-s.weighted, -s.jobs, s.skill))
