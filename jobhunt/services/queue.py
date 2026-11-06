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
from ..models.pipeline import (
    IngestRun,
    Job,
    JobCapability,
    JobExtraction,
    JobScore,
    JobScreen,
    JobSkill,
    JobSnapshot,
    Lead,
    Review,
)
from ..questions import SCREEN_VERSION
from ..schemas.config import SkillTaxonomy, skill_key


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
        # The current screen is the gate: a reject also hides jobs extracted before a new screen question existed
        # (e.g. degree requirements). `jobhunt screened-out` lists them in case Jev got one wrong.
        if job.id in screened_out:
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


async def find_job(session: AsyncSession, ref: str) -> Job:
    """A job by full id or a unique prefix of it (dashes optional), e.g. 'd58dfee2'."""
    needle = ref.strip().lower().replace("-", "")
    if len(needle) < 4 or any(ch not in "0123456789abcdef" for ch in needle):
        raise LookupError(f"{ref!r} doesn't look like a job id (use at least the first 4 characters).")
    ids = [job_id for job_id in (await session.execute(select(Job.id))).scalars() if job_id.hex.startswith(needle)]
    if not ids:
        raise LookupError(f"No job id starts with {ref!r}.")
    if len(ids) > 1:
        raise LookupError(f"{len(ids)} jobs start with {ref!r}; give more characters.")
    job = await session.get(Job, ids[0])
    assert job is not None
    return job


@dataclass
class JobDetail:
    job: Job
    sources: list[str]
    screen: JobScreen | None
    extraction: JobExtraction | None
    score: JobScore | None
    review: Review | None
    description: str | None


async def job_detail(session: AsyncSession, job: Job, profile_hash: str) -> JobDetail:
    async def newest(model: Any, *where: Any) -> Any:
        stmt = select(model).where(model.job_id == job.id, *where).order_by(desc(model.created_at)).limit(1)
        return (await session.execute(stmt)).scalar_one_or_none()

    sources = sorted(set((await session.execute(select(Lead.source).where(Lead.job_id == job.id))).scalars()))
    return JobDetail(
        job=job,
        sources=sources,
        screen=await newest(JobScreen, JobScreen.version == SCREEN_VERSION),
        extraction=await newest(JobExtraction),
        score=await newest(JobScore, JobScore.profile_hash == profile_hash),
        review=await newest(Review),
        description=await load_description(session, job.id),
    )


async def screened_out_jobs(session: AsyncSession, reason: str | None = None) -> list[tuple[Job, JobScreen]]:
    """Open jobs the current screen rejected (newest first), optionally only those whose reason contains `reason`."""
    stmt = (
        select(Job, JobScreen)
        .join(JobScreen, JobScreen.job_id == Job.id)
        .where(
            and_(
                Job.status == JobStatus.OPEN,
                JobScreen.version == SCREEN_VERSION,
                JobScreen.outcome == ScreenOutcome.REJECT,
            )
        )
        .order_by(desc(JobScreen.created_at))
    )
    rows = [(row[0], row[1]) for row in (await session.execute(stmt)).all()]
    seen: set[UUID] = set()
    result = []
    for job, screen in rows:
        if job.id in seen or (reason and reason.lower() not in screen.reason.lower()):
            continue
        seen.add(job.id)
        result.append((job, screen))
    return result


async def keep_screened_job(session: AsyncSession, job_id: UUID) -> int:
    """Override the screen for one job (Jev got it wrong): its rejects become passes, so it is extracted and queued."""
    screens = (
        (
            await session.execute(
                select(JobScreen).where(
                    and_(
                        JobScreen.job_id == job_id,
                        JobScreen.version == SCREEN_VERSION,
                        JobScreen.outcome == ScreenOutcome.REJECT,
                    )
                )
            )
        )
        .scalars()
        .all()
    )
    for screen in screens:
        screen.outcome = ScreenOutcome.PASS
        screen.reason = f"kept by hand (screen said: {screen.reason})"[:2000]
    await session.commit()
    return len(screens)


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
    last_fetched: int | None
    last_seen: int
    avg_seen: float
    zero_streak: int
    configured: bool = True

    @property
    def healthy(self) -> bool:
        """Removed sources never count. Otherwise: last run succeeded and it hasn't come back empty 3 times running."""
        if not self.configured:
            return True
        return self.last_status == "ok" and self.zero_streak < 3


def _fetched_nothing(run: IngestRun) -> bool:
    # Prefer the raw count: a board with postings but no entry-level matches is healthy.
    return run.items_fetched == 0 if run.items_fetched is not None else run.leads_seen == 0


async def source_health(
    session: AsyncSession, window_days: int = 30, configured: set[str] | None = None
) -> list[SourceHealth]:
    """Per-source summary of recent ingest runs.

    A source is unhealthy when its last run failed, or when it fetched nothing (not merely matched nothing) three
    runs in a row. Sources missing from `configured` are reported as removed and never count as unhealthy.
    """
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
            if run.status.value == "ok" and _fetched_nothing(run):
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
                last_fetched=latest.items_fetched,
                last_seen=latest.leads_seen,
                avg_seen=sum(run.leads_seen for run in ok_runs) / len(ok_runs) if ok_runs else 0.0,
                zero_streak=zero_streak,
                configured=configured is None or source in configured,
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
    group: str | None = None
    kind: str | None = None


async def _relevant_extractions(session: AsyncSession) -> Any:
    """Subquery of (extraction id, job id) for the newest relevant extraction of each open job."""
    newest = (
        select(JobExtraction.job_id, func.max(JobExtraction.created_at).label("created_at"))
        .group_by(JobExtraction.job_id)
        .subquery()
    )
    return (
        select(JobExtraction.id, JobExtraction.job_id)
        .join(newest, and_(newest.c.job_id == JobExtraction.job_id, newest.c.created_at == JobExtraction.created_at))
        .join(Job, Job.id == JobExtraction.job_id)
        .where(and_(JobExtraction.is_relevant.is_(True), Job.status == JobStatus.OPEN))
        .subquery()
    )


async def _job_values(session: AsyncSession, profile_hash: str) -> dict[UUID, float]:
    scores = (await session.execute(select(JobScore).where(JobScore.profile_hash == profile_hash))).scalars().all()
    return {score.job_id: score.value for score in scores}


async def _skipped_jobs(session: AsyncSession) -> set[UUID]:
    return {job_id for job_id, review in (await latest_reviews(session)).items() if review.decision == Decision.SKIP}


async def skill_report(
    session: AsyncSession,
    profile_hash: str,
    have: set[str],
    include_skipped: bool = False,
    taxonomy: SkillTaxonomy | None = None,
) -> list[SkillStat]:
    """Skill frequency across relevant open jobs (newest extraction per job).

    Spellings are merged case- and punctuation-insensitively and through `same_as` in the taxonomy. A job counts
    once per skill. `weighted` sums each job's value score (0.5 if unscored), so skills from jobs Ben actually
    wants count more. Jobs he skipped are excluded unless `include_skipped`.
    """
    relevant = await _relevant_extractions(session)
    rows = (
        await session.execute(
            select(JobSkill.skill, JobSkill.required, JobSkill.job_id).join(
                relevant, relevant.c.id == JobSkill.extraction_id
            )
        )
    ).all()
    values = await _job_values(session, profile_hash)
    skipped = set() if include_skipped else await _skipped_jobs(session)
    have_keys = {skill_key(name) for name in have}

    stats: dict[str, SkillStat] = {}
    seen: dict[str, set[UUID]] = defaultdict(set)
    for skill, required, job_id in rows:
        if job_id in skipped:
            continue
        cls = taxonomy.classify(skill) if taxonomy else None
        name = cls.name if cls else skill
        key = skill_key(name)
        stat = stats.setdefault(
            key,
            SkillStat(
                name, 0, 0, 0.0, key in have_keys or skill_key(skill) in have_keys, cls.group if cls else None,
                cls.kind if cls else None,
            ),
        )  # fmt: skip
        if job_id in seen[key]:
            continue
        seen[key].add(job_id)
        stat.jobs += 1
        stat.required_in += int(required)
        stat.weighted += values.get(job_id, 0.5)
    return sorted(stats.values(), key=lambda s: (-s.weighted, -s.jobs, s.skill))


@dataclass
class GroupStat:
    group: str
    name: str
    jobs: int = 0
    weighted: float = 0.0
    concept_only_jobs: int = 0
    tools: list[SkillStat] = field(default_factory=list)
    concepts: list[SkillStat] = field(default_factory=list)


async def skill_group_report(
    session: AsyncSession,
    profile_hash: str,
    have: set[str],
    taxonomy: SkillTaxonomy,
    include_skipped: bool = False,
) -> tuple[list[GroupStat], list[SkillStat]]:
    """Skills rolled up into groups, plus the skills no group claims yet.

    A group's `jobs` counts jobs that mention any of its skills, so a broad "AI" and a specific "SageMaker" in
    the same posting count once. `concept_only_jobs` are jobs that named only broad concepts in the group, never a
    specific tool: a hint that the posting cares about the area, not a product.
    """
    stats = await skill_report(session, profile_hash, have, include_skipped, taxonomy)
    relevant = await _relevant_extractions(session)
    rows = (
        await session.execute(
            select(JobSkill.skill, JobSkill.job_id).join(relevant, relevant.c.id == JobSkill.extraction_id)
        )
    ).all()
    values = await _job_values(session, profile_hash)
    skipped = set() if include_skipped else await _skipped_jobs(session)

    group_jobs: dict[str, set[UUID]] = defaultdict(set)
    tool_jobs: dict[str, set[UUID]] = defaultdict(set)
    for skill, job_id in rows:
        if job_id in skipped:
            continue
        cls = taxonomy.classify(skill)
        if cls is None:
            continue
        group_jobs[cls.group].add(job_id)
        if cls.kind == "tool":
            tool_jobs[cls.group].add(job_id)

    groups: dict[str, GroupStat] = {
        gid: GroupStat(gid, name) for gid, name in taxonomy.groups.items() if gid in group_jobs
    }
    for gid, stat in groups.items():
        jobs = group_jobs[gid]
        stat.jobs = len(jobs)
        stat.weighted = sum(values.get(job_id, 0.5) for job_id in jobs)
        stat.concept_only_jobs = len(jobs - tool_jobs[gid])
    unsorted: list[SkillStat] = []
    for stat_row in stats:
        if stat_row.group is None:
            unsorted.append(stat_row)
        elif stat_row.kind == "tool":
            groups[stat_row.group].tools.append(stat_row)
        else:
            groups[stat_row.group].concepts.append(stat_row)
    for group_stat in groups.values():
        group_stat.tools.sort(key=lambda t: (-t.jobs, t.skill))
        group_stat.concepts.sort(key=lambda c: (-c.jobs, c.skill))
    return sorted(groups.values(), key=lambda g: (-g.weighted, g.name)), unsorted


@dataclass
class CapabilityStat:
    capability: str
    jobs: int = 0
    weighted: float = 0.0
    phrases: int = 0
    example: str = ""


@dataclass
class CapabilityPhrase:
    capability: str
    phrase: str
    verbatim: bool
    company: str
    title: str
    job_id: UUID


async def capability_phrases(
    session: AsyncSession, capability: str | None = None, include_skipped: bool = False
) -> list[CapabilityPhrase]:
    """Every phrase behind one capability (or all capabilities), newest relevant extraction per open job."""
    relevant = await _relevant_extractions(session)
    stmt = (
        select(JobCapability.capability, JobCapability.phrase, JobCapability.verbatim, Job.company, Job.title, Job.id)
        .join(relevant, relevant.c.id == JobCapability.extraction_id)
        .join(Job, Job.id == JobCapability.job_id)
        .order_by(Job.company, Job.title)
    )
    if capability:
        stmt = stmt.where(JobCapability.capability == capability)
    skipped = set() if include_skipped else await _skipped_jobs(session)
    return [CapabilityPhrase(*row) for row in (await session.execute(stmt)).all() if row[5] not in skipped]


async def capability_report(
    session: AsyncSession, profile_hash: str, include_skipped: bool = False
) -> list[CapabilityStat]:
    """How many relevant jobs ask for each capability, weighted by job value like the skills report."""
    phrases = await capability_phrases(session, include_skipped=include_skipped)
    values = await _job_values(session, profile_hash)
    jobs: dict[str, set[UUID]] = defaultdict(set)
    stats: dict[str, CapabilityStat] = {}
    for item in phrases:
        stat = stats.setdefault(item.capability, CapabilityStat(item.capability))
        stat.phrases += 1
        if not stat.example or (item.verbatim and len(item.phrase) < len(stat.example)):
            stat.example = item.phrase
        jobs[item.capability].add(item.job_id)
    for key, stat in stats.items():
        stat.jobs = len(jobs[key])
        stat.weighted = sum(values.get(job_id, 0.5) for job_id in jobs[key])
    return sorted(stats.values(), key=lambda s: (-s.weighted, s.capability))
