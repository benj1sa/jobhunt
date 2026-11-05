"""LLM extraction of structured requirements from job-description snapshots."""

from dataclasses import dataclass, field
from logging import getLogger

from sqlalchemy import and_, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import JobStatus, ScreenOutcome
from ..models.pipeline import Job, JobExtraction, JobScreen, JobSkill, JobSnapshot
from ..questions import SCREEN_VERSION
from ..schemas.extraction import EXTRACTION_SCHEMA_VERSION, JobRequirements
from .filtering import judge_relevance
from .llm import OpenRouterClient
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
- Salary is annual base in USD; convert hourly by x2080. null if absent."""


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
    session: AsyncSession, model: str, limit: int | None
) -> list[tuple[Job, JobSnapshot]]:
    """Open jobs whose newest snapshot has no extraction for this model and schema version."""
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
        .where(and_(Job.status == JobStatus.OPEN, ~already, ~rejected))
        .order_by(Job.first_seen_at)
    )
    if limit:
        stmt = stmt.limit(limit)
    return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]


async def store_extraction(
    session: AsyncSession,
    job: Job,
    snapshot: JobSnapshot,
    model: str,
    requirements: JobRequirements,
    aliases: dict[str, str],
) -> JobExtraction:
    relevance = judge_relevance(requirements)
    extraction = JobExtraction(
        job_id=job.id,
        snapshot_id=snapshot.id,
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
        session.add(JobSkill(job_id=job.id, extraction_id=extraction.id, skill=skill, required=True))
    for skill in preferred:
        session.add(JobSkill(job_id=job.id, extraction_id=extraction.id, skill=skill, required=False))
    return extraction


async def extract_jobs(
    session: AsyncSession,
    llm: OpenRouterClient,
    model: str,
    aliases: dict[str, str],
    max_chars: int,
    limit: int | None = None,
) -> ExtractStats:
    """Extract requirements for every open job's newest unextracted snapshot. Commits per job."""
    stats = ExtractStats()
    for job, snapshot in await latest_snapshots_needing_extraction(session, model, limit):
        try:
            requirements = await llm.structured(
                model=model,
                system=EXTRACTION_SYSTEM,
                user=build_prompt(job, snapshot, max_chars),
                schema=JobRequirements,
            )
            extraction = await store_extraction(session, job, snapshot, model, requirements, aliases)
            await session.commit()
        except Exception as exc:
            logger.exception("Extraction failed for job %s (%s)", job.id, job.posting_url)
            await session.rollback()
            stats.failed += 1
            stats.errors.append(f"{job.company} - {job.title}: {type(exc).__name__}: {exc}"[:500])
            continue
        stats.extracted += 1
        stats.relevant += int(extraction.is_relevant)
    return stats
