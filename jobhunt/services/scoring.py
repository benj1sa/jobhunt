"""Value and likelihood scoring.

Every component is a number in [0, 1] with a short human-readable note. Components are combined with the weights
from `config/profile.toml`, then eligibility gates multiply likelihood. Everything is stored so the review screen
can show *why* a job ranks where it does.
"""

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import JobStatus
from ..models.pipeline import Job, JobExtraction, JobScore
from ..questions import FitSignals
from ..schemas.extraction import Degree, JobRequirements, Seniority, WorkMode
from ..schemas.profile import Profile
from .fit import FitProvider
from .queue import load_description
from .skills import normalize_skills

logger = getLogger(__name__)

LEVEL_SCORE = {
    Seniority.NEW_GRAD: 1.0,
    Seniority.ENTRY: 0.85,
    Seniority.UNKNOWN: 0.6,
    Seniority.MID: 0.25,
    Seniority.INTERN: 0.1,
    Seniority.SENIOR: 0.05,
}


@dataclass
class Component:
    score: float
    weight: float
    note: str


@dataclass
class ScoreResult:
    value: float
    likelihood: float
    priority: float
    value_parts: dict[str, Component] = field(default_factory=dict)
    likelihood_parts: dict[str, Component] = field(default_factory=dict)
    gates: list[str] = field(default_factory=list)
    fit: dict[str, Any] | None = None

    def components_json(self) -> dict[str, Any]:
        return {
            "value": {name: asdict(part) for name, part in self.value_parts.items()},
            "likelihood": {name: asdict(part) for name, part in self.likelihood_parts.items()},
            "gates": self.gates,
            "fit": self.fit,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _name_in(name: str, candidates: list[str]) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return False
    return any(c.strip().lower() in lowered or lowered in c.strip().lower() for c in candidates if c.strip())


def compute_score(
    job: Job,
    requirements: JobRequirements,
    profile: Profile,
    aliases: dict[str, str],
    fit: FitSignals | None,
    now: datetime | None = None,
) -> ScoreResult:
    """Pure function: score one job for one profile. No I/O, so it is easy to test and to explain."""
    now = now or datetime.now(tz=UTC)
    candidate = profile.candidate
    vw = profile.weights.value
    lw = profile.weights.likelihood

    # --- value -------------------------------------------------------------------------------------------------
    value_parts: dict[str, Component] = {}
    if fit is None:
        value_parts["interest"] = Component(0.5, vw.interest, "no fit provider run; neutral")
        value_parts["learning"] = Component(0.5, vw.learning, "no fit provider run; neutral")
    else:
        value_parts["interest"] = Component(_clamp(fit.interest), vw.interest, fit.interest_note)
        value_parts["learning"] = Component(_clamp(fit.learning), vw.learning, fit.learning_note)

    domain_hit = _name_in(requirements.domain, candidate.preferred_domains) if requirements.domain else False
    value_parts["domain"] = Component(
        1.0 if domain_hit else 0.4, vw.domain, f"{requirements.domain}{' (preferred)' if domain_hit else ''}"
    )

    salaries = [s for s in (requirements.salary_min_usd, requirements.salary_max_usd) if s]
    if salaries:
        midpoint = sum(salaries) / len(salaries)
        span = max(1, profile.compensation.target_usd - profile.compensation.floor_usd)
        comp = _clamp((midpoint - profile.compensation.floor_usd) / span)
        value_parts["compensation"] = Component(comp, vw.compensation, f"~${midpoint:,.0f}")
    else:
        value_parts["compensation"] = Component(0.5, vw.compensation, "salary not listed")

    dream = _name_in(job.company, profile.companies.dream)
    value_parts["company"] = Component(1.0 if dream else 0.5, vw.company, "dream company" if dream else "")

    locations = requirements.locations or ([job.location] if job.location else [])
    remote = requirements.work_mode == WorkMode.REMOTE or bool(job.is_remote)
    preferred_location = remote or any(_name_in(loc, candidate.preferred_locations) for loc in locations)
    value_parts["location"] = Component(
        1.0 if preferred_location else 0.3,
        vw.location,
        "remote" if remote else ("; ".join(locations)[:80] or "unknown"),
    )

    # --- likelihood --------------------------------------------------------------------------------------------
    likelihood_parts: dict[str, Component] = {}
    have = {s.lower() for s in normalize_skills(candidate.skills, aliases)}
    required = normalize_skills(requirements.required_skills, aliases)
    preferred = normalize_skills(requirements.preferred_skills, aliases)
    req_hits = [s for s in required if s.lower() in have]
    pref_hits = [s for s in preferred if s.lower() in have]
    if required or preferred:
        req_ratio = len(req_hits) / len(required) if required else 1.0
        pref_ratio = len(pref_hits) / len(preferred) if preferred else req_ratio
        skills_score = 0.8 * req_ratio + 0.2 * pref_ratio
        note = f"{len(req_hits)}/{len(required)} required, {len(pref_hits)}/{len(preferred)} preferred"
    else:
        skills_score, note = 0.6, "no skills listed"
    likelihood_parts["skills"] = Component(skills_score, lw.skills, note)

    years = requirements.min_years_experience
    gap = max(0.0, (years or 0) - candidate.years_experience)
    likelihood_parts["experience"] = Component(
        _clamp(1.0 - 0.3 * gap), lw.experience, "not stated" if years is None else f"{years}+ years asked"
    )
    likelihood_parts["level"] = Component(LEVEL_SCORE[requirements.seniority], lw.level, requirements.seniority.value)

    seen = job.posted_at or job.first_seen_at
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=UTC)
    age_days = max(0, (now - seen).days)
    freshness = (
        1.0 if age_days <= 3 else 0.85 if age_days <= 7 else 0.65 if age_days <= 14 else 0.4 if age_days <= 30 else 0.2
    )
    likelihood_parts["freshness"] = Component(freshness, lw.freshness, f"{age_days} days old")

    crowded = _name_in(job.company, profile.companies.high_competition)
    likelihood_parts["competition"] = Component(
        0.25 if crowded else 0.7, lw.competition, "high-volume brand" if crowded else ""
    )
    warm = _name_in(job.company, profile.companies.warm)
    likelihood_parts["warm"] = Component(1.0 if warm else 0.0, lw.warm, "warm contact" if warm else "")

    # --- gates -------------------------------------------------------------------------------------------------
    gate = 1.0
    gates: list[str] = []
    if requirements.requires_clearance and not candidate.has_clearance:
        gate *= 0.2
        gates.append("active clearance required (x0.2)")
    if requirements.requires_us_citizenship and not candidate.us_citizen:
        gate *= 0.0
        gates.append("US citizenship required (x0)")
    if candidate.needs_sponsorship and requirements.sponsors_visa.value == "no":
        gate *= 0.0
        gates.append("no sponsorship (x0)")
    if requirements.degree_required in (Degree.MASTERS, Degree.PHD) and candidate.degree == "bachelors":
        gate *= 0.3
        gates.append(f"{requirements.degree_required.value} required (x0.3)")

    value = sum(part.score * part.weight for part in value_parts.values())
    if _name_in(job.company, profile.companies.avoid):
        value = 0.0
        gates.append("company on avoid list (value 0)")
    likelihood = sum(part.score * part.weight for part in likelihood_parts.values()) * gate

    return ScoreResult(
        value=round(value, 4),
        likelihood=round(likelihood, 4),
        priority=round(math.sqrt(value * likelihood), 4),
        value_parts=value_parts,
        likelihood_parts=likelihood_parts,
        gates=gates,
        fit=asdict(fit) if fit else None,
    )


@dataclass
class ScoreStats:
    scored: int = 0
    fit_calls: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


async def _cached_fit(session: AsyncSession, job: Job, cache_key: str) -> FitSignals | None:
    rows = (await session.execute(select(JobScore.components).where(JobScore.job_id == job.id))).scalars().all()
    for components in rows:
        fit = components.get("fit")
        if fit and components.get("fit_key") == cache_key:
            return FitSignals(**fit)
    return None


async def latest_relevant_extractions(session: AsyncSession) -> list[tuple[Job, JobExtraction]]:
    newest = (
        select(JobExtraction.job_id, func.max(JobExtraction.created_at).label("created_at"))
        .group_by(JobExtraction.job_id)
        .subquery()
    )
    stmt = (
        select(Job, JobExtraction)
        .join(JobExtraction, JobExtraction.job_id == Job.id)
        .join(newest, and_(newest.c.job_id == JobExtraction.job_id, newest.c.created_at == JobExtraction.created_at))
        .where(and_(Job.status == JobStatus.OPEN, JobExtraction.is_relevant.is_(True)))
        .order_by(desc(Job.first_seen_at))
    )
    return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]


async def score_jobs(
    session: AsyncSession,
    profile: Profile,
    aliases: dict[str, str],
    fit_provider: FitProvider | None,
    rescore: bool = False,
) -> ScoreStats:
    """Score every relevant open job that has no score for the current profile hash (or all with `rescore`).

    A failed fit call fails that job's score (it is retried on the next run) rather than scoring it as neutral.
    """
    profile_hash = profile.content_hash
    cache_key = fit_provider.cache_key(profile) if fit_provider else None
    stats = ScoreStats()
    for job, extraction in await latest_relevant_extractions(session):
        existing = (
            await session.execute(
                select(JobScore).where(and_(JobScore.job_id == job.id, JobScore.profile_hash == profile_hash))
            )
        ).scalar_one_or_none()
        if existing is not None and not rescore:
            continue
        try:
            requirements = JobRequirements.model_validate(extraction.data)
            fit = await _cached_fit(session, job, cache_key) if cache_key else None
            if fit is None and fit_provider is not None:
                description = await load_description(session, job.id) or requirements.summary
                fit = await fit_provider.judge(job, requirements, description, profile)
                stats.fit_calls += 1
            result = compute_score(job, requirements, profile, aliases, fit)
        except Exception as exc:
            logger.exception("Scoring failed for job %s", job.id)
            stats.failed += 1
            stats.errors.append(f"{job.company} - {job.title}: {type(exc).__name__}: {exc}"[:500])
            continue
        components = result.components_json() | {"fit_key": cache_key if fit else None}
        if existing is not None:
            existing.value, existing.likelihood, existing.priority = result.value, result.likelihood, result.priority
            existing.components = components
        else:
            session.add(
                JobScore(
                    job_id=job.id,
                    profile_hash=profile_hash,
                    value=result.value,
                    likelihood=result.likelihood,
                    priority=result.priority,
                    components=components,
                )
            )
        await session.commit()
        stats.scored += 1
    return stats
