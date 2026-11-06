"""`jobhunt status`: what is waiting at each pipeline step, which steps call a model, and what they will cost.

Per-job token and cost figures come from `llm_usage` (what past runs actually used). Until a step has history,
the defaults below are used; they were measured on the first few hundred real jobs.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import JobStatus, LeadStatus, ScreenOutcome
from ..models.pipeline import Job, JobScore, JobScreen, JobSnapshot, Lead, LlmUsage
from ..questions import SCREEN_VERSION
from .extraction import latest_snapshots_needing_extraction
from .queue import build_queue
from .scoring import latest_relevant_extractions
from .screening import snapshots_needing_screen
from .usage import price

HISTORY_ROWS = 20  # recent llm_usage rows per step used for the per-job average


@dataclass(frozen=True)
class PerJob:
    """Typical usage for one job at one step."""

    input_tokens: float
    output_tokens: float
    cost: float
    from_history: bool


def default_per_job(stage: str, model: str) -> PerJob:
    tokens = {
        "screen": (2_500, 0),
        "extract": (5_200, 1_000),
        "fit": (2_500, 0) if model.startswith("jev") else (1_500, 300),
    }.get(stage, (2_000, 500))
    return PerJob(tokens[0], tokens[1], price(model, *tokens), from_history=False)


async def per_job_usage(session: AsyncSession, stage: str, model: str) -> PerJob:
    rows = (
        (
            await session.execute(
                select(LlmUsage)
                .where(
                    and_(
                        LlmUsage.stage == stage,
                        # Jev reports its resolved version (jev-1.14.0) for the jev-latest alias.
                        LlmUsage.model.like("jev%") if model.startswith("jev") else LlmUsage.model == model,
                    )
                )
                .order_by(LlmUsage.created_at.desc())
                .limit(HISTORY_ROWS)
            )
        )
        .scalars()
        .all()
    )
    items = sum(row.items for row in rows)
    if not items:
        return default_per_job(stage, model)
    return PerJob(
        sum(row.input_tokens for row in rows) / items,
        sum(row.output_tokens for row in rows) / items,
        sum(row.cost_usd for row in rows) / items,
        from_history=True,
    )


@dataclass
class StepStatus:
    command: str
    what: str
    model: str | None  # None: no AI in this step
    pending: int
    note: str = ""
    per_job: PerJob | None = None

    @property
    def est_input_tokens(self) -> float:
        return self.per_job.input_tokens * self.pending if self.per_job else 0.0

    @property
    def est_output_tokens(self) -> float:
        return self.per_job.output_tokens * self.pending if self.per_job else 0.0

    @property
    def est_cost(self) -> float:
        return self.per_job.cost * self.pending if self.per_job else 0.0


@dataclass
class Status:
    steps: list[StepStatus]
    queue_size: int
    needs_link: int
    spent_7d: dict[str, float] = field(default_factory=dict)
    spent_30d: float = 0.0
    next_command: str = ""

    @property
    def est_total(self) -> float:
        return sum(step.est_cost for step in self.steps)


async def _count(session: AsyncSession, stmt: object) -> int:
    return int((await session.execute(stmt)).scalar_one())  # type: ignore[call-overload]


async def screen_reject_rate(session: AsyncSession) -> float:
    total = await _count(
        session, select(func.count()).select_from(JobScreen).where(JobScreen.version == SCREEN_VERSION)
    )
    if total < 20:
        return 0.5
    rejected = await _count(
        session,
        select(func.count())
        .select_from(JobScreen)
        .where(and_(JobScreen.version == SCREEN_VERSION, JobScreen.outcome == ScreenOutcome.REJECT)),
    )
    return rejected / total


async def gather_status(
    session: AsyncSession,
    profile_hash: str,
    screen_model: str | None,
    extract_model: str | None,
    fit_model: str | None,
) -> Status:
    """`*_model` is None when that step is switched off (no API key, or fit provider 'none')."""
    new_leads = await _count(session, select(func.count()).select_from(Lead).where(Lead.status == LeadStatus.NEW))
    needs_link = await _count(
        session,
        select(func.count()).select_from(Lead).where(Lead.status.in_([LeadStatus.NEEDS_URL, LeadStatus.FAILED])),
    )
    has_snapshot = exists().where(JobSnapshot.job_id == Job.id)
    to_hydrate = await _count(
        session, select(func.count()).select_from(Job).where(and_(Job.status == JobStatus.OPEN, ~has_snapshot))
    )
    to_screen = len(await snapshots_needing_screen(session, None)) if screen_model else 0
    to_extract = len(await latest_snapshots_needing_extraction(session, extract_model, None)) if extract_model else 0
    scored = set(
        (await session.execute(select(JobScore.job_id).where(JobScore.profile_hash == profile_hash))).scalars()
    )
    to_score = sum(1 for job, _ in await latest_relevant_extractions(session) if job.id not in scored)

    reject_rate = await screen_reject_rate(session)
    # Unscreened snapshots count as extraction work today, but about `reject_rate` of them will be screened out.
    extract_after_screen = max(0, round(to_extract - to_screen * reject_rate))

    steps = [
        StepStatus("ingest", "find new postings on every source and watched board", None, 0, "always runs"),
        StepStatus("resolve", "turn new leads into jobs (one per posting)", None, new_leads),
        StepStatus("hydrate", "download full descriptions", None, to_hydrate),
        StepStatus(
            "screen",
            "Jev yes/no checks: US? entry level? relevant role?",
            screen_model,
            to_screen,
            "" if screen_model else "off: no TYPESAFE_API_KEY",
        ),
        StepStatus(
            "extract",
            "LLM pulls skills, requirements and capability phrases",
            extract_model,
            extract_after_screen,
            (f"~{reject_rate:.0%} of new jobs get screened out first" if to_screen else "")
            if extract_model
            else "off: no OPENROUTER_API_KEY",
        ),
        StepStatus(
            "score",
            "value x likelihood; the model judges fit to your profile",
            fit_model,
            to_score,
            "" if fit_model else "rule-based only (no fit provider)",
        ),
    ]
    stage_for = {"screen": "screen", "extract": "extract", "score": "fit"}
    for step in steps:
        if step.model:
            step.per_job = await per_job_usage(session, stage_for[step.command], step.model)
    # Jobs that will appear after hydrate/screen/extract also need a score; count them so the total isn't low.
    if fit_model and extract_after_screen:
        steps[-1].note = (steps[-1].note + "; " if steps[-1].note else "") + "+ the jobs extract finds relevant"

    queue_size = len(await build_queue(session, profile_hash))

    now = datetime.now(UTC)
    spent_7d: dict[str, float] = {}
    for stage, cost in (
        await session.execute(
            select(LlmUsage.stage, func.sum(LlmUsage.cost_usd))
            .where(LlmUsage.created_at >= now - timedelta(days=7))
            .group_by(LlmUsage.stage)
        )
    ).all():
        spent_7d[stage] = float(cost or 0)
    spent_30d = float(
        (
            await session.execute(
                select(func.coalesce(func.sum(LlmUsage.cost_usd), 0)).where(
                    LlmUsage.created_at >= now - timedelta(days=30)
                )
            )
        ).scalar_one()
    )

    if new_leads or to_hydrate or to_screen or to_extract or to_score:
        next_command = "jobhunt run --skip-ingest   (finish what's waiting, then `jobhunt review`)"
    elif queue_size:
        next_command = "jobhunt review"
    else:
        next_command = "jobhunt run (nothing waiting; this checks the sources for new postings)"
    return Status(steps, queue_size, needs_link, spent_7d, spent_30d, next_command)
