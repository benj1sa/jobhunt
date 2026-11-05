# Design: jobhunt job-search pipeline

Scope: discovery from many sources, deduplication into canonical jobs, description snapshots, LLM extraction of
requirements and skills, value/likelihood scoring, a terminal review queue, and a scheduled GitHub Actions run.

## Context

The project was generated from Rob's Awesome Python Template with `include_cli`, `include_sqlalchemy` and
`include_github_actions`. It follows the template's conventions: async SQLAlchemy 2.0 with `mapped_column`,
Alembic migrations that work on SQLite and PostgreSQL, one Pydantic `Settings` class, Typer commands wrapped in
`@syncify`, a module-level `getLogger(__name__)` per file, and pytest with in-memory SQLite fixtures.

Before this project, Ben tracked roles by hand in a CSV (`Company, Role, Link, Source`) gathered from LinkedIn,
SimplifyJobs, BuiltIn, Idealist, Wellfound, Google Jobs, HN Who's Hiring, ColorStack and CodePath Slack, and ATS
boards. About half of those links already point at an applicant tracking system (Greenhouse, Workday); the rest
point at boards that block automation or at aggregator pages.

No `docs/dev/architecture.md` existed in the project. This document serves as the architecture reference.

## References

- [Greenhouse Job Board API](https://developers.greenhouse.io/job-board.html) — list (`?content=true`) and single-job endpoints; `content` is HTML-entity-encoded, which drives the double-unescape in `html_to_text`.
- [Lever Postings API](https://github.com/lever/postings-api) — `?mode=json` listing and single posting with `descriptionPlain`, `lists` and `additionalPlain`, combined into one description.
- [Ashby public job posting API](https://developers.ashbyhq.com/docs/public-job-posting-api) — board endpoint only (no single-posting endpoint), so hydration fetches the board and selects by id; `id` is undocumented, so it falls back to the `jobUrl` path.
- [SimplifyJobs New-Grad-Positions](https://github.com/SimplifyJobs/New-Grad-Positions) — `.github/scripts/listings.json`; 19,063 listings (3,244 active) observed 2026-11-05, with two spellings of the software and AI categories.
- [Algolia HN Search API](https://hn.algolia.com/api) — `search_by_date` with `author_whoishiring` to find the monthly thread and `items/{id}` for its comments.
- [OpenRouter structured outputs](https://openrouter.ai/docs/features/structured-outputs) — `response_format.json_schema` with `strict`, plus `provider.require_parameters` to avoid providers that ignore the schema.
- [Textual](https://textual.textualize.io/) — async TUI framework; its `run_test` pilot is used for the TUI test.
- [Rob's robs-design skill](https://github.com/tedivm/opencode-config/tree/main/skills/robs-design) — the format of this document.

## Goals / Non-Goals

**Goals:**

- One canonical row per real posting, regardless of how many sources mention it, keyed by ATS posting id
- Idempotent stages (`ingest`, `resolve`, `hydrate`, `extract`, `score`) that can each be re-run safely
- Full description text kept forever, deduplicated by content hash
- Structured requirements and normalized skills per job, with deterministic relevance rules
- Value and likelihood scores built from visible, weighted components configured in `config/profile.toml`
- A keyboard review queue that opens the original posting and records decisions with reasons
- A daily GitHub Actions run with per-source health checks
- Skill-frequency report weighted by job value, to choose which projects to build

**Non-Goals:**

- Auto-applying or filling application forms
- Scraping LinkedIn, Wellfound, Indeed or Glassdoor (blocked or prohibited; handled as `needs_url`)
- Parsing email alerts (Phase 2; see Risks)
- Workable, SmartRecruiters, iCIMS, Oracle adapters (Phase 2; those jobs are URL-keyed and hydrated from HTML)
- A hosted web UI or multi-user support

## Decisions

### D1: Separate discovery (leads) from canonical jobs

**Decision:** Every source produces `RawLead` objects, stored as `lead` rows unique on `(source, url_normalized)`.
A resolver maps each lead to one `job` keyed by `canonical_key = <ats>:<board>:<external_id>` (or
`url:<normalized url>` when the link is not on a known ATS). Many leads point at one job, so deduplication is a
side effect of resolution and every source a job was "found on" is kept.

```python
# jobhunt/services/urls.py
@dataclass(frozen=True)
class AtsRef:
    ats: Ats
    board: str
    external_id: str

    @property
    def canonical_key(self) -> str:
        return f"{self.ats.value}:{self.board.lower()}:{self.external_id}"
```

`parse_ats` recognizes Greenhouse (`job-boards`/`boards`, embed `?for=&token=`), Lever, Ashby and Workday URLs,
and company career sites that embed Greenhouse via `?gh_jid=` when the domain is mapped in `config/boards.toml`.

**Alternative considered:** one scraper per site that writes finished jobs. Rejected: dedupe and description logic
would be duplicated per site, and a site that blocks bots would lose the job entirely instead of leaving a lead.

---

### D2: SQLite, GitHub Actions as producer, laptop as consumer

**Decision:** The scheduled workflow runs `ingest → resolve → hydrate → extract` and force-pushes `jobhunt.db` as a
single commit to a `data` branch. Locally, `make pull-data` runs `jobhunt pull`, which merges the remote file into
the local database by natural keys (`canonical_key`, `(source, url_normalized)`, `(job, content_hash)`,
`(snapshot, model, schema_version)`). Scores and reviews are only ever written locally.

**Alternatives considered:** a shared database file edited on both sides (Git cannot merge binary edits);
Supabase/PostgreSQL (more secrets and moving parts than a single-user tool needs; the models still run on
PostgreSQL if that changes).

**Trade-offs:** The repository must be private. The `data` branch is force-pushed, so it has no history; the
`ingest_run` table is the history.

---

### D3: OpenRouter structured outputs, validated by Pydantic

**Decision:** `OpenRouterClient.structured()` sends `response_format.json_schema` with `strict: true` and
`provider.require_parameters: true`, strips code fences, validates with Pydantic, and retries once with the
validation error appended. Models are settings (`extraction_model`, `fit_model`). Without `OPENROUTER_API_KEY` the
pipeline runs rules-only and stops before extraction.

`strict_json_schema()` post-processes `model_json_schema()` so every object has `additionalProperties: false` and
every property is required, which strict mode demands.

**Alternative considered:** the OpenAI SDK pointed at OpenRouter. Rejected: one more dependency for a single POST,
and the shared `HttpClient` already provides retries and concurrency limits.

---

### D4: Relevance depends only on the job; eligibility is a likelihood gate

**Decision:** `judge_relevance(requirements)` decides relevance from the extraction alone (role family, seniority,
years, US). Candidate-specific facts (citizenship, sponsorship, clearance, degree) are multiplicative gates in
scoring. Editing the profile therefore never invalidates extractions.

---

### D5: Scores are weighted, visible components

**Decision:** `compute_score()` is a pure function. Value = interest, learning, domain, compensation, company,
location. Likelihood = skills, experience, level, freshness, competition, warm contact, then multiplied by gates.
`priority = sqrt(value × likelihood)`. Weights live in `config/profile.toml` and must sum to 1.0 per axis.
Components, gates and the fit rationale are stored as JSON on `job_score`, keyed by the profile hash.

The fit judgment is categorical (`strong | moderate | weak`), not a number, because LLM numeric scores are poorly
calibrated. It is cached by a hash of the interests text and model, so changing a weight rescores without new
LLM calls.

**Alternative considered:** a single LLM "match score". Rejected: opaque, unstable between runs, and impossible to
tune.

---

### D6: Snapshots are immutable and deduplicated

**Decision:** Hydration stores `job_snapshot(description_text, raw)` unique on `(job_id, content_hash)`, where the
hash ignores whitespace and case. Extraction references the snapshot it read. A 404 closes a job only if it was
fetched successfully before; a first-fetch 404 is recorded as a hydrate error (most likely a wrong board slug).

---

### D7: Blocked boards become manual work, not scraping

**Decision:** Links on `linkedin.com`, `wellfound.com`, `angel.co`, `indeed.com`, `glassdoor.com` and `lnkd.in`
resolve to `needs_url`. `jobhunt unresolved` lists them and `jobhunt resolve <lead-id> <url>` attaches the real
posting. Aggregators (BuiltIn, Idealist, Vaia) are fetched once to find an ATS link; on network failure the lead
still resolves to a URL-keyed job rather than failing.

---

### D8: Template conventions, with one tooling override

**Decision:** Code follows the template's AGENTS.md, except type hints use built-in generics (`list[str]`,
`dict[str, str]`). The template's ruff version auto-fixes `List`/`Dict` to built-ins, and CI enforces ruff, so the
tooling wins over the written guideline. Python 3.11+ is required (`StrEnum`, `tomllib`, `typing.Self`).

---

### D9: Jev for the screen and for fit; LLM keeps extraction

**Decision:** TypeSafe's Jev (a System One model: typed questions in, calibrated probabilities out) is used where the
work is judgment, not generation. All wording, thresholds and decision tables live in `jobhunt/questions.py`.

_Screen before extraction_ ("Screen with a Noul first"): one request per new snapshot asks four atomic questions,
Nouls for role type, internship and US location and a Score over four experience levels. Code decides:

| Signals                                                                              | Outcome     | Effect                       |
| ------------------------------------------------------------------------------------ | ----------- | ---------------------------- |
| any clear no: role < 0.3, internship > 0.7, US < 0.3, P(new grad or 1-2 years) < 0.3 | `reject`    | skip extraction              |
| all clear yes                                                                        | `pass`      | extract                      |
| anything else                                                                        | `unsettled` | extract (the LLM settles it) |
| API error or unusable answer                                                         | no row      | extract; retried next run    |

_Fit_: one Noul per `interest_topics` and `growth_goals` entry in the profile plus a mentorship Noul, all in one
request over `{title, company, description}`. Code combines them:
`interest = 1 - prod(1 - weight_i * p_i)`, `learning = 0.6 * mean(growth) + 0.4 * mentorship`. Fit is cached by a
hash of the topics, `FIT_VERSION` and the Jev model.

```python
# jobhunt/questions.py (excerpt)
def decide_screen(response: SystemOneResponse) -> ScreenDecision:
    level = response.score("experience_level")
    signals = {
        "software_ai_ml_role": response.noul("software_ai_ml_role"),
        "internship": response.noul("internship"),
        "us_location": response.noul("us_location"),
        "entry_level": level.level_probability(0) + level.level_probability(1),
    }
    ...
```

**Alternatives considered:** Jev for all of extraction. Rejected: skills, summary and domain are generated text,
which Jev does not produce, and finding new skills is the point of the skills report. Choosing skills from
`skills.toml` would hide every skill not already listed. A single "is this a good fit?" question was also rejected:
the skill guidance (and D5) puts the decision in code and asks Jev only for atomic judgments.

**Trade-offs:** `jev-latest` can change answers between releases, so pin `JEV_MODEL` once the questions are tuned.
The 0.7/0.3 thresholds are TypeSafe's starting points; tune them by comparing screen outcomes with extraction's
relevance verdicts and your review decisions. Every request payload can be written to disk with `jobhunt jev-probe`.

## Data Storage

Enums are stored as VARCHAR of their values via `db_enum()` (`native_enum=False`, `values_callable`) so the same
migration works on SQLite and PostgreSQL.

```python
# jobhunt/models/enums.py
from enum import StrEnum


class LeadStatus(StrEnum):
    """Lifecycle of a discovered lead on its way to becoming a canonical job."""

    NEW = "new"
    RESOLVED = "resolved"
    # The link points at a board we cannot read automatically (LinkedIn, Wellfound, ...).
    NEEDS_URL = "needs_url"
    FAILED = "failed"


class JobStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class Ats(StrEnum):
    """Where a canonical job's description is fetched from."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    # Anything else: identified by its normalized URL, no structured hydration.
    URL = "url"


class Decision(StrEnum):
    APPLY = "apply"
    APPLIED = "applied"
    LATER = "later"
    SKIP = "skip"


class RunStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


class ScreenOutcome(StrEnum):
    """Result of the Jev screen that runs before LLM extraction."""

    PASS = "pass"
    UNSETTLED = "unsettled"
    REJECT = "reject"
```

```python
# jobhunt/models/pipeline.py
"""Database models for the job pipeline.

Discovery writes `Lead` rows. Resolution links each lead to one canonical `Job`.
Hydration stores immutable `JobSnapshot` rows. Extraction, scoring and review all
hang off the job so that re-running any stage never loses earlier stages' data.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base
from .enums import Ats, Decision, JobStatus, LeadStatus, RunStatus, ScreenOutcome


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def db_enum(enum_cls: type[StrEnum]) -> Enum:
    """Portable enum column: stored as VARCHAR of the member *values* on both SQLite and PostgreSQL."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=16,
        values_callable=lambda members: [member.value for member in members],
    )


class IngestRun(Base):
    """One execution of one source adapter. Used for health checks (a source returning 0 leads is a red flag)."""

    __tablename__ = "ingest_run"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[RunStatus] = mapped_column(db_enum(RunStatus), nullable=False, default=RunStatus.RUNNING)
    leads_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    leads_new: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Job(Base):
    """A canonical job posting, deduplicated across every source that mentioned it.

    `canonical_key` is `<ats>:<board>:<external_id>` for ATS-hosted jobs and `url:<normalized url>` otherwise.
    """

    __tablename__ = "job"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    canonical_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    ats: Mapped[Ats] = mapped_column(db_enum(Ats), nullable=False)
    board: Mapped[str | None] = mapped_column(String(256), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    location: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_remote: Mapped[bool | None] = mapped_column(nullable=True)
    posting_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    apply_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[JobStatus] = mapped_column(db_enum(JobStatus), nullable=False, default=JobStatus.OPEN)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    hydrated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hydrate_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Lead(Base):
    """A single sighting of a job on a discovery source.

    Unique per (source, url_normalized) so re-ingesting the same feed is idempotent.
    """

    __tablename__ = "lead"
    __table_args__ = (UniqueConstraint("source", "url_normalized", name="uq_lead_source_url"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    url_normalized: Mapped[str] = mapped_column(String(2048), nullable=False)
    location: Mapped[str | None] = mapped_column(String(512), nullable=True)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[LeadStatus] = mapped_column(db_enum(LeadStatus), nullable=False, default=LeadStatus.NEW, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    job_id: Mapped[UUID | None] = mapped_column(ForeignKey("job.id", ondelete="SET NULL"), nullable=True, index=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class JobSnapshot(Base):
    """An immutable copy of a job description as fetched. Deduplicated per job by content hash."""

    __tablename__ = "job_snapshot"
    __table_args__ = (UniqueConstraint("job_id", "content_hash", name="uq_snapshot_job_hash"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    description_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class JobScreen(Base):
    """Jev's cheap pre-check of one snapshot, deciding whether it is worth LLM extraction.

    `signals` holds the probabilities the decision used; `answers` the full typed answers for audit and tuning.
    """

    __tablename__ = "job_screen"
    __table_args__ = (UniqueConstraint("snapshot_id", "version", name="uq_screen_snapshot_version"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    snapshot_id: Mapped[UUID] = mapped_column(ForeignKey("job_snapshot.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[ScreenOutcome] = mapped_column(db_enum(ScreenOutcome), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    signals: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class JobExtraction(Base):
    """Structured requirements extracted from one snapshot by one model and schema version.

    `data` holds a validated `jobhunt.schemas.extraction.JobRequirements` dump.
    """

    __tablename__ = "job_extraction"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "model", "schema_version", name="uq_extraction_snapshot_model_schema"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    snapshot_id: Mapped[UUID] = mapped_column(ForeignKey("job_snapshot.id", ondelete="CASCADE"), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    is_relevant: Mapped[bool] = mapped_column(nullable=False)
    relevance_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class JobSkill(Base):
    """A normalized skill mentioned by a job, denormalized from the extraction for fast frequency queries."""

    __tablename__ = "job_skill"
    __table_args__ = (UniqueConstraint("extraction_id", "skill", name="uq_job_skill_extraction_skill"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    extraction_id: Mapped[UUID] = mapped_column(
        ForeignKey("job_extraction.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    required: Mapped[bool] = mapped_column(nullable=False)


class JobScore(Base):
    """Value and likelihood for a job under one version of the candidate profile.

    `components` keeps every weighted part so the review screen can explain the score.
    """

    __tablename__ = "job_score"
    __table_args__ = (UniqueConstraint("job_id", "profile_hash", name="uq_score_job_profile"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    likelihood: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    components: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Review(Base):
    """A decision Ben made about a job. Append-only; the latest row per job is the current state."""

    __tablename__ = "review"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    decision: Mapped[Decision] = mapped_column(db_enum(Decision), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
```

## Data Structures

### LLM extraction and fit (`jobhunt/schemas/extraction.py`)

```python
# jobhunt/schemas/extraction.py
"""Structured outputs requested from the LLM.

These models double as the JSON Schema sent to OpenRouter (`strict: true`), so every field is required and
"unknown" is an explicit enum value or `null` rather than an omitted key.
"""

import copy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Bump when JobRequirements changes shape; old extractions are kept and re-run alongside.
EXTRACTION_SCHEMA_VERSION = 1


class RoleFamily(StrEnum):
    SOFTWARE = "software"
    AI_ML = "ai_ml"
    MLOPS_DATA = "mlops_data"
    OTHER = "other"


class Seniority(StrEnum):
    INTERN = "intern"
    NEW_GRAD = "new_grad"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    UNKNOWN = "unknown"


class Degree(StrEnum):
    NONE = "none"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD = "phd"
    UNKNOWN = "unknown"


class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Tristate(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class JobRequirements(BaseModel):
    """What a job description asks for. Extracted once per snapshot."""

    model_config = ConfigDict(extra="forbid")

    role_family: RoleFamily = Field(description="Closest role family for the job title and duties.")
    seniority: Seniority = Field(
        description="Level the posting targets. new_grad = explicitly for recent/upcoming graduates."
    )
    min_years_experience: int | None = Field(
        description="Minimum years of professional experience required, or null if not stated."
    )
    degree_required: Degree = Field(description="Minimum degree required (not preferred).")
    locations: list[str] = Field(description="Work locations as written, e.g. 'New York, NY' or 'Remote - US'.")
    us_based: bool = Field(description="True if at least one location is in the United States or remote-US.")
    work_mode: WorkMode = Field(description="Remote, hybrid or onsite.")
    requires_clearance: bool = Field(
        description="True if an active security clearance is required (not merely 'ability to obtain')."
    )
    requires_us_citizenship: bool = Field(description="True if US citizenship is required.")
    sponsors_visa: Tristate = Field(description="Whether the employer sponsors work visas.")
    salary_min_usd: int | None = Field(description="Lower bound of annual base salary in USD, or null.")
    salary_max_usd: int | None = Field(description="Upper bound of annual base salary in USD, or null.")
    required_skills: list[str] = Field(
        description="Concrete skills, languages, frameworks and tools listed as required. Short canonical names."
    )
    preferred_skills: list[str] = Field(description="Skills listed as preferred, nice-to-have or a plus.")
    domain: str = Field(description="Industry or problem domain in 1-3 words, e.g. 'healthcare', 'defense', 'fintech'.")
    summary: str = Field(description="One sentence describing what the person in this role would actually do.")


class Alignment(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class FitJudgment(BaseModel):
    """The LLM's qualitative read on how well a job matches the candidate's interests.

    Categorical rather than numeric: LLMs are inconsistent at calibrated 0-1 scores but reliable at buckets.
    """

    model_config = ConfigDict(extra="forbid")

    interest_alignment: Alignment = Field(description="How well the work matches the candidate's stated interests.")
    learning_potential: Alignment = Field(
        description="How much the role would grow the candidate toward their stated goals."
    )
    rationale: str = Field(description="Two sentences max explaining both judgments.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON Schema for `model` that satisfies OpenAI/OpenRouter `strict: true` rules.

    Strict mode requires `additionalProperties: false` and every property listed in `required` on every object,
    and rejects `title`/`default` keywords on some providers.
    """
    schema: dict[str, Any] = copy.deepcopy(model.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema
```

### Candidate profile (`jobhunt/schemas/profile.py`)

```python
# jobhunt/schemas/profile.py
"""The candidate profile, loaded from `config/profile.toml`.

Everything that makes scoring personal lives here, so changing a weight is a one-line diff and the
profile hash recorded on each score tells you which version produced it.
"""

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Topic(BaseModel):
    """A named interest or growth goal. Each becomes one Jev Noul, so describe it as observable work."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short identifier shown in score notes, e.g. 'healthcare'.")
    description: str = Field(
        description="What the person in the job would be doing, e.g. 'healthcare, biomedical or clinical problems'."
    )
    weight: float = Field(default=1.0, ge=0, le=1, description="How much a match counts (0-1).")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Candidate name, used only in LLM prompts.")
    graduation: str = Field(description="Expected graduation, free text, e.g. 'December 2026'.")
    years_experience: float = Field(default=0.0, description="Full-time professional experience in years.")
    us_citizen: bool = Field(description="Whether the candidate is a US citizen.")
    needs_sponsorship: bool = Field(default=False, description="Whether the candidate needs visa sponsorship.")
    has_clearance: bool = Field(default=False, description="Whether the candidate holds an active clearance.")
    degree: str = Field(default="bachelors", description="Highest degree held or in progress.")
    skills: list[str] = Field(description="Skills the candidate can credibly claim. Normalized like job skills.")
    interests: str = Field(description="Free-text paragraph of interests and goals, given to the fit model.")
    preferred_locations: list[str] = Field(
        default_factory=list,
        description="Substrings matched against job locations (case-insensitive), e.g. 'Maryland', 'Remote'.",
    )
    preferred_domains: list[str] = Field(default_factory=list, description="Domains that raise value.")
    interest_topics: list[Topic] = Field(
        default_factory=list,
        description="Kinds of work you want, judged one by one by Jev to compute the interest component.",
    )
    growth_goals: list[Topic] = Field(
        default_factory=list,
        description="What you want a first job to grow, judged one by one by Jev for the learning component.",
    )


class Companies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dream: list[str] = Field(default_factory=list, description="Companies that add full company-value points.")
    warm: list[str] = Field(default_factory=list, description="Companies where the candidate has a warm contact.")
    high_competition: list[str] = Field(
        default_factory=list, description="Brand-name employers with very high applicant volume."
    )
    avoid: list[str] = Field(default_factory=list, description="Companies to drop from the queue entirely.")


class ValueWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interest: float = 0.30
    learning: float = 0.20
    domain: float = 0.10
    compensation: float = 0.15
    company: float = 0.10
    location: float = 0.15


class LikelihoodWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: float = 0.30
    experience: float = 0.20
    level: float = 0.15
    freshness: float = 0.15
    competition: float = 0.10
    warm: float = 0.10


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: ValueWeights = Field(default_factory=ValueWeights)
    likelihood: LikelihoodWeights = Field(default_factory=LikelihoodWeights)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Weights":
        for name, group in (("value", self.value), ("likelihood", self.likelihood)):
            total = sum(group.model_dump().values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"weights.{name} must sum to 1.0 (got {total:.2f})")
        return self


class Compensation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_usd: int = Field(default=70000, description="Salary at or below which compensation value is 0.")
    target_usd: int = Field(default=120000, description="Salary at or above which compensation value is 1.")


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    companies: Companies = Field(default_factory=Companies)
    weights: Weights = Field(default_factory=Weights)
    compensation: Compensation = Field(default_factory=Compensation)

    @property
    def content_hash(self) -> str:
        """Stable hash of the profile; a new hash means scores must be recomputed."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_profile(path: Path) -> Profile:
    with path.open("rb") as handle:
        return Profile.model_validate(tomllib.load(handle))
```

### TypeSafe System One request and response (`jobhunt/schemas/typesafe.py`)

```python
# jobhunt/schemas/typesafe.py
"""Request and response models for TypeSafe's System One API (`POST /v1/systemone`).

Shapes follow https://docs.typesafe.ai/api/system-one.md: questions and answers are keyed by names we choose,
and each answer's `type` matches its question.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str = Field(description="A concrete situation in which the statement is true.")
    false: str = Field(description="A concrete situation in which the statement is false.")


class NoulQuestion(BaseModel):
    """A yes/no question; the answer is the probability of yes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """Picks one option. Pair with a Noul when no option might fit."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseModel):
    """Rates on ordered levels; each level's position is its score, starting at 0."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(min_length=1, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    state: dict[str, Any] | str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, description="Probability that the statement is true.")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float = Field(description="Probability-weighted position across levels; may fall between integers.")
    confidence: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(description="Level index (as a string) -> probability.")

    def level_probability(self, level: int) -> float:
        return self.probabilities.get(str(level), 0.0)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str = Field(description="The model version that answered, e.g. 'jev-1.14.0'.")
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)

    def noul(self, name: str) -> float:
        answer = self.answers[name]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"Answer {name!r} is {answer.type}, not noul")
        return answer.noul

    def score(self, name: str) -> ScoreAnswer:
        answer = self.answers[name]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"Answer {name!r} is {answer.type}, not score")
        return answer
```

### Discovery contract (`jobhunt/sources/base.py`)

```python
# jobhunt/sources/base.py
"""The contract every discovery source implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..services.http import HttpClient
from ..services.urls import AtsRef


@dataclass
class RawLead:
    """One sighting of a job on a source, before resolution.

    `ats_ref` and `description_text` are optional shortcuts: sources that already know the ATS posting
    (board polling) or already hold the description (HN comments, board APIs) pass them along so the
    resolver and hydrator can skip a network round trip.
    """

    source: str
    company: str
    title: str
    url: str
    location: str | None = None
    posted_at: datetime | None = None
    ats_ref: AtsRef | None = None
    description_text: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Source(ABC):
    """A discovery source. `name` is stored on every lead and ingest run, so keep it stable."""

    name: str

    @abstractmethod
    async def fetch(self, client: HttpClient) -> list[RawLead]:
        """Return the source's current leads. Raise on failure; the ingest service records the error."""
```

## Interfaces

### CLI Commands

| Command      | Arguments / options                             | Description                                                          | Exit codes                             |
| ------------ | ----------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------- |
| `migrate`    | —                                               | Upgrade the database to the latest Alembic revision                  | 0, 1 if `alembic.ini` missing          |
| `ingest`     | `--source/-s simplify\|hn\|boards` (repeatable) | Discover leads; one `ingest_run` per source                          | 0, 1 on unknown source                 |
| `import-csv` | `PATH`, `--default-source`                      | Import `Company, Role, Link[, Source]` rows                          | 0, 1 on missing columns                |
| `add`        | `URL -c COMPANY -t TITLE -s SOURCE`             | Add and resolve one lead                                             | 0                                      |
| `resolve`    | `[LEAD_ID URL]`, `--limit`                      | Resolve new leads, or attach a URL to one lead                       | 0, 1 on unknown lead or blocked URL    |
| `hydrate`    | `--refresh`, `--limit`                          | Fetch descriptions; `--refresh` re-checks open jobs                  | 0                                      |
| `screen`     | `--limit`                                       | Jev screen of new descriptions                                       | 0, 1 without TypeSafe key              |
| `extract`    | `--limit`                                       | LLM extraction of requirements and skills (skips screen rejects)     | 0, 1 without API key                   |
| `score`      | `--rescore`, `--fit jev\|openrouter\|none`      | Score relevant jobs for the current profile                          | 0, 1 if the chosen provider has no key |
| `jev-probe`  | `--limit`, `--job`, `--out`, `--send`           | Write exact Jev payloads; `--send` calls the API and saves responses | 0                                      |
| `run`        | `--source`, `--skip-ingest`, `--skip-score`     | All stages in order                                                  | 0                                      |
| `review`     | —                                               | Open the TUI                                                         | 0                                      |
| `queue`      | `--limit`, `--all`, `--json`                    | Print the queue                                                      | 0                                      |
| `unresolved` | `--json`                                        | Leads needing a link or that failed                                  | 0                                      |
| `skills`     | `--top`, `--gaps`, `--json`                     | Skill frequency weighted by job value                                | 0                                      |
| `health`     | `--window-days`, `--strict`                     | Per-source run health                                                | 0, 1 with `--strict` when unhealthy    |
| `pull`       | `PATH`                                          | Merge a pipeline database into the local one                         | 0                                      |
| `export`     | `--all`                                         | Queue as CSV on stdout                                               | 0                                      |

### TUI — Review queue

| Key                   | Action                                                                       |
| --------------------- | ---------------------------------------------------------------------------- |
| arrows / mouse        | Move through the queue; the right pane shows the highlighted job             |
| `o`                   | Open the posting in the browser                                              |
| `a` / `d` / `l` / `s` | Decide apply / applied / later / skip, with an optional reason (Esc cancels) |
| `r`                   | Toggle showing reviewed and unscored jobs                                    |
| `g`                   | Reload from the database                                                     |
| `q`                   | Quit                                                                         |

Queue order: unreviewed scored jobs by priority, then `later` jobs. `apply`, `applied` and `skip` drop out unless
`r` is on.

## Accessibility

### TUI-Specific Requirements

- **Color independence:** Score components render as `#####-----` bars with numeric values, and decisions as words. No information is conveyed by color alone.
- **Terminal themes:** Textual's default theme tokens are used; no hard-coded colors.
- **Keyboard only:** Every action has a single-key binding listed in the footer. The reason dialog takes Enter to save and Escape to cancel.
- **Resize:** The two panes are percentage widths, and the detail pane scrolls.

### CLI-Specific Requirements

- **Help text:** Every command and option has `--help` text.
- **Errors:** Errors go to stderr with an actionable message, e.g. how to attach a link to a `needs_url` lead.
- **Structured output:** `queue`, `unresolved` and `skills` support `--json`; `export` writes CSV.

## Implementation Detail

| Module                                | Responsibility                                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `services/http.py`                    | `HttpClient`: semaphore-bounded concurrency, tenacity retries on 408/425/429/5xx and transport errors, `NotFoundError` on 404/410 |
| `services/urls.py`                    | URL normalization (drops tracking params, sorts query), ATS detection, blocked and aggregator hosts                               |
| `services/ats.py`                     | Greenhouse, Lever, Ashby list and fetch; Workday fetch via the `wday/cxs` JSON endpoint                                           |
| `services/text.py`                    | HTML to text (double-unescape, bullets preserved), whitespace-insensitive content hash                                            |
| `services/filtering.py`               | Title and location prefilters; final `judge_relevance`                                                                            |
| `sources/*.py`                        | `SimplifySource`, `HackerNewsSource`, `BoardSource` per configured board                                                          |
| `services/jobs.py`                    | `ingest` (concurrent fetch, sequential writes, per-source isolation), `resolve_leads`, `attach_url`, `hydrate_jobs`               |
| `services/llm.py`                     | `OpenRouterClient.structured()`                                                                                                   |
| `services/typesafe.py`                | `TypeSafeClient.system_one()`; validates responses, missing answers raise                                                         |
| `questions.py`                        | Every Jev question, threshold and decision table                                                                                  |
| `services/screening.py`               | Screen newest unscreened snapshots in batches of 20                                                                               |
| `services/fit.py`                     | `JevFitProvider` and `LlmFitProvider`, each with its own cache key                                                                |
| `services/extraction.py`              | Picks each open job's newest unextracted snapshot, stores extraction and normalized skills                                        |
| `services/scoring.py`                 | `compute_score` (pure) and `score_jobs` with fit caching                                                                          |
| `services/queue.py`                   | Queue, reviews, unresolved leads, source health, skill report                                                                     |
| `services/merge.py`                   | One-way natural-key merge for `jobhunt pull`                                                                                      |
| `tui.py`                              | Textual `ReviewApp`                                                                                                               |
| `.github/workflows/job-pipeline.yaml` | Daily run, weekly `hydrate --refresh`, health check, publish to `data`                                                            |

Every stage commits per item (lead, job, extraction, score), so a crash loses at most one item's work, and a
failing source, posting or LLM call is recorded and skipped rather than aborting the run.

## Migrations

One migration, `initial pipeline tables`, creates `ingest_run`, `job`, `lead`, `job_snapshot`, `job_extraction`,
`job_skill`, `job_score` and `review` in foreign-key order. It was generated with `make create_migration`.
`make check_ungenerated_migrations` passes. Enum columns are VARCHAR(16) with value lists, so SQLite and
PostgreSQL behave the same. Adding an ATS later means adding an `Ats` member and a migration that widens the
enum's check constraint.

## Testing Philosophy

### URL normalization and ATS detection

Each ATS URL shape is a parametrized case, including Workday locale prefixes in both cases, Greenhouse embeds, and
`?gh_jid=` career sites mapped through `boards.toml`. Round-trip tests make sure `posting_url_for(ref)` parses
back to the same `AtsRef`, because that invariant is what deduplication relies on.

### Source adapters

Adapters are tested against payloads shaped like the documented APIs (`tests/fixtures/payloads.py`) with `respx`.
The tests cover filtering (titles, locations, stale or inactive Simplify listings, unlisted Ashby jobs), the
Greenhouse double-encoding, and Lever section merging. Payloads are trimmed to the fields the code reads.

### Pipeline behavior

Service tests run against a real temporary SQLite database. They prove that re-ingesting is idempotent, that a
broken source never stops the others and shows up in health, that two URLs for one posting collapse into one job,
that blocked links become `needs_url` and can be attached by hand, and that hydration closes only jobs that were
fetched successfully before.

### LLM contract

OpenRouter is mocked. The tests verify that the request carries `strict` and `require_parameters`, that one
invalid reply is retried, that a second invalid reply raises `LlmError`, that API errors surface, and that
extraction does not repeat for an already-extracted snapshot.

### Scoring

`compute_score` is pure, so each component and gate is tested directly: clearance and citizenship gates,
freshness decay, experience gap, high-competition and avoid lists, neutral values without a fit model, and fit-cache
reuse across unrelated profile edits.

### Jev integration

The tests check the request shape against the documented API (Bearer auth, `noul` criteria with `true`/`false`,
four Score levels), that 529 is retried and 422 is not, and that a missing answer raises. Every row of the screen
decision table is covered, including the unsettled band, as is the fit arithmetic. Pipeline tests prove that
rejected snapshots skip extraction, that a failed screen call still gets extracted, that rejects leave the queue,
and that fit answers are reused across unrelated profile edits but re-asked when a topic changes. None of this
validates the _wording_ against the real model; that needs `jobhunt jev-probe --send` with representative,
hard and negative cases.

### Merge, CLI and TUI

Merge is tested for collapsing a locally-added job into the remote one and for idempotence. CLI tests run commands
end to end on the throwaway database. The TUI is started with Textual's pilot to check the empty state and key
handling, and `render_detail` is tested for escaping and skill matches.

## Documentation Plan

### `README.md`

**Audience:** Ben (the user)

Quick start, the daily commands table, source coverage, and how the scheduled run and `make pull-data` fit together.

### `docs/dev/design.md`

**Audience:** Developers and coding agents

This document. It is updated whenever a decision changes, before the code changes.

### `docs/dev/pipeline.md`

**Audience:** Ben operating the pipeline

Configuration files, GitHub secrets and first run, reading `jobhunt health`, fixing a wrong board slug, the
`needs_url` workflow, and cost control for LLM calls.

### `docs/dev/database.md`

**Audience:** Developers

The schema diagram regenerated by `make document_schema` (Paracelsus).

## Risks / Trade-offs

### Upstream format changes

**Risk:** Simplify's category names, Greenhouse's encoding, or Workday's JSON shape can change without notice, and an adapter silently returns zero leads.

**Mitigation:** Each source and board is its own `ingest_run`. `jobhunt health --strict` fails the scheduled workflow when a source errors or returns zero leads three runs in a row, and the database still publishes so nothing is lost.

### Wrong board slugs

**Risk:** `pinterest`, `trexsolutionsllc` and `anthropic` in `boards.toml` are unverified guesses. A wrong slug makes every posting 404.

**Mitigation:** The board shows up as failed in `health`. First-fetch 404s never close a job. Fix the slug from the company's real `job-boards.greenhouse.io/<slug>` URL.

### URL-keyed duplicates

**Risk:** A job on an unsupported ATS (Workable, iCIMS, Oracle) reached through two different URLs becomes two jobs.

**Mitigation:** Normalization removes most variation. Phase 2 adds adapters for the most common hosts (Workable, SmartRecruiters, iCIMS), chosen by the URL-host counts of URL-keyed jobs.

### JavaScript-rendered pages

**Risk:** URL-keyed jobs on JS-only career sites hydrate to almost no text.

**Mitigation:** Pages under 400 characters are flagged rather than stored, and stay in the queue unscored until a link is attached or an adapter exists.

### LLM cost and drift

**Risk:** Extraction runs on every new snapshot, and the fit model runs on every relevant job.

**Mitigation:** Title and location prefilters run before any LLM call. Extraction is cached per snapshot, model and schema version, and the fit judgment per job and interests text. The models are settings.

### Untested question wording

**Risk:** The Jev questions were written from TypeSafe's guidance but have not been probed against the live model, which this sandbox cannot reach.

**Mitigation:** Before relying on the screen, run `jobhunt jev-probe --limit 20 --send` on representative, hard and clearly-irrelevant jobs. Read individual answers. Compare screen rejects with extraction's verdicts for a week before trusting rejects to save money.

### Email alerts not ingested

**Risk:** LinkedIn, BuiltIn and Google Jobs alerts still need manual handling.

**Mitigation:** Phase 2: a Gmail-label source (OAuth refresh token as a repository secret) that turns alert emails into leads. Until then, `import-csv` and `add` cover them.
