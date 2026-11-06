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
from .enums import Ats, BoardOrigin, BoardStatus, Decision, JobStatus, LeadStatus, RunStatus, ScreenOutcome


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
    # Raw items the source returned before filtering (null for runs recorded before this column existed).
    items_fetched: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WatchedBoard(Base):
    """A company's whole Greenhouse, Lever or Ashby board, polled on every ingest.

    Boards in config/boards.toml are always polled and are not stored here. `token` is stored lowercased.
    """

    __tablename__ = "watched_board"
    __table_args__ = (UniqueConstraint("ats", "token", name="uq_watched_board_ats_token"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    ats: Mapped[str] = mapped_column(String(16), nullable=False)
    token: Mapped[str] = mapped_column(String(256), nullable=False)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    origin: Mapped[BoardOrigin] = mapped_column(db_enum(BoardOrigin), nullable=False)
    origin_detail: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[BoardStatus] = mapped_column(db_enum(BoardStatus), nullable=False, default=BoardStatus.ACTIVE)
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_ok_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Last time the board produced at least one lead that passed the title and location filters.
    last_lead_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_last: Mapped[int | None] = mapped_column(Integer, nullable=True)


class LlmUsage(Base):
    """Token use and cost of one pipeline step's model calls (one row per step, provider and model per run).

    `cost_usd` is what the provider reported when it reports cost (OpenRouter does), otherwise an estimate from
    list prices, flagged by `cost_estimated`. `jobhunt status` uses these rows to estimate the next run.
    """

    __tablename__ = "llm_usage"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    calls: Mapped[int] = mapped_column(Integer, nullable=False)
    items: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    cost_estimated: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


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


class JobCapability(Base):
    """A verbatim phrase from a posting and one capability it shows. A phrase tagged with two capabilities is two rows.

    `verbatim` is False when the phrase could not be found in the snapshot text, i.e. the model paraphrased.
    """

    __tablename__ = "job_capability"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    extraction_id: Mapped[UUID] = mapped_column(
        ForeignKey("job_extraction.id", ondelete="CASCADE"), nullable=False, index=True
    )
    capability: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    phrase: Mapped[str] = mapped_column(Text, nullable=False)
    verbatim: Mapped[bool] = mapped_column(nullable=False)


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
