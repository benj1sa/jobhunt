"""One-way merge of a pipeline database (produced by GitHub Actions) into the local review database.

Rows are matched on natural keys, never on UUIDs, so jobs created independently on both sides
(e.g. a `jobhunt add` locally and the same posting found by the scheduled run) collapse into one.
Reviews and scores are never merged: they belong to the local database only.
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

from sqlalchemy import and_, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..models.base import Base
from ..models.enums import LeadStatus
from ..models.pipeline import IngestRun, Job, JobExtraction, JobScreen, JobSkill, JobSnapshot, Lead

logger = getLogger(__name__)

M = TypeVar("M", bound=Base)


@dataclass
class MergeStats:
    jobs_new: int = 0
    jobs_updated: int = 0
    leads_new: int = 0
    snapshots_new: int = 0
    extractions_new: int = 0
    screens_new: int = 0
    runs_new: int = 0


def _copy(model: type[M], row: Base, **overrides: Any) -> M:
    values: dict[str, Any] = {column.key: getattr(row, column.key) for column in inspect(model).columns}
    values.update(overrides)
    instance: M = model(**values)
    return instance


async def merge_database(local: AsyncSession, remote_path: Path) -> MergeStats:
    if not remote_path.exists():
        raise FileNotFoundError(remote_path)
    engine = create_async_engine(f"sqlite+aiosqlite:///{remote_path}")
    stats = MergeStats()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as remote:
            job_map: dict[UUID, UUID] = {}
            for rjob in (await remote.execute(select(Job))).scalars().all():
                ljob = (
                    await local.execute(select(Job).where(Job.canonical_key == rjob.canonical_key))
                ).scalar_one_or_none()
                if ljob is None:
                    clash = await local.get(Job, rjob.id)
                    ljob = _copy(Job, rjob) if clash is None else _copy(Job, rjob, id=uuid4())
                    local.add(ljob)
                    await local.flush()
                    stats.jobs_new += 1
                else:
                    if rjob.last_seen_at > ljob.last_seen_at:
                        ljob.last_seen_at = rjob.last_seen_at
                        ljob.status = rjob.status
                    for attr in ("location", "is_remote", "apply_url", "posted_at", "hydrated_at"):
                        if getattr(ljob, attr) is None and getattr(rjob, attr) is not None:
                            setattr(ljob, attr, getattr(rjob, attr))
                    stats.jobs_updated += 1
                job_map[rjob.id] = ljob.id

            for rlead in (await remote.execute(select(Lead))).scalars().all():
                mapped_job = job_map.get(rlead.job_id) if rlead.job_id else None
                llead = (
                    await local.execute(
                        select(Lead).where(
                            and_(Lead.source == rlead.source, Lead.url_normalized == rlead.url_normalized)
                        )
                    )
                ).scalar_one_or_none()
                if llead is None:
                    clash = await local.get(Lead, rlead.id)
                    local.add(_copy(Lead, rlead, job_id=mapped_job, **({"id": uuid4()} if clash else {})))
                    stats.leads_new += 1
                elif llead.status != LeadStatus.RESOLVED and rlead.status == LeadStatus.RESOLVED:
                    llead.status, llead.job_id = rlead.status, mapped_job
            await local.flush()

            snapshot_map: dict[UUID, UUID] = {}
            for rsnap in (await remote.execute(select(JobSnapshot))).scalars().all():
                job_id = job_map[rsnap.job_id]
                existing = (
                    await local.execute(
                        select(JobSnapshot.id).where(
                            and_(JobSnapshot.job_id == job_id, JobSnapshot.content_hash == rsnap.content_hash)
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    lsnap = _copy(JobSnapshot, rsnap, job_id=job_id)
                    if await local.get(JobSnapshot, rsnap.id) is not None:
                        lsnap.id = uuid4()
                    local.add(lsnap)
                    await local.flush()
                    existing = lsnap.id
                    stats.snapshots_new += 1
                snapshot_map[rsnap.id] = existing

            for rscreen in (await remote.execute(select(JobScreen))).scalars().all():
                snapshot_id = snapshot_map[rscreen.snapshot_id]
                seen = (
                    await local.execute(
                        select(JobScreen.id).where(
                            and_(JobScreen.snapshot_id == snapshot_id, JobScreen.version == rscreen.version)
                        )
                    )
                ).scalar_one_or_none()
                if seen is None:
                    local.add(
                        _copy(JobScreen, rscreen, id=uuid4(), job_id=job_map[rscreen.job_id], snapshot_id=snapshot_id)
                    )
                    stats.screens_new += 1
            await local.flush()

            extraction_map: dict[UUID, UUID] = {}
            for rext in (await remote.execute(select(JobExtraction))).scalars().all():
                snapshot_id = snapshot_map[rext.snapshot_id]
                existing = (
                    await local.execute(
                        select(JobExtraction.id).where(
                            and_(
                                JobExtraction.snapshot_id == snapshot_id,
                                JobExtraction.model == rext.model,
                                JobExtraction.schema_version == rext.schema_version,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    lext = _copy(JobExtraction, rext, job_id=job_map[rext.job_id], snapshot_id=snapshot_id)
                    if await local.get(JobExtraction, rext.id) is not None:
                        lext.id = uuid4()
                    local.add(lext)
                    await local.flush()
                    extraction_map[rext.id] = lext.id
                    stats.extractions_new += 1

            # Skills only accompany extractions that were new to the local database.
            for rskill in (await remote.execute(select(JobSkill))).scalars().all():
                if rskill.extraction_id in extraction_map:
                    local.add(
                        _copy(
                            JobSkill,
                            rskill,
                            id=uuid4(),
                            job_id=job_map[rskill.job_id],
                            extraction_id=extraction_map[rskill.extraction_id],
                        )
                    )

            for rrun in (await remote.execute(select(IngestRun))).scalars().all():
                if await local.get(IngestRun, rrun.id) is None:
                    local.add(_copy(IngestRun, rrun))
                    stats.runs_new += 1

            await local.commit()
    finally:
        await engine.dispose()
    logger.info("Merged %s: %s", remote_path, stats)
    return stats
