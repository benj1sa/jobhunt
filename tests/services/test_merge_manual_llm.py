from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from jobhunt.models.base import Base
from jobhunt.models.enums import ScreenOutcome
from jobhunt.models.pipeline import Job, JobScreen, JobSnapshot, Lead
from jobhunt.schemas.extraction import FitJudgment
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import resolve_leads, upsert_leads
from jobhunt.services.llm import LlmError, OpenRouterClient
from jobhunt.services.manual import CsvFormatError, lead_from_args, leads_from_csv, source_slug
from jobhunt.services.merge import merge_database
from jobhunt.sources.base import RawLead


def test_source_slug():
    assert source_slug("ColorStack Board") == "colorstack-board"
    assert source_slug("Colorstack Slack") == source_slug("ColorStack Slack")
    assert source_slug("  ") == "manual"


def test_leads_from_csv(tmp_path: Path):
    path = tmp_path / "opps.csv"
    path.write_text(
        "Company,Role,Link,Source\n"
        "xloop,Software Engineer,https://wellfound.com/jobs/1,Wellfound\n"
        '"Steampunk, Inc.",MLOps Engineer,https://www.linkedin.com/jobs/view/2/,LinkedIn\n'
        "NoLink,Engineer,,LinkedIn\n"
    )
    leads = leads_from_csv(path)
    assert [(lead.company, lead.source) for lead in leads] == [("xloop", "wellfound"), ("Steampunk, Inc.", "linkedin")]


def test_leads_from_csv_requires_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("Company,Title\nA,B\n")
    with pytest.raises(CsvFormatError, match="link, role"):
        leads_from_csv(path)


def test_lead_from_args():
    lead = lead_from_args("https://x", "Acme", "SWE", "CodePath Slack")
    assert lead.source == "codepath-slack" and lead.extra["source_label"] == "CodePath Slack"


@pytest.mark.asyncio
@respx.mock
async def test_llm_gives_up_after_two_invalid_replies():
    respx.post("https://or.test/chat/completions").respond(json={"choices": [{"message": {"content": "{}"}}]})
    async with HttpClient() as client:
        llm = OpenRouterClient(client, SecretStr("k"), "https://or.test")
        with pytest.raises(LlmError):
            await llm.structured("m", "sys", "user", FitJudgment)


@pytest.mark.asyncio
@respx.mock
async def test_llm_strips_code_fences_and_surfaces_api_errors():
    content = '```json\n{"interest_alignment":"weak","learning_potential":"weak","rationale":"x"}\n```'
    route = respx.post("https://or.test/chat/completions")
    route.side_effect = [
        httpx.Response(200, json={"choices": [{"message": {"content": content}}]}),
        httpx.Response(200, json={"error": {"message": "no provider supports structured outputs"}}),
    ]
    async with HttpClient() as client:
        llm = OpenRouterClient(client, SecretStr("k"), "https://or.test")
        assert (await llm.structured("m", "s", "u", FitJudgment)).rationale == "x"
        with pytest.raises(LlmError, match="structured outputs"):
            await llm.structured("m", "s", "u", FitJudgment)


async def _remote_db(path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as session:
        await upsert_leads(
            session,
            [
                RawLead(source="simplify", company="Durable", title="AI Engineer",
                        url="https://job-boards.greenhouse.io/durable/jobs/1", description_text="Remote text."),
                RawLead(source="simplify", company="Other", title="SWE",
                        url="https://job-boards.greenhouse.io/other/jobs/2"),
            ],
        )  # fmt: skip
        await session.commit()
        await resolve_leads(session, [], client=None)
        snapshot = (await session.execute(select(JobSnapshot))).scalar_one()
        session.add(
            JobScreen(job_id=snapshot.job_id, snapshot_id=snapshot.id, version="screen-v1", model="jev-1.14.0",
                      outcome=ScreenOutcome.PASS, reason="ok", signals={}, answers={})
        )  # fmt: skip
        await session.commit()
    await engine.dispose()


@pytest.mark.asyncio
async def test_merge_collapses_jobs_by_canonical_key_and_is_idempotent(db_session, tmp_path: Path):
    remote = tmp_path / "remote.db"
    await _remote_db(remote)
    # The same posting was added locally by hand before the scheduled run found it.
    await upsert_leads(
        db_session,
        [RawLead(source="colorstack", company="Durable", title="AI Engineer",
                 url="https://boards.greenhouse.io/durable/jobs/1?gh_src=x")],
    )  # fmt: skip
    await db_session.commit()
    await resolve_leads(db_session, [], client=None)

    stats = await merge_database(db_session, remote)
    assert (stats.jobs_new, stats.jobs_updated, stats.leads_new, stats.snapshots_new) == (1, 1, 2, 1)
    assert (await db_session.execute(select(func.count()).select_from(Job))).scalar_one() == 2
    durable = (await db_session.execute(select(Job).where(Job.board == "durable"))).scalar_one()
    sources = set((await db_session.execute(select(Lead.source).where(Lead.job_id == durable.id))).scalars())
    assert sources == {"colorstack", "simplify"}

    assert stats.screens_new == 1
    again = await merge_database(db_session, remote)
    assert (again.jobs_new, again.leads_new, again.snapshots_new, again.screens_new) == (0, 0, 0, 0)
    assert (await db_session.execute(select(func.count()).select_from(JobSnapshot))).scalar_one() == 1


@pytest.mark.asyncio
async def test_merge_watched_boards_newest_poll_wins_but_manual_status_stays(db_session, tmp_path: Path):
    from datetime import timedelta

    from jobhunt.models.enums import BoardOrigin, BoardStatus
    from jobhunt.models.pipeline import WatchedBoard
    from jobhunt.services.boards import utcnow

    remote = tmp_path / "remote.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{remote}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    now = utcnow()
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add_all([
            WatchedBoard(ats="lever", token="a", company="A", origin=BoardOrigin.SEED, status=BoardStatus.DEAD,
                         status_reason="not found 2 times in a row", last_polled_at=now, consecutive_failures=2),
            WatchedBoard(ats="lever", token="b", company="B", origin=BoardOrigin.DISCOVERED, last_polled_at=now),
        ])  # fmt: skip
        await session.commit()
    await engine.dispose()

    db_session.add(
        WatchedBoard(ats="lever", token="a", company="A", origin=BoardOrigin.MANUAL, status=BoardStatus.ACTIVE,
                     status_reason="resumed by hand", last_polled_at=now - timedelta(days=1))
    )  # fmt: skip
    await db_session.commit()
    stats = await merge_database(db_session, remote)
    assert stats.boards_new == 1
    boards = {b.token: b for b in (await db_session.execute(select(WatchedBoard))).scalars()}
    assert boards["a"].status == BoardStatus.ACTIVE and boards["a"].consecutive_failures == 2
    assert boards["b"].origin == BoardOrigin.DISCOVERED
