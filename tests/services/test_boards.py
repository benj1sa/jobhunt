"""Board watchlist: seeds, discovery, polling lifecycle."""

from datetime import timedelta

import httpx
import pytest
import respx
from sqlalchemy import select

from jobhunt.models.enums import BoardOrigin, BoardStatus
from jobhunt.models.pipeline import WatchedBoard
from jobhunt.schemas.config import BoardConfig, BoardSeed
from jobhunt.services.boards import (
    PollOutcome,
    add_boards,
    boards_from_lines,
    boards_from_links,
    boards_from_simplify,
    parse_board_ref,
    record_polls,
    seed_boards,
    set_board_status,
    utcnow,
    watched_sources,
)
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import resolve_leads, upsert_leads
from jobhunt.sources.base import RawLead

CONFIGURED = [BoardConfig(ats="greenhouse", token="durable", company="Durable")]
UUID = "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"


async def boards(session) -> dict[str, WatchedBoard]:
    return {f"{b.ats}:{b.token}": b for b in (await session.execute(select(WatchedBoard))).scalars()}


def test_boards_from_simplify_reads_open_and_closed_listings_and_filters_categories():
    listings = [
        {"url": "https://job-boards.greenhouse.io/Stripe/jobs/1", "company_name": "Stripe", "category": "Software",
         "active": False},
        {"url": f"https://jobs.ashbyhq.com/ramp/{UUID}", "company_name": "Ramp", "category": "Quant"},
        {"url": "https://umd.wd1.myworkdayjobs.com/UMCP/job/X_1", "company_name": "UMD", "category": "Software"},
        {"url": "https://example.com/careers", "company_name": "Ex", "category": "Software"},
    ]  # fmt: skip
    assert boards_from_simplify(listings, []) == {("greenhouse", "stripe"): "Stripe", ("ashby", "ramp"): "Ramp"}
    assert set(boards_from_simplify(listings, ["Software"])) == {("greenhouse", "stripe")}


def test_boards_from_links_and_lines():
    text = (
        "| Ramp | https://jobs.ashbyhq.com/ramp | [apply](https://jobs.lever.co/palantir/" + UUID + ") "
        "https://boards.greenhouse.io/embed/job_app?for=figma&token=42 https://job-boards.greenhouse.io/v1/x"
    )
    assert set(boards_from_links(text)) == {("ashby", "ramp"), ("lever", "palantir"), ("greenhouse", "figma")}
    lines = "greenhouse:stripe\n# comment\nlever: Netflix  # trailing\nworkday:nope\nbad line\n"
    assert set(boards_from_lines(lines)) == {("greenhouse", "stripe"), ("lever", "netflix")}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("greenhouse:Stripe", ("greenhouse", "stripe")),
        (f"https://jobs.lever.co/palantir/{UUID}", ("lever", "palantir")),
        ("https://jobs.ashbyhq.com/ramp", ("ashby", "ramp")),
    ],
)
def test_parse_board_ref(value, expected):
    assert parse_board_ref(value) == expected


@pytest.mark.parametrize("value", ["stripe", "workday:umd", "https://example.com/jobs"])
def test_parse_board_ref_rejects(value):
    with pytest.raises(ValueError):
        parse_board_ref(value)


@pytest.mark.asyncio
async def test_add_boards_skips_configured_duplicates_and_respects_manual_pause(db_session):
    candidates = {("greenhouse", "durable"): "Durable", ("greenhouse", "stripe"): "Stripe", ("lever", "x"): "X"}
    assert await add_boards(db_session, candidates, BoardOrigin.SEED, "s", CONFIGURED) == 2
    assert await add_boards(db_session, candidates, BoardOrigin.SEED, "s", CONFIGURED) == 0

    await set_board_status(db_session, "greenhouse", "stripe", BoardStatus.PAUSED, "paused by hand")
    stored = await boards(db_session)
    stored["lever:x"].status, stored["lever:x"].status_reason = BoardStatus.PAUSED, "idle: no matching job in 60 days"
    await db_session.commit()
    await add_boards(db_session, candidates, BoardOrigin.DISCOVERED, "lead", CONFIGURED, reactivate_paused=True)
    stored = await boards(db_session)
    assert stored["lever:x"].status == BoardStatus.ACTIVE  # a new lead revives an idle board
    assert stored["greenhouse:stripe"].status == BoardStatus.PAUSED  # but never one you paused yourself


@pytest.mark.asyncio
async def test_resolving_leads_discovers_their_boards(db_session):
    await upsert_leads(
        db_session,
        [
            RawLead(source="simplify", company="Ramp", title="SWE", url=f"https://jobs.ashbyhq.com/ramp/{UUID}"),
            RawLead(source="simplify", company="Durable", title="AI", url="https://job-boards.greenhouse.io/durable/jobs/1"),
            RawLead(source="simplify", company="UMD", title="ML", url="https://umd.wd1.myworkdayjobs.com/UMCP/job/X_1"),
        ],
    )  # fmt: skip
    await db_session.commit()
    stats = await resolve_leads(db_session, CONFIGURED, client=None)
    assert stats.boards_discovered == 1  # Ramp; Durable is configured and Workday has no listing API
    stored = await boards(db_session)
    assert list(stored) == ["ashby:ramp"] and stored["ashby:ramp"].origin == BoardOrigin.DISCOVERED


@pytest.mark.asyncio
async def test_record_polls_retires_dead_and_idle_boards(db_session):
    candidates = {("greenhouse", "gone"): "Gone", ("greenhouse", "quiet"): "Quiet", ("greenhouse", "busy"): "Busy"}
    await add_boards(db_session, candidates, BoardOrigin.SEED, "s", [])
    stored = await boards(db_session)
    stored["greenhouse:quiet"].added_at = utcnow() - timedelta(days=90)
    stored["greenhouse:busy"].added_at = utcnow() - timedelta(days=90)
    await db_session.commit()

    outcomes = [
        PollOutcome("board:greenhouse:gone", ok=False, not_found=True, items=None, leads=0),
        PollOutcome("board:greenhouse:quiet", ok=True, not_found=False, items=40, leads=0),
        PollOutcome("board:greenhouse:busy", ok=True, not_found=False, items=40, leads=3),
    ]
    first = await record_polls(db_session, outcomes, idle_days=60, dead_after=2)
    assert first == {"paused": 1}
    second = await record_polls(db_session, outcomes, idle_days=60, dead_after=2)
    assert second == {"dead": 1}
    stored = await boards(db_session)
    assert stored["greenhouse:gone"].status == BoardStatus.DEAD
    assert stored["greenhouse:quiet"].status == BoardStatus.PAUSED
    assert stored["greenhouse:busy"].status == BoardStatus.ACTIVE and stored["greenhouse:busy"].items_last == 40


@pytest.mark.asyncio
async def test_watched_sources_orders_by_recent_leads_and_caps(db_session):
    await add_boards(
        db_session, {("lever", "a"): "A", ("lever", "b"): "B", ("lever", "c"): "C", ("lever", "d"): "D"},
        BoardOrigin.SEED, "s", [],
    )  # fmt: skip
    stored = await boards(db_session)
    stored["lever:c"].last_lead_at = utcnow()
    stored["lever:b"].last_lead_at = utcnow() - timedelta(days=5)
    stored["lever:d"].status = BoardStatus.DEAD
    await db_session.commit()
    sources = await watched_sources(db_session, [], limit=2)
    assert [s.name for s in sources] == ["board:lever:c", "board:lever:b"]
    assert len(await watched_sources(db_session, [], limit=10)) == 3


@pytest.mark.asyncio
@respx.mock
async def test_seed_boards_formats_and_isolates_failures(db_session):
    respx.get("https://seed.test/listings.json").respond(
        json=[{"url": "https://job-boards.greenhouse.io/stripe/jobs/1", "company_name": "Stripe"}]
    )
    respx.get("https://seed.test/README.md").respond(text="see https://jobs.ashbyhq.com/ramp and more")
    respx.get("https://seed.test/broken").mock(side_effect=httpx.ConnectError("down"))
    seeds = [
        BoardSeed(name="simplify", url="https://seed.test/listings.json", format="simplify"),
        BoardSeed(name="broken", url="https://seed.test/broken", format="lines"),
        BoardSeed(name="readme", url="https://seed.test/README.md", format="links"),
    ]
    async with HttpClient() as client:
        results = await seed_boards(db_session, client, seeds, [])
    by_name = {r.name: r for r in results}
    assert (by_name["simplify"].added, by_name["readme"].added) == (1, 1)
    assert by_name["broken"].error and "ConnectError" in by_name["broken"].error
    assert set(await boards(db_session)) == {"greenhouse:stripe", "ashby:ramp"}
