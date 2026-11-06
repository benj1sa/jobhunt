import time

import httpx
import pytest
import respx

from jobhunt.models.enums import Ats
from jobhunt.schemas.config import BoardConfig
from jobhunt.services.ats import ashby_list, fetch_posting, greenhouse_list, lever_fetch, workday_fetch
from jobhunt.services.http import HttpClient, NotFoundError
from jobhunt.services.urls import AtsRef
from jobhunt.settings import settings
from jobhunt.sources.boards import BoardSource
from jobhunt.sources.hn import HackerNewsSource, parse_comment
from jobhunt.sources.simplify import SimplifySource
from tests.fixtures import payloads


@pytest.mark.asyncio
@respx.mock
async def test_greenhouse_list_parses_and_decodes():
    respx.get("https://boards-api.greenhouse.io/v1/boards/durable/jobs").respond(json=payloads.GREENHOUSE_BOARD)
    async with HttpClient() as client:
        postings = await greenhouse_list(client, "durable")
    first = postings[0]
    assert first.ref == AtsRef(Ats.GREENHOUSE, "durable", "5156304008")
    assert first.description_text.startswith("Build LLM features.")
    assert first.posted_at is not None and first.posted_at.year == 2026


@pytest.mark.asyncio
@respx.mock
async def test_lever_fetch_combines_sections():
    posting_id = payloads.LEVER_POSTING["id"]
    respx.get(f"https://api.lever.co/v0/postings/acme/{posting_id}").respond(json=payloads.LEVER_POSTING)
    async with HttpClient() as client:
        posting = await lever_fetch(client, AtsRef(Ats.LEVER, "acme", posting_id))
    assert "We build things." in posting.description_text
    assert "Requirements" in posting.description_text and "- Go" in posting.description_text
    assert posting.is_remote is True


@pytest.mark.asyncio
@respx.mock
async def test_ashby_list_skips_unlisted_and_derives_id():
    respx.get("https://api.ashbyhq.com/posting-api/job-board/openai").respond(json=payloads.ASHBY_BOARD)
    async with HttpClient() as client:
        postings = await ashby_list(client, "openai")
    assert len(postings) == 1
    assert postings[0].ref.external_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.mark.asyncio
@respx.mock
async def test_ashby_fetch_missing_posting_is_not_found():
    respx.get("https://api.ashbyhq.com/posting-api/job-board/openai").respond(json=payloads.ASHBY_BOARD)
    async with HttpClient() as client:
        with pytest.raises(NotFoundError):
            await fetch_posting(client, AtsRef(Ats.ASHBY, "openai", "00000000-0000-0000-0000-000000000000"))


@pytest.mark.asyncio
@respx.mock
async def test_workday_fetch():
    respx.get("https://umd.wd1.myworkdayjobs.com/wday/cxs/umd/UMCP/job/MLOps-Engineer_JR104177").respond(
        json=payloads.WORKDAY_JOB
    )
    async with HttpClient() as client:
        posting = await workday_fetch(client, AtsRef(Ats.WORKDAY, "umd|wd1|UMCP", "MLOps-Engineer_JR104177"))
    assert posting.company == "University of Maryland"
    assert posting.description_text == "Deploy models on campus HPC."


@pytest.mark.asyncio
@respx.mock
async def test_http_client_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("jobhunt.services.http.wait_exponential_jitter", lambda **_: lambda _state: 0)
    route = respx.get("https://example.com/x").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
    )
    async with HttpClient() as client:
        assert await client.get_json("https://example.com/x") == {"ok": True}
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_board_source_filters_titles_and_locations():
    respx.get("https://boards-api.greenhouse.io/v1/boards/durable/jobs").respond(json=payloads.GREENHOUSE_BOARD)
    source = BoardSource(BoardConfig(ats="greenhouse", token="durable", company="Durable"))
    async with HttpClient() as client:
        leads = await source.fetch(client)
    assert [lead.title for lead in leads] == ["AI Engineer"]
    assert leads[0].ats_ref is not None and leads[0].description_text
    assert source.items_fetched == 3


@pytest.mark.asyncio
@respx.mock
async def test_simplify_source_filters():
    listings = [dict(item, date_posted=int(time.time())) for item in payloads.SIMPLIFY_LISTINGS]
    respx.get(settings.simplify_listings_url).respond(json=listings)
    async with HttpClient() as client:
        leads = await SimplifySource().fetch(client)
    assert [lead.company for lead in leads] == ["EquipmentShare"]
    assert leads[0].extra["sponsorship"] == "Other"


@pytest.mark.asyncio
@respx.mock
async def test_simplify_source_drops_stale_listings():
    respx.get(settings.simplify_listings_url).respond(json=payloads.SIMPLIFY_LISTINGS)  # date_posted = 1970
    async with HttpClient() as client:
        assert await SimplifySource().fetch(client) == []


def test_hn_parse_comment_extracts_entry_level_role():
    lead = parse_comment(payloads.HN_ITEM["children"][0])
    assert lead is not None
    assert lead.company == "Acme Health"
    assert lead.title == "Software Engineer (New Grad)"
    assert lead.location == "Baltimore, MD"
    assert lead.ats_ref == AtsRef(Ats.GREENHOUSE, "acmehealth", "777")


@pytest.mark.parametrize("index", [1, 2])
def test_hn_parse_comment_rejects(index):
    assert parse_comment(payloads.HN_ITEM["children"][index]) is None


@pytest.mark.asyncio
@respx.mock
async def test_hn_source_reads_latest_thread():
    respx.get("https://hn.algolia.com/api/v1/search_by_date").respond(json=payloads.HN_SEARCH)
    respx.get("https://hn.algolia.com/api/v1/items/41000000").respond(json=payloads.HN_ITEM)
    async with HttpClient() as client:
        leads = await HackerNewsSource().fetch(client)
    assert len(leads) == 1
