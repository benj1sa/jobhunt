"""Ingest → resolve → hydrate → extract → score → queue against a real SQLite database, with HTTP mocked."""

from datetime import UTC, datetime

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import func, select

from jobhunt.models.enums import Ats, Decision, JobStatus, LeadStatus, RunStatus
from jobhunt.models.pipeline import Job, JobExtraction, JobSkill, JobSnapshot, Lead
from jobhunt.schemas.config import BoardConfig
from jobhunt.schemas.extraction import JobRequirements
from jobhunt.services.extraction import extract_jobs
from jobhunt.services.fit import LlmFitProvider
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import attach_url, hydrate_jobs, ingest, resolve_leads, upsert_leads
from jobhunt.services.llm import OpenRouterClient
from jobhunt.services.queue import build_queue, record_review, skill_report, source_health
from jobhunt.services.scoring import score_jobs
from jobhunt.sources.base import RawLead, Source
from tests.fixtures import payloads

BOARDS = [BoardConfig(ats="greenhouse", token="pinterest", company="Pinterest", domains=["pinterestcareers.com"])]


class StaticSource(Source):
    def __init__(self, name: str, leads: list[RawLead]) -> None:
        self.name = name
        self._leads = leads

    async def fetch(self, client: HttpClient) -> list[RawLead]:
        return self._leads


class BrokenSource(Source):
    name = "broken"

    async def fetch(self, client: HttpClient) -> list[RawLead]:
        raise RuntimeError("upstream changed its format")


def lead(url: str, source: str = "csv", **kwargs) -> RawLead:
    return RawLead(source=source, company=kwargs.pop("company", "Durable"), title="AI Engineer", url=url, **kwargs)


async def count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


@pytest.mark.asyncio
async def test_upsert_leads_is_idempotent(db_session):
    url = "https://job-boards.greenhouse.io/durable/jobs/5156304008?utm_source=linkedin"
    assert await upsert_leads(db_session, [lead(url), lead(url.split("?")[0])]) == 1
    assert await upsert_leads(db_session, [lead(url)]) == 0
    assert await count(db_session, Lead) == 1


@pytest.mark.asyncio
async def test_ingest_isolates_failing_source(db_session):
    good = StaticSource("good", [lead("https://example.com/a", source="good")])
    async with HttpClient() as client:
        results = await ingest(db_session, client, [BrokenSource(), good])
    by_name = {result.source: result for result in results}
    assert by_name["broken"].status == RunStatus.FAILED
    assert "upstream changed" in (by_name["broken"].error or "")
    assert by_name["good"].status == RunStatus.OK and by_name["good"].new == 1
    health = {row.source: row for row in await source_health(db_session)}
    assert not health["broken"].healthy and health["good"].healthy


@pytest.mark.asyncio
async def test_resolve_dedupes_across_sources_and_flags_blocked(db_session):
    await upsert_leads(
        db_session,
        [
            lead("https://job-boards.greenhouse.io/durable/jobs/5156304008", source="greenhouse"),
            lead("https://boards.greenhouse.io/durable/jobs/5156304008?gh_src=abc", source="colorstack"),
            lead("https://www.pinterestcareers.com/jobs/7838591/x/?gh_jid=7838591", company="Pinterest"),
            lead("https://www.linkedin.com/jobs/view/4454811409/", source="linkedin"),
        ],
    )
    await db_session.commit()
    stats = await resolve_leads(db_session, BOARDS, client=None)
    assert (stats.resolved, stats.new_jobs, stats.needs_url) == (3, 2, 1)
    keys = set((await db_session.execute(select(Job.canonical_key))).scalars())
    assert keys == {"greenhouse:durable:5156304008", "greenhouse:pinterest:7838591"}

    blocked = (await db_session.execute(select(Lead).where(Lead.status == LeadStatus.NEEDS_URL))).scalar_one()
    async with HttpClient() as client:
        job = await attach_url(
            db_session, blocked.id, "https://job-boards.greenhouse.io/durable/jobs/5156304008", BOARDS, client
        )
    assert job.canonical_key == "greenhouse:durable:5156304008"
    assert await count(db_session, Job) == 2


@pytest.mark.asyncio
async def test_resolve_stores_pending_description_as_snapshot(db_session):
    raw = lead(
        "https://news.ycombinator.com/item?id=1",
        source="hn",
        description_text="We hire new grads to build Python services.",
    )
    await upsert_leads(db_session, [raw])
    await db_session.commit()
    await resolve_leads(db_session, [], client=None)
    snapshot = (await db_session.execute(select(JobSnapshot))).scalar_one()
    assert "new grads" in snapshot.description_text
    stored = (await db_session.execute(select(Lead))).scalar_one()
    assert "pending_description" not in stored.extra


@pytest.mark.asyncio
@respx.mock
async def test_hydrate_fetches_and_closes_only_previously_seen_jobs(db_session):
    await upsert_leads(
        db_session,
        [
            lead("https://job-boards.greenhouse.io/durable/jobs/5156304008"),
            lead("https://job-boards.greenhouse.io/wrongslug/jobs/1"),
        ],
    )
    await db_session.commit()
    await resolve_leads(db_session, [], client=None)
    respx.get("https://boards-api.greenhouse.io/v1/boards/durable/jobs/5156304008").respond(
        json=payloads.GREENHOUSE_JOB
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/wrongslug/jobs/1").respond(404)
    async with HttpClient() as client:
        stats = await hydrate_jobs(db_session, client)
        assert (stats.hydrated, stats.closed, stats.failed) == (1, 0, 1)

        # Same content again: no duplicate snapshot.
        again = await hydrate_jobs(db_session, client, refresh=True)
        assert again.unchanged == 1

        # Now the known-good posting disappears: it closes.
        respx.get("https://boards-api.greenhouse.io/v1/boards/durable/jobs/5156304008").respond(404)
        closed = await hydrate_jobs(db_session, client, refresh=True)
    assert closed.closed == 1
    statuses = dict((await db_session.execute(select(Job.board, Job.status))).all())
    assert statuses == {"durable": JobStatus.CLOSED, "wrongslug": JobStatus.OPEN}
    assert await count(db_session, JobSnapshot) == 1


def openrouter_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
@respx.mock
async def test_extract_score_queue_end_to_end(db_session, requirements: JobRequirements, profile, aliases):
    await upsert_leads(
        db_session,
        [lead("https://job-boards.greenhouse.io/acme/jobs/1", company="Acme", description_text="ML role text.")],
    )
    await db_session.commit()
    await resolve_leads(db_session, [], client=None)

    fit = '{"interest_alignment": "strong", "learning_potential": "moderate", "rationale": "Healthcare ML."}'
    route = respx.post("https://openrouter.test/chat/completions").mock(
        side_effect=[
            # First reply is invalid JSON for the schema; the client must retry once with the error.
            httpx.Response(200, json=openrouter_reply('{"role_family": "ai_ml"}')),
            httpx.Response(200, json=openrouter_reply(requirements.model_dump_json())),
            httpx.Response(200, json=openrouter_reply(fit)),
        ]
    )
    async with HttpClient() as client:
        llm = OpenRouterClient(client, SecretStr("test-key"), "https://openrouter.test")
        extracted = await extract_jobs(db_session, llm, "test/model", aliases, max_chars=1000)
        assert (extracted.extracted, extracted.relevant, extracted.failed) == (1, 1, 0)

        # A second run finds nothing new to extract.
        assert (await extract_jobs(db_session, llm, "test/model", aliases, max_chars=1000)).extracted == 0

        provider = LlmFitProvider(llm, "test/fit")
        scored = await score_jobs(db_session, profile, aliases, provider)
        assert (scored.scored, scored.fit_calls) == (1, 1)
        # Rescoring with the same interests reuses the cached fit judgment.
        rescored = await score_jobs(db_session, profile, aliases, provider, rescore=True)
        assert rescored.fit_calls == 0
    assert route.call_count == 3
    request_body = route.calls[0].request.content
    assert b'"strict":true' in request_body and b'"require_parameters":true' in request_body

    skills = set((await db_session.execute(select(JobSkill.skill))).scalars())
    assert {"Python", "PyTorch", "Kubernetes", "SQL"} <= skills
    extraction = (await db_session.execute(select(JobExtraction))).scalar_one()
    assert extraction.is_relevant

    items = await build_queue(db_session, profile.content_hash)
    assert len(items) == 1 and items[0].score is not None and items[0].score.priority > 0.5

    report = await skill_report(db_session, profile.content_hash, have={"python", "pytorch", "sql"})
    kubernetes = next(stat for stat in report if stat.skill == "Kubernetes")
    assert not kubernetes.have and kubernetes.required_in == 1

    await record_review(db_session, items[0].job.id, Decision.SKIP, "too much infra")
    assert await build_queue(db_session, profile.content_hash) == []
    reviewed = await build_queue(db_session, profile.content_hash, include_reviewed=True)
    assert reviewed[0].decision == Decision.SKIP
    # Skipped jobs no longer count toward the skills you should learn.
    assert await skill_report(db_session, profile.content_hash, have=set()) == []


@pytest.mark.asyncio
async def test_queue_hides_irrelevant_jobs(db_session, requirements, profile):
    now = datetime.now(tz=UTC)
    job = Job(
        canonical_key="url:x",
        ats=Ats.URL,
        company="X",
        title="Senior Engineer",
        posting_url="https://x",
        first_seen_at=now,
        last_seen_at=now,
    )
    db_session.add(job)
    await db_session.flush()
    snapshot = JobSnapshot(job_id=job.id, content_hash="h", description_text="t", raw={})
    db_session.add(snapshot)
    await db_session.flush()
    db_session.add(
        JobExtraction(
            job_id=job.id,
            snapshot_id=snapshot.id,
            model="m",
            schema_version=1,
            data=requirements.model_dump(mode="json"),
            is_relevant=False,
            relevance_reason="seniority is senior",
        )
    )
    await db_session.commit()
    assert await build_queue(db_session, profile.content_hash, include_reviewed=True, include_unscored=True) == []
