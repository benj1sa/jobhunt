"""Jev integration: request shape, decision tables, screening, fit, and failure handling. HTTP is mocked."""

import json

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import select

from jobhunt.models.enums import ScreenOutcome
from jobhunt.models.pipeline import JobExtraction, JobScore, JobScreen
from jobhunt.questions import NO, YES, combine_fit, decide_screen, fit_questions, screen_questions
from jobhunt.schemas.extraction import JobRequirements
from jobhunt.schemas.profile import Profile
from jobhunt.schemas.typesafe import SystemOneResponse
from jobhunt.services.extraction import extract_jobs
from jobhunt.services.fit import JevFitProvider
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import resolve_leads, upsert_leads
from jobhunt.services.llm import OpenRouterClient
from jobhunt.services.queue import build_queue
from jobhunt.services.scoring import score_jobs
from jobhunt.services.screening import screen_jobs
from jobhunt.services.typesafe import TypeSafeClient, TypeSafeError
from jobhunt.sources.base import RawLead
from tests.fixtures import jev

API = "https://ts.test/v1/systemone"


@pytest.fixture
def jev_profile(profile: Profile) -> Profile:
    data = profile.model_dump()
    data["candidate"]["interest_topics"] = [
        {"name": "applied ML", "description": "building machine learning models", "weight": 1.0},
        {"name": "healthcare", "description": "healthcare problems", "weight": 0.5},
    ]
    data["candidate"]["growth_goals"] = [{"name": "ML in production", "description": "deploying ML models"}]
    return Profile.model_validate(data)


def client(http: HttpClient) -> TypeSafeClient:
    return TypeSafeClient(http, SecretStr("ts-key"), "https://ts.test", "jev-latest")


async def add_job(session, url: str, text: str = "We hire new grads to build ML systems in Python.") -> None:
    await upsert_leads(
        session, [RawLead(source="csv", company="Acme", title="ML Engineer", url=url, description_text=text)]
    )
    await session.commit()
    await resolve_leads(session, [], client=None)


# --- request shape and client behavior -------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_request_shape_matches_api():
    route = respx.post(API).respond(json=jev.screen_response())
    async with HttpClient() as http:
        ts = client(http)
        await ts.system_one(ts.build_request(state={"posting": {"title": "x"}}, questions=screen_questions()))
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer ts-key"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest"
    assert body["questions"]["internship"]["type"] == "noul"
    assert set(body["questions"]["internship"]["criteria"]) == {"true", "false"}
    assert body["questions"]["experience_level"]["type"] == "score"
    assert len(body["questions"]["experience_level"]["criteria"]) == 4


@pytest.mark.asyncio
@respx.mock
async def test_overloaded_529_is_retried(monkeypatch):
    monkeypatch.setattr("jobhunt.services.http.wait_exponential_jitter", lambda **_: lambda _state: 0)
    route = respx.post(API).mock(side_effect=[httpx.Response(529), httpx.Response(200, json=jev.screen_response())])
    async with HttpClient() as http:
        ts = client(http)
        response = await ts.system_one(ts.build_request(state={"posting": {}}, questions=screen_questions()))
    assert route.call_count == 2 and response.model == "jev-1.14.0"


@pytest.mark.asyncio
@respx.mock
async def test_missing_answer_is_an_error_not_a_no():
    payload = jev.screen_response()
    del payload["answers"]["internship"]
    respx.post(API).respond(json=payload)
    async with HttpClient() as http:
        ts = client(http)
        with pytest.raises(TypeSafeError, match="internship"):
            await ts.system_one(ts.build_request(state={"posting": {}}, questions=screen_questions()))


@pytest.mark.asyncio
@respx.mock
async def test_validation_error_is_not_retried():
    route = respx.post(API).respond(422, json={"detail": [{"loc": ["body", "questions"], "msg": "bad"}]})
    async with HttpClient() as http:
        ts = client(http)
        with pytest.raises(httpx.HTTPStatusError):
            await ts.system_one(ts.build_request(state={"posting": {}}, questions=screen_questions()))
    assert route.call_count == 1


# --- decision tables -------------------------------------------------------------------------------------------


def screen(**kwargs) -> SystemOneResponse:
    return SystemOneResponse.model_validate(jev.screen_response(**kwargs))


def test_screen_pass_when_every_signal_is_clear():
    decision = decide_screen(screen())
    assert decision.outcome == ScreenOutcome.PASS
    assert decision.signals["entry_level"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"role": 0.1}, "not a software/AI/ML role"),
        ({"internship": 0.9}, "internship"),
        ({"us": 0.05}, "not in the US"),
        ({"levels": [0.05, 0.1, 0.35, 0.5], "new_grads": 0.1}, "not entry level"),
    ],
)
def test_screen_rejects_on_any_clear_no(kwargs, reason):
    decision = decide_screen(screen(**kwargs))
    assert decision.outcome == ScreenOutcome.REJECT
    assert reason in decision.reason


def test_explicit_new_grad_invitation_overrides_a_low_experience_level():
    # First live run: "New grads ... (max 2 years out of school)" scored 0.13 on the experience levels.
    decision = decide_screen(screen(levels=[0.05, 0.08, 0.6, 0.27], new_grads=0.95))
    assert decision.outcome == ScreenOutcome.PASS
    assert decision.signals["entry_level"] == pytest.approx(0.95)


def test_screen_middle_band_is_its_own_outcome():
    decision = decide_screen(screen(us=0.5, levels=[0.3, 0.3, 0.3, 0.1], new_grads=0.4))
    assert decision.outcome == ScreenOutcome.UNSETTLED
    assert "us_location" in decision.reason and "entry_level" in decision.reason


def test_thresholds_are_documented_defaults():
    assert (YES, NO) == (0.7, 0.3)


def test_fit_questions_one_per_topic_plus_mentorship(jev_profile):
    names = set(fit_questions(jev_profile))
    assert names == {"interest__applied_ml", "interest__healthcare", "growth__ml_in_production", "mentorship"}


def test_combine_fit_math(jev_profile):
    response = SystemOneResponse.model_validate(
        jev.response(
            {
                "interest__applied_ml": jev.noul(0.8),
                "interest__healthcare": jev.noul(0.6),
                "growth__ml_in_production": jev.noul(0.9),
                "mentorship": jev.noul(0.5),
            }
        )
    )
    signals = combine_fit(response, jev_profile)
    # 1 - (1 - 1.0*0.8) * (1 - 0.5*0.6) = 1 - 0.2*0.7
    assert signals.interest == pytest.approx(0.86)
    assert signals.learning == pytest.approx(0.6 * 0.9 + 0.4 * 0.5)
    assert signals.interest_note.startswith("applied ML 0.80")
    assert "healthcare 0.60?" in signals.interest_note  # unsettled band is marked
    assert signals.detail["provider"] == "jev"


# --- pipeline behavior -----------------------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_rejected_jobs_skip_extraction_and_failed_screens_do_not(
    db_session, requirements: JobRequirements, aliases
):
    await add_job(db_session, "https://job-boards.greenhouse.io/acme/jobs/1")
    await add_job(db_session, "https://job-boards.greenhouse.io/acme/jobs/2", text="Senior staff role, 10 years.")
    await add_job(db_session, "https://job-boards.greenhouse.io/acme/jobs/3", text="Something the API chokes on.")

    def reply(request: httpx.Request) -> httpx.Response:
        description = json.loads(request.content)["state"]["posting"]["description"]
        if "Senior" in description:
            return httpx.Response(200, json=jev.screen_response(levels=[0.01, 0.02, 0.07, 0.9], new_grads=0.02))
        if "chokes" in description:
            return httpx.Response(400, json={"error": "max_tokens_exceeded"})
        return httpx.Response(200, json=jev.screen_response())

    respx.post(API).mock(side_effect=reply)
    extract_route = respx.post("https://or.test/chat/completions").respond(
        json={"choices": [{"message": {"content": requirements.model_dump_json()}}]}
    )
    async with HttpClient() as http:
        stats = await screen_jobs(db_session, client(http), max_chars=5000)
        assert (stats.passed, stats.rejected, stats.failed) == (1, 1, 1)
        assert stats.input_tokens == 2400

        # Already-screened snapshots are not screened again; the failed one is retried.
        again = await screen_jobs(db_session, client(http), max_chars=5000)
        assert again.failed == 1 and again.passed == 0

        llm = OpenRouterClient(http, SecretStr("k"), "https://or.test")
        extracted = await extract_jobs(db_session, llm, "m", aliases, max_chars=1000)
    # The rejected job is skipped; the passed job AND the failed-screen job are extracted.
    assert extracted.extracted == 2 and extract_route.call_count == 2
    screens = (await db_session.execute(select(JobScreen))).scalars().all()
    assert {s.outcome for s in screens} == {ScreenOutcome.PASS, ScreenOutcome.REJECT}
    assert len((await db_session.execute(select(JobExtraction))).scalars().all()) == 2


@pytest.mark.asyncio
@respx.mock
async def test_screen_rejects_are_hidden_from_queue(db_session, profile):
    await add_job(db_session, "https://job-boards.greenhouse.io/acme/jobs/9")
    respx.post(API).respond(json=jev.screen_response(role=0.05))
    async with HttpClient() as http:
        await screen_jobs(db_session, client(http), max_chars=5000)
    assert await build_queue(db_session, profile.content_hash, include_reviewed=True, include_unscored=True) == []


@pytest.mark.asyncio
@respx.mock
async def test_jev_fit_scores_and_caches_by_topics(db_session, requirements, jev_profile, aliases):
    await add_job(db_session, "https://job-boards.greenhouse.io/acme/jobs/5")
    respx.post("https://or.test/chat/completions").respond(
        json={"choices": [{"message": {"content": requirements.model_dump_json()}}]}
    )
    fit_route = respx.post(API).respond(
        json=jev.response(
            {
                "interest__applied_ml": jev.noul(0.9),
                "interest__healthcare": jev.noul(0.2),
                "growth__ml_in_production": jev.noul(0.8),
                "mentorship": jev.noul(0.7),
            }
        )
    )
    async with HttpClient() as http:
        await extract_jobs(db_session, OpenRouterClient(http, SecretStr("k"), "https://or.test"), "m", aliases, 1000)
        provider = JevFitProvider(client(http), max_description_chars=5000)
        first = await score_jobs(db_session, jev_profile, aliases, provider)
        assert (first.scored, first.fit_calls) == (1, 1)

        # Changing a weight elsewhere in the profile reuses the Jev answers.
        reweighted = Profile.model_validate(
            jev_profile.model_dump() | {"compensation": {"floor_usd": 50000, "target_usd": 90000}}
        )
        second = await score_jobs(db_session, reweighted, aliases, provider)
        assert (second.scored, second.fit_calls) == (1, 0)

        # Changing an interest topic asks again.
        data = jev_profile.model_dump()
        data["candidate"]["interest_topics"][1]["description"] = "clinical research problems"
        third = await score_jobs(db_session, Profile.model_validate(data), aliases, provider)
        assert third.fit_calls == 1
    assert fit_route.call_count == 2
    sent = json.loads(fit_route.calls[0].request.content)
    assert "description" in sent["state"]["job"]
    score = (await db_session.execute(select(JobScore).limit(1))).scalars().first()
    assert score is not None
    assert score.components["value"]["interest"]["note"].startswith("applied ML 0.90")
    assert score.components["fit"]["detail"]["provider"] == "jev"


def test_degree_question_only_for_bachelors_profiles_and_rejects_grad_only_postings():
    assert "bachelors_accepted" in screen_questions("bachelors")
    assert "bachelors_accepted" not in screen_questions("masters")

    decision = decide_screen(screen(levels=[0.9, 0.08, 0.01, 0.01], new_grads=0.98, bachelors=0.04))
    assert decision.outcome == ScreenOutcome.REJECT and "master's or PhD" in decision.reason

    unsure = decide_screen(screen(levels=[0.9, 0.08, 0.01, 0.01], new_grads=0.98, bachelors=0.5))
    assert unsure.outcome == ScreenOutcome.UNSETTLED and "bachelors_accepted" in unsure.reason

    # A graduate-degree profile isn't asked, so a missing answer never blocks a pass.
    clear = decide_screen(screen(levels=[0.9, 0.08, 0.01, 0.01], new_grads=0.98, bachelors=None))
    assert clear.outcome == ScreenOutcome.PASS
