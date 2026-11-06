import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import select

from jobhunt.models.pipeline import LlmUsage
from jobhunt.services.fit import FitJudgment
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import resolve_leads, upsert_leads
from jobhunt.services.llm import OpenRouterClient
from jobhunt.services.status import default_per_job, gather_status, per_job_usage
from jobhunt.services.usage import UsageMeter, describe, price, save_usage
from jobhunt.sources.base import RawLead

FIT = '{"interest_alignment":"weak","learning_potential":"weak","rationale":"x"}'


def test_meter_attributes_calls_to_the_open_step_and_prefers_reported_cost():
    meter = UsageMeter()
    with meter.step("extract"):
        meter.record("openrouter", "anthropic/claude-haiku-5.5", 5_000, 1_000, cost=0.002)
        meter.record("openrouter", "anthropic/claude-haiku-5.5", 5_000, 1_000, cost=0.002)
    meter.record("typesafe", "jev-latest", 2_000, 0)  # outside any step

    meter.set_items("extract", 2)
    (extract,) = meter.take("extract")
    assert (extract.calls, extract.items, extract.input_tokens) == (2, 2, 10_000)
    assert extract.cost == pytest.approx(0.004) and not extract.estimated
    assert meter.take("extract") == []

    (other,) = meter.take("other")
    assert other.estimated
    assert other.cost == pytest.approx(price("jev-latest", 2_000, 0))
    assert "estimated" in describe([other])


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_client_reports_usage_from_the_response():
    respx.post("https://or.test/chat/completions").respond(
        json={
            "choices": [{"message": {"content": FIT}}],
            "usage": {"prompt_tokens": 1_200, "completion_tokens": 80, "cost": 0.00016},
        }
    )
    meter = UsageMeter()
    async with HttpClient() as client:
        llm = OpenRouterClient(client, SecretStr("k"), "https://or.test", meter=meter)
        with meter.step("fit"):
            await llm.structured("anthropic/claude-haiku-5.5", "s", "u", FitJudgment)
    (fit,) = meter.take("fit")
    assert (fit.input_tokens, fit.output_tokens, fit.cost) == (1_200, 80, pytest.approx(0.00016))


@pytest.mark.asyncio
async def test_saved_usage_drives_per_job_estimates(db_session):
    model = "anthropic/claude-haiku-5.5"
    assert not (await per_job_usage(db_session, "extract", model)).from_history
    assert default_per_job("extract", model).cost == pytest.approx(price(model, 5_200, 1_000))

    meter = UsageMeter()
    with meter.step("extract"):
        for _ in range(4):
            meter.record("openrouter", model, 6_000, 1_000, cost=0.0011)
    meter.set_items("extract", 2)  # 4 calls for 2 jobs (retries)
    await save_usage(db_session, meter.take("extract"))

    row = (await db_session.execute(select(LlmUsage))).scalar_one()
    assert (row.stage, row.calls, row.items) == ("extract", 4, 2)
    per_job = await per_job_usage(db_session, "extract", model)
    assert per_job.from_history
    assert per_job.input_tokens == 12_000 and per_job.cost == pytest.approx(0.0022)


@pytest.mark.asyncio
async def test_status_counts_waiting_work_and_recommends_next_command(db_session):
    await upsert_leads(
        db_session,
        [
            RawLead(source="simplify", company="Acme", title="ML Engineer",
                    url="https://job-boards.greenhouse.io/acme/jobs/1", description_text="ML role text."),
            RawLead(source="simplify", company="Beta", title="SWE", url="https://job-boards.greenhouse.io/beta/jobs/2"),
        ],
    )  # fmt: skip
    await db_session.commit()

    status = await gather_status(db_session, "profile", "jev-latest", "anthropic/claude-haiku-5.5", None)
    steps = {step.command: step for step in status.steps}
    assert steps["resolve"].pending == 2
    assert steps["resolve"].model is None and steps["resolve"].est_cost == 0
    assert status.next_command.startswith("jobhunt run")

    await resolve_leads(db_session, [], client=None)
    status = await gather_status(db_session, "profile", "jev-latest", "anthropic/claude-haiku-5.5", None)
    steps = {step.command: step for step in status.steps}
    assert steps["hydrate"].pending == 1  # Beta has no description yet
    assert steps["screen"].pending == 1
    # No screen history yet: assume half of what's screened is rejected before extraction.
    assert steps["extract"].pending in (0, 1)
    assert steps["screen"].est_cost > 0
    assert steps["score"].model is None and "rule-based" in steps["score"].note

    off = await gather_status(db_session, "profile", None, None, None)
    assert {step.command: step.pending for step in off.steps}["screen"] == 0
    assert off.est_total == 0
