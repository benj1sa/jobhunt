"""Capability evidence extraction, grouped skills, and LLM-assisted skill classification."""

import json
import tomllib
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import select

from jobhunt.models.pipeline import JobCapability, JobSkill
from jobhunt.schemas.config import Capability, load_skill_taxonomy, skill_key, suggested_path
from jobhunt.schemas.extraction import JobRequirements, extraction_json_schema
from jobhunt.services.extraction import extract_jobs
from jobhunt.services.http import HttpClient
from jobhunt.services.jobs import resolve_leads, upsert_leads
from jobhunt.services.llm import OpenRouterClient
from jobhunt.services.queue import capability_phrases, capability_report, skill_group_report, skill_report
from jobhunt.services.skill_classify import classify_skills
from jobhunt.sources.base import RawLead

API = "https://or.test/chat/completions"
CAPS = [
    Capability(id="customer_iteration", name="Customer-facing iteration", description="Users and feedback."),
    Capability(id="code_quality", name="Clean code", description="Maintainable code."),
]
POSTING = (
    "About the role\n- Work directly with customers to understand their problems and turn feedback into product "
    "improvements.\n- Write clean, maintainable code and contribute thoughtful feedback and review.\n- Python"
)

GROUPS_TOML = """
[groups.cloud]
name = "Cloud platforms"
[groups.ml_core]
name = "Machine learning (core)"

[skills]
"Amazon Web Services" = { group = "cloud", kind = "tool" }
"AWS" = { group = "cloud", kind = "tool", same_as = "Amazon Web Services" }
"Amazon SageMaker" = { group = "cloud", kind = "tool" }
"Cloud computing" = { group = "cloud", kind = "concept" }
"Machine Learning" = { group = "ml_core", kind = "concept" }
"""


def reply(payload: str) -> dict:
    return {"choices": [{"message": {"content": payload}}]}


def requirements_with(evidence: list[dict], skills: list[str], base: JobRequirements) -> str:
    return JobRequirements.model_validate(
        base.model_dump() | {"capability_evidence": evidence, "required_skills": skills, "preferred_skills": []}
    ).model_dump_json()


async def add_job(session, n: int, text: str = POSTING) -> None:
    await upsert_leads(
        session,
        [RawLead(source="csv", company=f"Co{n}", title="SWE", url=f"https://job-boards.greenhouse.io/c/jobs/{n}",
                 description_text=text)],
    )  # fmt: skip
    await session.commit()
    await resolve_leads(session, [], client=None)


def test_extraction_schema_injects_capability_enum():
    schema = extraction_json_schema(["customer_iteration", "code_quality"])
    items = schema["$defs"]["CapabilityEvidence"]["properties"]["capabilities"]["items"]
    assert items["enum"] == ["customer_iteration", "code_quality", "other"]
    assert "capability_evidence" in schema["required"]


def test_skill_key_merges_spelling_but_keeps_c_family_apart():
    assert skill_key("Test-Driven Development") == skill_key("test driven development")
    assert len({skill_key("C"), skill_key("C++"), skill_key("C#")}) == 3


@pytest.mark.asyncio
@respx.mock
async def test_capability_evidence_is_stored_checked_and_reported(db_session, requirements, aliases):
    await add_job(db_session, 1)
    evidence = [
        {"phrase": "Work directly with customers to understand their problems and turn feedback into product "
                   "improvements.", "capabilities": ["customer_iteration"]},
        {"phrase": "Write clean, maintainable code and contribute thoughtful feedback and review.",
         "capabilities": ["code_quality", "made_up_id"]},
        {"phrase": "Be a great communicator", "capabilities": ["other"]},
    ]  # fmt: skip
    route = respx.post(API).respond(json=reply(requirements_with(evidence, ["Python"], requirements)))
    async with HttpClient() as http:
        llm = OpenRouterClient(http, SecretStr("k"), "https://or.test")
        stats = await extract_jobs(db_session, llm, "m", aliases, 5000, capabilities=CAPS)
    assert stats.extracted == 1
    body = json.loads(route.calls[0].request.content)
    assert "customer_iteration:" in body["messages"][0]["content"]
    sent_enum = body["response_format"]["json_schema"]["schema"]["$defs"]["CapabilityEvidence"]
    assert "customer_iteration" in sent_enum["properties"]["capabilities"]["items"]["enum"]

    rows = (await db_session.execute(select(JobCapability))).scalars().all()
    by_tag = {(r.capability, r.verbatim) for r in rows}
    # The unknown id became "other"; the invented phrase is flagged as not verbatim.
    assert ("customer_iteration", True) in by_tag and ("code_quality", True) in by_tag
    assert ("other", True) in by_tag and ("other", False) in by_tag

    phrases = await capability_phrases(db_session, "customer_iteration")
    assert [p.company for p in phrases] == ["Co1"] and phrases[0].phrase.startswith("Work directly")
    report = {s.capability: s for s in await capability_report(db_session, "no-profile")}
    assert report["customer_iteration"].jobs == 1 and report["other"].phrases == 2


@pytest.mark.asyncio
@respx.mock
async def test_rerun_replaces_extraction_instead_of_duplicating(db_session, requirements, aliases):
    await add_job(db_session, 2)
    first = requirements_with([{"phrase": "Write clean, maintainable code", "capabilities": ["code_quality"]}],
                              ["Python", "AWS"], requirements)  # fmt: skip
    second = requirements_with([], ["Go"], requirements)
    respx.post(API).mock(side_effect=[httpx.Response(200, json=reply(first)), httpx.Response(200, json=reply(second))])
    async with HttpClient() as http:
        llm = OpenRouterClient(http, SecretStr("k"), "https://or.test")
        await extract_jobs(db_session, llm, "m", aliases, 5000, capabilities=CAPS)
        assert (await extract_jobs(db_session, llm, "m", aliases, 5000, capabilities=CAPS)).extracted == 0
        assert (await extract_jobs(db_session, llm, "m", aliases, 5000, capabilities=CAPS, rerun=True)).extracted == 1
    assert set((await db_session.execute(select(JobSkill.skill))).scalars()) == {"Go"}
    assert (await db_session.execute(select(JobCapability))).scalars().all() == []


@pytest.mark.asyncio
@respx.mock
async def test_grouped_skills_merge_spellings_and_separate_broad_from_specific(
    db_session, requirements, aliases, tmp_path: Path
):
    path = tmp_path / "skill_groups.toml"
    path.write_text(GROUPS_TOML)
    taxonomy = load_skill_taxonomy(path)
    payloads = [
        requirements_with([], ["Amazon Web Services", "Amazon SageMaker"], requirements),
        requirements_with([], ["AWS", "Machine Learning"], requirements),
        requirements_with([], ["cloud computing", "Machine learning", "Rust"], requirements),
    ]
    for n in (10, 11, 12):
        await add_job(db_session, n)
    respx.post(API).mock(side_effect=[httpx.Response(200, json=reply(p)) for p in payloads])
    async with HttpClient() as http:
        await extract_jobs(db_session, OpenRouterClient(http, SecretStr("k"), "https://or.test"), "m", aliases, 5000)

    flat = {s.skill: s for s in await skill_report(db_session, "p", {"rust"}, taxonomy=taxonomy)}
    assert flat["Amazon Web Services"].jobs == 2  # "AWS" merged via same_as
    assert flat["Machine Learning"].jobs == 2  # case-insensitive merge
    assert flat["Rust"].have and flat["Rust"].group is None

    groups, unsorted = await skill_group_report(db_session, "p", set(), taxonomy)
    cloud = next(g for g in groups if g.group == "cloud")
    assert cloud.jobs == 3 and cloud.concept_only_jobs == 1  # job 12 said only "cloud computing"
    assert [t.skill for t in cloud.tools] == ["Amazon Web Services", "Amazon SageMaker"]
    assert [s.skill for s in unsorted] == ["Rust"]


@pytest.mark.asyncio
@respx.mock
async def test_classify_writes_reviewable_suggestions_that_load(tmp_path: Path):
    path = tmp_path / "skill_groups.toml"
    path.write_text(GROUPS_TOML)
    taxonomy = load_skill_taxonomy(path)
    items = {"items": [
        {"skill": "Google Vertex AI", "group": "cloud", "kind": "tool", "same_as": None},
        {"skill": "aws s3", "group": "cloud", "kind": "tool", "same_as": "amazon web services"},
        {"skill": "Google Vertex AI", "group": "cloud", "kind": "tool", "same_as": None},
        {"skill": "Palantir", "group": "skip", "kind": "tool", "same_as": None},
    ]}  # fmt: skip
    route = respx.post(API).respond(json=reply(json.dumps(items)))
    async with HttpClient() as http:
        llm = OpenRouterClient(http, SecretStr("k"), "https://or.test")
        stats = await classify_skills(llm, "m", taxonomy, ["Google Vertex AI", "AWS S3", "Palantir"], path)
    assert (stats.suggested, stats.skipped) == (2, 2)
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"]["json_schema"]["schema"]["$defs"]["SkillSuggestion"]["properties"]["group"][
        "enum"
    ] == ["cloud", "ml_core", "skip"]
    suggested = tomllib.loads(suggested_path(path).read_text())["skills"]
    assert suggested["AWS S3"]["same_as"] == "Amazon Web Services"
    reloaded = load_skill_taxonomy(path)
    assert reloaded.classify("aws s3").name == "Amazon Web Services"
    assert reloaded.classify("Google Vertex AI").group == "cloud"
