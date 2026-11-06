"""CLI commands that don't need the network, run against the throwaway database set up in conftest."""

import json
from pathlib import Path

from typer.testing import CliRunner

from jobhunt.cli import app

runner = CliRunner()


def test_import_resolve_and_list_unresolved(tmp_path: Path):
    path = tmp_path / "opps.csv"
    path.write_text(
        "Company,Role,Link,Source\n"
        "Crowe,ML Software Engineer,https://www.linkedin.com/jobs/view/4454811409/,LinkedIn\n"
        "Durable,AI Engineer,https://job-boards.greenhouse.io/durable/jobs/5156304008,Greenhouse\n"
    )
    result = runner.invoke(app, ["import-csv", str(path)])
    assert result.exit_code == 0, result.output
    assert "2 new leads" in result.output

    result = runner.invoke(app, ["resolve"])
    assert result.exit_code == 0, result.output
    assert "1 need a link" in result.output

    result = runner.invoke(app, ["unresolved", "--json"])
    rows = json.loads(result.output)
    assert [row["company"] for row in rows] == ["Crowe"]


def test_import_csv_reports_bad_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("Name\nx\n")
    result = runner.invoke(app, ["import-csv", str(path)])
    assert result.exit_code == 1


def test_extract_requires_api_key():
    result = runner.invoke(app, ["extract"])
    assert result.exit_code == 1


def test_queue_and_skills_run_on_empty_scores():
    assert runner.invoke(app, ["queue", "--json"]).exit_code == 0
    assert runner.invoke(app, ["skills", "--json"]).exit_code == 0
    assert runner.invoke(app, ["health"]).exit_code == 0


def test_unknown_source_group_fails():
    result = runner.invoke(app, ["ingest", "--source", "myspace"])
    assert result.exit_code == 1


def test_jev_probe_writes_separate_payloads_without_sending(tmp_path: Path):
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("Company,Role,Link\nProbeCo,ML Engineer,https://news.ycombinator.com/item?id=777\n")
    assert runner.invoke(app, ["import-csv", str(csv_path)]).exit_code == 0

    # Give the job a description without the network.
    import asyncio

    from sqlalchemy import select

    from jobhunt.models.pipeline import Job, Lead
    from jobhunt.services.db import get_session
    from jobhunt.services.jobs import add_snapshot, resolve_leads

    async def describe() -> str:
        async with get_session() as session:
            await resolve_leads(session, [], client=None)
            lead = (await session.execute(select(Lead).where(Lead.company == "ProbeCo"))).scalar_one()
            job = await session.get(Job, lead.job_id)
            assert job is not None
            await add_snapshot(session, job, "New grads build ML systems.", {})
            await session.commit()
            return str(job.id)

    job_id = asyncio.run(describe())
    out = tmp_path / "probe"
    result = runner.invoke(app, ["jev-probe", "--job", job_id, "--out", str(out)])
    assert result.exit_code == 0, result.output
    screen = json.loads((out / f"{job_id}.screen.request.json").read_text())
    fit = json.loads((out / f"{job_id}.fit.request.json").read_text())
    assert set(screen["questions"]) == {
        "software_ai_ml_role",
        "internship",
        "us_location",
        "open_to_new_grads",
        "bachelors_accepted",
        "experience_level",
    }
    assert "mentorship" in fit["questions"]
    assert not list(out.glob("*.response.json"))


def test_jev_probe_send_requires_key():
    assert runner.invoke(app, ["jev-probe", "--send"]).exit_code == 1


def test_capability_and_grouped_skill_commands_run_on_empty_data():
    assert runner.invoke(app, ["skills"]).exit_code == 0
    assert runner.invoke(app, ["skills", "--unsorted", "--json"]).exit_code == 0
    assert runner.invoke(app, ["skills", "--group", "nope"]).exit_code == 1
    result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0 and "No capability evidence yet" in result.output
    assert runner.invoke(app, ["capabilities", "customer_iteration", "--json"]).exit_code == 0
    assert runner.invoke(app, ["capabilities", "not_a_capability"]).exit_code == 1
    assert runner.invoke(app, ["skills-classify"]).exit_code == 1  # no API key in tests


def test_boards_commands():
    assert runner.invoke(app, ["boards", "add", "greenhouse:clitestco"]).exit_code == 0
    assert "already watched" in runner.invoke(app, ["boards", "add", "greenhouse:clitestco"]).output
    assert runner.invoke(app, ["boards", "pause", "greenhouse:clitestco"]).exit_code == 0
    assert runner.invoke(app, ["boards", "resume", "https://job-boards.greenhouse.io/clitestco/jobs/1"]).exit_code == 0
    assert runner.invoke(app, ["boards", "pause", "greenhouse:not-watched"]).exit_code == 1
    assert runner.invoke(app, ["boards", "add", "workday:umd"]).exit_code == 1
    summary = runner.invoke(app, ["boards"])
    assert summary.exit_code == 0 and "Watched boards" in summary.output
    assert runner.invoke(app, ["health", "--boards"]).exit_code == 0


def test_status_and_help_explain_the_steps():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "Next:" in result.output and "extract" in result.output
    help_text = runner.invoke(app, ["--help"]).output
    assert help_text.index("Start here") < help_text.index("Pipeline steps") < help_text.index("Maintenance")


def test_chatgpt_provider_without_sign_in_says_how_to_fix(monkeypatch, tmp_path: Path):
    from jobhunt.settings import settings

    monkeypatch.setattr(settings, "llm_provider", "chatgpt")
    monkeypatch.setattr(settings, "chatgpt_credentials_path", tmp_path / "none.json")
    result = runner.invoke(app, ["extract"])
    assert result.exit_code == 1 and "jobhunt chatgpt login" in " ".join(result.output.split())
    status = runner.invoke(app, ["chatgpt", "status"])
    assert status.exit_code == 0 and "Not signed in" in status.output


def test_show_finds_a_job_by_id_prefix(tmp_path: Path):
    import asyncio

    from sqlalchemy import select

    from jobhunt.models.pipeline import Job
    from jobhunt.services.db import get_session
    from jobhunt.services.jobs import resolve_leads, upsert_leads
    from jobhunt.sources.base import RawLead

    async def make() -> str:
        async with get_session() as session:
            await upsert_leads(
                session,
                [RawLead(source="simplify", company="ShowCo", title="ML Engineer",
                         url="https://job-boards.greenhouse.io/showco/jobs/42",
                         description_text="PhD or MS in Computer Science required.")],
            )  # fmt: skip
            await session.commit()
            await resolve_leads(session, [], client=None)
            job = (await session.execute(select(Job).where(Job.company == "ShowCo"))).scalar_one()
            return job.id.hex

    job_id = asyncio.run(make())
    result = runner.invoke(app, ["show", job_id[:8]])
    assert result.exit_code == 0, result.output
    assert "ShowCo" in result.output and "PhD or MS" in result.output and "not screened yet" in result.output
    assert runner.invoke(app, ["show", "zzzz"]).exit_code == 1


def test_skills_and_capabilities_export_csv():
    for args in (["skills", "--csv"], ["capabilities", "--csv"], ["capabilities", "--phrases", "--csv"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)


def test_echo_csv_writes_a_header_and_rounds_floats(capsys):
    from jobhunt.cli import _echo_csv

    _echo_csv([{"skill": "PyTorch", "weighted": 3.14159, "jobs": 37}])
    assert capsys.readouterr().out.splitlines() == ["skill,weighted,jobs", "PyTorch,3.14,37"]
