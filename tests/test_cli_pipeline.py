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
    assert set(screen["questions"]) == {"software_ai_ml_role", "internship", "us_location", "experience_level"}
    assert "mentorship" in fit["questions"]
    assert not list(out.glob("*.response.json"))


def test_jev_probe_send_requires_key():
    assert runner.invoke(app, ["jev-probe", "--send"]).exit_code == 1
