import asyncio
import csv
import json
import logging
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from .settings import settings

app = typer.Typer(help="Collect, score and review entry-level SWE / AI / ML job postings.", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger(__name__)


def syncify(f: Callable[..., Any]) -> Callable[..., Any]:
    """This simple decorator converts an async function into a sync function,
    allowing it to work with Typer.
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose or settings.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # httpx logs every request at INFO; keep that out of normal runs.
    logging.getLogger("httpx").setLevel(logging.DEBUG if verbose else logging.WARNING)


def _fail(message: str) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(code=1)


def _print_errors(errors: list[str], limit: int = 10) -> None:
    for error in errors[:limit]:
        err_console.print(f"  [yellow]![/] {error}")
    if len(errors) > limit:
        err_console.print(f"  ... and {len(errors) - limit} more (run with -v for details)")


# --- Setup ------------------------------------------------------------------------------------------------------


@app.command(help="Create or upgrade the database schema to the latest migration.")
def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    config_path = Path("alembic.ini")
    if not config_path.exists():
        _fail("alembic.ini not found. Run this from the project root.")
    command.upgrade(Config(str(config_path)), "head")
    console.print("Database is up to date.")


@app.command(help="Install testing data for local development.")
@syncify
async def test_data() -> None:
    from . import __version__
    from .services.db import get_session, test_data

    typer.echo(f"{settings.project_name} - {__version__}")

    async with get_session() as session:
        await test_data(session)

    typer.echo("Development data installed successfully.")


@app.command(help=f"Display the current installed version of {settings.project_name}.")
def version() -> None:
    from . import __version__

    typer.echo(f"{settings.project_name} - {__version__}")


# --- Pipeline stages --------------------------------------------------------------------------------------------


@app.command(help="Discover leads from sources (simplify, hn, boards). Default: all.")
@syncify
async def ingest(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source group to run; repeatable.")] = None,
) -> None:
    from .services.context import app_context
    from .services.jobs import ingest as run_ingest

    async with app_context() as ctx:
        try:
            sources = ctx.sources(source)
        except ValueError as exc:
            _fail(str(exc))
        results = await run_ingest(ctx.session, ctx.http, sources)
    table = Table("Source", "Status", "Seen", "New", "Error")
    for result in results:
        style = "green" if result.status.value == "ok" else "red"
        table.add_row(
            result.source, f"[{style}]{result.status.value}[/]", str(result.seen), str(result.new), result.error or ""
        )
    console.print(table)


@app.command("import-csv", help="Import leads from a CSV with Company, Role, Link and optional Source columns.")
@syncify
async def import_csv(
    path: Annotated[Path, typer.Argument(help="CSV file to import.", exists=True, dir_okay=False)],
    default_source: Annotated[str, typer.Option(help="Source label for rows without one.")] = "manual",
) -> None:
    from .services.context import app_context
    from .services.jobs import upsert_leads
    from .services.manual import CsvFormatError, leads_from_csv

    try:
        leads = leads_from_csv(path, default_source=default_source)
    except CsvFormatError as exc:
        _fail(str(exc))
    async with app_context() as ctx:
        created = await upsert_leads(ctx.session, leads)
        await ctx.session.commit()
    console.print(f"Read {len(leads)} rows; {created} new leads. Run `jobhunt resolve` next.")


@app.command(help="Add one job by link.")
@syncify
async def add(
    url: Annotated[str, typer.Argument(help="Link to the posting.")],
    company: Annotated[str, typer.Option("--company", "-c", help="Company name.")],
    title: Annotated[str, typer.Option("--title", "-t", help="Job title.")],
    source: Annotated[str, typer.Option("--source", "-s", help="Where you found it, e.g. 'ColorStack Slack'.")],
) -> None:
    from .services.context import app_context
    from .services.jobs import resolve_leads, upsert_leads
    from .services.manual import lead_from_args

    async with app_context() as ctx:
        await upsert_leads(ctx.session, [lead_from_args(url, company, title, source)])
        await ctx.session.commit()
        stats = await resolve_leads(ctx.session, ctx.boards, ctx.http)
    if stats.needs_url:
        console.print("Saved, but this board blocks automation. Attach the real posting with `jobhunt resolve`.")
    else:
        console.print("Added. Run `jobhunt run --skip-ingest` to hydrate, extract and score it.")


@app.command(help="Resolve new leads to canonical jobs, or attach a posting link to one lead.")
@syncify
async def resolve(
    lead_id: Annotated[str | None, typer.Argument(help="Lead to resolve by hand (see `jobhunt unresolved`).")] = None,
    url: Annotated[str | None, typer.Argument(help="The company's own ATS posting link for that lead.")] = None,
    limit: Annotated[int | None, typer.Option(help="Resolve at most this many leads.")] = None,
) -> None:
    from .services.context import app_context
    from .services.jobs import attach_url, resolve_leads

    async with app_context() as ctx:
        if lead_id:
            if not url:
                _fail("Give both a lead id and a posting URL.")
            try:
                job = await attach_url(ctx.session, UUID(lead_id), url or "", ctx.boards, ctx.http)
            except (LookupError, ValueError) as exc:
                _fail(str(exc))
            console.print(f"Linked to {job.company} - {job.title} ({job.canonical_key}).")
            return
        stats = await resolve_leads(ctx.session, ctx.boards, ctx.http, limit=limit)
    console.print(
        f"Resolved {stats.resolved} ({stats.new_jobs} new jobs), {stats.needs_url} need a link, {stats.failed} failed."
    )
    _print_errors(stats.errors)


@app.command(help="Fetch full descriptions for jobs that don't have one yet.")
@syncify
async def hydrate(
    refresh: Annotated[bool, typer.Option(help="Re-fetch every open job (detects closed postings).")] = False,
    limit: Annotated[int | None, typer.Option(help="Hydrate at most this many jobs.")] = None,
) -> None:
    from .services.context import app_context
    from .services.jobs import hydrate_jobs

    async with app_context() as ctx:
        stats = await hydrate_jobs(ctx.session, ctx.http, refresh=refresh, limit=limit)
    console.print(
        f"Hydrated {stats.hydrated}, unchanged {stats.unchanged}, closed {stats.closed}, failed {stats.failed}."
    )
    _print_errors(stats.errors)


@app.command(help="Extract structured requirements and skills with the LLM.")
@syncify
async def extract(
    limit: Annotated[int | None, typer.Option(help="Extract at most this many jobs.")] = None,
) -> None:
    from .services.context import app_context
    from .services.extraction import extract_jobs

    async with app_context() as ctx:
        if ctx.llm is None:
            _fail("OPENROUTER_API_KEY is not set; extraction needs it.")
        assert ctx.llm is not None
        stats = await extract_jobs(
            ctx.session, ctx.llm, settings.extraction_model, ctx.aliases, settings.llm_max_description_chars, limit
        )
    console.print(f"Extracted {stats.extracted} ({stats.relevant} relevant), failed {stats.failed}.")
    _print_errors(stats.errors)


@app.command(help="Screen new descriptions with Jev so obvious misfits skip LLM extraction.")
@syncify
async def screen(
    limit: Annotated[int | None, typer.Option(help="Screen at most this many jobs.")] = None,
) -> None:
    from .services.context import app_context
    from .services.screening import screen_jobs

    async with app_context() as ctx:
        if ctx.jev is None:
            _fail("TYPESAFE_API_KEY is not set; the screen needs it.")
        assert ctx.jev is not None
        stats = await screen_jobs(ctx.session, ctx.jev, settings.screen_max_description_chars, limit)
    console.print(
        f"Screened: {stats.passed} pass, {stats.unsettled} unsettled, {stats.rejected} rejected, "
        f"{stats.failed} failed ({stats.input_tokens:,} input tokens)."
    )
    _print_errors(stats.errors)


@app.command(help="Score relevant jobs for value and likelihood using config/profile.toml.")
@syncify
async def score(
    rescore: Annotated[bool, typer.Option(help="Recompute scores that already exist for this profile.")] = False,
    fit: Annotated[
        str | None,
        typer.Option(help="Fit provider: jev, openrouter or none. Default: the FIT_PROVIDER setting (auto)."),
    ] = None,
) -> None:
    from .services.context import app_context
    from .services.scoring import score_jobs

    if fit not in (None, "jev", "openrouter", "none"):
        _fail("--fit must be jev, openrouter or none.")
    async with app_context() as ctx:
        try:
            provider = ctx.fit_provider(fit)
        except ValueError as exc:
            _fail(str(exc))
        stats = await score_jobs(ctx.session, ctx.profile(), ctx.aliases, provider, rescore=rescore)
    name = provider.name if provider else "none"
    console.print(f"Scored {stats.scored} jobs ({stats.fit_calls} {name} fit calls), failed {stats.failed}.")
    _print_errors(stats.errors)


@app.command(help="Run the whole pipeline: ingest, resolve, hydrate, screen, extract, score.")
@syncify
async def run(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source group; repeatable.")] = None,
    skip_ingest: Annotated[bool, typer.Option(help="Start from resolve (e.g. after import-csv).")] = False,
    skip_score: Annotated[bool, typer.Option(help="Stop after extraction (the scheduled workflow does this).")] = False,
) -> None:
    from .services.context import app_context
    from .services.extraction import extract_jobs
    from .services.jobs import hydrate_jobs, resolve_leads
    from .services.jobs import ingest as run_ingest
    from .services.scoring import score_jobs
    from .services.screening import screen_jobs

    async with app_context() as ctx:
        if not skip_ingest:
            results = await run_ingest(ctx.session, ctx.http, ctx.sources(source))
            failed = [r.source for r in results if r.status.value != "ok"]
            console.print(
                f"Ingest: {sum(r.new for r in results)} new leads from {len(results)} sources"
                + (f"; [red]failed: {', '.join(failed)}[/]" if failed else "")
            )
        resolved = await resolve_leads(ctx.session, ctx.boards, ctx.http)
        console.print(f"Resolve: {resolved.new_jobs} new jobs, {resolved.needs_url} need a link")
        hydrated = await hydrate_jobs(ctx.session, ctx.http)
        console.print(f"Hydrate: {hydrated.hydrated} described, {hydrated.closed} closed, {hydrated.failed} failed")
        if ctx.jev is not None:
            screened = await screen_jobs(ctx.session, ctx.jev, settings.screen_max_description_chars)
            console.print(
                f"Screen: {screened.passed} pass, {screened.unsettled} unsettled, {screened.rejected} rejected, "
                f"{screened.failed} failed"
            )
        if ctx.llm is None:
            console.print("[yellow]OPENROUTER_API_KEY not set: skipping extraction and scoring.[/]")
            return
        extracted = await extract_jobs(
            ctx.session, ctx.llm, settings.extraction_model, ctx.aliases, settings.llm_max_description_chars
        )
        console.print(f"Extract: {extracted.extracted} jobs, {extracted.relevant} relevant, {extracted.failed} failed")
        if not skip_score:
            scored = await score_jobs(ctx.session, ctx.profile(), ctx.aliases, ctx.fit_provider())
            console.print(f"Score: {scored.scored} jobs")


# --- Review and reports -----------------------------------------------------------------------------------------


@app.command(help="Open the interactive review queue.")
def review() -> None:
    from .tui import ReviewApp

    ReviewApp().run()


@app.command(help="Print the review queue.")
@syncify
async def queue(
    limit: Annotated[int, typer.Option(help="Show at most this many jobs.")] = 25,
    all_jobs: Annotated[bool, typer.Option("--all", help="Include reviewed and unscored jobs.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import build_queue

    async with app_context() as ctx:
        items = await build_queue(
            ctx.session, ctx.profile().content_hash, include_reviewed=all_jobs, include_unscored=all_jobs
        )
    rows: list[dict[str, Any]] = [
        {
            "priority": item.score.priority if item.score else None,
            "value": item.score.value if item.score else None,
            "likelihood": item.score.likelihood if item.score else None,
            "company": item.job.company,
            "title": item.job.title,
            "location": item.job.location,
            "sources": item.sources,
            "url": item.job.posting_url,
            "decision": item.decision.value if item.decision else None,
            "job_id": str(item.job.id),
        }
        for item in items[:limit]
    ]
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    table = Table("Pri", "Val", "Lik", "Company", "Title", "Sources", "State")
    for row in rows:
        fmt = lambda v: f"{v:.2f}" if v is not None else "-"
        table.add_row(
            fmt(row["priority"]), fmt(row["value"]), fmt(row["likelihood"]),
            row["company"], row["title"], ", ".join(row["sources"]), row["decision"] or "",
        )  # fmt: skip
    console.print(table)
    console.print(f"{len(items)} jobs in queue.")


@app.command(help="List leads that need a posting link or failed to resolve.")
@syncify
async def unresolved(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import unresolved_leads

    async with app_context() as ctx:
        leads = await unresolved_leads(ctx.session)
        rows = [
            {"lead_id": str(lead.id), "company": lead.company, "title": lead.title, "source": lead.source,
             "status": lead.status.value, "url": lead.url}
            for lead in leads
        ]  # fmt: skip
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    table = Table("Lead id", "Company", "Title", "Source", "Status", "Link")
    for row in rows:
        table.add_row(row["lead_id"], row["company"], row["title"], row["source"], row["status"], row["url"])
    console.print(table)
    console.print("Attach a posting with: jobhunt resolve <lead id> <company ATS link>")


@app.command(help="Most-requested skills across relevant jobs, weighted by job value.")
@syncify
async def skills(
    top: Annotated[int, typer.Option(help="Number of skills to show.")] = 30,
    gaps: Annotated[bool, typer.Option(help="Only show skills you don't list in your profile.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import skill_report
    from .services.skills import normalize_skills

    async with app_context() as ctx:
        profile = ctx.profile()
        have = {s.lower() for s in normalize_skills(profile.candidate.skills, ctx.aliases)}
        report = await skill_report(ctx.session, profile.content_hash, have)
    if gaps:
        report = [stat for stat in report if not stat.have]
    report = report[:top]
    if as_json:
        typer.echo(json.dumps([stat.__dict__ for stat in report], indent=2))
        return
    table = Table("Skill", "Weighted", "Jobs", "Required in", "You have it")
    for stat in report:
        table.add_row(
            stat.skill, f"{stat.weighted:.1f}", str(stat.jobs), str(stat.required_in), "yes" if stat.have else "no"
        )
    console.print(table)


@app.command(help="Show per-source ingest health. With --strict, exits 1 if any source is unhealthy.")
@syncify
async def health(
    window_days: Annotated[int, typer.Option(help="Look at runs from the last N days.")] = 30,
    strict: Annotated[bool, typer.Option(help="Exit with code 1 when a source is unhealthy.")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import source_health

    async with app_context() as ctx:
        report = await source_health(ctx.session, window_days)
    table = Table("Source", "Last run", "Status", "Last seen", "Avg seen", "Zero streak", "Error")
    for row in report:
        style = "green" if row.healthy else "red"
        table.add_row(
            row.source,
            row.last_run.strftime("%Y-%m-%d %H:%M") if row.last_run else "-",
            f"[{style}]{row.last_status}[/]",
            str(row.last_seen),
            f"{row.avg_seen:.1f}",
            str(row.zero_streak),
            (row.last_error or "")[:60],
        )
    console.print(table)
    if strict and any(not row.healthy for row in report):
        raise typer.Exit(code=1)


@app.command(help="Merge jobs from a pipeline database (e.g. the GitHub Actions data branch) into this one.")
@syncify
async def pull(
    path: Annotated[Path, typer.Argument(help="Path to the remote SQLite database file.", exists=True)],
) -> None:
    from .services.context import app_context
    from .services.merge import merge_database

    async with app_context() as ctx:
        stats = await merge_database(ctx.session, path)
    console.print(
        f"Merged: {stats.jobs_new} new jobs, {stats.jobs_updated} updated, {stats.leads_new} leads, "
        f"{stats.snapshots_new} descriptions, {stats.screens_new} screens, {stats.extractions_new} extractions."
    )


@app.command(help="Export the queue (or every job with --all) as CSV to stdout.")
@syncify
async def export(
    all_jobs: Annotated[bool, typer.Option("--all", help="Include reviewed and unscored jobs.")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import build_queue

    async with app_context() as ctx:
        items = await build_queue(
            ctx.session, ctx.profile().content_hash, include_reviewed=all_jobs, include_unscored=all_jobs
        )
    writer = csv.writer(sys.stdout)
    writer.writerow(["Company", "Role", "Link", "Source", "Priority", "Value", "Likelihood", "Decision"])
    for item in items:
        writer.writerow(
            [
                item.job.company,
                item.job.title,
                item.job.posting_url,
                "; ".join(item.sources),
                item.score.priority if item.score else "",
                item.score.value if item.score else "",
                item.score.likelihood if item.score else "",
                item.decision.value if item.decision else "",
            ]
        )


@app.command("jev-probe", help="Build (and with --send, send) Jev screen and fit requests for a few jobs.")
@syncify
async def jev_probe(
    out: Annotated[Path, typer.Option(help="Directory for request and response JSON files.")] = Path("jev-probe"),
    limit: Annotated[int, typer.Option(help="Number of jobs to probe.")] = 5,
    job_id: Annotated[list[str] | None, typer.Option("--job", help="Specific job id; repeatable.")] = None,
    send: Annotated[
        bool, typer.Option(help="Actually call the TypeSafe API (uses TYPESAFE_API_KEY and sends job text).")
    ] = False,
) -> None:
    """Write the exact payloads to disk so any answer can be rerun, then optionally send them."""
    from sqlalchemy import func, select

    from .models.pipeline import Job, JobSnapshot
    from .questions import decide_screen, fit_questions, fit_state, screen_questions, screen_state
    from .schemas.typesafe import SystemOneRequest
    from .services.context import app_context
    from .services.queue import load_description

    async with app_context() as ctx:
        if send and ctx.jev is None:
            _fail("--send needs TYPESAFE_API_KEY.")
        profile = ctx.profile()
        stmt = select(Job).where(Job.id.in_(select(JobSnapshot.job_id)))
        if job_id:
            stmt = stmt.where(Job.id.in_([UUID(value) for value in job_id]))
        else:
            stmt = stmt.order_by(func.random()).limit(limit)
        jobs = (await ctx.session.execute(stmt)).scalars().all()
        out.mkdir(parents=True, exist_ok=True)
        table = Table("Job", "Kind", "Result")
        for job in jobs:
            description = await load_description(ctx.session, job.id) or ""
            chars = settings.screen_max_description_chars
            requests = {
                "screen": SystemOneRequest(
                    model=settings.jev_model,
                    state=screen_state(job.title, job.company, job.location, description[:chars]),
                    questions=screen_questions(),
                ),
                "fit": SystemOneRequest(
                    model=settings.jev_model,
                    state=fit_state(job.title, job.company, description[:chars]),
                    questions=fit_questions(profile),
                ),
            }
            for kind, request in requests.items():
                request_path = out / f"{job.id}.{kind}.request.json"
                response_path = out / f"{job.id}.{kind}.response.json"
                request_path.write_text(request.model_dump_json(indent=2, exclude_none=True))
                if not send or ctx.jev is None:
                    table.add_row(f"{job.company} - {job.title}"[:50], kind, f"wrote {request_path}")
                    continue
                response = await ctx.jev.system_one(request)
                response_path.write_text(response.model_dump_json(indent=2))
                if kind == "screen":
                    decision = decide_screen(response)
                    result = f"{decision.outcome.value}: " + ", ".join(
                        f"{k}={v:.2f}" for k, v in decision.signals.items()
                    )
                else:
                    result = ", ".join(f"{k}={a.noul:.2f}" for k, a in response.answers.items() if a.type == "noul")
                table.add_row(f"{job.company} - {job.title}"[:50], kind, result)
    console.print(table)
    if not send:
        console.print("Payloads written. Review them, then rerun with --send to call the API.")


if __name__ == "__main__":
    app()
