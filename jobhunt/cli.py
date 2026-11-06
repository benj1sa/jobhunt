import asyncio
import csv
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .services.llm import LlmConfigError
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
    from rich.logging import RichHandler

    # Warnings and errors only by default: per-job problems are summarized after each step instead of logged
    # line by line. Logs go to stderr so `--csv > file` stays clean; the live progress bar redirects them above
    # itself. -v shows everything.
    logging.basicConfig(
        level=logging.DEBUG if verbose or settings.debug else logging.WARNING,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err_console, show_time=False, show_path=False, markup=False)],
    )
    # httpx logs every request at INFO; keep that out of normal runs.
    logging.getLogger("httpx").setLevel(logging.DEBUG if verbose else logging.WARNING)


class _LiveStep:
    """A one-line progress bar for one pipeline step; see services/progress.py."""

    def __init__(self, bar: Any, label: str) -> None:
        self._bar = bar
        self._label = label
        self._task: Any = None
        self.failed = 0
        self.started = time.monotonic()

    def start(self, total: int) -> None:
        if self._task is None:
            self._task = self._bar.add_task(self._label, total=total, counts="", item="")
        else:
            self._bar.update(self._task, total=total)

    def advance(self, n: int = 1, *, ok: bool = True, item: str | None = None) -> None:
        if self._task is None:
            self.start(0)
        self.failed += 0 if ok else n
        fields: dict[str, Any] = {"counts": f"[red]{self.failed} failed[/]" if self.failed else ""}
        if item:
            fields["item"] = escape(item if len(item) <= 48 else item[:47] + "…")
        self._bar.update(self._task, advance=n, **fields)

    def note(self, text: str) -> None:
        if self._task is None:
            self.start(0)
        self._bar.update(self._task, item=f"[yellow]{escape(text)}[/]")

    @property
    def took(self) -> str:
        seconds = int(time.monotonic() - self.started)
        return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


@contextmanager
def _step_progress(label: str) -> Iterator[_LiveStep]:
    """Live progress for a step. It disappears when the step ends (only the summary line stays), and it is
    switched off when output isn't a terminal (e.g. piped to a file or in CI)."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<8}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[counts]}"),
        TextColumn("[dim]{task.fields[item]}"),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    ) as bar:
        yield _LiveStep(bar, label)


def _print_stopped(stopped: str | None, done: int, not_started: int) -> None:
    if stopped:
        console.print(
            f"[yellow]Stopped early:[/] {escape(stopped)}\n"
            f"  {done} finished and saved; {not_started} not sent. They are picked up by the next run."
        )


def _fail(message: str) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(code=1)


async def _report_usage(ctx: Any, stage: str, items: int) -> float:
    """Print and save what this step's model calls used. Returns the step's cost in USD."""
    from .services.usage import describe, save_usage

    ctx.meter.set_items(stage, items)
    steps = ctx.meter.take(stage)
    if not steps:
        return 0.0
    await save_usage(ctx.session, steps)
    console.print(f"  [dim]AI usage ({stage}): {describe(steps)}[/]")
    return float(sum(step.cost for step in steps))


def _print_reasons(reasons: "Counter[str]", limit: int = 8) -> None:
    """Grouped failure reasons, e.g. '28 x apply.workable.com: JavaScript-only page'."""
    for reason, count in reasons.most_common(limit):
        err_console.print(f"  [yellow]![/] {count} x {reason}")
    if len(reasons) > limit:
        err_console.print(f"  ... and {len(reasons) - limit} more kinds (run with -v for details)")


def _print_errors(errors: list[str], limit: int = 10) -> None:
    for error in errors[:limit]:
        err_console.print(f"  [yellow]![/] {error}")
    if len(errors) > limit:
        err_console.print(f"  ... and {len(errors) - limit} more (run with -v for details)")


# --- Setup ------------------------------------------------------------------------------------------------------


@app.command(help="Create or upgrade the database schema to the latest migration.", rich_help_panel="5. Maintenance")
def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    config_path = Path("alembic.ini")
    if not config_path.exists():
        _fail("alembic.ini not found. Run this from the project root.")
    command.upgrade(Config(str(config_path)), "head")
    console.print("Database is up to date.")


@app.command(help="Install testing data for local development.", rich_help_panel="5. Maintenance")
@syncify
async def test_data() -> None:
    from . import __version__
    from .services.db import get_session, test_data

    typer.echo(f"{settings.project_name} - {__version__}")

    async with get_session() as session:
        await test_data(session)

    typer.echo("Development data installed successfully.")


@app.command(
    help=f"Display the current installed version of {settings.project_name}.", rich_help_panel="5. Maintenance"
)
def version() -> None:
    from . import __version__

    typer.echo(f"{settings.project_name} - {__version__}")


# --- Pipeline stages --------------------------------------------------------------------------------------------


def _changes_note(changes: "Counter[str]") -> str:
    parts = [f"{count} marked {name}" for name, count in changes.items() if count]
    return ", ".join(parts)


async def _ingest_all(ctx: Any, groups: list[str] | None) -> tuple[list[Any], "Counter[str]", set[str]]:
    """Ingest configured sources plus watched boards, record each board's poll, and prune old run rows."""
    from .services.boards import PollOutcome, prune_ingest_runs, record_polls
    from .services.jobs import ingest as run_ingest

    try:
        configured = ctx.sources(groups)
        sources = await ctx.all_sources(groups)
    except ValueError as exc:
        _fail(str(exc))
    configured_names = {source.name for source in configured}
    watched = {source.name for source in sources if source.name not in configured_names}
    with _step_progress("ingest") as progress:
        results = await run_ingest(ctx.session, ctx.http, sources, progress)
    outcomes = [
        PollOutcome(r.source, r.status.value == "ok", r.not_found, r.items, r.seen)
        for r in results
        if r.source in watched
    ]
    changes = await record_polls(ctx.session, outcomes, settings.board_idle_days, settings.board_dead_after_failures)
    await prune_ingest_runs(ctx.session, settings.ingest_run_retention_days)
    return results, changes, watched


@app.command(
    help="Step 1: Find new postings on every source and watched board. No AI.",
    rich_help_panel="2. Pipeline steps (`run` does these in order)",
)
@syncify
async def ingest(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source group to run; repeatable.")] = None,
) -> None:
    from .services.context import app_context

    async with app_context() as ctx:
        results, changes, watched = await _ingest_all(ctx, source)
    table = Table("Source", "Status", "Fetched", "Leads", "New", "Error")
    boards_ok = boards_failed = boards_new = boards_leads = 0
    for result in results:
        if result.source in watched:
            boards_ok += int(result.status.value == "ok")
            boards_failed += int(result.status.value != "ok")
            boards_new += result.new
            boards_leads += result.seen
            continue
        style = "green" if result.status.value == "ok" else "red"
        table.add_row(
            result.source, f"[{style}]{result.status.value}[/]", "-" if result.items is None else str(result.items),
            str(result.seen), str(result.new), result.error or "",
        )  # fmt: skip
    if boards_ok or boards_failed:
        table.add_row(
            f"watched boards ({boards_ok + boards_failed})", f"{boards_ok} ok, {boards_failed} failed", "-",
            str(boards_leads), str(boards_new), _changes_note(changes),
        )  # fmt: skip
    console.print(table)


@app.command(
    "import-csv",
    help="Import leads from a CSV with Company, Role, Link and optional Source columns.",
    rich_help_panel="3. Add jobs by hand",
)
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


@app.command(help="Add one job by link.", rich_help_panel="3. Add jobs by hand")
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


@app.command(
    help="Step 2: Turn new leads into jobs, or attach a posting link to one lead. No AI.",
    rich_help_panel="2. Pipeline steps (`run` does these in order)",
)
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
        with _step_progress("resolve") as progress:
            stats = await resolve_leads(ctx.session, ctx.boards, ctx.http, limit=limit, progress=progress)
    console.print(
        f"Resolved {stats.resolved} ({stats.new_jobs} new jobs), {stats.needs_url} need a link, {stats.failed} failed."
    )
    _print_errors(stats.errors)


@app.command(
    help="Step 3: Download full descriptions. No AI.", rich_help_panel="2. Pipeline steps (`run` does these in order)"
)
@syncify
async def hydrate(
    refresh: Annotated[bool, typer.Option(help="Re-fetch every open job (detects closed postings).")] = False,
    limit: Annotated[int | None, typer.Option(help="Hydrate at most this many jobs.")] = None,
) -> None:
    from .services.context import app_context
    from .services.jobs import hydrate_jobs

    async with app_context() as ctx:
        with _step_progress("hydrate") as progress:
            stats = await hydrate_jobs(ctx.session, ctx.http, refresh=refresh, limit=limit, progress=progress)
    console.print(
        f"Hydrated {stats.hydrated}, unchanged {stats.unchanged}, closed {stats.closed}, failed {stats.failed}."
    )
    _print_reasons(stats.reasons)


@app.command(
    help="Step 5: AI (OpenRouter ~$0.001/job, or your ChatGPT plan): skills, requirements, capability phrases.",
    rich_help_panel="2. Pipeline steps (`run` does these in order)",
)
@syncify
async def extract(
    limit: Annotated[int | None, typer.Option(help="Extract at most this many jobs.")] = None,
    rerun: Annotated[
        bool, typer.Option(help="Re-extract jobs already extracted (e.g. after editing config/capabilities.toml).")
    ] = False,
) -> None:
    from .services.context import app_context
    from .services.extraction import extract_jobs

    async with app_context() as ctx:
        if ctx.llm is None:
            _fail(ctx.llm_missing)
        assert ctx.llm is not None
        with ctx.meter.step("extract"), _step_progress("extract") as progress:
            stats = await extract_jobs(
                ctx.session,
                ctx.llm,
                ctx.extraction_model,
                ctx.aliases,
                settings.llm_max_description_chars,
                limit,
                capabilities=ctx.capabilities(),
                rerun=rerun,
                concurrency=settings.llm_concurrency,
                progress=progress,
            )
        console.print(
            f"Extracted {stats.extracted} ({stats.relevant} relevant), failed {stats.failed} ({progress.took})."
        )
        await _report_usage(ctx, "extract", stats.extracted + stats.failed)
    _print_errors(stats.errors)
    _print_stopped(stats.stopped, stats.extracted, stats.not_started)
    if stats.stopped:
        raise typer.Exit(code=1)


@app.command(
    help="Step 4: AI (Jev, ~$0.0001/job): yes/no checks so misfits skip extraction.",
    rich_help_panel="2. Pipeline steps (`run` does these in order)",
)
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
        with ctx.meter.step("screen"), _step_progress("screen") as progress:
            stats = await screen_jobs(
                ctx.session,
                ctx.jev,
                settings.screen_max_description_chars,
                limit,
                progress,
                degree=ctx.profile().candidate.degree,
            )
        console.print(
            f"Screened: {stats.passed} pass, {stats.unsettled} unsettled, {stats.rejected} rejected, "
            f"{stats.failed} failed."
        )
        await _report_usage(ctx, "screen", stats.passed + stats.unsettled + stats.rejected + stats.failed)
    _print_reasons(stats.reject_reasons)
    _print_errors(stats.errors)


@app.command(
    help="Step 6: Value and likelihood from config/profile.toml. AI fit check (Jev, OpenRouter or ChatGPT) if set up.",
    rich_help_panel="2. Pipeline steps (`run` does these in order)",
)
@syncify
async def score(
    rescore: Annotated[bool, typer.Option(help="Recompute scores that already exist for this profile.")] = False,
    fit: Annotated[
        str | None,
        typer.Option(help="Fit provider: jev, llm (the LLM_PROVIDER model) or none. Default: FIT_PROVIDER (auto)."),
    ] = None,
) -> None:
    from .services.context import app_context
    from .services.scoring import score_jobs

    if fit not in (None, "jev", "llm", "openrouter", "chatgpt", "none"):
        _fail("--fit must be jev, llm, openrouter, chatgpt or none.")
    async with app_context() as ctx:
        try:
            provider = ctx.fit_provider(fit)
        except ValueError as exc:
            _fail(str(exc))
        try:
            with ctx.meter.step("fit"), _step_progress("score") as progress:
                stats = await score_jobs(
                    ctx.session, ctx.profile(), ctx.aliases, provider, rescore=rescore, progress=progress
                )
        except LlmConfigError as exc:
            await _report_usage(ctx, "fit", 0)
            _fail(str(exc))
        name = provider.name if provider else "none"
        console.print(f"Scored {stats.scored} jobs ({stats.fit_calls} {name} fit calls), failed {stats.failed}.")
        await _report_usage(ctx, "fit", stats.fit_calls)
    _print_errors(stats.errors)


@app.command(
    help="Do everything new: ingest, resolve, hydrate, screen, extract, score. Prints AI cost at the end.",
    rich_help_panel="1. Start here",
)
@syncify
async def run(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source group; repeatable.")] = None,
    skip_ingest: Annotated[bool, typer.Option(help="Start from resolve (e.g. after import-csv).")] = False,
    skip_score: Annotated[bool, typer.Option(help="Stop after extraction (the scheduled workflow does this).")] = False,
) -> None:
    from .services.context import app_context
    from .services.extraction import extract_jobs
    from .services.jobs import hydrate_jobs, resolve_leads
    from .services.scoring import score_jobs
    from .services.screening import screen_jobs

    async with app_context() as ctx:
        if not skip_ingest:
            results, changes, watched = await _ingest_all(ctx, source)
            failed = [r.source for r in results if r.status.value != "ok" and r.source not in watched]
            boards_failed = sum(1 for r in results if r.status.value != "ok" and r.source in watched)
            console.print(
                f"Ingest: {sum(r.new for r in results)} new leads from {len(results)} sources "
                f"({len(watched)} watched boards"
                + (f", {boards_failed} failed" if boards_failed else "")
                + (f", {_changes_note(changes)}" if changes else "")
                + ")"
                + (f"; [red]failed: {', '.join(failed)}[/]" if failed else "")
            )
        with _step_progress("resolve") as progress:
            resolved = await resolve_leads(ctx.session, ctx.boards, ctx.http, progress=progress)
        console.print(
            f"Resolve: {resolved.new_jobs} new jobs, {resolved.needs_url} need a link, "
            f"{resolved.boards_discovered} new boards watched ({progress.took})"
        )
        with _step_progress("hydrate") as progress:
            hydrated = await hydrate_jobs(ctx.session, ctx.http, progress=progress)
        console.print(
            f"Hydrate: {hydrated.hydrated} described, {hydrated.closed} closed, {hydrated.failed} failed "
            f"({progress.took})"
        )
        _print_reasons(hydrated.reasons, limit=5)
        spent = 0.0
        if ctx.jev is not None:
            with ctx.meter.step("screen"), _step_progress("screen") as progress:
                screened = await screen_jobs(
                    ctx.session,
                    ctx.jev,
                    settings.screen_max_description_chars,
                    progress=progress,
                    degree=ctx.profile().candidate.degree,
                )
            console.print(
                f"Screen: {screened.passed} pass, {screened.unsettled} unsettled, {screened.rejected} rejected, "
                f"{screened.failed} failed ({progress.took})"
            )
            _print_reasons(screened.reject_reasons, limit=5)
            spent += await _report_usage(
                ctx, "screen", screened.passed + screened.unsettled + screened.rejected + screened.failed
            )
        if ctx.llm is None:
            console.print(f"[yellow]Skipping extraction and scoring: {ctx.llm_missing}[/]")
            return
        with ctx.meter.step("extract"), _step_progress("extract") as progress:
            extracted = await extract_jobs(
                ctx.session,
                ctx.llm,
                ctx.extraction_model,
                ctx.aliases,
                settings.llm_max_description_chars,
                capabilities=ctx.capabilities(),
                concurrency=settings.llm_concurrency,
                progress=progress,
            )
        console.print(
            f"Extract: {extracted.extracted} jobs, {extracted.relevant} relevant, {extracted.failed} failed "
            f"({progress.took})"
        )
        spent += await _report_usage(ctx, "extract", extracted.extracted + extracted.failed)
        if not skip_score:
            try:
                with ctx.meter.step("fit"), _step_progress("score") as progress:
                    scored = await score_jobs(
                        ctx.session, ctx.profile(), ctx.aliases, ctx.fit_provider(), progress=progress
                    )
            except LlmConfigError as exc:
                await _report_usage(ctx, "fit", 0)
                _fail(str(exc))
            console.print(f"Score: {scored.scored} jobs ({scored.fit_calls} fit calls, {progress.took})")
            spent += await _report_usage(ctx, "fit", scored.fit_calls)
        console.print(f"[b]AI cost of this run: ${spent:.4f}[/b]. `jobhunt status` shows what's next.")
        # Shown last so it isn't lost above the score line: the run scored what was extracted before stopping.
        _print_stopped(extracted.stopped, extracted.extracted, extracted.not_started)


# --- Review and reports -----------------------------------------------------------------------------------------


@app.command(
    help="What's waiting at each step, which steps use AI, estimated tokens and cost, and what to run next.",
    rich_help_panel="1. Start here",
)
@syncify
async def status() -> None:
    from .services.context import app_context
    from .services.status import gather_status

    async with app_context() as ctx:
        provider = ctx.fit_provider()
        fit_model = None
        if provider is not None:
            fit_model = settings.jev_model if provider.name == "jev" else ctx.fit_model
        result = await gather_status(
            ctx.session,
            ctx.profile().content_hash,
            screen_model=settings.jev_model if ctx.jev is not None else None,
            extract_model=ctx.extraction_model if ctx.llm is not None else None,
            fit_model=fit_model,
        )

    def tokens(n: float) -> str:
        return "" if not n else f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1000:.0f}k"

    table = Table(title="Pipeline status (`jobhunt run` does steps 1-6)", title_justify="left")
    for col in ("step", "what it does", "AI model", "waiting", "est. tokens in/out", "est. cost", "notes"):
        table.add_column(col, justify="right" if col in ("waiting", "est. cost") else "left")
    for n, step in enumerate(result.steps, 1):
        per_job = ""
        if step.per_job:
            basis = "your history" if step.per_job.from_history else "typical"
            per_job = f"${step.per_job.cost:.5f}/job ({basis})"
        notes = "; ".join(part for part in (per_job, step.note) if part)
        io = (
            f"{tokens(step.est_input_tokens)} / {tokens(step.est_output_tokens) or '0'}"
            if step.est_input_tokens
            else ""
        )
        table.add_row(
            f"{n}. {step.command}",
            step.what,
            step.model or "[dim]none[/]",
            str(step.pending) if step.command != "ingest" else "",
            io,
            f"${step.est_cost:.4f}" if step.model else "",
            notes,
        )
    console.print(table)
    console.print(f"Estimated AI cost to finish what's waiting: [b]${result.est_total:.4f}[/b]")
    spent = ", ".join(f"{stage} ${cost:.4f}" for stage, cost in sorted(result.spent_7d.items())) or "nothing"
    console.print(f"Spent on AI: last 7 days {spent}; last 30 days ${result.spent_30d:.4f}")
    console.print(
        f"Review queue: {result.queue_size} jobs.  Leads needing a link: {result.needs_link} (`jobhunt unresolved`)."
    )
    console.print(f"[b]Next:[/b] {result.next_command}")


@app.command(help="Open the interactive review queue.", rich_help_panel="1. Start here")
def review() -> None:
    from .tui import ReviewApp

    ReviewApp().run()


@app.command(help="Print the review queue.", rich_help_panel="4. Explore what jobs ask for")
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


@app.command(help="List leads that need a posting link or failed to resolve.", rich_help_panel="3. Add jobs by hand")
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


@app.command(
    help="Everything about one job: links, screen answers, extracted requirements, score, and the description.",
    rich_help_panel="4. Explore what jobs ask for",
)
@syncify
async def show(
    job_ref: Annotated[str, typer.Argument(help="Job id, or its first few characters (e.g. d58dfee2).")],
    full: Annotated[bool, typer.Option(help="Print the whole description (paged) instead of the first part.")] = False,
    open_link: Annotated[bool, typer.Option("--open", help="Also open the posting in your browser.")] = False,
) -> None:
    import webbrowser

    from rich.panel import Panel

    from .services.context import app_context
    from .services.queue import find_job, job_detail

    async with app_context() as ctx:
        try:
            job = await find_job(ctx.session, job_ref)
        except LookupError as exc:
            _fail(str(exc))
        detail = await job_detail(ctx.session, job, ctx.profile().content_hash)

    lines = [
        f"[b]{escape(job.title)}[/b] at [b]{escape(job.company)}[/b]",
        f"{escape(job.location or 'location not listed')}{' (remote)' if job.is_remote else ''} · {job.status.value}"
        f" · first seen {job.first_seen_at:%b %d}" + (f" · posted {job.posted_at:%b %d}" if job.posted_at else ""),
        f"Posting: {job.posting_url}"
        + (f"\nApply: {job.apply_url}" if job.apply_url and job.apply_url != job.posting_url else ""),
        f"Found via: {', '.join(detail.sources) or '-'} · id {job.id.hex}",
    ]
    if job.hydrate_error:
        lines.append(f"[yellow]Description problem:[/] {escape(job.hydrate_error)}")
    if detail.screen:
        color = {"pass": "green", "reject": "red"}.get(detail.screen.outcome.value, "yellow")
        signals = ", ".join(f"{name} {value:.2f}" for name, value in detail.screen.signals.items())
        lines += ["", f"[b]Screen[/b] [{color}]{detail.screen.outcome.value}[/]: {escape(detail.screen.reason)}"]
        lines.append(f"[dim]{signals}[/]")
    else:
        lines += ["", "[b]Screen[/b] not screened yet"]
    if detail.extraction:
        data = detail.extraction.data
        lines += [
            "",
            f"[b]Extracted[/b] ({detail.extraction.model}): "
            + (
                "relevant"
                if detail.extraction.is_relevant
                else f"[red]not relevant[/] ({detail.extraction.relevance_reason})"
            ),
            (
                f"Seniority {data.get('seniority')} · degree {data.get('degree_required')} · "
                f"min years {data.get('min_years_experience')} · {data.get('work_mode')}"
            ),
            f"Required: {escape(', '.join(data.get('required_skills') or []) or '-')}",
            f"Preferred: {escape(', '.join(data.get('preferred_skills') or []) or '-')}",
        ]
        if data.get("summary"):
            lines.append(f"[dim]{escape(str(data['summary']))}[/]")
    else:
        lines += ["", "[b]Extracted[/b] not yet"]
    if detail.score:
        gates = (detail.score.components or {}).get("gates") or []
        lines += [
            "",
            f"[b]Score[/b] priority {detail.score.priority:.2f} (value {detail.score.value:.2f}, "
            f"likelihood {detail.score.likelihood:.2f})" + (f" · {escape('; '.join(gates))}" if gates else ""),
        ]
    if detail.review:
        lines.append(
            f"[b]Your review[/b] {detail.review.decision.value}"
            + (f": {escape(detail.review.reason)}" if detail.review.reason else "")
        )
    console.print(Panel("\n".join(lines), expand=False))

    text = detail.description or ""
    if not text:
        console.print("[dim]No description stored.[/]")
    elif full:
        with console.pager():
            console.print(text, markup=False, highlight=False)
    else:
        preview = text[:1500]
        console.print(preview, markup=False, highlight=False)
        if len(text) > len(preview):
            console.print(
                f"[dim]… {len(text) - len(preview):,} more characters: add --full, or --open for the page.[/]"
            )
    if detail.screen and detail.screen.outcome.value == "reject":
        console.print(f"Wrongly screened out? `jobhunt screened-out --keep {job.id.hex[:8]}`")
    if open_link:
        webbrowser.open(job.posting_url)


@app.command(
    "screened-out",
    help="Jobs the Jev screen rejected (e.g. master's/PhD required), with why. --keep <job id> overrides one.",
    rich_help_panel="4. Explore what jobs ask for",
)
@syncify
async def screened_out(
    reason: Annotated[
        str | None, typer.Option(help="Only rejects whose reason contains this, e.g. 'master' or 'internship'.")
    ] = None,
    keep: Annotated[str | None, typer.Option(help="Job id to put back: it gets extracted and queued.")] = None,
    limit: Annotated[int, typer.Option(help="Rows to show.")] = 50,
) -> None:
    from collections import Counter

    from .services.context import app_context
    from .services.queue import find_job, keep_screened_job, screened_out_jobs

    async with app_context() as ctx:
        if keep:
            try:
                kept_job = await find_job(ctx.session, keep)
            except LookupError as exc:
                _fail(str(exc))
            changed = await keep_screened_job(ctx.session, kept_job.id)
            if not changed:
                _fail("That job has no screen reject to override.")
            console.print("Kept. `jobhunt run --skip-ingest` extracts and scores it.")
            return
        rows = await screened_out_jobs(ctx.session, reason)
    counts = Counter(part for _, screen in rows for part in screen.reason.split("; "))
    console.print(f"{len(rows)} open jobs screened out: " + ", ".join(f"{n} {why}" for why, n in counts.most_common()))
    table = Table("Id", "Company", "Title", "Why", "Link")
    for job, screen in rows[:limit]:
        table.add_row(job.id.hex[:8], job.company, job.title, screen.reason, job.posting_url)
    console.print(table)
    if len(rows) > limit:
        console.print(f"... {len(rows) - limit} more (use --limit or --reason).")
    console.print("Details: `jobhunt show <id>`. Jev got one wrong? `jobhunt screened-out --keep <id>` puts it back.")


def _echo_csv(rows: list[dict[str, Any]]) -> None:
    """Rows as CSV on stdout, for `> file.csv` (opens in Sheets/Excel)."""
    import io

    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()})
    typer.echo(buffer.getvalue(), nl=False)


def _have_skills(ctx: Any) -> set[str]:
    from .services.skills import normalize_skills

    return {s.lower() for s in normalize_skills(ctx.profile().candidate.skills, ctx.aliases)}


def _tool_list(tools: list[Any], limit: int) -> str:
    shown = [f"{t.skill} {t.jobs}{'*' if t.have else ''}" for t in tools[:limit]]
    return ", ".join(shown) + (f", +{len(tools) - limit} more" if len(tools) > limit else "")


@app.command(
    help="Skills across relevant jobs, grouped into families, weighted by job value.",
    rich_help_panel="4. Explore what jobs ask for",
)
@syncify
async def skills(
    group: Annotated[str | None, typer.Option(help="Show every skill in one group (id from the table).")] = None,
    flat: Annotated[bool, typer.Option(help="One row per skill instead of grouped.")] = False,
    unsorted: Annotated[bool, typer.Option(help="Skills no group claims yet (add them to skill_groups.toml).")] = False,
    gaps: Annotated[bool, typer.Option(help="Only skills you don't list in your profile.")] = False,
    top: Annotated[int, typer.Option(help="Rows to show (flat and unsorted views; 0 = all).")] = 40,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    as_csv: Annotated[
        bool, typer.Option("--csv", help="Every skill as CSV, one row each with its group (`> skills.csv`).")
    ] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import GroupStat, SkillStat, skill_group_report, skill_report

    if as_csv:  # one flat table is what a spreadsheet wants, whatever the view
        flat = True
        top = top if top != 40 else 0

    report: list[SkillStat] = []
    groups: list[GroupStat] = []
    leftovers: list[SkillStat] = []
    async with app_context() as ctx:
        profile = ctx.profile()
        have = _have_skills(ctx)
        taxonomy = ctx.taxonomy()
        if flat or unsorted:
            report = await skill_report(ctx.session, profile.content_hash, have, taxonomy=taxonomy)
        else:
            groups, leftovers = await skill_group_report(ctx.session, profile.content_hash, have, taxonomy)

    if flat or unsorted:
        rows = [s for s in report if (not unsorted or s.group is None) and (not gaps or not s.have)]
        rows = rows[:top] if top else rows
        if group:
            rows = [s for s in rows if s.group == group]
        if as_csv:
            names = taxonomy.groups
            _echo_csv(
                [
                    {"skill": s.skill, "group": s.group or "", "group_name": names.get(s.group or "", ""),
                     "kind": s.kind or "", "jobs": s.jobs, "required_in": s.required_in, "weighted": s.weighted,
                     "you_have_it": "yes" if s.have else "no"}
                    for s in rows
                ]
            )  # fmt: skip
            return
        if as_json:
            typer.echo(json.dumps([s.__dict__ for s in rows], indent=2))
            return
        table = Table("Skill", "Group", "Kind", "Weighted", "Jobs", "Required in", "You have it")
        for s in rows:
            table.add_row(
                s.skill, s.group or "-", s.kind or "-", f"{s.weighted:.1f}", str(s.jobs), str(s.required_in),
                "yes" if s.have else "no",
            )  # fmt: skip
        console.print(table)
        if unsorted:
            console.print("Add these to config/skill_groups.toml, or run `jobhunt skills-classify`.")
        return

    if group:
        match = next((g for g in groups if g.group == group), None)
        if match is None:
            _fail(f"No jobs mention group {group!r}. Groups: {', '.join(g.group for g in groups)}")
        assert match is not None
        if as_json:
            typer.echo(json.dumps({"group": match.__dict__ | {
                "tools": [t.__dict__ for t in match.tools], "concepts": [c.__dict__ for c in match.concepts]}},
                default=str, indent=2))  # fmt: skip
            return
        console.print(
            f"[b]{match.name}[/b]: {match.jobs} jobs; {match.concept_only_jobs} named only broad concepts, no tool."
        )
        table = Table("Skill", "Kind", "Jobs", "Required in", "Weighted", "You have it")
        for s in sorted(match.tools + match.concepts, key=lambda x: (-x.jobs, x.skill)):
            if gaps and s.have:
                continue
            table.add_row(
                s.skill, s.kind or "-", str(s.jobs), str(s.required_in), f"{s.weighted:.1f}", "yes" if s.have else "no"
            )
        console.print(table)
        return

    if as_json:
        typer.echo(json.dumps([
            g.__dict__ | {"tools": [t.__dict__ for t in g.tools], "concepts": [c.__dict__ for c in g.concepts]}
            for g in groups], indent=2))  # fmt: skip
        return
    table = Table("Group", "Jobs", "Weighted", "Broad only", "Most-asked specific tools (jobs; * = you have it)")
    for g in groups:
        tools = [t for t in g.tools if not (gaps and t.have)]
        table.add_row(f"{g.name}\n[dim]{g.group}[/dim]", str(g.jobs), f"{g.weighted:.1f}", str(g.concept_only_jobs),
                      _tool_list(tools, 6) or "[dim]concepts only[/dim]")  # fmt: skip
    console.print(table)
    unsorted_mentions = sum(s.jobs for s in leftovers)
    console.print(
        f"{len(leftovers)} skills ({unsorted_mentions} mentions) aren't in a group yet: "
        "`jobhunt skills --unsorted` or `jobhunt skills-classify`. Details: `jobhunt skills --group <id>`."
    )


@app.command(
    "skills-classify",
    help="AI (OpenRouter LLM, cents): place unsorted skills into groups (writes a suggestions file).",
    rich_help_panel="4. Explore what jobs ask for",
)
@syncify
async def skills_classify(
    min_jobs: Annotated[int, typer.Option(help="Only skills mentioned by at least this many jobs.")] = 1,
    limit: Annotated[int, typer.Option(help="At most this many skills per run.")] = 300,
) -> None:
    from .schemas.config import suggested_path
    from .services.context import app_context
    from .services.queue import skill_report
    from .services.skill_classify import classify_skills

    async with app_context() as ctx:
        if ctx.llm is None:
            _fail(ctx.llm_missing)
        assert ctx.llm is not None
        taxonomy = ctx.taxonomy()
        report = await skill_report(ctx.session, ctx.profile().content_hash, set(), taxonomy=taxonomy)
        todo = [s.skill for s in report if s.group is None and s.jobs >= min_jobs][:limit]
        if not todo:
            console.print("Every skill already has a group.")
            return
        with ctx.meter.step("classify"), _step_progress("classify") as progress:
            stats = await classify_skills(
                ctx.llm,
                ctx.extraction_model,
                taxonomy,
                todo,
                settings.skill_groups_path,
                progress=progress,
                concurrency=settings.llm_concurrency,
            )
        await _report_usage(ctx, "classify", len(todo) - stats.not_started)
    console.print(
        f"Proposed groups for {stats.suggested} of {len(todo)} skills ({stats.skipped} skipped as non-skills, "
        f"{progress.took}). Review {suggested_path(settings.skill_groups_path)}."
    )
    _print_errors(stats.errors)
    _print_stopped(stats.stopped, len(todo) - stats.not_started, stats.not_started)
    if stats.stopped:
        raise typer.Exit(code=1)


@app.command(
    help="What postings expect you to DO (customer iteration, testing, ...), with the phrases behind it.",
    rich_help_panel="4. Explore what jobs ask for",
)
@syncify
async def capabilities(
    capability: Annotated[
        str | None, typer.Argument(help="Capability id to list every phrase for (ids are in the table).")
    ] = None,
    other: Annotated[bool, typer.Option(help="List phrases tagged 'other' (candidates for new capabilities).")] = False,
    every_phrase: Annotated[
        bool, typer.Option("--phrases", help="Every phrase for every capability (pair with --csv or --json).")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    as_csv: Annotated[bool, typer.Option("--csv", help="CSV output (`> capabilities.csv`).")] = False,
) -> None:
    from .services.context import app_context
    from .services.queue import capability_phrases, capability_report

    async with app_context() as ctx:
        names = {c.id: c.name for c in ctx.capabilities()}
        if every_phrase:
            phrases = await capability_phrases(ctx.session, None)
            rows = [
                {"capability": p.capability, "capability_name": names.get(p.capability, "Other"), "company": p.company,
                 "title": p.title, "phrase": p.phrase, "verbatim": "yes" if p.verbatim else "no",
                 "job_id": p.job_id.hex[:8]}
                for p in phrases
            ]  # fmt: skip
            if as_json:
                typer.echo(json.dumps(rows, indent=2))
            elif as_csv:
                _echo_csv(rows)
            else:
                console.print(f"{len(rows)} phrases. Add --csv (`> phrases.csv`) or --json to export them.")
            return
        wanted = "other" if other else capability
        if wanted:
            if wanted != "other" and wanted not in names:
                _fail(f"Unknown capability {wanted!r}. Ids: {', '.join(names)}")
            phrases = await capability_phrases(ctx.session, wanted)
            report = []
        else:
            report = await capability_report(ctx.session, ctx.profile().content_hash)
            phrases = []

    if wanted:
        if as_json:
            typer.echo(json.dumps([p.__dict__ for p in phrases], default=str, indent=2))
            return
        if as_csv:
            _echo_csv(
                [{"company": p.company, "title": p.title, "phrase": p.phrase,
                  "verbatim": "yes" if p.verbatim else "no", "job_id": p.job_id.hex[:8]} for p in phrases]
            )  # fmt: skip
            return
        jobs = len({p.job_id for p in phrases})
        console.print(f"[b]{names.get(wanted, 'Other')}[/b]: {len(phrases)} phrases from {jobs} jobs")
        table = Table("Company", "Title", "Phrase")
        for p in phrases:
            mark = "" if p.verbatim else " [dim](paraphrased)[/dim]"
            table.add_row(p.company[:28], p.title[:40], p.phrase + mark)
        console.print(table)
        return

    if as_json:
        typer.echo(json.dumps([s.__dict__ | {"name": names.get(s.capability, s.capability)} for s in report], indent=2))
        return
    if as_csv:
        _echo_csv(
            [{"capability": s.capability, "name": names.get(s.capability, "Other (unmatched)"), "jobs": s.jobs,
              "weighted": s.weighted, "phrases": s.phrases, "example": s.example} for s in report]
        )  # fmt: skip
        return
    if not report:
        console.print("No capability evidence yet. Run `jobhunt extract` (or `jobhunt extract --rerun`) first.")
        return
    table = Table("Capability", "Jobs", "Weighted", "Phrases", "Shortest example")
    for s in report:
        label = names.get(s.capability, "Other (unmatched)")
        table.add_row(
            f"{label}\n[dim]{s.capability}[/dim]", str(s.jobs), f"{s.weighted:.1f}", str(s.phrases), s.example
        )
    console.print(table)
    console.print("Every phrase behind one: `jobhunt capabilities <id>`.")


@app.command(
    help="Show per-source ingest health. With --strict, exits 1 if any source is unhealthy.",
    rich_help_panel="5. Maintenance",
)
@syncify
async def health(
    window_days: Annotated[int, typer.Option(help="Look at runs from the last N days.")] = 30,
    strict: Annotated[bool, typer.Option(help="Exit with code 1 when a source is unhealthy.")] = False,
    boards: Annotated[
        bool, typer.Option("--boards", help="List every watched board instead of a summary row.")
    ] = False,
) -> None:
    """Configured sources get a row each. Watched boards are summarized: they retire themselves (dead / paused), so
    individual failures are expected and only a widespread failure counts against --strict."""
    from .services.context import app_context
    from .services.queue import source_health

    async with app_context() as ctx:
        configured = {source.name for source in ctx.sources()}
        watched = {source.name for source in await ctx.all_sources()} - configured
        report = await source_health(ctx.session, window_days, configured | watched)
    table = Table("Source", "Last run", "Status", "Fetched", "Leads", "Avg leads", "Empty streak", "Error")
    watched_rows = [row for row in report if row.source in watched]
    for row in report:
        if row.source in watched and not boards:
            continue
        if not row.configured:
            status = "[dim]inactive[/]"  # removed from config, or a watched board that is now paused or dead
        else:
            style = "green" if row.healthy else "red"
            status = f"[{style}]{row.last_status}[/]"
        table.add_row(
            row.source,
            row.last_run.strftime("%Y-%m-%d %H:%M") if row.last_run else "-",
            status,
            "-" if row.last_fetched is None else str(row.last_fetched),
            str(row.last_seen),
            f"{row.avg_seen:.1f}",
            str(row.zero_streak),
            (row.last_error or "")[:60],
        )
    failing_boards = [row for row in watched_rows if row.last_status != "ok"]
    if watched_rows and not boards:
        share = len(failing_boards) / len(watched_rows)
        style = "red" if share > 0.5 else "green"
        table.add_row(
            f"watched boards ({len(watched_rows)} polled)", "-", f"[{style}]{len(failing_boards)} failing[/]",
            str(sum(r.last_fetched or 0 for r in watched_rows)), str(sum(r.last_seen for r in watched_rows)),
            "-", "-", "see `jobhunt boards` / `health --boards`",
        )  # fmt: skip
    console.print(table)
    configured_unhealthy = any(not row.healthy for row in report if row.source in configured)
    widespread = bool(watched_rows) and len(failing_boards) / len(watched_rows) > 0.5
    if strict and (configured_unhealthy or widespread):
        raise typer.Exit(code=1)


@app.command(
    help="Merge jobs from a pipeline database (e.g. the GitHub Actions data branch) into this one.",
    rich_help_panel="5. Maintenance",
)
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
        f"{stats.snapshots_new} descriptions, {stats.screens_new} screens, {stats.extractions_new} extractions, "
        f"{stats.usage_new} usage records. Next: `jobhunt status`."
    )


@app.command(
    help="Export the queue (or every job with --all) as CSV to stdout.", rich_help_panel="4. Explore what jobs ask for"
)
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


@app.command(
    "jev-probe",
    help="AI (Jev, only with --send): build and send Jev screen and fit requests for a few jobs.",
    rich_help_panel="5. Maintenance",
)
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
                    questions=screen_questions(ctx.profile().candidate.degree),
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


boards_app = typer.Typer(help="The company-board watchlist (Greenhouse, Lever, Ashby boards polled in full).")
app.add_typer(boards_app, name="boards", rich_help_panel="5. Maintenance")

chatgpt_app = typer.Typer(help="Run extraction and fit on your ChatGPT plan instead of OpenRouter (laptop only).")
app.add_typer(chatgpt_app, name="chatgpt", rich_help_panel="5. Maintenance")


def _chatgpt_auth() -> Any:
    from .services.chatgpt_auth import ChatGPTAuth

    return ChatGPTAuth(settings.chatgpt_credentials_path)


@chatgpt_app.command("login", help="Sign in with ChatGPT in your browser and allow jobhunt to use your plan.")
@syncify
async def chatgpt_login(
    no_browser: Annotated[bool, typer.Option(help="Only print the sign-in link (open it yourself).")] = False,
) -> None:
    import webbrowser

    from .services.chatgpt_auth import ChatGPTAuthError

    def show(url: str) -> None:
        console.print(f"Opening ChatGPT sign-in. If no browser opens, visit:\n[link={url}]{url}[/link]")
        if not no_browser:
            webbrowser.open(url)

    try:
        creds = await _chatgpt_auth().login(open_browser=show)
    except ChatGPTAuthError as exc:
        _fail(str(exc))
    console.print(
        f"[green]Signed in as {creds.email or creds.subject}.[/] Tokens saved to {settings.chatgpt_credentials_path}."
    )
    if settings.llm_provider != "chatgpt":
        console.print("Set [b]LLM_PROVIDER=chatgpt[/b] in .env to use it (keep the GitHub workflow on OpenRouter).")
    await _chatgpt_models_table(warn_only=True)


@chatgpt_app.command("logout", help="Revoke and forget the ChatGPT sign-in.")
@syncify
async def chatgpt_logout() -> None:
    console.print("Signed out." if await _chatgpt_auth().logout() else "Not signed in.")


@chatgpt_app.command("models", help="List the models your ChatGPT plan offers.")
@syncify
async def chatgpt_models() -> None:
    await _chatgpt_models_table(warn_only=False)


async def _chatgpt_models_table(warn_only: bool) -> None:
    import httpx

    from .services.chatgpt import ChatGPTClient
    from .services.llm import LlmConfigError

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            models = await ChatGPTClient(client, _chatgpt_auth()).models()
        except LlmConfigError as exc:
            if warn_only:
                console.print(f"[yellow]Couldn't list plan models: {exc}[/]")
                return
            _fail(str(exc))
    ids = {model_id for model_id, _ in models}
    if not warn_only:
        table = Table("Model id", "Name")
        for model_id, name in models:
            table.add_row(model_id, name)
        console.print(table)
    for setting, value in (
        ("CHATGPT_MODEL", settings.chatgpt_model),
        ("CHATGPT_FIT_MODEL", settings.chatgpt_fit_model),
    ):
        if ids and value not in ids:
            console.print(
                f"[yellow]{setting}={value} isn't in your plan's list. Pick one of: {', '.join(sorted(ids))}[/]"
            )


@chatgpt_app.command("status", help="Show the ChatGPT sign-in and whether jobhunt is using it.")
def chatgpt_status() -> None:
    import time

    creds = _chatgpt_auth().load()
    if not creds.signed_in:
        console.print("Not signed in. Run `jobhunt chatgpt login`.")
    else:
        minutes = max(0, int((creds.expires_at - time.time()) / 60))
        console.print(
            f"Signed in as {creds.email or creds.subject}; access token refreshes in {minutes} min "
            f"(automatic). Plan usage permission: {'yes' if creds.has_plan_scopes else '[red]no[/]'}."
        )
    in_use = settings.llm_provider == "chatgpt"
    console.print(
        f"LLM_PROVIDER={settings.llm_provider}: "
        + (
            f"extraction uses {settings.chatgpt_model}, LLM fit uses {settings.chatgpt_fit_model}."
            if in_use
            else "not using ChatGPT."
        )
    )


@boards_app.callback(invoke_without_command=True)
@syncify
async def boards_summary(ctx_typer: typer.Context) -> None:
    """Summary of the watchlist when run without a subcommand."""
    if ctx_typer.invoked_subcommand is not None:
        return
    from .services.boards import board_summary
    from .services.context import app_context

    async with app_context() as ctx:
        summary = await board_summary(ctx.session)
        configured = len(ctx.boards)
    console.print(
        f"Watched boards: {dict(summary.by_status) or 'none'}  ·  by origin: {dict(summary.by_origin) or '-'}  ·  "
        f"active by ATS: {dict(summary.by_ats) or '-'}  ·  never polled yet: {summary.never_polled}  ·  "
        f"plus {configured} from config/boards.toml"
    )
    if summary.productive:
        table = Table("Board", "Company", "Origin", "Last matching job", "Postings last poll")
        for board in summary.productive:
            table.add_row(
                f"{board.ats}:{board.token}", board.company, board.origin.value,
                board.last_lead_at.strftime("%Y-%m-%d") if board.last_lead_at else "-",
                "-" if board.items_last is None else str(board.items_last),
            )  # fmt: skip
        console.print(table)
    console.print("Subcommands: seed, add, pause, resume. `jobhunt boards --help` for details.")


@boards_app.command("seed", help="Import boards from the lists in config/board_seeds.toml.")
@syncify
async def boards_seed(
    name: Annotated[list[str] | None, typer.Option("--name", help="Only this seed (repeatable).")] = None,
) -> None:
    from .schemas.config import load_board_seeds
    from .services.boards import seed_boards
    from .services.context import app_context

    seeds = load_board_seeds(settings.board_seeds_path)
    if name:
        seeds = [seed for seed in seeds if seed.name in name]
    if not seeds:
        _fail(f"No seeds to run. Check {settings.board_seeds_path}.")
    async with app_context() as ctx:
        results = await seed_boards(ctx.session, ctx.http, seeds, ctx.boards)
    table = Table("Seed", "Boards in list", "Newly watched", "Error")
    for result in results:
        table.add_row(result.name, str(result.found), str(result.added), result.error or "")
    console.print(table)
    console.print("They're polled on the next `jobhunt run`; boards that 404 twice are retired automatically.")


@boards_app.command("add", help="Watch a board: ats:token (greenhouse:stripe) or any board/posting link.")
@syncify
async def boards_add(
    board: Annotated[str, typer.Argument(help="ats:token or a link.")],
    company: Annotated[str | None, typer.Option(help="Display name.")] = None,
) -> None:
    from .models.enums import BoardOrigin
    from .services.boards import add_boards, parse_board_ref
    from .services.context import app_context

    try:
        key = parse_board_ref(board)
    except ValueError as exc:
        _fail(str(exc))
    async with app_context() as ctx:
        added = await add_boards(ctx.session, {key: company or key[1]}, BoardOrigin.MANUAL, "cli", ctx.boards)
    console.print(f"Now watching {key[0]}:{key[1]}." if added else f"{key[0]}:{key[1]} was already watched.")


async def _set_status(board: str, status_value: str, reason: str) -> None:
    from .models.enums import BoardStatus
    from .services.boards import parse_board_ref, set_board_status
    from .services.context import app_context

    try:
        ats, token = parse_board_ref(board)
    except ValueError as exc:
        _fail(str(exc))
    async with app_context() as ctx:
        try:
            await set_board_status(ctx.session, ats, token, BoardStatus(status_value), reason)
        except LookupError as exc:
            _fail(str(exc))
    console.print(f"{ats}:{token} is now {status_value}.")


@boards_app.command("pause", help="Stop polling a board (it stays paused even if new leads point at it).")
@syncify
async def boards_pause(board: Annotated[str, typer.Argument(help="ats:token or a link.")]) -> None:
    await _set_status(board, "paused", "paused by hand")


@boards_app.command("resume", help="Poll a paused or dead board again.")
@syncify
async def boards_resume(board: Annotated[str, typer.Argument(help="ats:token or a link.")]) -> None:
    await _set_status(board, "active", "resumed by hand")


if __name__ == "__main__":
    app()


# `jobhunt --help` lists panels in order of first appearance: sort so "1. Start here" comes first and the pipeline
# steps read in the order `run` performs them.
_START_ORDER = {"status": 0, "run": 1, "review": 2}
app.registered_commands.sort(
    key=lambda command: (
        command.rich_help_panel or "",
        _START_ORDER.get(getattr(command.callback, "__name__", ""), 9),
        str(command.help or "")[:7],
    )
)
