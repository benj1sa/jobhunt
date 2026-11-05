"""Keyboard-driven review queue built with Textual.

Left: the queue sorted by priority. Right: everything needed to decide on the highlighted job, including the
score breakdown. Decisions are appended to the `review` table immediately, so quitting never loses work.
"""

import webbrowser
from logging import getLogger
from typing import Any, ClassVar

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static

from .models.enums import Decision
from .services.context import app_context
from .services.queue import QueueItem, build_queue, load_description, record_review
from .services.skills import normalize_skills

logger = getLogger(__name__)

DESCRIPTION_PREVIEW_CHARS = 3500


class ReasonScreen(ModalScreen[str | None]):
    """Ask for an optional one-line reason. Enter submits, Escape cancels the decision."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, decision: Decision) -> None:
        super().__init__()
        self.decision = decision

    def compose(self) -> ComposeResult:
        with Vertical(id="reason-box"):
            yield Label(f"Reason for '{self.decision.value}' (optional, Enter to save, Esc to cancel):")
            yield Input(placeholder="e.g. needs 3+ years of Go", id="reason-input")

    @on(Input.Submitted)
    def submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


def _bar(score: float, width: int = 10) -> str:
    """Text bar that reads without color: '#######---'."""
    filled = round(max(0.0, min(1.0, score)) * width)
    return "#" * filled + "-" * (width - filled)


def render_detail(item: QueueItem, description: str | None, have: set[str], aliases: dict[str, str]) -> str:
    job = item.job
    req: dict[str, Any] = item.requirements
    lines: list[str] = [
        f"[b]{escape(job.title)}[/b]",
        f"{escape(job.company)}  ·  {escape(job.location or 'location not listed')}",
        f"[u]{escape(job.posting_url)}[/u]",
        f"Found on: {escape(', '.join(item.sources) or 'unknown')}",
    ]
    if item.decision:
        lines.append(f"Current decision: [b]{item.decision.value}[/b] {escape(item.review_reason or '')}")
    if item.score:
        components = item.score.components
        lines += [
            "",
            (
                f"[b]Priority {item.score.priority:.2f}[/b]   value {item.score.value:.2f}   "
                f"likelihood {item.score.likelihood:.2f}"
            ),
        ]
        for axis in ("value", "likelihood"):
            lines.append(f"[b]{axis.title()}[/b]")
            for name, part in components.get(axis, {}).items():
                lines.append(
                    f"  {name:<12} {_bar(part['score'])} {part['score']:.2f} x{part['weight']:.2f}  "
                    f"{escape(part.get('note') or '')}"
                )
        for gate in components.get("gates", []):
            lines.append(f"  [b]GATE[/b] {escape(gate)}")
        if fit := components.get("fit"):
            detail = fit.get("detail") or {}
            lines.append(
                f"  fit by {escape(str(detail.get('provider', '?')))} ({escape(str(detail.get('model', '')))})"
            )
            if rationale := detail.get("rationale"):
                lines += ["", f"[i]{escape(rationale)}[/i]"]
    if req:
        required = normalize_skills(req.get("required_skills", []), aliases)
        preferred = normalize_skills(req.get("preferred_skills", []), aliases)
        mark = lambda skills: ", ".join(f"{escape(s)} (have)" if s.lower() in have else escape(s) for s in skills)
        lines += [
            "",
            f"[b]Summary[/b] {escape(req.get('summary', ''))}",
            (
                f"Level: {req.get('seniority')}  ·  years: {req.get('min_years_experience') or 'n/s'}  ·  "
                f"mode: {req.get('work_mode')}  ·  domain: {escape(req.get('domain', ''))}"
            ),
            f"[b]Required[/b] {mark(required) or 'none listed'}",
            f"[b]Preferred[/b] {mark(preferred) or 'none listed'}",
        ]
    if description:
        preview = description[:DESCRIPTION_PREVIEW_CHARS]
        lines += ["", "[b]Description[/b]", escape(preview) + ("…" if len(description) > len(preview) else "")]
    elif job.hydrate_error:
        lines += ["", f"[b]No description:[/b] {escape(job.hydrate_error)}"]
    return "\n".join(lines)


class ReviewApp(App[None]):
    TITLE = "jobhunt review"
    CSS = """
    #queue { width: 45%; }
    #detail-scroll { width: 55%; padding: 0 1; }
    #reason-box { width: 70; height: auto; padding: 1 2; border: thick $accent; background: $surface; }
    ReasonScreen { align: center middle; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("o", "open", "Open posting"),
        Binding("a", "decide('apply')", "Apply"),
        Binding("d", "decide('applied')", "Applied"),
        Binding("l", "decide('later')", "Later"),
        Binding("s", "decide('skip')", "Skip"),
        Binding("r", "toggle_reviewed", "Show reviewed"),
        Binding("g", "reload", "Reload"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.items: list[QueueItem] = []
        self.show_reviewed = False
        self.have: set[str] = set()
        self.aliases: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="queue", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="detail-scroll"):
                yield Static("Loading…", id="detail")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#queue", DataTable)
        table.add_columns("Pri", "Val", "Lik", "Company", "Title", "State")
        await self.load_queue()

    async def load_queue(self, keep_row: int = 0) -> None:
        async with app_context() as ctx:
            profile = ctx.profile()
            self.aliases = ctx.aliases
            self.have = {s.lower() for s in normalize_skills(profile.candidate.skills, ctx.aliases)}
            self.items = await build_queue(
                ctx.session,
                profile.content_hash,
                include_reviewed=self.show_reviewed,
                include_unscored=self.show_reviewed,
            )
        table = self.query_one("#queue", DataTable)
        table.clear()
        for item in self.items:
            score = item.score
            table.add_row(
                f"{score.priority:.2f}" if score else "-",
                f"{score.value:.2f}" if score else "-",
                f"{score.likelihood:.2f}" if score else "-",
                item.job.company[:24],
                item.job.title[:48],
                item.decision.value if item.decision else "",
                key=str(item.job.id),
            )
        self.sub_title = f"{len(self.items)} jobs" + (" (incl. reviewed)" if self.show_reviewed else "")
        if self.items:
            table.move_cursor(row=min(keep_row, len(self.items) - 1))
            self.show_detail(table.cursor_row)
        else:
            self.query_one("#detail", Static).update(
                "Queue is empty. Run `jobhunt run` (or `jobhunt pull`) to fill it, or press r to show reviewed jobs."
            )

    def current(self) -> QueueItem | None:
        row = self.query_one("#queue", DataTable).cursor_row
        return self.items[row] if 0 <= row < len(self.items) else None

    @on(DataTable.RowHighlighted)
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_detail(event.cursor_row)

    @work(exclusive=True, group="detail")
    async def show_detail(self, row: int) -> None:
        if not 0 <= row < len(self.items):
            return
        item = self.items[row]
        if item.description is None:
            async with app_context() as ctx:
                item.description = await load_description(ctx.session, item.job.id) or ""
        self.query_one("#detail", Static).update(render_detail(item, item.description, self.have, self.aliases))
        self.query_one("#detail-scroll", VerticalScroll).scroll_home(animate=False)

    def action_open(self) -> None:
        if item := self.current():
            webbrowser.open(item.job.posting_url)
            self.notify(f"Opened {item.job.posting_url}")

    def action_decide(self, decision_value: str) -> None:
        item = self.current()
        if item is None:
            return
        decision = Decision(decision_value)
        row = self.query_one("#queue", DataTable).cursor_row

        async def save(reason: str | None) -> None:
            if reason is None:
                self.notify("Cancelled")
                return
            async with app_context() as ctx:
                await record_review(ctx.session, item.job.id, decision, reason)
            self.notify(f"{decision.value}: {item.job.company} - {item.job.title}")
            await self.load_queue(keep_row=row)

        self.push_screen(ReasonScreen(decision), save)

    async def action_toggle_reviewed(self) -> None:
        self.show_reviewed = not self.show_reviewed
        await self.load_queue()

    async def action_reload(self) -> None:
        await self.load_queue(keep_row=self.query_one("#queue", DataTable).cursor_row)
