"""Progress reporting for long pipeline steps.

Services call `progress.start(total)` and then `progress.advance(ok=..., item=...)` once per finished item. The CLI
passes a live one-line bar (see `cli._step_progress`) that disappears when the step ends, leaving only the step's
summary line, so the terminal holds one line per step instead of one per job. Everything else gets `NULL_PROGRESS`.
"""

from typing import Protocol


class Progress(Protocol):
    def start(self, total: int) -> None: ...

    def advance(self, n: int = 1, *, ok: bool = True, item: str | None = None) -> None: ...

    def note(self, text: str) -> None:
        """Show a status message in place of the current item (e.g. 'waiting 20s for the rate limit')."""
        ...


class NullProgress:
    def start(self, total: int) -> None:
        return None

    def advance(self, n: int = 1, *, ok: bool = True, item: str | None = None) -> None:
        return None

    def note(self, text: str) -> None:
        return None


NULL_PROGRESS = NullProgress()
