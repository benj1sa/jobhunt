from enum import StrEnum


class LeadStatus(StrEnum):
    """Lifecycle of a discovered lead on its way to becoming a canonical job."""

    NEW = "new"
    RESOLVED = "resolved"
    # The link points at a board we cannot read automatically (LinkedIn, Wellfound, ...).
    NEEDS_URL = "needs_url"
    FAILED = "failed"


class JobStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class Ats(StrEnum):
    """Where a canonical job's description is fetched from."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    # Anything else: identified by its normalized URL, no structured hydration.
    URL = "url"


class Decision(StrEnum):
    APPLY = "apply"
    APPLIED = "applied"
    LATER = "later"
    SKIP = "skip"


class RunStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


class ScreenOutcome(StrEnum):
    """Result of the Jev screen that runs before LLM extraction."""

    PASS = "pass"
    UNSETTLED = "unsettled"
    REJECT = "reject"


class BoardOrigin(StrEnum):
    """How a company board got onto the watchlist."""

    DISCOVERED = "discovered"  # a lead from any source resolved to this board
    SEED = "seed"  # imported from a seed list in config/board_seeds.toml
    MANUAL = "manual"  # added with `jobhunt boards add`


class BoardStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"  # no matching jobs for a long time; reactivated when a new lead points at it
    DEAD = "dead"  # board not found repeatedly (moved, renamed or closed)
