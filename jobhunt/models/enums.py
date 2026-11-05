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
