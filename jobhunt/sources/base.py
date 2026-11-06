"""The contract every discovery source implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..services.http import HttpClient
from ..services.urls import AtsRef


@dataclass
class RawLead:
    """One sighting of a job on a source, before resolution.

    `ats_ref` and `description_text` are optional shortcuts: sources that already know the ATS posting
    (board polling) or already hold the description (HN comments, board APIs) pass them along so the
    resolver and hydrator can skip a network round trip.
    """

    source: str
    company: str
    title: str
    url: str
    location: str | None = None
    posted_at: datetime | None = None
    ats_ref: AtsRef | None = None
    description_text: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Source(ABC):
    """A discovery source. `name` is stored on every lead and ingest run, so keep it stable."""

    name: str
    # Raw items read before any filtering, set by `fetch`. Health checks use it to tell "the source is broken"
    # (nothing fetched) apart from "nothing matched the filters today" (items fetched, zero leads).
    items_fetched: int | None = None

    @abstractmethod
    async def fetch(self, client: HttpClient) -> list[RawLead]:
        """Return the source's current leads. Raise on failure; the ingest service records the error."""
