"""Clients for the public job-board APIs of each supported applicant tracking system.

Each ATS exposes two operations:
- `list_board(token)`: every open posting on a company board (used for discovery from `config/boards.toml`)
- `fetch_posting(ref)`: one posting with its full description (used for hydration)

Both return `AtsPosting`, so the rest of the pipeline never sees ATS-specific JSON except in `raw`.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import Any

from ..models.enums import Ats
from .http import HttpClient, NotFoundError
from .text import html_to_text
from .urls import AtsRef, posting_url_for

logger = getLogger(__name__)

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards"
LEVER_API = "https://api.lever.co/v0/postings"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board"


class UnsupportedAtsError(Exception):
    """Raised when asked to list a board on an ATS that has no listing API here (e.g. Workday)."""


@dataclass
class AtsPosting:
    ref: AtsRef
    title: str
    company: str | None
    location: str | None
    is_remote: bool | None
    posting_url: str
    apply_url: str | None
    posted_at: datetime | None
    description_text: str
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.debug("Unparseable timestamp %r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _from_epoch_ms(value: Any) -> datetime | None:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    return None


# --- Greenhouse -------------------------------------------------------------------------------------------------


def _greenhouse_posting(token: str, data: dict[str, Any]) -> AtsPosting:
    ref = AtsRef(Ats.GREENHOUSE, token, str(data["id"]))
    location = (data.get("location") or {}).get("name")
    return AtsPosting(
        ref=ref,
        title=data.get("title", "").strip(),
        company=data.get("company_name"),
        location=location,
        is_remote=("remote" in location.lower()) if location else None,
        posting_url=data.get("absolute_url") or posting_url_for(ref),
        apply_url=data.get("absolute_url"),
        posted_at=_parse_iso(data.get("first_published")) or _parse_iso(data.get("updated_at")),
        description_text=html_to_text(data.get("content") or ""),
        raw=data,
    )


async def greenhouse_list(client: HttpClient, token: str, content: bool = False) -> list[AtsPosting]:
    """List a board. Without `content`, postings carry no description (hydration fetches it for matches only),
    which keeps polling thousands of boards cheap."""
    params = {"content": "true"} if content else None
    payload = await client.get_json(f"{GREENHOUSE_API}/{token}/jobs", params=params)
    return [_greenhouse_posting(token, job) for job in payload.get("jobs", [])]


async def greenhouse_fetch(client: HttpClient, ref: AtsRef) -> AtsPosting:
    data = await client.get_json(f"{GREENHOUSE_API}/{ref.board}/jobs/{ref.external_id}")
    return _greenhouse_posting(ref.board, data)


# --- Lever ------------------------------------------------------------------------------------------------------


def _lever_posting(token: str, data: dict[str, Any]) -> AtsPosting:
    ref = AtsRef(Ats.LEVER, token, str(data["id"]).lower())
    categories = data.get("categories") or {}
    parts: list[str] = [data.get("openingPlain") or "", data.get("descriptionPlain") or ""]
    for section in data.get("lists") or []:
        items = html_to_text(section.get("content") or "")
        parts.append(f"{section.get('text', '')}\n{items}")
    parts.append(data.get("additionalPlain") or "")
    workplace = (data.get("workplaceType") or "").lower()
    return AtsPosting(
        ref=ref,
        title=(data.get("text") or "").strip(),
        company=None,
        location=categories.get("location"),
        is_remote=(workplace == "remote") if workplace else None,
        posting_url=data.get("hostedUrl") or posting_url_for(ref),
        apply_url=data.get("applyUrl"),
        posted_at=_from_epoch_ms(data.get("createdAt")),
        description_text="\n\n".join(part.strip() for part in parts if part and part.strip()),
        raw=data,
    )


async def lever_list(client: HttpClient, token: str) -> list[AtsPosting]:
    payload = await client.get_json(f"{LEVER_API}/{token}", params={"mode": "json"})
    return [_lever_posting(token, job) for job in payload]


async def lever_fetch(client: HttpClient, ref: AtsRef) -> AtsPosting:
    data = await client.get_json(f"{LEVER_API}/{ref.board}/{ref.external_id}", params={"mode": "json"})
    return _lever_posting(ref.board, data)


# --- Ashby ------------------------------------------------------------------------------------------------------


def _ashby_posting(token: str, data: dict[str, Any]) -> AtsPosting:
    # The public API's `id` field is not documented; the posting UUID is always the last path segment of jobUrl.
    job_url: str = data.get("jobUrl") or ""
    external_id = str(data.get("id") or job_url.rstrip("/").rsplit("/", 1)[-1]).lower()
    ref = AtsRef(Ats.ASHBY, token, external_id)
    description = data.get("descriptionPlain") or html_to_text(data.get("descriptionHtml") or "")
    return AtsPosting(
        ref=ref,
        title=(data.get("title") or "").strip(),
        company=None,
        location=data.get("location"),
        is_remote=data.get("isRemote"),
        posting_url=job_url or posting_url_for(ref),
        apply_url=data.get("applyUrl"),
        posted_at=_parse_iso(data.get("publishedAt")),
        description_text=description.strip(),
        raw=data,
    )


async def ashby_list(client: HttpClient, token: str) -> list[AtsPosting]:
    payload = await client.get_json(f"{ASHBY_API}/{token}", params={"includeCompensation": "true"})
    return [_ashby_posting(token, job) for job in payload.get("jobs", []) if job.get("isListed", True)]


async def ashby_fetch(client: HttpClient, ref: AtsRef) -> AtsPosting:
    # Ashby has no public single-posting endpoint, so fetch the board and pick the posting.
    for posting in await ashby_list(client, ref.board):
        if posting.ref.external_id == ref.external_id:
            return posting
    raise NotFoundError(f"Ashby posting {ref.external_id} not on board {ref.board}")


# --- Workday ----------------------------------------------------------------------------------------------------


async def workday_fetch(client: HttpClient, ref: AtsRef) -> AtsPosting:
    """Fetch one posting from Workday's candidate-experience JSON endpoint (the same one its job page calls)."""
    tenant, wd, site = ref.board.split("|")
    api = f"https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/job/{ref.external_id}"
    data = await client.get_json(api)
    info = data.get("jobPostingInfo") or {}
    location = info.get("location")
    return AtsPosting(
        ref=ref,
        title=(info.get("title") or "").strip(),
        company=(data.get("hiringOrganization") or {}).get("name"),
        location=location,
        is_remote=("remote" in location.lower()) if location else None,
        posting_url=info.get("externalUrl") or posting_url_for(ref),
        apply_url=info.get("externalUrl"),
        posted_at=_parse_iso(info.get("startDate")),
        description_text=html_to_text(info.get("jobDescription") or ""),
        raw=data,
    )


# --- Dispatch ---------------------------------------------------------------------------------------------------


async def list_board(client: HttpClient, ats: str, token: str) -> list[AtsPosting]:
    match ats:
        case "greenhouse":
            return await greenhouse_list(client, token)
        case "lever":
            return await lever_list(client, token)
        case "ashby":
            return await ashby_list(client, token)
    raise UnsupportedAtsError(ats)


async def fetch_posting(client: HttpClient, ref: AtsRef) -> AtsPosting:
    """Fetch one posting. Raises `NotFoundError` when the posting has been taken down."""
    match ref.ats:
        case Ats.GREENHOUSE:
            return await greenhouse_fetch(client, ref)
        case Ats.LEVER:
            return await lever_fetch(client, ref)
        case Ats.ASHBY:
            return await ashby_fetch(client, ref)
        case Ats.WORKDAY:
            return await workday_fetch(client, ref)
    raise UnsupportedAtsError(ref.ats.value)
