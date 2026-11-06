"""Discovery from Hacker News monthly "Ask HN: Who is hiring?" threads via the Algolia HN API.

Comments are free text. By convention the first line is pipe-separated (`Company | Role | Location | ...`).
Only comments that mention entry-level hiring and a software/AI/ML role are kept.
"""

import re
from datetime import datetime
from logging import getLogger
from typing import Any

from ..services.filtering import NON_US_MARKERS, US_STATE_CODES, US_WORDS, location_passes
from ..services.http import HttpClient
from ..services.text import html_to_text
from ..services.urls import find_ats_links
from ..settings import settings
from .base import RawLead, Source

logger = getLogger(__name__)

ALGOLIA = "https://hn.algolia.com/api/v1"
ENTRY_LEVEL = re.compile(
    r"new[- ]?grad|entry[- ]level|junior|early[- ]career|recent grad|university grad", re.IGNORECASE
)
ROLE_SEGMENT = re.compile(r"engineer|developer|scientist|\bswe\b|\bml\b|\bai\b", re.IGNORECASE)


def looks_like_location(segment: str) -> bool:
    return bool(
        re.search(r"\b(remote|onsite|on-site|hybrid)\b", segment, re.IGNORECASE)
        or US_STATE_CODES.search(segment)
        or US_WORDS.search(segment)
        or NON_US_MARKERS.search(segment)
    )


def parse_comment(comment: dict[str, Any]) -> RawLead | None:
    """Turn one top-level HN comment into a lead, or None if it isn't an entry-level software/AI/ML posting."""
    text = html_to_text(comment.get("text") or "")
    if not text or not ENTRY_LEVEL.search(text):
        return None
    first_line = text.split("\n", 1)[0]
    segments = [segment.strip() for segment in first_line.split("|") if segment.strip()]
    if len(segments) < 2:
        return None
    company = segments[0][:200]
    title = next((segment for segment in segments[1:] if ROLE_SEGMENT.search(segment)), None)
    if title is None:
        return None
    location = next(
        (segment for segment in segments[1:] if segment is not title and looks_like_location(segment)), None
    )
    if not location_passes(first_line):
        return None
    comment_url = f"https://news.ycombinator.com/item?id={comment['id']}"
    ats_links = find_ats_links(comment.get("text") or "")
    ats_ref = next(iter(ats_links.values()), None)
    created = comment.get("created_at")
    return RawLead(
        source="hn",
        company=company,
        title=title[:300],
        url=comment_url,
        location=location,
        posted_at=datetime.fromisoformat(created) if created else None,
        # Single ATS link: the comment *is* that posting. Several links: keep the comment as its own job.
        ats_ref=ats_ref if len(ats_links) == 1 else None,
        description_text=text,
        extra={"hn_comment_id": comment["id"], "ats_links": sorted(ats_links)},
    )


class HackerNewsSource(Source):
    name = "hn"

    async def fetch(self, client: HttpClient) -> list[RawLead]:
        search = await client.get_json(
            f"{ALGOLIA}/search_by_date",
            params={"tags": "story,author_whoishiring", "hitsPerPage": 20},
        )
        threads = [hit for hit in search.get("hits", []) if "who is hiring" in (hit.get("title") or "").lower()]
        leads: list[RawLead] = []
        self.items_fetched = 0
        for thread in threads[: settings.hn_months_back]:
            item = await client.get_json(f"{ALGOLIA}/items/{thread['objectID']}")
            self.items_fetched += len(item.get("children") or [])
            for comment in item.get("children") or []:
                if lead := parse_comment(comment):
                    leads.append(lead)
        logger.info("HN: %d threads read, %d entry-level leads", min(len(threads), settings.hn_months_back), len(leads))
        return leads
