"""Discovery from the SimplifyJobs New-Grad-Positions listings JSON."""

from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any

from ..services.filtering import location_passes, title_passes
from ..services.http import HttpClient
from ..settings import settings
from .base import RawLead, Source

logger = getLogger(__name__)

# Simplify's category labels have changed over time; both spellings are observed in the live file.
RELEVANT_CATEGORIES = {
    "Software",
    "Software Engineering",
    "AI/ML/Data",
    "Data Science, AI & Machine Learning",
}


class SimplifySource(Source):
    name = "simplify"

    async def fetch(self, client: HttpClient) -> list[RawLead]:
        listings: list[dict[str, Any]] = await client.get_json(settings.simplify_listings_url)
        cutoff = datetime.now(tz=UTC) - timedelta(days=settings.simplify_max_age_days)
        leads: list[RawLead] = []
        for item in listings:
            if not (item.get("active") and item.get("is_visible", True)):
                continue
            if item.get("category") not in RELEVANT_CATEGORIES:
                continue
            posted = datetime.fromtimestamp(item.get("date_posted") or 0, tz=UTC)
            if posted < cutoff:
                continue
            title = str(item.get("title") or "")
            locations: list[str] = item.get("locations") or []
            location = "; ".join(locations) or None
            if not title_passes(title) or not any(location_passes(loc) for loc in (locations or [""])):
                continue
            leads.append(
                RawLead(
                    source=self.name,
                    company=str(item.get("company_name") or "Unknown"),
                    title=title,
                    url=str(item["url"]),
                    location=location,
                    posted_at=posted,
                    extra={
                        "simplify_id": item.get("id"),
                        "sponsorship": item.get("sponsorship"),
                        "degrees": item.get("degrees"),
                    },
                )
            )
        logger.info("Simplify: %d listings, %d leads after filters", len(listings), len(leads))
        return leads
