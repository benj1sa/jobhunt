"""Leads entered by hand: a CSV export (Company, Role, Link, Source) or a single `jobhunt add`."""

import csv
import re
from logging import getLogger
from pathlib import Path

from ..sources.base import RawLead

logger = getLogger(__name__)

REQUIRED_COLUMNS = {"company", "role", "link"}


class CsvFormatError(ValueError):
    """The CSV is missing a required column."""


def source_slug(label: str) -> str:
    """'ColorStack Board' -> 'colorstack-board'. Lowercased so 'Colorstack Slack' and 'ColorStack Slack' match."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "manual"


def leads_from_csv(path: Path, default_source: str = "manual") -> list[RawLead]:
    """Read leads from a CSV with Company, Role, Link and optional Source columns (header names case-insensitive)."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = {name.strip().lower(): name for name in reader.fieldnames or []}
        missing = REQUIRED_COLUMNS - headers.keys()
        if missing:
            raise CsvFormatError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        leads: list[RawLead] = []
        for line_number, row in enumerate(reader, start=2):
            link = (row.get(headers["link"]) or "").strip()
            if not link:
                logger.warning("%s line %d has no link; skipped", path, line_number)
                continue
            source_label = (row.get(headers["source"]) if "source" in headers else None) or default_source
            leads.append(
                RawLead(
                    source=source_slug(source_label),
                    company=(row.get(headers["company"]) or "").strip(),
                    title=(row.get(headers["role"]) or "").strip(),
                    url=link,
                    extra={"source_label": source_label.strip()},
                )
            )
    return leads


def lead_from_args(url: str, company: str, title: str, source: str) -> RawLead:
    return RawLead(
        source=source_slug(source),
        company=company,
        title=title,
        url=url,
        extra={"source_label": source},
    )
