"""Skill-name normalization shared by extraction, scoring and the skills report."""

import re
from collections.abc import Iterable


def normalize_skill(name: str, aliases: dict[str, str]) -> str | None:
    """Map a raw skill string to its canonical name, or None if it is empty after cleanup."""
    cleaned = re.sub(r"\s+", " ", name).strip().strip(".,;:")
    if not cleaned or len(cleaned) > 80:
        return None
    canonical = aliases.get(cleaned.lower())
    if canonical:
        return canonical
    # Keep the extractor's casing (e.g. "FastAPI") unless it gave us an all-lowercase word.
    return cleaned[0].upper() + cleaned[1:] if cleaned.islower() else cleaned


def normalize_skills(names: Iterable[str], aliases: dict[str, str]) -> list[str]:
    """Normalize and deduplicate, preserving first-seen order."""
    seen: dict[str, None] = {}
    for name in names:
        if (skill := normalize_skill(name, aliases)) is not None:
            seen.setdefault(skill, None)
    return list(seen)
