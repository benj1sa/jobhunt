"""Loaders for the hand-edited TOML files in `config/`."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ats: Literal["greenhouse", "lever", "ashby"] = Field(description="Applicant tracking system hosting the board.")
    token: str = Field(description="Board slug on that ATS.")
    company: str = Field(description="Display name of the company.")
    domains: list[str] = Field(
        default_factory=list,
        description="Company career-site domains that embed this board (used to resolve ?gh_jid= links).",
    )


class BoardsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board: list[BoardConfig] = Field(default_factory=list)


class SkillsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: dict[str, list[str]] = Field(default_factory=dict)


def load_boards(path: Path) -> list[BoardConfig]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return BoardsFile.model_validate(tomllib.load(handle)).board


def load_skill_aliases(path: Path) -> dict[str, str]:
    """Return a lookup of lowercased alias (and canonical name) -> canonical name."""
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        parsed = SkillsFile.model_validate(tomllib.load(handle))
    lookup: dict[str, str] = {}
    for canonical, aliases in parsed.aliases.items():
        lookup[canonical.strip().lower()] = canonical
        for alias in aliases:
            lookup[alias.strip().lower()] = canonical
    return lookup
