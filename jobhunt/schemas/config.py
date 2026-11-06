"""Loaders for the hand-edited TOML files in `config/`."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


def skill_key(name: str) -> str:
    """Comparison key for skill names: case- and punctuation-insensitive ("Test-Driven Development" ==
    "test driven development"), but keeps + and # so C, C++ and C# stay distinct."""
    return re.sub(r"[^a-z0-9+#]", "", name.lower())


class SkillGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class SkillEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str
    kind: Literal["concept", "tool"]
    same_as: str | None = None


class SkillGroupsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: dict[str, SkillGroup] = Field(default_factory=dict)
    skills: dict[str, SkillEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _groups_exist(self) -> "SkillGroupsFile":
        unknown = {entry.group for entry in self.skills.values()} - set(self.groups)
        if unknown:
            raise ValueError(f"skills reference undefined groups: {', '.join(sorted(unknown))}")
        return self


@dataclass(frozen=True)
class SkillClass:
    """Where one skill sits: its display name after `same_as` merging, its group and its kind."""

    name: str
    group: str
    kind: str


@dataclass
class SkillTaxonomy:
    groups: dict[str, str]
    by_key: dict[str, SkillClass]

    def classify(self, skill: str) -> SkillClass | None:
        return self.by_key.get(skill_key(skill))


def suggested_path(path: Path) -> Path:
    """`config/skill_groups.toml` -> `config/skill_groups.suggested.toml` (written by `jobhunt skills-classify`)."""
    return path.with_name(f"{path.stem}.suggested{path.suffix}")


def load_skill_taxonomy(path: Path) -> SkillTaxonomy:
    """Load the hand-edited groups file, then the LLM-suggested file. Hand-edited entries always win."""
    if not path.exists():
        return SkillTaxonomy(groups={}, by_key={})
    with path.open("rb") as handle:
        parsed = SkillGroupsFile.model_validate(tomllib.load(handle))
    entries = dict(parsed.skills)
    suggested = suggested_path(path)
    if suggested.exists():
        with suggested.open("rb") as handle:
            extra = SkillGroupsFile.model_validate({"groups": parsed.groups, **tomllib.load(handle)})
        for name, entry in extra.skills.items():
            entries.setdefault(name, entry)
    by_key: dict[str, SkillClass] = {}
    for name, entry in entries.items():
        by_key.setdefault(skill_key(name), SkillClass(name=entry.same_as or name, group=entry.group, kind=entry.kind))
    return SkillTaxonomy(groups={gid: group.name for gid, group in parsed.groups.items()}, by_key=by_key)


class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)


class CapabilitiesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: list[Capability] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "CapabilitiesFile":
        ids = [c.id for c in self.capability]
        if len(ids) != len(set(ids)) or "other" in ids:
            raise ValueError("capability ids must be unique and must not be 'other'")
        return self


def load_capabilities(path: Path) -> list[Capability]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return CapabilitiesFile.model_validate(tomllib.load(handle)).capability


class BoardSeed(BaseModel):
    """A list of company boards to import into the watchlist.

    format = "simplify": a SimplifyJobs listings.json (any repo); every Greenhouse, Lever or Ashby link, open or
             closed, becomes a board. `categories` optionally limits which listings count.
    format = "links": any text (a README, a CSV, a web page); every ATS posting or board link in it counts.
    format = "lines": one `ats:token` per line, e.g. `greenhouse:stripe`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    url: str
    format: Literal["simplify", "links", "lines"]
    categories: list[str] = Field(default_factory=list)


class BoardSeedsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: list[BoardSeed] = Field(default_factory=list)


def load_board_seeds(path: Path) -> list[BoardSeed]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return BoardSeedsFile.model_validate(tomllib.load(handle)).seed
