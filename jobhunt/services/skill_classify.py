"""LLM-assisted upkeep of config/skill_groups.toml: propose a group and kind for skills no group claims yet.

Proposals go to `skill_groups.suggested.toml`, never into the hand-edited file. They take effect immediately
(hand-edited entries win), and you can move, fix or delete them like any other config line.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.config import SkillTaxonomy, suggested_path
from ..schemas.extraction import strict_json_schema
from .llm import LlmConfigError, LlmError, OpenRouterClient

logger = getLogger(__name__)

BATCH_SIZE = 60

CLASSIFY_SYSTEM = """You sort skill names from job postings into a fixed set of groups.
For each skill:
- group: the single best group id from the allowed list, or "skip" for things that are not skills
  (company names, job titles, vague words, certifications nobody would study for).
- kind: "tool" for anything you could install, call, buy or name on a resume as a product, library, language,
  protocol or standard (e.g. SageMaker, PyTorch, gRPC, FHIR). "concept" for a broad area or practice
  (e.g. Machine Learning, Security, Distributed systems).
- same_as: if the skill is just another spelling or a near-synonym of one of the listed known skills, that
  known skill's exact name; otherwise null."""


class SkillSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    group: str
    kind: Literal["concept", "tool"]
    same_as: str | None = Field(description="Exact name of a known skill this duplicates, or null.")


class SkillSuggestions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SkillSuggestion]


@dataclass
class ClassifyStats:
    suggested: int = 0
    skipped: int = 0
    failed_batches: int = 0
    errors: list[str] = field(default_factory=list)


def _schema(group_ids: list[str]) -> dict[str, Any]:
    schema = strict_json_schema(SkillSuggestions)
    schema["$defs"]["SkillSuggestion"]["properties"]["group"] = {"type": "string", "enum": [*group_ids, "skip"]}
    return schema


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # JSON string escaping is valid TOML basic-string escaping


async def classify_skills(
    llm: OpenRouterClient,
    model: str,
    taxonomy: SkillTaxonomy,
    skills: list[str],
    groups_path: Path,
) -> ClassifyStats:
    """Ask the LLM to place `skills`, then append valid proposals to the suggested file."""
    stats = ClassifyStats()
    group_ids = list(taxonomy.groups)
    known = sorted({cls.name for cls in taxonomy.by_key.values()})
    known_keys = {name.lower(): name for name in known}
    group_list = "\n".join(f"- {gid}: {name}" for gid, name in taxonomy.groups.items())
    lines: list[str] = []
    for start in range(0, len(skills), BATCH_SIZE):
        batch = skills[start : start + BATCH_SIZE]
        user = (
            f"Allowed groups:\n{group_list}\n\nKnown skills (for same_as):\n{', '.join(known)}\n\n"
            "Skills to sort (return one item per skill, same spelling):\n" + "\n".join(f"- {s}" for s in batch)
        )
        try:
            result = await llm.structured(
                model=model, system=CLASSIFY_SYSTEM, user=user, schema=SkillSuggestions, json_schema=_schema(group_ids)
            )
        except LlmConfigError:
            raise
        except LlmError as exc:
            stats.failed_batches += 1
            stats.errors.append(str(exc)[:300])
            continue
        wanted = {s.lower(): s for s in batch}
        for item in result.items:
            original = wanted.pop(item.skill.lower(), None)  # pop: a repeated item would be a duplicate TOML key
            if original is None or item.group not in taxonomy.groups:
                stats.skipped += 1
                continue
            same_as = known_keys.get(item.same_as.lower()) if item.same_as else None
            extra = f", same_as = {_toml_string(same_as)}" if same_as else ""
            lines.append(f'{_toml_string(original)} = {{ group = "{item.group}", kind = "{item.kind}"{extra} }}')
            stats.suggested += 1

    if lines:
        target = suggested_path(groups_path)
        is_new = not target.exists()
        with target.open("a", encoding="utf-8") as handle:
            if is_new:
                handle.write(
                    "# LLM-proposed entries for skill_groups.toml (`jobhunt skills-classify`). Hand-edited entries\n"
                    "# in skill_groups.toml win. Fix or delete lines freely; move good ones into the main file.\n"
                    "[skills]\n"
                )
            handle.write(f"\n# {date.today().isoformat()} via {model}\n" + "\n".join(lines) + "\n")
    return stats
