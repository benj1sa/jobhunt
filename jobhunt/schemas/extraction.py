"""Structured outputs requested from the LLM.

These models double as the JSON Schema sent to OpenRouter (`strict: true`), so every field is required and
"unknown" is an explicit enum value or `null` rather than an omitted key.
"""

import copy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Bump when JobRequirements changes shape; old extractions are kept and re-run alongside.
# v2: capability_evidence (verbatim responsibility phrases tagged with config/capabilities.toml ids).
EXTRACTION_SCHEMA_VERSION = 2


class RoleFamily(StrEnum):
    SOFTWARE = "software"
    AI_ML = "ai_ml"
    MLOPS_DATA = "mlops_data"
    OTHER = "other"


class Seniority(StrEnum):
    INTERN = "intern"
    NEW_GRAD = "new_grad"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    UNKNOWN = "unknown"


class Degree(StrEnum):
    NONE = "none"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD = "phd"
    UNKNOWN = "unknown"


class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Tristate(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class CapabilityEvidence(BaseModel):
    """One responsibility or expectation from the posting, copied verbatim, tagged with capability ids."""

    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(description="The phrase exactly as written in the posting (trim to the relevant clause).")
    capabilities: list[str] = Field(description="One to three capability ids it demonstrates, or 'other'.")


class JobRequirements(BaseModel):
    """What a job description asks for. Extracted once per snapshot."""

    model_config = ConfigDict(extra="forbid")

    role_family: RoleFamily = Field(description="Closest role family for the job title and duties.")
    seniority: Seniority = Field(
        description="Level the posting targets. new_grad = explicitly for recent/upcoming graduates."
    )
    min_years_experience: int | None = Field(
        description="Minimum years of professional experience required, or null if not stated."
    )
    degree_required: Degree = Field(description="Minimum degree required (not preferred).")
    locations: list[str] = Field(description="Work locations as written, e.g. 'New York, NY' or 'Remote - US'.")
    us_based: bool = Field(description="True if at least one location is in the United States or remote-US.")
    work_mode: WorkMode = Field(description="Remote, hybrid or onsite.")
    requires_clearance: bool = Field(
        description="True if an active security clearance is required (not merely 'ability to obtain')."
    )
    requires_us_citizenship: bool = Field(description="True if US citizenship is required.")
    sponsors_visa: Tristate = Field(description="Whether the employer sponsors work visas.")
    salary_min_usd: int | None = Field(description="Lower bound of annual base salary in USD, or null.")
    salary_max_usd: int | None = Field(description="Upper bound of annual base salary in USD, or null.")
    required_skills: list[str] = Field(
        description="Concrete skills, languages, frameworks and tools listed as required. Short canonical names."
    )
    preferred_skills: list[str] = Field(description="Skills listed as preferred, nice-to-have or a plus.")
    domain: str = Field(description="Industry or problem domain in 1-3 words, e.g. 'healthcare', 'defense', 'fintech'.")
    summary: str = Field(description="One sentence describing what the person in this role would actually do.")
    capability_evidence: list[CapabilityEvidence] = Field(
        default_factory=list,
        description=(
            "Every responsibility or expectation about how the person works (not tools, degrees, years, benefits "
            "or company description), copied verbatim and tagged with capability ids."
        ),
    )


class Alignment(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class FitJudgment(BaseModel):
    """The LLM's qualitative read on how well a job matches the candidate's interests.

    Categorical rather than numeric: LLMs are inconsistent at calibrated 0-1 scores but reliable at buckets.
    """

    model_config = ConfigDict(extra="forbid")

    interest_alignment: Alignment = Field(description="How well the work matches the candidate's stated interests.")
    learning_potential: Alignment = Field(
        description="How much the role would grow the candidate toward their stated goals."
    )
    rationale: str = Field(description="Two sentences max explaining both judgments.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON Schema for `model` that satisfies OpenAI/OpenRouter `strict: true` rules.

    Strict mode requires `additionalProperties: false` and every property listed in `required` on every object,
    rejects `title`/`default` keywords on some providers, and (OpenAI) allows no other keyword beside `$ref`.
    """
    schema: dict[str, Any] = copy.deepcopy(model.model_json_schema())
    defs: dict[str, Any] = schema.get("$defs", {})

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if "$ref" in node and len(node) > 1:
                # OpenAI strict mode rejects keywords next to $ref ("$ref cannot have keywords {'description'}").
                # Keep the field's description by moving it onto the referenced definition when that has none.
                target = defs.get(str(node["$ref"]).rsplit("/", 1)[-1])
                if isinstance(target, dict) and "description" in node:
                    target.setdefault("description", node["description"])
                for key in [key for key in node if key != "$ref"]:
                    del node[key]
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


def extraction_json_schema(capability_ids: list[str]) -> dict[str, Any]:
    """Strict schema for JobRequirements with the capability ids from config injected as an enum."""
    schema = strict_json_schema(JobRequirements)
    evidence = schema["$defs"]["CapabilityEvidence"]["properties"]["capabilities"]
    evidence["items"] = {"type": "string", "enum": [*capability_ids, "other"]}
    return schema
