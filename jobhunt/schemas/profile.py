"""The candidate profile, loaded from `config/profile.toml`.

Everything that makes scoring personal lives here, so changing a weight is a one-line diff and the
profile hash recorded on each score tells you which version produced it.
"""

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Topic(BaseModel):
    """A named interest or growth goal. Each becomes one Jev Noul, so describe it as observable work."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short identifier shown in score notes, e.g. 'healthcare'.")
    description: str = Field(
        description="What the person in the job would be doing, e.g. 'healthcare, biomedical or clinical problems'."
    )
    weight: float = Field(default=1.0, ge=0, le=1, description="How much a match counts (0-1).")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Candidate name, used only in LLM prompts.")
    graduation: str = Field(description="Expected graduation, free text, e.g. 'December 2026'.")
    years_experience: float = Field(default=0.0, description="Full-time professional experience in years.")
    us_citizen: bool = Field(description="Whether the candidate is a US citizen.")
    needs_sponsorship: bool = Field(default=False, description="Whether the candidate needs visa sponsorship.")
    has_clearance: bool = Field(default=False, description="Whether the candidate holds an active clearance.")
    degree: str = Field(default="bachelors", description="Highest degree held or in progress.")
    skills: list[str] = Field(description="Skills the candidate can credibly claim. Normalized like job skills.")
    interests: str = Field(description="Free-text paragraph of interests and goals, given to the fit model.")
    preferred_locations: list[str] = Field(
        default_factory=list,
        description="Substrings matched against job locations (case-insensitive), e.g. 'Maryland', 'Remote'.",
    )
    preferred_domains: list[str] = Field(default_factory=list, description="Domains that raise value.")
    interest_topics: list[Topic] = Field(
        default_factory=list,
        description="Kinds of work you want, judged one by one by Jev to compute the interest component.",
    )
    growth_goals: list[Topic] = Field(
        default_factory=list,
        description="What you want a first job to grow, judged one by one by Jev for the learning component.",
    )


class Companies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dream: list[str] = Field(default_factory=list, description="Companies that add full company-value points.")
    warm: list[str] = Field(default_factory=list, description="Companies where the candidate has a warm contact.")
    high_competition: list[str] = Field(
        default_factory=list, description="Brand-name employers with very high applicant volume."
    )
    avoid: list[str] = Field(default_factory=list, description="Companies to drop from the queue entirely.")


class ValueWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interest: float = 0.30
    learning: float = 0.20
    domain: float = 0.10
    compensation: float = 0.15
    company: float = 0.10
    location: float = 0.15


class LikelihoodWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: float = 0.30
    experience: float = 0.20
    level: float = 0.15
    freshness: float = 0.15
    competition: float = 0.10
    warm: float = 0.10


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: ValueWeights = Field(default_factory=ValueWeights)
    likelihood: LikelihoodWeights = Field(default_factory=LikelihoodWeights)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Weights":
        for name, group in (("value", self.value), ("likelihood", self.likelihood)):
            total = sum(group.model_dump().values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"weights.{name} must sum to 1.0 (got {total:.2f})")
        return self


class Compensation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_usd: int = Field(default=70000, description="Salary at or below which compensation value is 0.")
    target_usd: int = Field(default=120000, description="Salary at or above which compensation value is 1.")


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    companies: Companies = Field(default_factory=Companies)
    weights: Weights = Field(default_factory=Weights)
    compensation: Compensation = Field(default_factory=Compensation)

    @property
    def content_hash(self) -> str:
        """Stable hash of the profile; a new hash means scores must be recomputed."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_profile(path: Path) -> Profile:
    with path.open("rb") as handle:
        return Profile.model_validate(tomllib.load(handle))
