"""Fit providers: who judges interest and learning for the value score.

Both produce `FitSignals`, so scoring doesn't care which ran. Each provider has a cache key covering everything its
judgment depends on; editing anything else in the profile (weights, salary, companies) reuses cached judgments.
"""

import hashlib
import json
from logging import getLogger
from typing import Protocol

from ..models.pipeline import Job
from ..questions import FIT_VERSION, FitSignals, combine_fit, fit_questions, fit_state
from ..schemas.extraction import Alignment, FitJudgment, JobRequirements
from ..schemas.profile import Profile
from .llm import OpenRouterClient
from .typesafe import TypeSafeClient

logger = getLogger(__name__)

ALIGNMENT_SCORE = {Alignment.STRONG: 1.0, Alignment.MODERATE: 0.6, Alignment.WEAK: 0.2}

FIT_SYSTEM = """You judge how well a job fits one candidate's interests and growth goals.
Be honest and specific; most jobs are a moderate fit. Judge only interest alignment and learning potential,
not whether the candidate is qualified."""


def _hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


class FitProvider(Protocol):
    name: str

    def cache_key(self, profile: Profile) -> str: ...

    async def judge(
        self, job: Job, requirements: JobRequirements, description: str, profile: Profile
    ) -> FitSignals: ...


class JevFitProvider:
    """One Jev request per job: a Noul per interest topic and growth goal, combined in `questions.combine_fit`."""

    name = "jev"

    def __init__(self, client: TypeSafeClient, max_description_chars: int) -> None:
        self._client = client
        self._max_chars = max_description_chars

    def cache_key(self, profile: Profile) -> str:
        topics = [t.model_dump() for t in profile.candidate.interest_topics + profile.candidate.growth_goals]
        return _hash("jev", FIT_VERSION, self._client.model, json.dumps(topics, sort_keys=True))

    async def judge(self, job: Job, requirements: JobRequirements, description: str, profile: Profile) -> FitSignals:
        if not profile.candidate.interest_topics and not profile.candidate.growth_goals:
            raise ValueError("Jev fit needs interest_topics or growth_goals in config/profile.toml")
        request = self._client.build_request(
            state=fit_state(job.title, job.company, description[: self._max_chars]),
            questions=fit_questions(profile),
        )
        return combine_fit(await self._client.system_one(request), profile)


class LlmFitProvider:
    """One OpenRouter call per job, judging the free-text interests paragraph."""

    name = "openrouter"

    def __init__(self, llm: OpenRouterClient, model: str) -> None:
        self._llm = llm
        self._model = model

    def cache_key(self, profile: Profile) -> str:
        return _hash("openrouter", self._model, profile.candidate.interests)

    async def judge(self, job: Job, requirements: JobRequirements, description: str, profile: Profile) -> FitSignals:
        judgment = await self._llm.structured(
            model=self._model,
            system=FIT_SYSTEM,
            user=(
                f"Candidate interests and goals:\n{profile.candidate.interests.strip()}\n\n"
                f"Job: {job.title} at {job.company}\nDomain: {requirements.domain}\n"
                f"What the role does: {requirements.summary}\n"
                f"Required skills: {', '.join(requirements.required_skills) or 'not listed'}"
            ),
            schema=FitJudgment,
        )
        return FitSignals(
            interest=ALIGNMENT_SCORE[judgment.interest_alignment],
            learning=ALIGNMENT_SCORE[judgment.learning_potential],
            interest_note=f"{judgment.interest_alignment.value} alignment",
            learning_note=f"{judgment.learning_potential.value} learning",
            detail={"provider": "openrouter", "model": self._model, "rationale": judgment.rationale},
        )
