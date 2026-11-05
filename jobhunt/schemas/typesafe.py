"""Request and response models for TypeSafe's System One API (`POST /v1/systemone`).

Shapes follow https://docs.typesafe.ai/api/system-one.md: questions and answers are keyed by names we choose,
and each answer's `type` matches its question.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str = Field(description="A concrete situation in which the statement is true.")
    false: str = Field(description="A concrete situation in which the statement is false.")


class NoulQuestion(BaseModel):
    """A yes/no question; the answer is the probability of yes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """Picks one option. Pair with a Noul when no option might fit."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseModel):
    """Rates on ordered levels; each level's position is its score, starting at 0."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(min_length=1, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    state: dict[str, Any] | str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, description="Probability that the statement is true.")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float = Field(description="Probability-weighted position across levels; may fall between integers.")
    confidence: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(description="Level index (as a string) -> probability.")

    def level_probability(self, level: int) -> float:
        return self.probabilities.get(str(level), 0.0)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str = Field(description="The model version that answered, e.g. 'jev-1.14.0'.")
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)

    def noul(self, name: str) -> float:
        answer = self.answers[name]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"Answer {name!r} is {answer.type}, not noul")
        return answer.noul

    def score(self, name: str) -> ScoreAnswer:
        answer = self.answers[name]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"Answer {name!r} is {answer.type}, not score")
        return answer
