"""Every Jev question, threshold and decision table in one place.

Jev makes judgments; code makes decisions. Each function here either builds a request (state + questions) or turns
answers into a decision with rules you can read. Change wording here, bump the matching *_VERSION, and the pipeline
re-asks instead of reusing cached answers. Probe wording changes against the live API before relying on them
(`jobhunt jev-probe`).
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .models.enums import ScreenOutcome
from .schemas.profile import Profile, Topic
from .schemas.typesafe import NoulCriteria, NoulQuestion, Question, ScoreQuestion, SystemOneResponse

# Starting thresholds from TypeSafe's guidance. Tune them on your own labeled outcomes, not on a handful of cases.
YES = 0.7
NO = 0.3

# v2: added `open_to_new_grads` (direct, observable) because reading "max 2 years out of school" as a level was
# unreliable in the first live run, and named the new-grad-program case in the internship criteria.
SCREEN_VERSION = "screen-v2"
FIT_VERSION = "fit-v1"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "topic"


# --- Screen: should this posting go to (paid) LLM extraction? ---------------------------------------------------
#
# Truth table (code below):
#   entry_level = max(open_to_new_grads, P(experience level is "new grad" or "1-2 years"))
#   any signal clearly no  (role < NO, internship > YES, US < NO, entry_level < NO)    -> REJECT, skip extraction
#   every signal clearly yes                                                          -> PASS
#   anything else                                                                     -> UNSETTLED
# PASS and UNSETTLED both go to extraction, where the LLM's reading settles it; only REJECT saves the call.

EXPERIENCE_LEVELS = [
    "The posting is for new or recent graduates, or says no professional experience is needed.",
    "The posting asks for about one to two years of professional experience, or calls the role junior or entry level.",
    "The posting asks for about three to five years of professional experience, or calls the role mid-level.",
    "The posting asks for six or more years of experience, or calls the role senior, staff, principal or lead.",
]


def screen_state(title: str, company: str, location: str | None, description: str) -> dict[str, Any]:
    return {
        "posting": {
            "title": title,
            "company": company,
            "listed_location": location or "not listed",
            "description": description,
        }
    }


def screen_questions() -> dict[str, Question]:
    return {
        "software_ai_ml_role": NoulQuestion(
            instructions="Is the job in `posting` an engineering role that builds software, AI systems or ML systems?",
            criteria=NoulCriteria(
                true=(
                    "The person mainly writes software, builds or deploys AI or machine learning models, or runs ML "
                    "infrastructure, as a software, AI, ML or MLOps engineer would."
                ),
                false=(
                    "The job is mainly something else, such as sales or solutions engineering, hardware or "
                    "electrical design, IT support, data analysis without engineering, recruiting or management."
                ),
            ),
        ),
        "internship": NoulQuestion(
            instructions="Is the job in `posting` an internship or co-op?",
            criteria=NoulCriteria(
                true="The posting describes a temporary internship or co-op position for current students.",
                false=(
                    "The posting describes a regular full-time or part-time position, including a full-time "
                    "program or rotation for new college graduates."
                ),
            ),
        ),
        "us_location": NoulQuestion(
            instructions="Can the job in `posting` be done from within the United States?",
            criteria=NoulCriteria(
                true="The posting lists a location in the United States, or is remote with the United States allowed.",
                false="Every location in the posting is outside the United States, such as 'London, UK' or 'Remote - Canada'.",
            ),
        ),
        "open_to_new_grads": NoulQuestion(
            instructions="Does `posting` say that new or recent graduates can apply for this job?",
            criteria=NoulCriteria(
                true=(
                    "The posting invites new or recent graduates, for example 'new grad', 'early career', "
                    "'recent graduates welcome' or 'up to 2 years out of school'."
                ),
                false="The posting asks for several years of professional experience and does not mention graduates.",
            ),
        ),
        "experience_level": ScoreQuestion(
            instructions="How much professional experience does the job in `posting` ask for?",
            criteria=EXPERIENCE_LEVELS,
        ),
    }


@dataclass(frozen=True)
class ScreenDecision:
    outcome: ScreenOutcome
    reason: str
    signals: dict[str, float]


def decide_screen(response: SystemOneResponse) -> ScreenDecision:
    level = response.score("experience_level")
    signals = {
        "software_ai_ml_role": response.noul("software_ai_ml_role"),
        "internship": response.noul("internship"),
        "us_location": response.noul("us_location"),
        # Probability mass on "new grad" and "1-2 years". Read the levels, not the averaged position.
        "experience_entry_levels": level.level_probability(0) + level.level_probability(1),
        "open_to_new_grads": response.noul("open_to_new_grads"),
    }
    # Either signal is enough evidence that the role is entry level; reject only when both say no.
    signals["entry_level"] = max(signals["experience_entry_levels"], signals["open_to_new_grads"])
    clear_no = {
        "not a software/AI/ML role": signals["software_ai_ml_role"] < NO,
        "internship": signals["internship"] > YES,
        "not in the US": signals["us_location"] < NO,
        "not entry level": signals["entry_level"] < NO,
    }
    clear_yes = (
        signals["software_ai_ml_role"] > YES
        and signals["internship"] < NO
        and signals["us_location"] > YES
        and signals["entry_level"] > YES
    )
    if any(clear_no.values()):
        return ScreenDecision(ScreenOutcome.REJECT, "; ".join(k for k, v in clear_no.items() if v), signals)
    if clear_yes:
        return ScreenDecision(ScreenOutcome.PASS, "all screen signals clear", signals)
    decisive = ("software_ai_ml_role", "internship", "us_location", "entry_level")
    unsettled = [name for name in decisive if NO <= signals[name] <= YES]
    return ScreenDecision(ScreenOutcome.UNSETTLED, "unsettled: " + ", ".join(unsettled), signals)


# --- Fit: how well does the work match Ben's interests and growth goals? ----------------------------------------
#
# One Noul per interest topic and per growth goal (one judgment each), plus one about early-career support.
# Code combines them:
#   interest = 1 - prod(1 - weight_i * p_i)          chance that at least one weighted interest is a real match
#   learning = 0.6 * mean(growth goals) + 0.4 * mentorship   (mentorship alone if no growth goals)


def fit_state(title: str, company: str, description: str) -> dict[str, Any]:
    return {"job": {"title": title, "company": company, "description": description}}


def _interest_question(topic: Topic) -> NoulQuestion:
    return NoulQuestion(
        instructions=f"Would the person in the job in `job` spend much of their time on {topic.description}?",
        criteria=NoulCriteria(
            true=f"The duties described in `job` center on {topic.description}.",
            false=f"The duties are about other work, even if the company itself works on {topic.description}.",
        ),
    )


def _growth_question(topic: Topic) -> NoulQuestion:
    return NoulQuestion(
        instructions=f"Would the person in the job in `job` get hands-on experience {topic.description}?",
        criteria=NoulCriteria(
            true=f"The duties in `job` include {topic.description}.",
            false=f"The duties do not include {topic.description}.",
        ),
    )


def fit_questions(profile: Profile) -> dict[str, Question]:
    questions: dict[str, Question] = {}
    for topic in profile.candidate.interest_topics:
        questions[f"interest__{_slug(topic.name)}"] = _interest_question(topic)
    for goal in profile.candidate.growth_goals:
        questions[f"growth__{_slug(goal.name)}"] = _growth_question(goal)
    questions["mentorship"] = NoulQuestion(
        instructions="Does `job` describe support for early-career engineers?",
        criteria=NoulCriteria(
            true="The posting mentions mentors, a structured onboarding or rotation program, or training for new graduates.",
            false="The posting says nothing about support for new engineers, or expects the hire to work independently from day one.",
        ),
    )
    return questions


@dataclass
class FitSignals:
    """Interest and learning in [0, 1], whichever provider produced them, with notes for the review screen."""

    interest: float
    learning: float
    interest_note: str
    learning_note: str
    detail: dict[str, Any] = field(default_factory=dict)


def _describe(names_and_probs: list[tuple[str, float]]) -> str:
    shown = [f"{name} {p:.2f}{'?' if NO <= p <= YES else ''}" for name, p in names_and_probs if p >= NO]
    return ", ".join(shown) if shown else "no clear match"


def combine_fit(response: SystemOneResponse, profile: Profile) -> FitSignals:
    interests = [
        (topic, response.noul(f"interest__{_slug(topic.name)}")) for topic in profile.candidate.interest_topics
    ]
    miss = 1.0
    for topic, probability in interests:
        miss *= 1.0 - topic.weight * probability
    interest = 1.0 - miss if interests else 0.5

    growth = [(goal, response.noul(f"growth__{_slug(goal.name)}")) for goal in profile.candidate.growth_goals]
    mentorship = response.noul("mentorship")
    if growth:
        learning = 0.6 * (sum(p for _, p in growth) / len(growth)) + 0.4 * mentorship
    else:
        learning = mentorship

    ranked = sorted(((t.name, p) for t, p in interests), key=lambda item: -item[1])
    return FitSignals(
        interest=round(interest, 4),
        learning=round(learning, 4),
        interest_note=_describe(ranked[:3]),
        learning_note=_describe([(g.name, p) for g, p in growth] + [("mentorship", mentorship)]),
        detail={
            "provider": "jev",
            "model": response.model,
            "version": FIT_VERSION,
            "answers": {name: answer.model_dump() for name, answer in response.answers.items()},
        },
    )
