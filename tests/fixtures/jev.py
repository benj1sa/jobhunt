"""Builders for System One responses shaped like https://docs.typesafe.ai/api/system-one.md."""

from typing import Any


def noul(probability: float) -> dict[str, Any]:
    return {"type": "noul", "noul": probability}


def score(probabilities: list[float]) -> dict[str, Any]:
    position = sum(index * p for index, p in enumerate(probabilities))
    return {
        "type": "score",
        "score": position,
        "confidence": max(probabilities),
        "legend": {str(i): f"level {i}" for i in range(len(probabilities))},
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
    }


def response(answers: dict[str, dict[str, Any]], model: str = "jev-1.14.0") -> dict[str, Any]:
    return {"model": model, "answers": answers, "usage": {"input_tokens": 1200, "output_tokens": 8}}


def screen_response(
    role: float = 0.95,
    internship: float = 0.05,
    us: float = 0.9,
    levels: list[float] | None = None,
    new_grads: float = 0.5,
    bachelors: float | None = 0.9,
) -> dict[str, Any]:
    answers = {
        "software_ai_ml_role": noul(role),
        "internship": noul(internship),
        "us_location": noul(us),
        "open_to_new_grads": noul(new_grads),
        "experience_level": score(levels or [0.7, 0.2, 0.08, 0.02]),
    }
    if bachelors is not None:  # None: the question wasn't asked (graduate-degree profile)
        answers["bachelors_accepted"] = noul(bachelors)
    return response(answers)
