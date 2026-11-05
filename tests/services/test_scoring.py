from datetime import UTC, datetime, timedelta

import pytest

from jobhunt.questions import FitSignals
from jobhunt.schemas.extraction import JobRequirements, strict_json_schema
from jobhunt.schemas.profile import Profile
from jobhunt.services.fit import LlmFitProvider
from jobhunt.services.scoring import compute_score

NOW = datetime(2026, 11, 3, tzinfo=UTC)


def variant(requirements: JobRequirements, **changes) -> JobRequirements:
    return JobRequirements.model_validate(requirements.model_dump() | changes)


def test_score_components_are_explainable(job, requirements, profile, aliases):
    fit = FitSignals(interest=1.0, learning=0.6, interest_note="strong alignment", learning_note="moderate learning")
    result = compute_score(job, requirements, profile, aliases, fit, now=NOW)
    value_parts = result.value_parts
    assert value_parts["interest"].score == 1.0
    assert value_parts["interest"].note == "strong alignment"
    assert value_parts["domain"].score == 1.0  # healthcare is preferred
    assert value_parts["company"].note == "dream company"
    # "Baltimore, MD" matches neither "Maryland" nor "Remote": substring matching is literal, so list "MD" too.
    assert value_parts["location"].score == 0.3
    likelihood_parts = result.likelihood_parts
    # Python + PyTorch of Python/PyTorch/Kubernetes; SQL preferred.
    assert likelihood_parts["skills"].note == "2/3 required, 1/1 preferred"
    assert likelihood_parts["level"].score == 1.0
    assert likelihood_parts["freshness"].score == 1.0
    assert result.priority == pytest.approx((result.value * result.likelihood) ** 0.5, abs=1e-3)
    assert set(result.components_json()) == {"value", "likelihood", "gates", "fit"}


def test_weights_must_sum_to_one(profile):
    data = profile.model_dump()
    data["weights"]["value"]["interest"] = 0.9
    with pytest.raises(ValueError, match="must sum to 1.0"):
        Profile.model_validate(data)


def test_clearance_gate_cuts_likelihood(job, requirements, profile, aliases):
    base = compute_score(job, requirements, profile, aliases, None, now=NOW)
    gated = compute_score(job, variant(requirements, requires_clearance=True), profile, aliases, None, now=NOW)
    assert gated.likelihood == pytest.approx(base.likelihood * 0.2, abs=1e-3)
    assert gated.gates == ["active clearance required (x0.2)"]


def test_citizenship_gate_only_for_non_citizens(job, requirements, profile, aliases):
    citizen_only = variant(requirements, requires_us_citizenship=True)
    assert compute_score(job, citizen_only, profile, aliases, None, now=NOW).gates == []
    noncitizen = Profile.model_validate(
        profile.model_dump() | {"candidate": profile.candidate.model_dump() | {"us_citizen": False}}
    )
    assert compute_score(job, citizen_only, noncitizen, aliases, None, now=NOW).likelihood == 0


def test_stale_posting_and_experience_lower_likelihood(job, requirements, profile, aliases):
    fresh = compute_score(job, requirements, profile, aliases, None, now=NOW)
    job.posted_at = NOW - timedelta(days=40)
    stale = compute_score(job, variant(requirements, min_years_experience=3), profile, aliases, None, now=NOW)
    assert stale.likelihood < fresh.likelihood
    assert stale.likelihood_parts["freshness"].score == 0.2
    assert stale.likelihood_parts["experience"].score == pytest.approx(1 - 0.3 * 2.5)


def test_high_competition_and_avoid(job, requirements, profile, aliases):
    job.company = "Google"
    crowded = compute_score(job, requirements, profile, aliases, None, now=NOW)
    assert crowded.likelihood_parts["competition"].score == 0.25
    avoided = Profile.model_validate(profile.model_dump() | {"companies": {"avoid": ["google"]}})
    assert compute_score(job, requirements, avoided, aliases, None, now=NOW).value == 0


def test_no_fit_model_is_neutral(job, requirements, profile, aliases):
    result = compute_score(job, requirements, profile, aliases, None, now=NOW)
    assert result.value_parts["interest"].score == 0.5
    assert result.fit is None


def test_llm_fit_cache_key_ignores_unrelated_profile_edits(profile):
    provider = LlmFitProvider(llm=None, model="m")  # type: ignore[arg-type]
    edited = Profile.model_validate(profile.model_dump() | {"compensation": {"floor_usd": 1, "target_usd": 2}})
    assert provider.cache_key(profile) == provider.cache_key(edited)
    assert profile.content_hash != edited.content_hash


def test_strict_schema_requires_every_property():
    schema = strict_json_schema(JobRequirements)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "title" not in schema
