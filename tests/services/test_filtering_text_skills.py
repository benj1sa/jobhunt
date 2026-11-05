import pytest

from jobhunt.schemas.extraction import JobRequirements
from jobhunt.services.filtering import judge_relevance, location_passes, title_passes
from jobhunt.services.skills import normalize_skill, normalize_skills
from jobhunt.services.text import content_hash, html_to_text


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer",
        "Software Engineer II",
        "AI Engineer",
        "MLOps Engineer",
        "Machine Learning Software Engineer",
        "Software Engineer - New Grad",
        "University Grad Software Engineer, 2027 (USA)",
        "Full Stack Engineer",
        "Software Developer",
        "Applied AI Engineer",
    ],
)
def test_title_passes(title):
    assert title_passes(title)


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Staff ML Engineer",
        "Software Engineer Intern",
        "Software Engineer III",
        "Sales Engineer",
        "Hardware Engineer",
        "Engineering Manager",
        "Claude Corps",
        "Product Designer",
    ],
)
def test_title_rejected(title):
    assert not title_passes(title)


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (None, True),
        ("Remote", True),
        ("New York, NY", True),
        ("College Park, MD", True),
        ("Remote in USA", True),
        ("Remote - Canada", False),
        ("London, UK", False),
        ("Bengaluru", False),
        ("Hybrid in Berlin", False),
        ("Toronto, ON; Seattle, WA", True),
    ],
)
def test_location_passes(location, expected):
    assert location_passes(location) is expected


def test_html_to_text_decodes_greenhouse_double_encoding():
    encoded = "&amp;lt;p&amp;gt;Build &amp;lt;b&amp;gt;models&amp;lt;/b&amp;gt;&amp;lt;/p&amp;gt;&amp;lt;ul&amp;gt;&amp;lt;li&amp;gt;Python&amp;lt;/li&amp;gt;&amp;lt;/ul&amp;gt;"
    text = html_to_text(encoded)
    assert "Build models" in text
    assert "- Python" in text
    assert "<" not in text


def test_html_to_text_drops_scripts():
    assert html_to_text("<p>Hi</p><script>evil()</script>") == "Hi"


def test_content_hash_ignores_whitespace_and_case():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")


def test_normalize_skill(aliases):
    assert normalize_skill("  torch ", aliases) == "PyTorch"
    assert normalize_skill("kubernetes", aliases) == "Kubernetes"
    assert normalize_skill("FastAPI", aliases) == "FastAPI"
    assert normalize_skill(" , ", aliases) is None


def test_normalize_skills_dedupes_in_order(aliases):
    assert normalize_skills(["pytorch", "Python", "torch", "python"], aliases) == ["PyTorch", "Python"]


def test_judge_relevance_accepts_new_grad(requirements):
    assert judge_relevance(requirements).relevant


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"seniority": "senior"}, "seniority is senior"),
        ({"min_years_experience": 5}, "requires 5+ years"),
        ({"us_based": False}, "not US-based"),
        ({"role_family": "other"}, "not a software/AI/ML role"),
    ],
)
def test_judge_relevance_rejects(requirements: JobRequirements, changes, reason):
    result = judge_relevance(JobRequirements.model_validate(requirements.model_dump() | changes))
    assert not result.relevant
    assert reason in result.reason
