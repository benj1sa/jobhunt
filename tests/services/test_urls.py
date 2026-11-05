import pytest

from jobhunt.models.enums import Ats
from jobhunt.schemas.config import BoardConfig
from jobhunt.services.urls import (
    AtsRef,
    find_ats_links,
    is_aggregator,
    is_blocked,
    normalize_url,
    parse_ats,
    posting_url_for,
)

BOARDS = [BoardConfig(ats="greenhouse", token="pinterest", company="Pinterest", domains=["pinterestcareers.com"])]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://WWW.Example.com/jobs/1/?utm_source=x&utm_medium=y", "https://example.com/jobs/1"),
        ("https://example.com:443/a#frag", "https://example.com/a"),
        ("https://example.com/a?b=2&a=1&gclid=z", "https://example.com/a?a=1&b=2"),
        ("example.com/careers", "https://example.com/careers"),
        ("https://example.com/", "https://example.com/"),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_normalize_url_keeps_identifying_params():
    assert "gh_jid=123" in normalize_url("https://careers.x.com/job?gh_jid=123&utm_campaign=a")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://job-boards.greenhouse.io/durable/jobs/5156304008",
            AtsRef(Ats.GREENHOUSE, "durable", "5156304008"),
        ),
        (
            "https://boards.greenhouse.io/embed/job_app?for=acme&token=42",
            AtsRef(Ats.GREENHOUSE, "acme", "42"),
        ),
        (
            "https://jobs.lever.co/palantir/3F2A9C1E-5B7D-4E8A-9C21-7D4E5F6A8B90",
            AtsRef(Ats.LEVER, "palantir", "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"),
        ),
        (
            "https://jobs.ashbyhq.com/openai/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90/application",
            AtsRef(Ats.ASHBY, "openai", "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"),
        ),
        (
            "https://umd.wd1.myworkdayjobs.com/en-US/UMCP/job/MLOps-Engineer_JR104177?utm_source=google",
            AtsRef(Ats.WORKDAY, "umd|wd1|UMCP", "MLOps-Engineer_JR104177"),
        ),
        (
            "https://intel.wd1.myworkdayjobs.com/en-us/external/job/US-Oregon/Engineer_JR1",
            AtsRef(Ats.WORKDAY, "intel|wd1|external", "US-Oregon/Engineer_JR1"),
        ),
        (
            "https://www.pinterestcareers.com/jobs/7838591/university-grad/?gh_jid=7838591",
            AtsRef(Ats.GREENHOUSE, "pinterest", "7838591"),
        ),
    ],
)
def test_parse_ats(url, expected):
    assert parse_ats(url, BOARDS) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/jobs/view/4454811409/",
        "https://example.com/careers?gh_jid=123",  # unknown career site, no board mapping
        "https://job-boards.greenhouse.io/durable",  # board page, not a posting
        "https://jobs.lever.co/palantir",
    ],
)
def test_parse_ats_returns_none(url):
    assert parse_ats(url, BOARDS) is None


def test_canonical_key_lowercases_board():
    assert AtsRef(Ats.GREENHOUSE, "Durable", "1").canonical_key == "greenhouse:durable:1"


def test_blocked_and_aggregator_hosts():
    assert is_blocked("https://www.linkedin.com/jobs/view/1/")
    assert is_blocked("https://wellfound.com/l/2CoC9W")
    assert not is_blocked("https://job-boards.greenhouse.io/x/jobs/1")
    assert is_aggregator("https://builtin.com/job/ai-engineer/1")
    assert not is_aggregator("https://linkedin.com/x")


def test_find_ats_links_in_html():
    html = (
        '<a href="https://job-boards.greenhouse.io/acme/jobs/1?gh_src=x">Apply</a> and '
        "https://jobs.lever.co/beta/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90."
    )
    links = find_ats_links(html)
    assert set(links) == {"greenhouse:acme:1", "lever:beta:3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"}


@pytest.mark.parametrize(
    ("ref", "url"),
    [
        (AtsRef(Ats.GREENHOUSE, "a", "1"), "https://job-boards.greenhouse.io/a/jobs/1"),
        (
            AtsRef(Ats.LEVER, "a", "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"),
            "https://jobs.lever.co/a/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90",
        ),
        (
            AtsRef(Ats.ASHBY, "a", "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"),
            "https://jobs.ashbyhq.com/a/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90",
        ),
        (AtsRef(Ats.WORKDAY, "t|wd5|Site", "Job_1"), "https://t.wd5.myworkdayjobs.com/Site/job/Job_1"),
    ],
)
def test_posting_url_round_trips(ref, url):
    assert posting_url_for(ref) == url
    assert parse_ats(url) == ref
