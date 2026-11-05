"""Recorded-shape API payloads, trimmed to the fields the adapters read.

Shapes follow the public docs (Greenhouse Job Board API, Lever Postings API, Ashby posting API, Workday cxs,
Algolia HN API) and the live SimplifyJobs listings.json observed on 2026-11-05.
"""

from typing import Any

GREENHOUSE_JOB: dict[str, Any] = {
    "id": 5156304008,
    "title": "AI Engineer",
    "updated_at": "2026-10-30T12:00:00-04:00",
    "first_published": "2026-10-28T09:00:00-04:00",
    "company_name": "Durable",
    "location": {"name": "New York, NY"},
    "absolute_url": "https://job-boards.greenhouse.io/durable/jobs/5156304008",
    "content": "&lt;p&gt;Build LLM features.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;",
}

GREENHOUSE_BOARD: dict[str, Any] = {
    "jobs": [
        GREENHOUSE_JOB,
        {**GREENHOUSE_JOB, "id": 1, "title": "Senior Staff Engineer", "absolute_url": "https://x/1"},
        {**GREENHOUSE_JOB, "id": 2, "title": "Software Engineer", "location": {"name": "London, UK"}},
    ]
}

LEVER_POSTING: dict[str, Any] = {
    "id": "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90",
    "text": "Software Engineer, New Grad",
    "hostedUrl": "https://jobs.lever.co/acme/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90",
    "applyUrl": "https://jobs.lever.co/acme/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90/apply",
    "categories": {"location": "Remote - US", "commitment": "Full-time"},
    "descriptionPlain": "We build things.",
    "lists": [{"text": "Requirements", "content": "<li>Go</li><li>SQL</li>"}],
    "additionalPlain": "Benefits.",
    "workplaceType": "remote",
    "createdAt": 1793800000000,
}

ASHBY_BOARD: dict[str, Any] = {
    "jobs": [
        {
            "title": "Software Engineer, Emerging Talent",
            "location": "San Francisco, CA",
            "isListed": True,
            "isRemote": False,
            "descriptionPlain": "Applied engineering.",
            "publishedAt": "2026-10-01T00:00:00.000Z",
            "jobUrl": "https://jobs.ashbyhq.com/openai/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "applyUrl": "https://jobs.ashbyhq.com/openai/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/application",
        },
        {
            "title": "Hidden",
            "isListed": False,
            "jobUrl": "https://jobs.ashbyhq.com/openai/ffffffff-0000-0000-0000-000000000000",
        },
    ]
}

WORKDAY_JOB: dict[str, Any] = {
    "jobPostingInfo": {
        "title": "Machine Learning Operations (MLOps) Engineer",
        "jobDescription": "<p>Deploy models on campus HPC.</p>",
        "location": "College Park, MD",
        "startDate": "2026-10-15",
        "externalUrl": "https://umd.wd1.myworkdayjobs.com/UMCP/job/MLOps-Engineer_JR104177",
    },
    "hiringOrganization": {"name": "University of Maryland"},
}

SIMPLIFY_LISTINGS: list[dict[str, Any]] = [
    {
        "source": "Simplify",
        "category": "Software",
        "company_name": "EquipmentShare",
        "id": "fb9cc829",
        "title": "Junior Software Engineer",
        "active": True,
        "date_updated": 0,
        "date_posted": 0,  # replaced with "now" by the test
        "url": "https://www.equipmentshare.com/careers/openings/?gh_jid=8319404",
        "locations": ["Columbia, MO"],
        "company_url": "",
        "is_visible": True,
        "sponsorship": "Other",
        "degrees": ["Bachelor's"],
    },
    {
        "source": "Simplify",
        "category": "Hardware",
        "company_name": "Chips",
        "id": "x1",
        "title": "Hardware Engineer",
        "active": True,
        "date_posted": 0,
        "url": "https://chips.example/1",
        "locations": ["Austin, TX"],
        "is_visible": True,
    },
    {
        "source": "Simplify",
        "category": "Software",
        "company_name": "Closed Co",
        "id": "x2",
        "title": "Software Engineer",
        "active": False,
        "date_posted": 0,
        "url": "https://closed.example/1",
        "locations": ["Remote"],
        "is_visible": True,
    },
]

HN_SEARCH: dict[str, Any] = {"hits": [{"objectID": "41000000", "title": "Ask HN: Who is hiring? (November 2026)"}]}

HN_ITEM: dict[str, Any] = {
    "id": 41000000,
    "children": [
        {
            "id": 41000001,
            "created_at": "2026-11-02T15:00:00.000Z",
            "text": "Acme Health | Software Engineer (New Grad) | Baltimore, MD | Onsite<p>We hire new grads. "
            "Apply: https://job-boards.greenhouse.io/acmehealth/jobs/777</p>",
        },
        {
            "id": 41000002,
            "created_at": "2026-11-02T15:00:00.000Z",
            "text": "BigCo | Staff Engineer | Remote<p>10+ years required.</p>",
        },
        {
            "id": 41000003,
            "created_at": "2026-11-02T15:00:00.000Z",
            "text": "EuroCo | Junior Developer | Berlin, Germany<p>Junior friendly.</p>",
        },
    ],
}
