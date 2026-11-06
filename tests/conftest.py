import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# The CLI and services build their engine from settings at import time, so point them at a throwaway
# database before anything from `jobhunt` is imported.
_TEST_DIR = Path(tempfile.mkdtemp(prefix="jobhunt-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR / 'cli.db'}"
# Never read the developer's real .env: tests must not see (or spend) real API keys.
os.environ["JOBHUNT_ENV_FILE"] = str(_TEST_DIR / "no.env")
for _key in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "FIT_PROVIDER", "EXTRACTION_MODEL", "JEV_MODEL"):
    os.environ.pop(_key, None)

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from jobhunt.models.base import Base
from jobhunt.models.pipeline import Job
from jobhunt.schemas.extraction import JobRequirements
from jobhunt.schemas.profile import Profile
from jobhunt.services.db import test_data

# Tables for the database the CLI tests use.
_sync_engine = create_engine(os.environ["DATABASE_URL"])
Base.metadata.create_all(_sync_engine)
_sync_engine.dispose()


@pytest_asyncio.fixture
async def db_session_maker(tmpdir):
    """Creates a test database engine, complete with fake data."""
    test_database_url = f"sqlite+aiosqlite:///{tmpdir}/test_database.db"
    engine = create_async_engine(test_database_url, future=True, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        await test_data(session)

    yield async_session_maker

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_session_maker):
    async with db_session_maker() as session:
        yield session


@pytest.fixture
def profile() -> Profile:
    return Profile.model_validate(
        {
            "candidate": {
                "name": "Test",
                "graduation": "December 2026",
                "years_experience": 0.5,
                "us_citizen": True,
                "skills": ["Python", "pytorch", "SQL", "React"],
                "interests": "Applied ML in healthcare.",
                "preferred_locations": ["Maryland", "Remote"],
                "preferred_domains": ["healthcare"],
            },
            "companies": {"dream": ["Acme"], "warm": ["Comcast"], "high_competition": ["Google"]},
        }
    )


@pytest.fixture
def aliases() -> dict[str, str]:
    return {"pytorch": "PyTorch", "torch": "PyTorch", "python": "Python", "k8s": "Kubernetes"}


@pytest.fixture
def requirements() -> JobRequirements:
    return JobRequirements.model_validate(
        {
            "role_family": "ai_ml",
            "seniority": "new_grad",
            "min_years_experience": 0,
            "degree_required": "bachelors",
            "locations": ["Baltimore, MD"],
            "us_based": True,
            "work_mode": "hybrid",
            "requires_clearance": False,
            "requires_us_citizenship": False,
            "sponsors_visa": "unknown",
            "salary_min_usd": 100000,
            "salary_max_usd": 140000,
            "required_skills": ["Python", "torch", "Kubernetes"],
            "preferred_skills": ["SQL"],
            "domain": "healthcare",
            "summary": "Build ML models for clinical data.",
        }
    )


@pytest.fixture
def job() -> Job:
    now = datetime(2026, 11, 1, tzinfo=UTC)
    return Job(
        canonical_key="greenhouse:acme:1",
        ats="greenhouse",
        board="acme",
        external_id="1",
        company="Acme",
        title="ML Engineer, New Grad",
        location="Baltimore, MD",
        posting_url="https://job-boards.greenhouse.io/acme/jobs/1",
        posted_at=now,
        first_seen_at=now,
        last_seen_at=now,
    )
