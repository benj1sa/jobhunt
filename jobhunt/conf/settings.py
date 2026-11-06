import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

from .db import DatabaseSettings


class Settings(DatabaseSettings):
    model_config = SettingsConfigDict(
        # JOBHUNT_ENV_FILE lets tests ignore a developer's real .env (and its API keys).
        env_file=os.environ.get("JOBHUNT_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = Field(default="jobhunt", description="Human-readable project name.")
    debug: bool = Field(default=False, description="Enable SQL echo and debug logging.")

    # Files Ben edits by hand. Kept out of the database so they diff cleanly in git.
    profile_path: Path = Field(
        default=Path("config/profile.toml"),
        description="Candidate profile: skills, interests, preferences and score weights.",
    )
    boards_path: Path = Field(
        default=Path("config/boards.toml"),
        description="Watchlist of company ATS boards (Greenhouse, Lever, Ashby) to poll directly.",
    )
    skills_aliases_path: Path = Field(
        default=Path("config/skills.toml"),
        description="Alias map used to normalize extracted skill names (e.g. 'pytorch' -> 'PyTorch').",
    )
    board_seeds_path: Path = Field(
        default=Path("config/board_seeds.toml"),
        description="Lists of company boards to import into the watchlist with `jobhunt boards seed`.",
    )
    board_idle_days: int = Field(
        default=60,
        description="Pause a watched board after this many days without a job that passes the filters.",
    )
    board_dead_after_failures: int = Field(
        default=2,
        description="Mark a watched board dead after this many consecutive 'not found' polls.",
    )
    max_active_boards: int = Field(
        default=3000,
        description="Poll at most this many watched boards per run (most recently productive first).",
    )
    ingest_run_retention_days: int = Field(
        default=90,
        description="Delete ingest_run rows older than this (health only looks at the last 30 days).",
    )
    skill_groups_path: Path = Field(
        default=Path("config/skill_groups.toml"),
        description="Skill families and concept/tool kinds used by the grouped skills report.",
    )
    capabilities_path: Path = Field(
        default=Path("config/capabilities.toml"),
        description="Capability taxonomy that extraction tags responsibility phrases with.",
    )

    http_timeout: float = Field(default=20.0, description="Timeout in seconds for a single HTTP request.")
    http_concurrency: int = Field(default=8, description="Maximum concurrent outbound HTTP requests.")
    http_user_agent: str = Field(
        default="jobhunt/0.1 (personal job search tool)",
        description="User-Agent header sent to job boards.",
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key. Leave unset to run the pipeline on rules only (no extraction or LLM fit).",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    extraction_model: str = Field(
        default="anthropic/claude-haiku-5.5",
        description="OpenRouter model used to extract structured requirements from job descriptions.",
    )
    fit_model: str = Field(
        default="anthropic/claude-sonnet-5.5",
        description="OpenRouter model used to judge role fit against the candidate profile.",
    )
    llm_concurrency: int = Field(
        default=8,
        description="LLM extraction calls in flight at once (results are still stored one job at a time).",
    )
    llm_max_description_chars: int = Field(
        default=24000,
        description="Job description text is truncated to this many characters before being sent to the LLM.",
    )

    typesafe_api_key: SecretStr | None = Field(
        default=None,
        description="TypeSafe API key for Jev. Enables the screen before extraction and the Jev fit provider.",
    )
    typesafe_base_url: str = Field(default="https://api.typesafe.ai", description="TypeSafe API base URL.")
    jev_model: str = Field(
        default="jev-latest",
        description="Jev model or alias. Pin a version (e.g. 'jev-1.14.0') to keep answers stable across releases.",
    )
    screen_max_description_chars: int = Field(
        default=24000,
        description="Description characters sent to the Jev screen (default context is 32k tokens per question).",
    )
    fit_provider: Literal["auto", "jev", "openrouter", "none"] = Field(
        default="auto",
        description="Who judges fit: 'auto' uses Jev when TYPESAFE_API_KEY is set, else OpenRouter, else nothing.",
    )

    simplify_listings_url: str = Field(
        default="https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/.github/scripts/listings.json",
        description="URL of the SimplifyJobs New-Grad listings JSON.",
    )
    simplify_max_age_days: int = Field(
        default=45,
        description="Ignore SimplifyJobs listings posted more than this many days ago.",
    )
    hn_months_back: int = Field(
        default=1,
        description="How many monthly HN 'Who is hiring?' threads to read (1 = latest only).",
    )
