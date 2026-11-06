"""Wiring shared by the CLI and the TUI: settings → database session, HTTP client, LLM client and config files."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import getLogger

from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.config import (
    BoardConfig,
    Capability,
    SkillTaxonomy,
    load_boards,
    load_capabilities,
    load_skill_aliases,
    load_skill_taxonomy,
)
from ..schemas.profile import Profile, load_profile
from ..settings import settings
from ..sources.base import Source
from ..sources.boards import BoardSource
from ..sources.hn import HackerNewsSource
from ..sources.simplify import SimplifySource
from .boards import watched_sources
from .chatgpt import PREFIX as CHATGPT_PREFIX
from .chatgpt import ChatGPTClient
from .chatgpt_auth import ChatGPTAuth
from .db import get_session
from .fit import FitProvider, JevFitProvider, LlmFitProvider
from .http import HttpClient
from .llm import LlmClient, OpenRouterClient
from .typesafe import TypeSafeClient
from .usage import UsageMeter

logger = getLogger(__name__)

SOURCE_GROUPS = ("simplify", "hn", "boards")


@dataclass
class AppContext:
    session: AsyncSession
    http: HttpClient
    llm: LlmClient | None
    jev: TypeSafeClient | None
    meter: UsageMeter
    boards: list[BoardConfig]
    aliases: dict[str, str]

    @property
    def extraction_model(self) -> str:
        """Model id stored with each extraction ("chatgpt/<model>" on the plan)."""
        if settings.llm_provider == "chatgpt":
            return CHATGPT_PREFIX + settings.chatgpt_model
        return settings.extraction_model

    @property
    def fit_model(self) -> str:
        if settings.llm_provider == "chatgpt":
            return CHATGPT_PREFIX + settings.chatgpt_fit_model
        return settings.fit_model

    @property
    def llm_missing(self) -> str:
        """Why `llm` is None, as an instruction."""
        if settings.llm_provider == "chatgpt":
            return "Not signed in to ChatGPT (LLM_PROVIDER=chatgpt). Run `jobhunt chatgpt login`."
        return "OPENROUTER_API_KEY is not set (or set LLM_PROVIDER=chatgpt and run `jobhunt chatgpt login`)."

    def taxonomy(self) -> SkillTaxonomy:
        return load_skill_taxonomy(settings.skill_groups_path)

    def capabilities(self) -> list[Capability]:
        return load_capabilities(settings.capabilities_path)

    def profile(self) -> Profile:
        return load_profile(settings.profile_path)

    def fit_provider(self, override: str | None = None) -> FitProvider | None:
        """Pick the fit provider from settings (or an override). Raises if the chosen provider has no API key."""
        choice = override or settings.fit_provider
        if choice == "none":
            return None
        if choice == "jev" or (choice == "auto" and self.jev is not None):
            if self.jev is None:
                raise ValueError("fit provider 'jev' needs TYPESAFE_API_KEY")
            return JevFitProvider(self.jev, settings.screen_max_description_chars)
        if choice in ("llm", "openrouter", "chatgpt") or (choice == "auto" and self.llm is not None):
            if self.llm is None:
                raise ValueError(f"fit provider {choice!r} needs an LLM: {self.llm_missing}")
            return LlmFitProvider(self.llm, self.fit_model)
        return None

    async def all_sources(self, groups: list[str] | None = None) -> list[Source]:
        """Configured sources plus every active watched board (when the "boards" group is included)."""
        sources = self.sources(groups)
        if groups is None or "boards" in groups:
            sources.extend(await watched_sources(self.session, self.boards, settings.max_active_boards))
        return sources

    def sources(self, groups: list[str] | None = None) -> list[Source]:
        selected = set(groups or SOURCE_GROUPS)
        unknown = selected - set(SOURCE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown source group(s): {', '.join(sorted(unknown))}. Choose from {SOURCE_GROUPS}.")
        result: list[Source] = []
        if "simplify" in selected:
            result.append(SimplifySource())
        if "hn" in selected:
            result.append(HackerNewsSource())
        if "boards" in selected:
            result.extend(BoardSource(board) for board in self.boards)
        return result


@asynccontextmanager
async def app_context() -> AsyncGenerator[AppContext, None]:
    async with get_session() as session, HttpClient() as http:
        meter = UsageMeter()
        llm: LlmClient | None = None
        if settings.llm_provider == "chatgpt":
            auth = ChatGPTAuth(settings.chatgpt_credentials_path)
            if auth.load().signed_in:
                llm = ChatGPTClient(http.client, auth, meter, settings.chatgpt_reasoning_effort)
        elif settings.openrouter_api_key:
            llm = OpenRouterClient(http, settings.openrouter_api_key, settings.openrouter_base_url, meter)
        jev = (
            TypeSafeClient(http, settings.typesafe_api_key, settings.typesafe_base_url, settings.jev_model, meter)
            if settings.typesafe_api_key
            else None
        )
        yield AppContext(
            session=session,
            http=http,
            llm=llm,
            jev=jev,
            meter=meter,
            boards=load_boards(settings.boards_path),
            aliases=load_skill_aliases(settings.skills_aliases_path),
        )
