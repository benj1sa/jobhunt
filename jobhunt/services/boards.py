"""The company-board watchlist: discovery, seeding, polling lifecycle and summaries.

Greenhouse, Lever and Ashby have no cross-company search, so coverage comes from knowing board names. Boards join
the watchlist three ways (a lead resolved to them, a seed list, or by hand), are polled in full on every ingest,
and leave it on their own: repeated 404s mark a board dead, and a long stretch with no matching job pauses it.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any

from sqlalchemy import and_, delete, nulls_last, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.enums import Ats, BoardOrigin, BoardStatus
from ..models.pipeline import IngestRun, Job, WatchedBoard
from ..schemas.config import BoardConfig, BoardSeed
from ..sources.boards import BoardSource
from .http import HttpClient
from .urls import find_ats_links, parse_ats

logger = getLogger(__name__)

WATCHABLE = {Ats.GREENHOUSE.value, Ats.LEVER.value, Ats.ASHBY.value}
BOARD_LINK = re.compile(
    r"https?://(?:(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io|jobs(?:\.eu)?\.lever\.co|jobs\.ashbyhq\.com)"
    r"/([A-Za-z0-9][\w.-]*)",
)
NOT_BOARDS = {"embed", "v1", "api", "jobs"}


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


def board_key(ats: str, token: str) -> tuple[str, str]:
    return ats.lower(), token.strip().lower()


# --- Adding boards ----------------------------------------------------------------------------------------------


async def _existing_keys(session: AsyncSession) -> dict[tuple[str, str], WatchedBoard]:
    boards = (await session.execute(select(WatchedBoard))).scalars().all()
    return {board_key(board.ats, board.token): board for board in boards}


async def add_boards(
    session: AsyncSession,
    candidates: dict[tuple[str, str], str],
    origin: BoardOrigin,
    origin_detail: str,
    configured: list[BoardConfig],
    reactivate_paused: bool = False,
) -> int:
    """Insert unseen boards. `candidates` maps (ats, token) -> company name. Returns how many were added.

    With `reactivate_paused`, boards paused for being idle become active again (a fresh lead proves they hire).
    Boards paused by hand and dead boards are never touched.
    """
    existing = await _existing_keys(session)
    configured_keys = {board_key(board.ats, board.token) for board in configured}
    added = 0
    inserted: set[tuple[str, str]] = set()
    for (ats, token), company in candidates.items():
        key = board_key(ats, token)
        if key in configured_keys or key in inserted or key[0] not in WATCHABLE or not key[1]:
            continue
        board = existing.get(key)
        if board is not None:
            if (
                reactivate_paused
                and board.status == BoardStatus.PAUSED
                and (board.status_reason or "").startswith("idle")
            ):
                board.status = BoardStatus.ACTIVE
                board.status_reason = "reactivated: new lead"
            continue
        session.add(
            WatchedBoard(
                ats=key[0],
                token=key[1],
                company=(company or token)[:256],
                origin=origin,
                origin_detail=origin_detail[:512],
                status=BoardStatus.ACTIVE,
            )
        )
        inserted.add(key)
        added += 1
    await session.commit()
    return added


async def discover_boards(session: AsyncSession, configured: list[BoardConfig]) -> int:
    """Watch the full board of every company whose jobs reached us through any source."""
    rows = (
        await session.execute(
            select(Job.ats, Job.board, Job.company).where(
                and_(Job.ats.in_([Ats.GREENHOUSE, Ats.LEVER, Ats.ASHBY]), Job.board.is_not(None))
            )
        )
    ).all()
    candidates = {board_key(ats.value, board): company for ats, board, company in rows if board}
    return await add_boards(session, candidates, BoardOrigin.DISCOVERED, "lead", configured, reactivate_paused=True)


def boards_from_simplify(listings: list[dict[str, Any]], categories: list[str]) -> dict[tuple[str, str], str]:
    wanted = set(categories)
    found: dict[tuple[str, str], str] = {}
    for item in listings:
        if wanted and item.get("category") not in wanted:
            continue
        ref = parse_ats(str(item.get("url") or ""))
        if ref is not None and ref.ats.value in WATCHABLE:
            found.setdefault(board_key(ref.ats.value, ref.board), str(item.get("company_name") or ref.board))
    return found


def boards_from_links(text: str) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for ref in find_ats_links(text).values():
        if ref.ats.value in WATCHABLE:
            found.setdefault(board_key(ref.ats.value, ref.board), ref.board)
    # Board pages without a posting id, e.g. https://jobs.ashbyhq.com/ramp
    for match in BOARD_LINK.finditer(text):
        host_token = match.group(1)
        if host_token.lower() in NOT_BOARDS:
            continue
        url = match.group(0).lower()
        ats = "greenhouse" if "greenhouse" in url else "lever" if "lever" in url else "ashby"
        found.setdefault(board_key(ats, host_token), host_token)
    return found


def boards_from_lines(text: str) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        ats, token = (part.strip() for part in line.split(":", 1))
        if ats.lower() in WATCHABLE and token:
            found.setdefault(board_key(ats, token), token)
    return found


@dataclass
class SeedResult:
    name: str
    found: int = 0
    added: int = 0
    error: str | None = None


async def seed_boards(
    session: AsyncSession, client: HttpClient, seeds: list[BoardSeed], configured: list[BoardConfig]
) -> list[SeedResult]:
    results: list[SeedResult] = []
    for seed in seeds:
        result = SeedResult(seed.name)
        try:
            if seed.format == "simplify":
                candidates = boards_from_simplify(await client.get_json(seed.url), seed.categories)
            else:
                _, text = await client.get_text(seed.url)
                candidates = boards_from_links(text) if seed.format == "links" else boards_from_lines(text)
        except Exception as exc:  # noqa: BLE001 - a broken seed URL must not stop the other seeds
            logger.warning("Seed %s failed: %s", seed.name, exc)
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            results.append(result)
            continue
        result.found = len(candidates)
        result.added = await add_boards(session, candidates, BoardOrigin.SEED, seed.name, configured)
        results.append(result)
    return results


# --- Polling ----------------------------------------------------------------------------------------------------


async def watched_sources(session: AsyncSession, configured: list[BoardConfig], limit: int) -> list[BoardSource]:
    """Active watched boards as sources, most recently productive first, capped at `limit`."""
    configured_keys = {board_key(board.ats, board.token) for board in configured}
    boards = (
        (
            await session.execute(
                select(WatchedBoard)
                .where(WatchedBoard.status == BoardStatus.ACTIVE)
                .order_by(nulls_last(WatchedBoard.last_lead_at.desc()), WatchedBoard.added_at)
            )
        )
        .scalars()
        .all()
    )
    sources: list[BoardSource] = []
    for board in boards:
        if board_key(board.ats, board.token) in configured_keys:
            continue
        if len(sources) >= limit:
            break
        config = BoardConfig.model_validate({"ats": board.ats, "token": board.token, "company": board.company})
        sources.append(BoardSource(config))
    return sources


@dataclass
class PollOutcome:
    source: str
    ok: bool
    not_found: bool
    items: int | None
    leads: int


async def record_polls(
    session: AsyncSession, outcomes: list[PollOutcome], idle_days: int, dead_after: int
) -> Counter[str]:
    """Update watched boards after an ingest run and apply the dead / idle rules. Returns transition counts."""
    by_name = {f"board:{b.ats}:{b.token}": b for b in (await session.execute(select(WatchedBoard))).scalars()}
    now = utcnow()
    changes: Counter[str] = Counter()
    for outcome in outcomes:
        board = by_name.get(outcome.source)
        if board is None:
            continue
        board.last_polled_at = now
        if outcome.ok:
            board.last_ok_at = now
            board.consecutive_failures = 0
            board.items_last = outcome.items
            if outcome.leads:
                board.last_lead_at = now
        else:
            board.consecutive_failures += 1
            if outcome.not_found and board.consecutive_failures >= dead_after:
                board.status = BoardStatus.DEAD
                board.status_reason = f"not found {board.consecutive_failures} times in a row"
                changes["dead"] += 1
                continue
        reference = _aware(board.last_lead_at) or _aware(board.added_at) or now
        if board.status == BoardStatus.ACTIVE and now - reference > timedelta(days=idle_days):
            board.status = BoardStatus.PAUSED
            board.status_reason = f"idle: no matching job in {idle_days} days"
            changes["paused"] += 1
    await session.commit()
    return changes


async def prune_ingest_runs(session: AsyncSession, retention_days: int) -> None:
    cutoff = utcnow() - timedelta(days=retention_days)
    await session.execute(delete(IngestRun).where(IngestRun.started_at < cutoff))
    await session.commit()


# --- Manual control and summary ---------------------------------------------------------------------------------


def parse_board_ref(value: str) -> tuple[str, str]:
    """`greenhouse:stripe` or a board/posting URL -> (ats, token)."""
    if "://" in value:
        ref = parse_ats(value)
        if ref is not None:
            return board_key(ref.ats.value, ref.board)
        found = boards_from_links(value)
        if len(found) == 1:
            return next(iter(found))
        raise ValueError(f"Not a Greenhouse, Lever or Ashby board link: {value}")
    if ":" not in value:
        raise ValueError("Use ats:token, e.g. greenhouse:stripe, or paste a board link.")
    ats, token = value.split(":", 1)
    if ats.lower() not in WATCHABLE:
        raise ValueError(f"Unsupported ATS {ats!r}; use greenhouse, lever or ashby.")
    return board_key(ats, token)


async def set_board_status(
    session: AsyncSession, ats: str, token: str, status: BoardStatus, reason: str
) -> WatchedBoard:
    board = (
        await session.execute(select(WatchedBoard).where(and_(WatchedBoard.ats == ats, WatchedBoard.token == token)))
    ).scalar_one_or_none()
    if board is None:
        raise LookupError(f"{ats}:{token} is not on the watchlist")
    board.status = status
    board.status_reason = reason
    board.consecutive_failures = 0
    await session.commit()
    return board


@dataclass
class BoardSummary:
    by_status: Counter[str] = field(default_factory=Counter)
    by_origin: Counter[str] = field(default_factory=Counter)
    by_ats: Counter[str] = field(default_factory=Counter)
    productive: list[WatchedBoard] = field(default_factory=list)
    never_polled: int = 0


async def board_summary(session: AsyncSession, top: int = 15) -> BoardSummary:
    boards = list((await session.execute(select(WatchedBoard))).scalars().all())
    summary = BoardSummary()
    for board in boards:
        summary.by_status[board.status.value] += 1
        summary.by_origin[board.origin.value] += 1
        if board.status == BoardStatus.ACTIVE:
            summary.by_ats[board.ats] += 1
        summary.never_polled += int(board.last_polled_at is None)
    with_leads = [b for b in boards if b.last_lead_at is not None]
    summary.productive = sorted(with_leads, key=lambda b: _aware(b.last_lead_at) or utcnow(), reverse=True)[:top]
    return summary
