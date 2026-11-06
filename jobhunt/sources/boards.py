"""Discovery by polling company ATS boards listed in `config/boards.toml`."""

from logging import getLogger

from ..schemas.config import BoardConfig
from ..services.ats import list_board
from ..services.filtering import location_passes, title_passes
from ..services.http import HttpClient
from .base import RawLead, Source

logger = getLogger(__name__)


class BoardSource(Source):
    """One company board. Each board is its own source so `jobhunt health` shows a dead slug immediately."""

    def __init__(self, board: BoardConfig) -> None:
        self.board = board
        self.name = f"board:{board.ats}:{board.token}"

    async def fetch(self, client: HttpClient) -> list[RawLead]:
        postings = await list_board(client, self.board.ats, self.board.token)
        self.items_fetched = len(postings)
        leads: list[RawLead] = []
        for posting in postings:
            if not (title_passes(posting.title) and location_passes(posting.location)):
                continue
            leads.append(
                RawLead(
                    source=self.name,
                    company=self.board.company,
                    title=posting.title,
                    url=posting.posting_url,
                    location=posting.location,
                    posted_at=posting.posted_at,
                    ats_ref=posting.ref,
                    description_text=posting.description_text or None,
                    raw=posting.raw,
                    extra={"is_remote": posting.is_remote, "apply_url": posting.apply_url},
                )
            )
        logger.info("Board %s: %d postings, %d passed filters", self.name, len(postings), len(leads))
        return leads
