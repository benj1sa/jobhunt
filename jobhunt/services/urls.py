"""URL normalization and applicant-tracking-system detection.

This is the heart of deduplication: two links are the same job if they normalize to the same ATS reference.
"""

import re
from dataclasses import dataclass

from yarl import URL

from ..models.enums import Ats
from ..schemas.config import BoardConfig

# Query parameters that only carry tracking and never identify a posting.
TRACKING_PARAMS = re.compile(
    r"^(utm_.*|gclid|fbclid|mc_[ce]id|ref|refid|source|src|trk|trackingid|lipi|_ga)$", re.IGNORECASE
)

# Boards that block automated access or forbid it in their terms. Leads here need a human-supplied ATS link.
BLOCKED_HOSTS = ("linkedin.com", "wellfound.com", "angel.co", "indeed.com", "glassdoor.com", "lnkd.in")

# Aggregators whose pages are fetchable and usually link to the real ATS posting.
AGGREGATOR_HOSTS = ("builtin.com", "talents.vaia.com", "idealist.org", "simplify.jobs", "ycombinator.com")

UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
GREENHOUSE_PATH = re.compile(r"^/(?P<board>[\w.-]+)/jobs/(?P<id>\d+)")
LEVER_PATH = re.compile(rf"^/(?P<board>[\w.-]+)/(?P<id>{UUID_RE})")
ASHBY_PATH = re.compile(rf"^/(?P<board>[^/]+)/(?P<id>{UUID_RE})")
WORKDAY_HOST = re.compile(r"^(?P<tenant>[\w-]+)\.(?P<wd>wd\d+)\.myworkdayjobs\.com$")
WORKDAY_PATH = re.compile(r"^/(?:[a-zA-Z]{2}-[a-zA-Z]{2}/)?(?P<site>[^/]+)/job/(?P<path>.+)$")


@dataclass(frozen=True)
class AtsRef:
    """Identifies one posting on one ATS board."""

    ats: Ats
    board: str
    external_id: str

    @property
    def canonical_key(self) -> str:
        return f"{self.ats.value}:{self.board.lower()}:{self.external_id}"


def normalize_url(raw: str) -> str:
    """Normalize a link for storage and comparison.

    Lowercases scheme and host, drops `www.`, default ports, fragments, trailing slashes and tracking parameters,
    and sorts the remaining query so parameter order never creates a duplicate.
    """
    url = URL(raw.strip())
    if not url.is_absolute():
        url = URL(f"https://{raw.strip()}")
    host = (url.host or "").lower().removeprefix("www.")
    query = sorted((key, value) for key, value in url.query.items() if not TRACKING_PARAMS.match(key))
    path = url.path.rstrip("/") or "/"
    port = url.port if url.port not in (80, 443) else None
    built = URL.build(scheme=url.scheme.lower() or "https", host=host, port=port, path=path, encoded=False)
    return str(built.with_query(query) if query else built)


def host_matches(host: str, domains: tuple[str, ...] | list[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def is_blocked(url: str) -> bool:
    return host_matches(URL(url).host or "", BLOCKED_HOSTS)


def is_aggregator(url: str) -> bool:
    return host_matches(URL(url).host or "", AGGREGATOR_HOSTS)


def parse_ats(raw_url: str, boards: list[BoardConfig] | None = None) -> AtsRef | None:
    """Detect which ATS posting a link points at, or None if it isn't a recognizable ATS link."""
    url = URL(raw_url.strip())
    host = (url.host or "").lower()
    path = url.path

    if host.endswith("greenhouse.io"):
        if (match := GREENHOUSE_PATH.match(path)) and match["board"] != "embed":
            return AtsRef(Ats.GREENHOUSE, match["board"], match["id"])
        # Embedded application form: boards.greenhouse.io/embed/job_app?for=<board>&token=<id>
        if "for" in url.query and url.query.get("token", "").isdigit():
            return AtsRef(Ats.GREENHOUSE, url.query["for"], url.query["token"])
        if (gh_board := url.query.get("for")) and (gh_job := url.query.get("gh_jid")):
            return AtsRef(Ats.GREENHOUSE, gh_board, gh_job)
        return None

    if host in ("jobs.lever.co", "jobs.eu.lever.co"):
        if match := LEVER_PATH.match(path):
            return AtsRef(Ats.LEVER, match["board"], match["id"].lower())
        return None

    if host == "jobs.ashbyhq.com":
        if match := ASHBY_PATH.match(path):
            return AtsRef(Ats.ASHBY, match["board"], match["id"].lower())
        return None

    if wd_host := WORKDAY_HOST.match(host):
        if wd_path := WORKDAY_PATH.match(path):
            wd_board = f"{wd_host['tenant']}|{wd_host['wd']}|{wd_path['site']}"
            return AtsRef(Ats.WORKDAY, wd_board, wd_path["path"].rstrip("/"))
        return None

    # Company career sites that embed Greenhouse: https://careers.example.com/jobs?gh_jid=123
    if (job_id := url.query.get("gh_jid")) and job_id.isdigit() and boards:
        site_host = host.removeprefix("www.")
        for board in boards:
            if board.ats == "greenhouse" and host_matches(site_host, board.domains):
                return AtsRef(Ats.GREENHOUSE, board.token, job_id)

    return None


ATS_LINK_RE = re.compile(
    r"https?://(?:"
    r"(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/[^\s\"'<>]+"
    r"|jobs(?:\.eu)?\.lever\.co/[^\s\"'<>]+"
    r"|jobs\.ashbyhq\.com/[^\s\"'<>]+"
    r"|[\w-]+\.wd\d+\.myworkdayjobs\.com/[^\s\"'<>]+"
    r")",
    re.IGNORECASE,
)


def find_ats_links(text: str, boards: list[BoardConfig] | None = None) -> dict[str, AtsRef]:
    """Find every recognizable ATS posting link in free text or HTML, keyed by canonical key."""
    found: dict[str, AtsRef] = {}
    for match in ATS_LINK_RE.finditer(text):
        candidate = match.group(0).rstrip(".,);]").replace("&amp;", "&")
        if ref := parse_ats(candidate, boards):
            found.setdefault(ref.canonical_key, ref)
    return found


def posting_url_for(ref: AtsRef) -> str:
    """The human-facing posting URL for an ATS reference."""
    match ref.ats:
        case Ats.GREENHOUSE:
            return f"https://job-boards.greenhouse.io/{ref.board}/jobs/{ref.external_id}"
        case Ats.LEVER:
            return f"https://jobs.lever.co/{ref.board}/{ref.external_id}"
        case Ats.ASHBY:
            return f"https://jobs.ashbyhq.com/{ref.board}/{ref.external_id}"
        case Ats.WORKDAY:
            tenant, wd, site = ref.board.split("|")
            return f"https://{tenant}.{wd}.myworkdayjobs.com/{site}/job/{ref.external_id}"
        case _:
            return ref.external_id
