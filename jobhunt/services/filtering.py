"""Cheap rule-based filters applied before any LLM call.

Titles and locations are screened at discovery time so large boards and the Simplify feed only produce
plausible leads. The final relevance decision (seniority, years, US) is made after extraction.
"""

import re
from dataclasses import dataclass

from ..schemas.extraction import JobRequirements, RoleFamily, Seniority

ROLE_WORDS = re.compile(r"\b(engineer(ing)?|developer|programmer|swe|scientist|mlops|sde)\b", re.IGNORECASE)

TECH_WORDS = re.compile(
    r"(software|\bswe\b|\bsde\b|full[- ]?stack|back[- ]?end|front[- ]?end|\bml\b|machine learning|\bai\b|a\.i\.|"
    r"artificial intelligence|mlops|\bllm|\bdata\b|platform|infrastructure|devops|applied|research engineer|"
    r"developer|new grad|university grad|early career|graduate|forward deployed|product engineer|"
    r"computer vision|\bnlp\b|deep learning|cloud|web|mobile|ios|android|systems engineer|embedded software)",
    re.IGNORECASE,
)

SENIOR_WORDS = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|manager|director|head|vp|vice president|architect|distinguished|"
    r"fellow|iii|iv|v|intern|internship|co-?op|contract(or)?|part[- ]time|phd)\b",
    re.IGNORECASE,
)

NON_SOFTWARE_ENGINEER = re.compile(
    r"\b(sales|solutions?|support|field|mechanical|electrical|civil|hardware|manufacturing|quality|process|"
    r"chemical|structural|network|test|validation|facilities|customer)\s+engineer",
    re.IGNORECASE,
)

# State codes are matched case-sensitively so words like "in" or "or" never count as Indiana or Oregon.
US_STATE_CODES = re.compile(
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|"
    r"ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|US|USA)\b"
)

US_WORDS = re.compile(
    r"\b(united states|u\.s\.|new york|san francisco|seattle|boston|austin|chicago|los angeles|"
    r"washington|denver|atlanta|philadelphia|maryland|virginia|california|texas|massachusetts)\b",
    re.IGNORECASE,
)

NON_US_MARKERS = re.compile(
    r"\b(canada|toronto|vancouver|montreal|united kingdom|uk|london|ireland|dublin|germany|berlin|munich|"
    r"france|paris|netherlands|amsterdam|spain|madrid|poland|india|bangalore|bengaluru|hyderabad|singapore|"
    r"japan|tokyo|australia|sydney|israel|tel aviv|mexico|brazil|emea|apac|latam|europe)\b",
    re.IGNORECASE,
)


def title_passes(title: str) -> bool:
    """True if a title looks like an entry-level software, AI or ML role."""
    if not ROLE_WORDS.search(title) and not re.search(r"\bnew grad|university grad|early career", title, re.IGNORECASE):
        return False
    if SENIOR_WORDS.search(title) or NON_SOFTWARE_ENGINEER.search(title):
        return False
    return bool(TECH_WORDS.search(title))


def location_passes(location: str | None) -> bool:
    """True if the location is in the US, remote, or unknown. Case-sensitive state codes avoid 'in'/'or' hits."""
    if not location or not location.strip():
        return True
    if US_STATE_CODES.search(location) or US_WORDS.search(location):
        return True
    return not NON_US_MARKERS.search(location)


@dataclass(frozen=True)
class Relevance:
    relevant: bool
    reason: str


def judge_relevance(requirements: JobRequirements) -> Relevance:
    """Final, deterministic relevance decision from the extracted requirements.

    Depends only on the job, never on the profile, so editing the profile never invalidates extractions.
    Candidate eligibility (citizenship, sponsorship, clearance) is applied as a likelihood gate in scoring.
    """
    problems: list[str] = []
    if requirements.role_family == RoleFamily.OTHER:
        problems.append("not a software/AI/ML role")
    if requirements.seniority in (Seniority.MID, Seniority.SENIOR, Seniority.INTERN):
        problems.append(f"seniority is {requirements.seniority.value}")
    if requirements.min_years_experience is not None and requirements.min_years_experience > 2:
        problems.append(f"requires {requirements.min_years_experience}+ years")
    if not requirements.us_based:
        problems.append("not US-based")
    if problems:
        return Relevance(relevant=False, reason="; ".join(problems))
    return Relevance(relevant=True, reason=f"{requirements.seniority.value} {requirements.role_family.value} role")
