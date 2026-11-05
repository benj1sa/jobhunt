"""Turning job-description HTML into clean, hashable plain text."""

import hashlib
import html
import re

from bs4 import BeautifulSoup

BLOCK_TAGS = ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "ul", "ol")


def html_to_text(markup: str) -> str:
    """Convert HTML (including entity-encoded HTML, as Greenhouse returns) into readable plain text.

    List items become "- " bullets and block elements become line breaks so section structure survives.
    """
    # Greenhouse double-encodes: "&lt;p&gt;" arrives as text. Unescape until stable (max 2 passes).
    for _ in range(2):
        if "&lt;" not in markup and "&amp;" not in markup:
            break
        markup = html.unescape(markup)
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for item in soup.find_all("li"):
        item.insert_before("\n- ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")
    text = soup.get_text()
    return clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Hash of whitespace-insensitive text so cosmetic re-renders don't create new snapshots."""
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(collapsed.encode()).hexdigest()
