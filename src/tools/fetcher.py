"""Async URL content fetcher with HTML cleaning."""

import logging

import httpx
from bs4 import BeautifulSoup

from src.errors import FetchError

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0  # seconds
_MAX_CHARS = 8_000  # truncate very long pages


def clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace, returning plain text.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with normalised whitespace, truncated to _MAX_CHARS.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse multiple spaces / newlines
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_CHARS]


async def fetch_url_content(url: str) -> str:
    """Fetch and clean the text content of a URL.

    Args:
        url: The URL to retrieve.

    Returns:
        Plain text content of the page (truncated to _MAX_CHARS).

    Raises:
        FetchError: On HTTP error or network failure.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
            response = await client.get(url, headers={"User-Agent": "Cortex/0.1"})
            response.raise_for_status()
            return clean_html(response.text)
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except Exception as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
