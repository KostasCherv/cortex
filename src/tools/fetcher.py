"""Async URL content fetcher with HTML cleaning."""

import ipaddress
import logging
import socket
import urllib.parse

import httpx
from bs4 import BeautifulSoup

from src.errors import FetchError

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0  # seconds
_MAX_CHARS = 8_000  # truncate very long pages

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


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


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"Blocked URL: scheme '{parsed.scheme}' not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise FetchError("Blocked URL: missing hostname")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise FetchError(f"DNS resolution failed for {hostname}: {exc}") from exc
    for info in infos:
        ip_str = info[4][0]
        # Strip IPv6 zone ID if present (e.g. "fe80::1%eth0")
        ip_str = ip_str.split("%")[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if any(ip in net for net in _BLOCKED_NETWORKS):
            raise FetchError(f"Blocked URL: {hostname} resolves to private/reserved address {ip}")


async def fetch_url_content(url: str) -> str:
    """Fetch and clean the text content of a URL.

    Args:
        url: The URL to retrieve.

    Returns:
        Plain text content of the page (truncated to _MAX_CHARS).

    Raises:
        FetchError: On HTTP error or network failure.
    """
    _validate_url(url)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:  # intentional: _validate_url already blocks private IPs
            response = await client.get(url, headers={"User-Agent": "Cortex/0.1"})
            response.raise_for_status()
            return clean_html(response.text)
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except Exception as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
