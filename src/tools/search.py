"""Tavily search tool with exponential-backoff retry."""

import time
import functools
import logging
from typing import Callable, TypeVar

from src.config import settings
from src.errors import SearchError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator that retries a function with exponential back-off on exception.

    Args:
        max_attempts: Total number of attempts before raising.
        base_delay: Initial sleep in seconds; doubled on each retry.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                            attempt, max_attempts, fn.__name__, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts, fn.__name__, exc,
                        )
            raise SearchError(f"'{fn.__name__}' failed after {max_attempts} attempts") from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


@with_retry(max_attempts=3, base_delay=1.0)
def perform_search(query: str, max_results: int | None = None) -> list[dict]:
    """Execute a Tavily web search and return structured results.

    Args:
        query: The search query string.
        max_results: Override for the configured max result count.

    Returns:
        List of dicts with ``url``, ``title``, and ``content`` keys.

    Raises:
        SearchError: If the search fails after all retries.
    """
    from tavily import TavilyClient

    if not settings.tavily_api_key:
        raise SearchError("TAVILY_API_KEY is not set.")

    client = TavilyClient(api_key=settings.tavily_api_key)
    n = max_results or settings.max_search_results

    response = client.search(query=query, max_results=n, include_raw_content=True)
    results = response.get("results", [])

    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "raw_content": r.get("raw_content") or "",
        }
        for r in results
    ]
