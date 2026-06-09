"""Free reference lookup tools backed by public HTTP APIs."""

from __future__ import annotations

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

_HTTP_USER_AGENT = "ResearchAgent/1.0 (rag-chat)"

_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_OPEN_LIBRARY_API = "https://openlibrary.org/search.json"

_WIKIPEDIA_MAX_QUERY_LENGTH = 300
_WIKIPEDIA_TOP_K = 5
_WIKIPEDIA_EXTRACT_MAX_CHARS = 1000
_OPEN_LIBRARY_MAX_QUERY_LENGTH = 300
_OPEN_LIBRARY_TOP_K = 5


class _WikipediaInput(BaseModel):
    query: str = Field(description="Topic to look up on Wikipedia")


class _OpenLibraryInput(BaseModel):
    query: str = Field(description="Book title, author, or subject to search in Open Library")


def _http_headers() -> dict[str, str]:
    return {"User-Agent": _HTTP_USER_AGENT}


async def wikipedia_lookup(query: str) -> str:
    normalized_query = query.strip()[:_WIKIPEDIA_MAX_QUERY_LENGTH]
    if not normalized_query:
        return "No Wikipedia query provided."

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": normalized_query,
        "gsrlimit": _WIKIPEDIA_TOP_K,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(_WIKIPEDIA_API, params=params, headers=_http_headers())
            response.raise_for_status()
            if not response.text.strip():
                return "Wikipedia returned an empty response. Try again or use web search."
            try:
                data = response.json()
            except ValueError:
                return "Wikipedia returned an unexpected response. Try again or use web search."
    except httpx.HTTPError as exc:
        return f"Wikipedia lookup failed: {exc}"

    summaries: list[str] = []
    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("missing") is not None:
            continue
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        summaries.append(
            f"Page: {title}\nSummary: {extract[:_WIKIPEDIA_EXTRACT_MAX_CHARS]}"
        )

    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


async def open_library_lookup(query: str) -> str:
    normalized_query = query.strip()[:_OPEN_LIBRARY_MAX_QUERY_LENGTH]
    if not normalized_query:
        return "No Open Library query provided."

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                _OPEN_LIBRARY_API,
                params={"q": normalized_query, "limit": _OPEN_LIBRARY_TOP_K},
                headers=_http_headers(),
            )
            response.raise_for_status()
            if not response.text.strip():
                return "Open Library returned an empty response. Try again or use web search."
            try:
                data = response.json()
            except ValueError:
                return "Open Library returned an unexpected response. Try again or use web search."
    except httpx.HTTPError as exc:
        return f"Open Library lookup failed: {exc}"

    docs = data.get("docs") or []
    if not docs:
        return "No Open Library books found for that query."

    summaries: list[str] = []
    for doc in docs:
        title = (doc.get("title") or "").strip()
        if not title:
            continue
        authors = doc.get("author_name") or []
        year = doc.get("first_publish_year")
        key = doc.get("key") or ""
        url = f"https://openlibrary.org{key}" if key else ""
        lines = [f"Title: {title}"]
        if authors:
            lines.append(f"Authors: {', '.join(str(name) for name in authors)}")
        if year:
            lines.append(f"First published: {year}")
        if url:
            lines.append(f"URL: {url}")
        summaries.append("\n".join(lines))

    if not summaries:
        return "No Open Library books found for that query."
    return "\n\n".join(summaries)


async def wikipedia_lookup_with_artifact(query: str) -> tuple[str, dict[str, list[dict[str, str]]]]:
    text = await wikipedia_lookup(query)
    if text.startswith("No ") or "failed:" in text or "unexpected response" in text or "empty response" in text:
        return text, {"results": []}

    parts = []
    for block in text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        title = lines[0].removeprefix("Page: ").strip()
        extract = lines[1].removeprefix("Summary: ").strip()
        parts.append({"title": title, "extract": extract})
    return text, {"results": parts}


async def open_library_lookup_with_artifact(query: str) -> tuple[str, dict[str, list[dict[str, str]]]]:
    text = await open_library_lookup(query)
    if text.startswith("No ") or "failed:" in text or "unexpected response" in text or "empty response" in text:
        return text, {"results": []}

    parts: list[dict[str, str]] = []
    for block in text.split("\n\n"):
        item: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("Title: "):
                item["title"] = line.removeprefix("Title: ").strip()
            elif line.startswith("Authors: "):
                item["authors"] = line.removeprefix("Authors: ").strip()
            elif line.startswith("First published: "):
                item["year"] = line.removeprefix("First published: ").strip()
            elif line.startswith("URL: "):
                item["url"] = line.removeprefix("URL: ").strip()
        if item.get("title"):
            parts.append(item)
    return text, {"results": parts}


def make_wikipedia_tool() -> BaseTool:
    return StructuredTool.from_function(
        coroutine=wikipedia_lookup_with_artifact,
        name="wikipedia",
        description=(
            "Look up encyclopedic information on Wikipedia. "
            "Use for people, places, concepts, companies, and historical topics."
        ),
        args_schema=_WikipediaInput,
        response_format="content_and_artifact",
    )


def make_open_library_tool() -> BaseTool:
    return StructuredTool.from_function(
        coroutine=open_library_lookup_with_artifact,
        name="open_library",
        description=(
            "Search Open Library for book metadata. "
            "Use for titles, authors, publication years, and bibliographic details."
        ),
        args_schema=_OpenLibraryInput,
        response_format="content_and_artifact",
    )
