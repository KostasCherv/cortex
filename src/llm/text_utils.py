"""Utilities for extracting plain text from LLM response shapes."""

from __future__ import annotations


def extract_llm_text(response: object) -> str:
    """Extract plain text from provider-specific LLM response shapes."""
    content = response.content if hasattr(response, "content") else response  # type: ignore[union-attr]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()
