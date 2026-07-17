# scripts/finetune/router_prompt.py
"""Router system prompt and message formatter for training data generation."""

from __future__ import annotations

ROUTER_SYSTEM_PROMPT = """You are a chat routing assistant. Given a user message and optional context, \
output ONLY a JSON object with no markdown fences or explanation.

JSON schema:
{
  "action": "answer_direct" | "answer_from_rag" | "web_search" | "asset_price" | "search_finance_tools" | "ask_clarifying",
  "reason": "one sentence explaining the routing decision",
  "query": "search query — required for web_search and search_finance_tools, empty string otherwise",
  "symbols": ["TICKER"] — required list for asset_price, empty list otherwise,
  "currency": "ISO currency code if relevant, empty string otherwise"
}

Routing rules:
- answer_direct: question answerable from general knowledge, no live data or documents needed
- answer_from_rag: question about uploaded documents or the knowledge base; select this even when no context has been retrieved yet, because retrieval happens after routing
- web_search: needs current or live information (news, recent events, live prices)
- asset_price: user wants current price or quote for a specific stock or crypto (symbols required)
- search_finance_tools: user needs financial ratios, statements, or structured data via a tool (query required)
- ask_clarifying: message is too ambiguous, incomplete, or multi-intent to route confidently"""


def format_user_turn(
    *,
    message: str,
    rag_context: str = "",
    history: list[dict[str, str]] | None = None,
) -> str:
    """Format a user turn for a router training record."""
    parts: list[str] = [f"User message: {message.strip()}"]

    if rag_context.strip():
        parts.append(f"Pre-retrieved RAG context: yes — {rag_context.strip()}")
    else:
        parts.append(
            "Pre-retrieved RAG context: none. Resource availability: unknown; "
            "context is retrieved after routing. If the message asks about uploaded "
            "documents or the knowledge base, choose answer_from_rag."
        )

    if history:
        last = history[-1]
        snippet = str(last.get("content", ""))[:200]
        parts.append(f"Last exchange — {last.get('role', 'user')}: {snippet}")
    else:
        parts.append("Conversation history: none")

    return "\n".join(parts)


def build_training_record(
    *,
    message: str,
    rag_context: str = "",
    history: list[dict[str, str]] | None = None,
    assistant_json: str,
) -> dict[str, list[dict[str, str]]]:
    """Build a messages-format training record."""
    return {
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_user_turn(
                    message=message, rag_context=rag_context, history=history
                ),
            },
            {"role": "assistant", "content": assistant_json},
        ]
    }
