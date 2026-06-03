# Composio Agent Integration Design

**Date:** 2026-06-03  
**Status:** Approved  
**Scope:** Replace Alpha Vantage MCP + Tavily with Composio as the single tool provider for the RAG chat agent. Migrate from manual decide-then-dispatch routing to native LLM tool-calling via `bind_tools`.

---

## 1. Goals

- Single tool provider: Composio replaces Alpha Vantage, Tavily, and all custom tool-dispatch code
- Native LLM tool-calling: `llm.bind_tools(tools)` replaces the custom routing/planning prompts
- Production-ready: per-request streaming status, full LangFuse/LangSmith observability
- Service account model: the app holds one Composio account; users do not connect their own

---

## 2. What Gets Deleted

### Files removed entirely
- `src/tools/alpha_vantage_mcp_client.py`
- `src/tools/alpha_vantage_mcp.py`
- `src/tools/asset_prices.py`
- `src/tools/asset_price_provider.py`
- `src/tools/web_search.py`
- `src/tools/search.py`
- `src/tools/fetcher.py`

### Prompts removed
- `src/prompts/web_search_decision.j2`
- `src/prompts/finance_tool_selection.j2`
- `src/prompts/finance_tool_call_plan.j2`
- `src/prompts/web_search_repair.j2`

### Functions removed from `src/api/endpoints.py`
- `_decide_chat_action`
- `_resolve_web_context`
- `_select_finance_tool_candidate`
- `_plan_finance_tool_call`
- `_tool_call_payload_to_context_row`
- `_tool_definition_to_context_row`
- `_tool_match_to_context_row`
- `_quote_asset_prices`
- `_search_web`

### Config fields removed from `src/config.py`
- `tavily_api_key`
- `max_search_results`
- `web_search_provider`
- `asset_price_provider`
- `alpha_vantage_api_key`
- `alpha_vantage_mcp_url`
- `alpha_vantage_mcp_tool_refresh_seconds`

---

## 3. What Gets Added

### `src/tools/composio_toolset.py` (new)

Thin wrapper around `ComposioToolSet` from `composio-langchain`. Responsibilities:
- Holds the process-wide singleton `ComposioToolSet(api_key=settings.composio_api_key)`
- Exposes `get_tools() -> list[BaseTool]` — returns the cached LangChain tool list
- Exposes `get_connected_app_names() -> list[str]` — used to inject the app list into the system prompt
- Initializes at startup; refreshes the tool catalog every `composio_tool_refresh_seconds` (default 3600)
- Raises `ComposioError` (new error type in `src/errors.py`) on misconfiguration; logs a warning and disables tools if Composio is unreachable at startup

### `src/errors.py` addition
```python
class ComposioError(CortexError):
    """Raised when a Composio toolset operation fails."""
```

### New config fields in `src/config.py`
- `composio_api_key: str` — env `COMPOSIO_API_KEY`, default `""`
- `composio_enabled: bool` — env `COMPOSIO_ENABLED`, default `True`
- `composio_apps: list[str]` — env `COMPOSIO_APPS` (comma-separated), default `[]` (empty = all connected apps)
- `composio_tool_refresh_seconds: int` — default `3600`
- `composio_max_agent_turns: int` — default `5`

### `_run_agent_loop()` in `src/api/endpoints.py`

Replaces the entire decide-then-dispatch block. Signature:
```python
async def _run_agent_loop(
    *,
    messages: list[BaseMessage],
    sse_queue: asyncio.Queue,
    metadata: dict[str, object],
) -> str
```

Logic:
1. Load tools from the Composio toolset singleton
2. `llm_with_tools = get_llm(temperature=0.0).bind_tools(tools)`
3. Loop up to `settings.composio_max_agent_turns` times:
   - Wrap LLM call in `observe_llm_generation` + `start_step_span`
   - `response = await llm_with_tools.ainvoke(messages)`
   - If no `tool_calls` → break
   - For each tool call:
     - Emit `tool_start` SSE event: `{"tool": name, "input_summary": truncated_args}`
     - Wrap execution in `start_step_span(run_type="tool")`
     - Execute via `tool.arun(args)` (the LangChain `BaseTool` handles Composio dispatch)
     - On success: emit `tool_end` SSE event `{"tool": name, "status": "ok"}`
     - On error: emit `tool_end` SSE event `{"tool": name, "status": "error"}`, append `ToolMessage` with error text so LLM can recover
   - Append `AIMessage` + `ToolMessage`s to messages
4. Return final response text

---

## 4. Prompts

### `src/prompts/rag_chat_system.j2` (rewritten)

The main agent system prompt. Sections:

**Tool-use policy** (always present):
> You have access to tools. Use them proactively when the user needs real-world data, current information, or actions in external apps. Do not claim you cannot access the web, fetch URLs, or retrieve data when you have tools available to do so.

**Connected apps block** (dynamic, injected when `composio_apps` is non-empty):
```jinja2
{% if composio_apps %}
You are connected to the following apps: {{ composio_apps | join(', ') }}.
Use these tools when the user's request involves any of these services.
{% endif %}
```

**RAG context block** (unchanged):
```jinja2
{% if rag_context %}
Retrieved context from the user's documents:
{{ rag_context }}
{% endif %}
```

**System instructions override** (unchanged — per-agent custom instructions):
```jinja2
{% if system_instructions %}
Additional instructions: {{ system_instructions }}
{% endif %}
```

**Behavioral guidelines**:
- Answer greetings and simple conversational messages directly without calling tools
- Prefer RAG context for questions about the user's uploaded documents
- Use tools for anything requiring fresh data, current prices, web content, or external app actions
- When tool results are available, ground your answer in them; do not fabricate data

### Prompts kept unchanged
- `followup_answer.j2` — follow-up chat on research reports; no tool-calling needed
- All research graph prompts (`summarize.j2`, `report.j2`, etc.)
- All planner/itinerary prompts

---

## 5. SSE Event Protocol

Two new event types added to the existing SSE stream. The frontend chat component handles them to show status indicators.

| Event | Payload | When |
|---|---|---|
| `tool_start` | `{"tool": "TOOL_NAME", "input_summary": "..."}` | Before each tool execution |
| `tool_end` | `{"tool": "TOOL_NAME", "status": "ok"\|"error"}` | After each tool execution |
| `token` | `{"text": "..."}` | Unchanged — final answer chunks |

`input_summary` is the first 120 characters of the serialized tool arguments, to give the user context without leaking large payloads.

---

## 6. Observability

No new infrastructure. Existing wrappers applied inside `_run_agent_loop`:

- Each LLM turn: `observe_llm_generation(step_name="agent_loop.turn_{n}", model=..., prompt=...)`
- Each tool call: `start_step_span(name="agent_loop.tool.{tool_name}", run_type="tool", tags=["external", "composio"])`

This produces a per-request trace tree in LangFuse/LangSmith: one span per LLM turn, one span per tool call, all nested under the parent request span.

---

## 7. Startup / Shutdown Lifecycle

**Startup** (`validate_session_store_configuration`):
```python
if settings.composio_enabled:
    try:
        await initialize_composio_toolset()
        logger.info("[startup] Composio toolset loaded with %d tools.", tool_count)
    except ComposioError as exc:
        logger.warning("[startup] Composio unavailable; tool-calling disabled: %s", exc)
```

**Shutdown** (`shutdown_background_clients`):
```python
await shutdown_composio_toolset()
```

If Composio fails at startup, `_run_agent_loop` falls back to running the LLM without tools (no crash).

---

## 8. Error Handling

| Scenario | Behaviour |
|---|---|
| Composio unreachable at startup | Log warning, `tools=[]`, agent answers without tools |
| Tool call raises an exception | Emit `tool_end status=error`, append error `ToolMessage`, LLM recovers in next turn |
| Max turns reached without final answer | Return last LLM response text as-is |
| Tool returns oversized payload | Truncate to 6000 chars before appending as `ToolMessage` |
| `composio_enabled=False` | Skip toolset init entirely; `_run_agent_loop` uses empty tool list |

---

## 9. Testing Strategy

**Unit tests** (new file `tests/test_tools_composio_toolset.py`):
- Toolset singleton init and refresh
- `get_tools()` returns `BaseTool` instances
- `get_connected_app_names()` reflects the catalog
- Startup failure produces warning and empty tool list

**Unit tests** (new file `tests/test_agent_loop.py`):
- Single-turn: no tool calls → returns LLM response directly
- Multi-turn: tool call → result appended → LLM produces final answer
- Tool error: exception → error `ToolMessage` → LLM recovers
- Max-turns guard: loop exits at `composio_max_agent_turns` regardless
- `tool_start` / `tool_end` SSE events emitted in correct order

**Regression**: `uv run pytest --tb=short` — all existing tests pass; research graph, planner, itinerary, RAG ingestion unaffected.

**Integration smoke test** (manual):
- Set `COMPOSIO_API_KEY`, start server, confirm catalog loaded in logs
- Send a chat message requiring web search → confirm `tool_start`/`tool_end` SSE events in browser DevTools → confirm grounded answer

---

## 10. Files Changed Summary

| File | Action |
|---|---|
| `src/tools/composio_toolset.py` | New |
| `src/tools/alpha_vantage_mcp_client.py` | Delete |
| `src/tools/alpha_vantage_mcp.py` | Delete |
| `src/tools/asset_prices.py` | Delete |
| `src/tools/asset_price_provider.py` | Delete |
| `src/tools/web_search.py` | Delete |
| `src/tools/search.py` | Delete |
| `src/tools/fetcher.py` | Delete |
| `src/errors.py` | Add `ComposioError` |
| `src/config.py` | Remove old fields, add Composio fields |
| `src/api/endpoints.py` | Remove dispatch functions, add `_run_agent_loop` |
| `src/prompts/rag_chat_system.j2` | Rewrite |
| `src/prompts/web_search_decision.j2` | Delete |
| `src/prompts/finance_tool_selection.j2` | Delete |
| `src/prompts/finance_tool_call_plan.j2` | Delete |
| `src/prompts/web_search_repair.j2` | Delete |
| `pyproject.toml` | Add `composio-langchain` |
| `tests/test_tools_composio_toolset.py` | New |
| `tests/test_agent_loop.py` | New |
