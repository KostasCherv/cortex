# Chat Tool Toggles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `+` button to the chat input that opens a tool-selection popover, giving users explicit control over web search (on by default) and Composio integrations (off by default) per chat session.

**Architecture:** A `ToolConfig` record flows from UI state → transport → API request body. The backend replaces keyword-based auto-detection with explicit user flags. Web search is wired as a native LangChain `StructuredTool` in `_run_agent_loop`. Composio binding is gated by the `tools.composio` flag. A new `web_used` SSE event notifies the UI when web search fires.

**Tech Stack:** React + TypeScript (shadcn/ui `Popover`, `Switch`), FastAPI (Pydantic), LangChain Core (`StructuredTool`), Tavily (existing `perform_search_cached`)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/api/endpoints.py` | Modify | Add `RagChatTools`, extend `RagChatRequest`, update `_run_agent_loop` signature + web search tool, update all callers |
| `src/api/rag_chat_helpers.py` | Modify | Add `allow_web_search` to `RagChatPrepared`; update both `prepare_*` functions to accept + apply `tools` flags |
| `ui/src/components/chat/toolConfig.ts` | Create | Tool registry (`TOOL_DEFINITIONS`, `ToolConfig`, `defaultToolConfig`) |
| `ui/src/components/chat/ToolMenuButton.tsx` | Create | `+` button + Popover list of tool toggles |
| `ui/src/components/chat/transports.ts` | Modify | Add `tools?: ToolConfig` to `streamMessage`; add `onWebUsed?` to `StreamCallbacks` |
| `ui/src/components/chat/ChatThreadContainer.tsx` | Modify | Add `toolConfig` state; render `ToolMenuButton`; pass tools to transport; handle `onWebUsed` |
| `ui/src/api/client.ts` | Modify | Add `tools?` param + `onWebUsed?` to both stream functions; include in request body; handle `web_used` SSE event |
| `ui/src/types.ts` | Modify | Add `{ type: 'web_used'; provider: string }` to `RagChatStreamEvent` |

---

## Task 1: Backend — `RagChatTools` model + extend `RagChatRequest`

**Files:**
- Modify: `src/api/endpoints.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing test for new request field**

```python
# In tests/test_api.py, add to the existing test class or as a standalone test:
def test_rag_chat_request_default_tools():
    from src.api.endpoints import RagChatRequest, RagChatTools
    req = RagChatRequest(message="hello")
    assert req.tools.web_search is True
    assert req.tools.composio is False

def test_rag_chat_tools_explicit():
    from src.api.endpoints import RagChatRequest
    req = RagChatRequest(message="hello", tools={"web_search": False, "composio": True})
    assert req.tools.web_search is False
    assert req.tools.composio is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_api.py::test_rag_chat_request_default_tools tests/test_api.py::test_rag_chat_tools_explicit -v
```
Expected: `ImportError` or `AttributeError` — `RagChatTools` not defined yet.

- [ ] **Step 3: Add `RagChatTools` and update `RagChatRequest` in `endpoints.py`**

Find the existing `RagChatRequest` class (around line 310) and replace it:

```python
class RagChatTools(BaseModel):
    web_search: bool = True
    composio: bool = False


class RagChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    tools: RagChatTools = Field(default_factory=RagChatTools)
```

`Field` is already imported from pydantic at the top of the file — confirm with `grep "from pydantic" src/api/endpoints.py`. If `Field` is not imported, add it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_api.py::test_rag_chat_request_default_tools tests/test_api.py::test_rag_chat_tools_explicit -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/endpoints.py tests/test_api.py
git commit -m "feat(rag_chat): add RagChatTools model with web_search and composio flags"
```

---

## Task 2: Backend — Web search as LangChain `StructuredTool` in `_run_agent_loop`

**Files:**
- Modify: `src/api/endpoints.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

```python
def test_run_agent_loop_returns_tuple():
    """_run_agent_loop must return (answer: str, web_used: bool)."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from langchain_core.messages import HumanMessage

    mock_response = MagicMock()
    mock_response.content = "test answer"
    mock_response.tool_calls = []

    with patch("src.api.endpoints.get_llm") as mock_get_llm, \
         patch("src.api.endpoints.settings") as mock_settings:
        mock_settings.composio_max_agent_turns = 1
        mock_settings.composio_enabled = False
        mock_settings.tavily_api_key = None  # disable web search
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = llm

        from src.api.endpoints import _run_agent_loop
        result = asyncio.run(_run_agent_loop(
            messages=[HumanMessage(content="hi")],
            metadata={},
            bind_tools=False,
            allow_web_search=False,
        ))
        assert isinstance(result, tuple)
        assert len(result) == 2
        answer, web_used = result
        assert isinstance(answer, str)
        assert web_used is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_api.py::test_run_agent_loop_returns_tuple -v
```
Expected: FAIL — `_run_agent_loop` currently returns `str`, not `tuple`.

- [ ] **Step 3: Build the web search `StructuredTool` factory**

Add this helper near `_run_agent_loop` in `endpoints.py` (after the existing imports, before the function). The `perform_search_cached` import goes at module level alongside the other tool imports:

```python
# Add to the imports block at the top of endpoints.py, near other tool imports:
from langchain_core.tools import StructuredTool
from pydantic import BaseModel as _PydanticBase

from src.tools.search import perform_search_cached
```

Then add the factory function just before `_run_agent_loop`:

```python
class _WebSearchInput(_PydanticBase):
    query: str


def _make_web_search_tool(web_used_flag: list[bool]) -> StructuredTool:
    """Return a LangChain StructuredTool that sets web_used_flag[0] = True on first call."""

    async def _search(query: str) -> str:
        web_used_flag[0] = True
        results = await perform_search_cached(query, max_results=5)
        lines = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = (r.get("content") or "")[:400]
            lines.append(f"[{title}]({url})\n{content}")
        return "\n\n".join(lines) if lines else "No results found."

    return StructuredTool.from_function(
        coroutine=_search,
        name="web_search",
        description="Search the web for up-to-date information. Use when the answer requires current data.",
        args_schema=_WebSearchInput,
    )
```

- [ ] **Step 4: Update `_run_agent_loop` to accept `allow_web_search` and return `tuple[str, bool]`**

Replace the entire `_run_agent_loop` function signature and body:

```python
async def _run_agent_loop(
    *,
    messages: list[BaseMessage],
    metadata: dict[str, object],
    on_event=None,
    bind_tools: bool = True,
    allow_web_search: bool = True,
) -> tuple[str, bool]:
    """Run an agentic tool-calling loop and return (answer, web_used).

    bind_tools: when False, skip Composio router session and tool schema binding.
    allow_web_search: when True, bind the Tavily web search tool to the LLM.
    """
    llm = get_llm(temperature=0.0)
    max_turns = settings.composio_max_agent_turns
    loop_messages = list(messages)
    last_response_text = ""
    web_used_flag: list[bool] = [False]

    web_tools: list = []
    if allow_web_search and settings.tavily_api_key:
        web_tools = [_make_web_search_tool(web_used_flag)]

    async def _invoke_turn(llm_target: object, turn: int) -> object:
        with start_step_span(
            name=f"agent_loop.turn_{turn}",
            run_type="llm",
            node_name="agent_loop",
            inputs={"turn": turn, "bind_tools": bind_tools},
            metadata=metadata,
            tags=["llm", "agent_loop"],
        ):
            return await llm_target.ainvoke(loop_messages)  # type: ignore[union-attr]

    if not bind_tools or not settings.composio_enabled:
        base_llm = llm.bind_tools(web_tools) if web_tools else llm
        for turn in range(max_turns):
            response = await _invoke_turn(base_llm, turn)
            last_response_text = _extract_llm_text(
                response.content if hasattr(response, "content") else response
            )
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break
            loop_messages.append(response)
            web_tool_map = {t.name: t for t in web_tools}
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_id = tc.get("id", tool_name)
                tool_args = tc.get("args", {})
                matched = web_tool_map.get(tool_name)
                result_text = ""
                if matched:
                    try:
                        raw = await matched.arun(tool_args)
                        result_text = str(raw)[:6000]
                    except Exception as exc:
                        result_text = f"Error: {exc}"
                else:
                    result_text = f"Tool '{tool_name}' not available."
                loop_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tool_id)
                )
        return last_response_text, web_used_flag[0]

    manager = get_composio_toolset_manager()
    user_id = settings.composio_user_id

    async with manager.router_tools_context(user_id) as composio_tools:
        all_tools = list(composio_tools) + web_tools
        llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm
        tool_map = {t.name: t for t in all_tools}

        for turn in range(max_turns):
            response = await _invoke_turn(llm_with_tools, turn)
            last_response_text = _extract_llm_text(
                response.content if hasattr(response, "content") else response
            )
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            loop_messages.append(response)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_id = tc.get("id", tool_name)
                tool_args = tc.get("args", {})
                input_summary = str(tool_args)[:120]

                if on_event:
                    await on_event({"type": "tool_start", "tool": tool_name, "input_summary": input_summary})

                tool_result = ""
                tool_status = "ok"
                try:
                    with start_step_span(
                        name=f"agent_loop.tool.{tool_name}",
                        run_type="tool",
                        node_name="agent_loop",
                        inputs={"tool": tool_name, "args": tool_args},
                        metadata=metadata,
                        tags=["external", "composio"],
                    ):
                        matched_tool = tool_map.get(tool_name)
                        if matched_tool is None:
                            raise ValueError(f"Tool '{tool_name}' not found in catalog.")
                        raw_result = await matched_tool.arun(tool_args)
                        tool_result = str(raw_result)[:6000]
                except Exception as exc:
                    tool_result = f"Error executing {tool_name}: {exc}"
                    tool_status = "error"

                if on_event:
                    await on_event({"type": "tool_end", "tool": tool_name, "status": tool_status})

                loop_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_id)
                )

    return last_response_text, web_used_flag[0]
```

- [ ] **Step 5: Update all `_run_agent_loop` callers to unpack the tuple**

There are 4 call sites. Update each one:

**Non-streaming agent chat** (~line 1852):
```python
answer, _ = await _run_agent_loop(
    messages=prepared.messages,
    metadata={"agent_id": agent_id, "user_id": current_user.user_id},
    bind_tools=prepared.bind_tools,
    allow_web_search=prepared.allow_web_search,
)
```

**Streaming agent chat** (~line 1982):
```python
loop_task = asyncio.create_task(
    _run_agent_loop(
        messages=prepared.messages,
        metadata={"agent_id": agent_id, "user_id": current_user.user_id},
        on_event=on_event,
        bind_tools=prepared.bind_tools,
        allow_web_search=prepared.allow_web_search,
    )
)
# ...after loop_task completes:
answer, web_used = loop_task.result()
```

**Non-streaming workspace chat** (~line 2196):
```python
answer, _ = await _run_agent_loop(
    messages=prepared.messages,
    metadata={"user_id": current_user.user_id},
    bind_tools=prepared.bind_tools,
    allow_web_search=prepared.allow_web_search,
)
```

**Streaming workspace chat** (~line 2317):
```python
loop_task = asyncio.create_task(
    _run_agent_loop(
        messages=prepared.messages,
        metadata={"user_id": current_user.user_id},
        on_event=on_event,
        bind_tools=prepared.bind_tools,
        allow_web_search=prepared.allow_web_search,
    )
)
# ...after loop_task completes:
answer, web_used = loop_task.result()
```

For the **streaming endpoints**, also emit the `web_used` SSE event after the loop result is known. Add this right after `answer, web_used = loop_task.result()`:

```python
answer, web_used = loop_task.result()
if web_used:
    yield f"data: {json.dumps({'type': 'web_used', 'provider': settings.web_search_provider})}\n\n"
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_api.py::test_run_agent_loop_returns_tuple -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/api/endpoints.py tests/test_api.py
git commit -m "feat(rag_chat): add web search tool to agent loop with web_used tracking"
```

---

## Task 3: Backend — Update `RagChatPrepared` + `prepare_*` functions

**Files:**
- Modify: `src/api/rag_chat_helpers.py`
- Test: `tests/test_rag_chat_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag_chat_helpers.py  (create file if it doesn't exist)
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.endpoints import RagChatTools


def test_rag_chat_prepared_has_allow_web_search():
    from src.api.rag_chat_helpers import RagChatPrepared
    from langchain_core.messages import HumanMessage
    from src.rag_engine import RagQueryResult
    prepared = RagChatPrepared(
        agent=None,
        resource_ids=[],
        rag_context=MagicMock(spec=RagQueryResult),
        chat_session_id="sess-1",
        messages=[HumanMessage(content="hi")],
        bind_tools=False,
        tool_skip_reason=None,
        composio_apps=[],
        allow_web_search=True,
    )
    assert prepared.allow_web_search is True


@pytest.mark.asyncio
async def test_prepare_workspace_respects_composio_false():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings

    tools = RagChatTools(web_search=True, composio=False)

    with patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", return_value=[]), \
         patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new_callable=AsyncMock, return_value="sess-1"), \
         patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr, \
         patch("src.api.rag_chat_helpers.retrieve_context_for_query", new_callable=AsyncMock, return_value=MagicMock(context="", chunks=[])), \
         patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new_callable=AsyncMock, return_value=""), \
         patch("src.api.rag_chat_helpers.list_rag_chat_messages", new_callable=AsyncMock, return_value=[]):
        mock_mgr.return_value.get_connected_app_names.return_value = ["slack"]
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="latest news",
            session_id=None,
            timings=RagChatTimings(),
            tools=tools,
        )
        # composio=False means bind_tools must be False regardless of message content
        assert result.bind_tools is False
        assert result.allow_web_search is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_rag_chat_helpers.py -v
```
Expected: FAIL — `RagChatPrepared` has no `allow_web_search`, `prepare_workspace_rag_chat` has no `tools` param.

- [ ] **Step 3: Add `allow_web_search` to `RagChatPrepared`**

In `src/api/rag_chat_helpers.py`, update the dataclass:

```python
@dataclass
class RagChatPrepared:
    agent: Any
    resource_ids: list[str]
    rag_context: RagQueryResult
    chat_session_id: str
    messages: list[BaseMessage]
    bind_tools: bool
    tool_skip_reason: str | None
    composio_apps: list[str]
    allow_web_search: bool = True
```

- [ ] **Step 4: Update `prepare_agent_rag_chat` to accept and apply `tools`**

Add `tools` parameter and replace the `should_bind_composio_tools` call:

```python
async def prepare_agent_rag_chat(
    *,
    agent_id: str,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,
) -> RagChatPrepared | None:
```

Inside, after `composio_apps = get_composio_toolset_manager().get_connected_app_names()`, replace the existing `bind_tools, tool_skip_reason = should_bind_composio_tools(...)` block with:

```python
    if tools is not None:
        bind_tools = tools.composio and settings.composio_enabled
        tool_skip_reason = None if bind_tools else "user_disabled"
        allow_web_search = tools.web_search
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason
```

And update the final `return RagChatPrepared(...)` to include `allow_web_search=allow_web_search`.

- [ ] **Step 5: Update `prepare_workspace_rag_chat` the same way**

Same change pattern as Step 4, for `prepare_workspace_rag_chat`:

```python
async def prepare_workspace_rag_chat(
    *,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,
) -> RagChatPrepared:
```

Replace the `bind_tools` assignment block:

```python
    if tools is not None:
        bind_tools = tools.composio and settings.composio_enabled
        tool_skip_reason = None if bind_tools else "user_disabled"
        allow_web_search = tools.web_search
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason
```

Update `return RagChatPrepared(...)` to include `allow_web_search=allow_web_search`.

Add the import for `RagChatTools` at the top of `rag_chat_helpers.py` (inside the function to avoid circular import, or use `TYPE_CHECKING`):

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.api.endpoints import RagChatTools
```

- [ ] **Step 6: Update endpoint callers to pass `body.tools`**

In `endpoints.py`, update all 4 `prepare_*` calls to pass `tools=body.tools`:

```python
prepared = await prepare_agent_rag_chat(
    agent_id=agent_id,
    user_id=current_user.user_id,
    normalized_message=normalized_message,
    session_id=body.session_id,
    timings=timings,
    tools=body.tools,
)
```

```python
prepared = await prepare_workspace_rag_chat(
    user_id=current_user.user_id,
    normalized_message=normalized_message,
    session_id=body.session_id,
    timings=timings,
    tools=body.tools,
)
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_rag_chat_helpers.py -v
```
Expected: PASS

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest --cov=src --cov-report=term-missing -q
```
Expected: No new failures.

- [ ] **Step 9: Commit**

```bash
git add src/api/rag_chat_helpers.py src/api/endpoints.py tests/test_rag_chat_helpers.py
git commit -m "feat(rag_chat): pass user tool flags through prepare functions, bypass auto-detection"
```

---

## Task 4: Frontend — Tool registry `toolConfig.ts`

**Files:**
- Create: `ui/src/components/chat/toolConfig.ts`

- [ ] **Step 1: Create the file**

```typescript
// ui/src/components/chat/toolConfig.ts
import { Globe, Puzzle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export type ToolDefinition = {
  id: string
  label: string
  icon: LucideIcon
  defaultEnabled: boolean
}

export type ToolConfig = Record<string, boolean>

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  { id: 'web_search', label: 'Web search', icon: Globe, defaultEnabled: true },
  { id: 'composio', label: 'Connected apps', icon: Puzzle, defaultEnabled: false },
]

export function defaultToolConfig(): ToolConfig {
  return Object.fromEntries(TOOL_DEFINITIONS.map((t) => [t.id, t.defaultEnabled]))
}
```

- [ ] **Step 2: Verify the file compiles**

```bash
cd ui && npx tsc --noEmit 2>&1 | grep toolConfig
```
Expected: no output (no errors).

- [ ] **Step 3: Commit**

```bash
git add ui/src/components/chat/toolConfig.ts
git commit -m "feat(ui): add chat tool registry (toolConfig.ts)"
```

---

## Task 5: Frontend — `ToolMenuButton` component

**Files:**
- Create: `ui/src/components/chat/ToolMenuButton.tsx`

The project uses shadcn/ui. `Popover`, `Switch`, and `Button` already exist. Confirm with:
```bash
ls ui/src/components/ui/popover.tsx ui/src/components/ui/switch.tsx ui/src/components/ui/button.tsx
```
If `popover.tsx` is missing, add it first: `cd ui && npx shadcn@latest add popover`.

- [ ] **Step 1: Create `ToolMenuButton.tsx`**

```tsx
// ui/src/components/chat/ToolMenuButton.tsx
import { Plus } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import { TOOL_DEFINITIONS, defaultToolConfig } from './toolConfig'
import type { ToolConfig } from './toolConfig'

type Props = {
  toolConfig: ToolConfig
  onToggle: (id: string, enabled: boolean) => void
  disabled?: boolean
}

export function ToolMenuButton({ toolConfig, onToggle, disabled }: Props) {
  const [open, setOpen] = useState(false)
  const defaults = defaultToolConfig()
  const hasNonDefault = TOOL_DEFINITIONS.some((t) => toolConfig[t.id] !== defaults[t.id])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className={cn('relative shrink-0 h-9 w-9', disabled && 'opacity-50 pointer-events-none')}
          aria-label="Toggle tools"
          disabled={disabled}
        >
          <Plus size={16} />
          {hasNonDefault && (
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        className="w-56 p-1"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <div className="flex flex-col gap-0.5">
          {TOOL_DEFINITIONS.map((tool) => {
            const Icon = tool.icon
            const enabled = toolConfig[tool.id] ?? tool.defaultEnabled
            return (
              <button
                key={tool.id}
                type="button"
                className="flex items-center justify-between gap-3 rounded-md px-2 py-2 text-sm hover:bg-muted transition-colors"
                onClick={() => onToggle(tool.id, !enabled)}
              >
                <span className="flex items-center gap-2 text-foreground">
                  <Icon size={15} className="shrink-0 text-muted-foreground" />
                  {tool.label}
                </span>
                <Switch
                  checked={enabled}
                  onCheckedChange={(v) => onToggle(tool.id, v)}
                  onClick={(e) => e.stopPropagation()}
                  aria-label={`Toggle ${tool.label}`}
                />
              </button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd ui && npx tsc --noEmit 2>&1 | grep ToolMenuButton
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add ui/src/components/chat/ToolMenuButton.tsx
git commit -m "feat(ui): add ToolMenuButton component with tool toggles popover"
```

---

## Task 6: Frontend — Update `types.ts` + `client.ts`

**Files:**
- Modify: `ui/src/types.ts`
- Modify: `ui/src/api/client.ts`

- [ ] **Step 1: Add `web_used` event to `RagChatStreamEvent` in `types.ts`**

Find `RagChatStreamEvent` and add the new variant:

```typescript
export type RagChatStreamEvent =
  | {
      type: 'session'
      session_id: string
      web_used?: boolean
      web_provider?: string | null
    }
  | { type: 'chunk'; text: string }
  | { type: 'citations'; citations: RagCitation[] }
  | { type: 'suggestions'; suggestions: string[] }
  | { type: 'web_used'; provider: string }
  | { type: 'done' }
  | { type: 'error'; error: string }
```

- [ ] **Step 2: Add `onWebUsed?` to `RagAgentChatStreamOptions` in `client.ts`**

```typescript
type RagAgentChatStreamOptions = {
  signal?: AbortSignal
  onSession: (sessionId: string, webUsed?: boolean, webProvider?: string | null) => void
  onChunk: (text: string) => void
  onCitations: (citations: RagCitation[]) => void
  onSuggestions?: (suggestions: string[]) => void
  onWebUsed?: () => void
  onDone: () => void
  onError?: (error: string) => void
}
```

- [ ] **Step 3: Handle `web_used` event in the `handleEvent` function**

Both `streamRagAgentChat` and `streamRagWorkspaceChat` have identical `handleEvent` functions. In each one, add before the `done` check:

```typescript
if (parsed.type === 'web_used') {
  options.onWebUsed?.()
  return false
}
```

- [ ] **Step 4: Add `tools` param to both stream functions and pass in request body**

Update `streamRagAgentChat`:

```typescript
export async function streamRagAgentChat(
  agentId: string,
  message: string,
  sessionId: string | null,
  accessToken: string | null,
  options: RagAgentChatStreamOptions,
  tools?: Record<string, boolean>,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, session_id: sessionId, tools }),
    signal: options.signal,
  })
  // ... rest unchanged
```

Update `streamRagWorkspaceChat`:

```typescript
export async function streamRagWorkspaceChat(
  message: string,
  sessionId: string | null,
  accessToken: string | null,
  options: RagAgentChatStreamOptions,
  tools?: Record<string, boolean>,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, session_id: sessionId, tools }),
    signal: options.signal,
  })
  // ... rest unchanged
```

- [ ] **Step 5: Verify compilation**

```bash
cd ui && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add ui/src/types.ts ui/src/api/client.ts
git commit -m "feat(ui): add tools param and web_used event handling to stream API client"
```

---

## Task 7: Frontend — Wire `transports.ts` + `ChatThreadContainer`

**Files:**
- Modify: `ui/src/components/chat/transports.ts`
- Modify: `ui/src/components/chat/ChatThreadContainer.tsx`

- [ ] **Step 1: Update `StreamCallbacks` and `ChatTransport` in `transports.ts`**

```typescript
import type { ToolConfig } from './toolConfig'

export type StreamCallbacks = {
  signal?: AbortSignal
  onSession: (sessionId: string, webUsed?: boolean, webProvider?: string | null) => void
  onChunk: (text: string) => void
  onCitations: (citations: RagChatMessage['citations']) => void
  onSuggestions?: (suggestions: string[]) => void
  onWebUsed?: () => void
  onDone: () => void
  onError?: (error: string) => void
}

export type ChatTransport = {
  key: string
  listSessions: (accessToken: string) => Promise<RagChatSessionSummary[]>
  loadSessionMessages: (
    sessionId: string,
    accessToken: string,
  ) => Promise<{ session_id: string; messages: RagChatMessage[] }>
  streamMessage: (
    message: string,
    sessionId: string | null,
    accessToken: string,
    callbacks: StreamCallbacks,
    tools?: ToolConfig,
  ) => Promise<void>
  deleteLastExchange: (sessionId: string, accessToken: string) => Promise<void>
}
```

- [ ] **Step 2: Update `createAgentChatTransport` and `workspaceChatTransport` to pass tools**

```typescript
export function createAgentChatTransport(agentId: string): ChatTransport {
  return {
    // ...
    streamMessage: async (message, sessionId, accessToken, callbacks, tools) => {
      await streamRagAgentChat(agentId, message, sessionId, accessToken, {
        signal: callbacks.signal,
        onSession: callbacks.onSession,
        onChunk: callbacks.onChunk,
        onCitations: callbacks.onCitations,
        onSuggestions: callbacks.onSuggestions,
        onWebUsed: callbacks.onWebUsed,
        onDone: callbacks.onDone,
        onError: callbacks.onError,
      }, tools)
    },
    // ...
  }
}

export const workspaceChatTransport: ChatTransport = {
  // ...
  streamMessage: async (message, sessionId, accessToken, callbacks, tools) => {
    await streamRagWorkspaceChat(message, sessionId, accessToken, {
      signal: callbacks.signal,
      onSession: callbacks.onSession,
      onChunk: callbacks.onChunk,
      onCitations: callbacks.onCitations,
      onSuggestions: callbacks.onSuggestions,
      onWebUsed: callbacks.onWebUsed,
      onDone: callbacks.onDone,
      onError: callbacks.onError,
    }, tools)
  },
  // ...
}
```

- [ ] **Step 3: Add `toolConfig` state and `ToolMenuButton` to `ChatThreadContainer`**

At the top of the component, add the import and state:

```typescript
import { ToolMenuButton } from './ToolMenuButton'
import { defaultToolConfig } from './toolConfig'
import type { ToolConfig } from './toolConfig'

// Inside ChatThreadContainer, after existing useState declarations:
const [toolConfig, setToolConfig] = useState<ToolConfig>(defaultToolConfig)
```

Reset tool config when transport changes (add to the existing transport.key effect):

```typescript
useEffect(() => {
  currentTransportKeyRef.current = transport.key
  setToolConfig(defaultToolConfig())
}, [transport.key])
```

- [ ] **Step 4: Pass `toolConfig` to `transport.streamMessage` in the `send` function**

Find the `transport.streamMessage(question, sessionId, accessToken, {` call and add `toolConfig` as the last argument:

```typescript
await transport.streamMessage(question, sessionId, accessToken, {
  signal: controller.signal,
  onSession: (nextSessionId, webUsed) => { ... },
  onChunk: (textChunk) => { ... },
  onCitations: (citations) => { ... },
  onSuggestions: (suggestions) => { ... },
  onWebUsed: () => {
    if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
    setWebUsedLastReply(true)
  },
  onDone: () => { ... },
  onError: (streamError) => { ... },
}, toolConfig)
```

Note: remove `setWebUsedLastReply(Boolean(webUsed))` from `onSession` since web_used now arrives via the separate `onWebUsed` event. Keep the rest of `onSession` unchanged.

- [ ] **Step 5: Render `ToolMenuButton` in the input row**

Find the input area `<div className="flex gap-2 items-end">` and add `ToolMenuButton` as the first child, before `Textarea`:

```tsx
<div className="flex gap-2 items-end">
  <ToolMenuButton
    toolConfig={toolConfig}
    onToggle={(id, enabled) =>
      setToolConfig((prev) => ({ ...prev, [id]: enabled }))
    }
    disabled={chatting}
  />
  <Textarea
    className="resize-none min-h-10 max-h-32 text-sm"
    // ...existing props unchanged
  />
  {/* ...existing send/stop button unchanged */}
</div>
```

- [ ] **Step 6: Verify full compilation**

```bash
cd ui && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 7: Run existing UI tests**

```bash
cd ui && npm test -- --run
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add ui/src/components/chat/transports.ts ui/src/components/chat/ChatThreadContainer.tsx
git commit -m "feat(ui): wire tool toggles into ChatThreadContainer and transports"
```

---

## Task 8: Integration smoke test

- [ ] **Step 1: Start backend**

```bash
uv run uvicorn src.api.endpoints:app --reload --port 8000
```

- [ ] **Step 2: Start UI dev server**

```bash
cd ui && npm run dev
```

- [ ] **Step 3: Verify the `+` button appears in the chat input**

Open the chat page. The `+` button should appear to the left of the textarea.

- [ ] **Step 4: Verify toggle popover opens**

Click `+`. Should show "Web search" (on) and "Connected apps" (off).

- [ ] **Step 5: Verify tool config flows to backend**

Open browser DevTools → Network. Send a chat message with web search ON. Confirm the request body contains `{"message": "...", "tools": {"web_search": true, "composio": false}}`.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest --cov=src --cov-report=term-missing -q
```
Expected: no failures.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat(chat): tool selection toggles — web search on by default, Composio opt-in"
```
