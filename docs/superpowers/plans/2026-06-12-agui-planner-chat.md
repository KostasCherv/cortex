# AG-UI Planner Chat Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled SSE event format in Planner Chat with the AG-UI open protocol, gaining standard event types, eliminating custom parsing boilerplate, and unlocking future tool-call/reasoning visualization.

**Architecture:** The `planner_graph` (LangGraph) and all React components stay untouched. Only the streaming layer changes: the Python backend replaces `_sse_line` dict calls with `EventEncoder` + typed AG-UI event models (`ag-ui-protocol` package); the TypeScript frontend replaces `PlannerChatStreamEvent` parsing with AG-UI event type matching. The `PlannerStreamCallbacks` public interface is preserved so callers need zero changes.

**Tech Stack:** `ag-ui-protocol>=0.1.0` (Python), `@ag-ui/core` (npm), FastAPI `StreamingResponse`, Vitest (frontend tests), pytest (backend tests).

---

## File Map

| File | Change |
|------|--------|
| `pyproject.toml` | Add `ag-ui-protocol>=0.1.0` dependency |
| `src/api/planner_chat.py` | Replace `_sse_line` + `json` with `EventEncoder` + AG-UI event models |
| `tests/test_planner_chat_api.py` | Update event type/field assertions for AG-UI format |
| `ui/package.json` | Add `@ag-ui/core` |
| `ui/src/api/plannerChatClient.ts` | Rewrite `handleEvent` to parse AG-UI events |
| `ui/src/api/plannerChatClient.test.ts` | New: unit tests for AG-UI event parsing |

**Not changed:** `planner_graph/`, `planner_thread_store`, `ui/src/types.ts`, all React pages/components, all other streaming endpoints.

---

## Event mapping reference

| Old custom event | AG-UI wire format (camelCase, JSON) |
|---|---|
| `{"type":"session","thread_id":"…"}` | `{"type":"RUN_STARTED","threadId":"…","runId":"…"}` |
| `{"type":"chunk","text":"…"}` | `{"type":"TEXT_MESSAGE_START",…}` + `{"type":"TEXT_MESSAGE_CHUNK","messageId":"…","delta":"…"}` + `{"type":"TEXT_MESSAGE_END",…}` |
| `{"type":"plan","plan":{…},"markdown":"…",…}` | `{"type":"CUSTOM","name":"plan","value":{"type":"plan","plan":{…},"markdown":"…",…}}` |
| `{"type":"done"}` | `{"type":"RUN_FINISHED","threadId":"…","runId":"…"}` |
| `{"type":"error","error":"…"}` | `{"type":"RUN_ERROR","message":"…"}` |

Note: AG-UI Python SDK uses `alias_generator=to_camel`, so all field names are camelCase in JSON output (`threadId`, `runId`, `messageId`, `delta`).

---

## Task 1: Add Python ag-ui-protocol dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency to pyproject.toml**

In `pyproject.toml`, inside the `dependencies` list, add:
```toml
"ag-ui-protocol>=0.1.0",
```

- [ ] **Step 2: Sync the lockfile**

```bash
uv sync
```
Expected: resolves and installs `ag-ui-protocol` without conflicts.

- [ ] **Step 3: Verify the import chain works**

```bash
uv run python -c "
from ag_ui.core import (
    RunStartedEvent, RunFinishedEvent, RunErrorEvent,
    TextMessageStartEvent, TextMessageChunkEvent, TextMessageEndEvent,
    CustomEvent,
)
from ag_ui.encoder import EventEncoder
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add ag-ui-protocol dependency"
```

---

## Task 2: Update backend tests for AG-UI event format (RED)

**Files:**
- Modify: `tests/test_planner_chat_api.py`

Write the updated assertions *before* changing the backend so the tests fail (RED) now and pass (GREEN) after Task 3. The `_parse_sse` helper is unchanged — it parses `data: …` lines regardless of event schema.

- [ ] **Step 1: Update `test_new_thread_session_event_returned`**

Replace:
```python
session_events = [e for e in events if e["type"] == "session"]
assert len(session_events) == 1
assert "thread_id" in session_events[0]
```
With:
```python
session_events = [e for e in events if e["type"] == "RUN_STARTED"]
assert len(session_events) == 1
assert "threadId" in session_events[0]
```

- [ ] **Step 2: Update `test_new_thread_clarification_chunk_and_done`**

Replace:
```python
chunk_events = [e for e in events if e["type"] == "chunk"]
done_events = [e for e in events if e["type"] == "done"]
assert any(question in e["text"] for e in chunk_events)
assert len(done_events) == 1
```
With:
```python
chunk_events = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
done_events = [e for e in events if e["type"] == "RUN_FINISHED"]
assert any(question in e["delta"] for e in chunk_events)
assert len(done_events) == 1
```

- [ ] **Step 3: Update `test_new_thread_final_plan_returns_plan_event`**

Replace:
```python
plan_events = [e for e in events if e["type"] == "plan"]
done_events = [e for e in events if e["type"] == "done"]
assert len(plan_events) == 1
assert plan_events[0]["markdown"] == final_plan.markdown
assert len(done_events) == 1
```
With:
```python
plan_events = [e for e in events if e.get("type") == "CUSTOM" and e.get("name") == "plan"]
done_events = [e for e in events if e["type"] == "RUN_FINISHED"]
assert len(plan_events) == 1
assert plan_events[0]["value"]["markdown"] == final_plan.markdown
assert len(done_events) == 1
```

- [ ] **Step 4: Update `test_graph_error_returns_error_event`**

Replace:
```python
error_events = [e for e in events if e["type"] == "error"]
assert len(error_events) == 1
assert error_events[0]["error"] == "some_error_code"
```
With:
```python
error_events = [e for e in events if e["type"] == "RUN_ERROR"]
assert len(error_events) == 1
assert error_events[0]["message"] == "some_error_code"
```

- [ ] **Step 5: Update `test_graph_invocation_exception_returns_error_event`**

Replace:
```python
error_events = [e for e in events if e["type"] == "error"]
assert len(error_events) == 1
assert "Graph execution failed" in error_events[0]["error"]
```
With:
```python
error_events = [e for e in events if e["type"] == "RUN_ERROR"]
assert len(error_events) == 1
assert "Graph execution failed" in error_events[0]["message"]
```

- [ ] **Step 6: Update `test_existing_thread_continues_conversation`**

Replace:
```python
session_events = [e for e in events if e["type"] == "session"]
assert session_events[0]["thread_id"] == thread_id
```
With:
```python
session_events = [e for e in events if e["type"] == "RUN_STARTED"]
assert session_events[0]["threadId"] == thread_id
```

- [ ] **Step 7: Run tests — confirm RED**

```bash
uv run pytest tests/test_planner_chat_api.py -v
```
Expected: multiple FAILED (old `session`/`chunk`/`done`/`error` events not found in responses that still emit the old format).

- [ ] **Step 8: Commit**

```bash
git add tests/test_planner_chat_api.py
git commit -m "test(planner-chat): update assertions for AG-UI protocol event format"
```

---

## Task 3: Update backend to emit AG-UI events (GREEN)

**Files:**
- Modify: `src/api/planner_chat.py`

- [ ] **Step 1: Replace imports at top of file**

Remove `import json`.  
Add after the existing stdlib imports:
```python
from ag_ui.core import (
    CustomEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
```

- [ ] **Step 2: Delete the `_sse_line` helper**

Remove these lines entirely:
```python
def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
```

- [ ] **Step 3: Rewrite `_stream_planner_turn`**

Replace the entire `_stream_planner_turn` function with:

```python
async def _stream_planner_turn(
    message: str,
    thread_id: str,
    user_id: str,
) -> AsyncGenerator[str, None]:
    encoder = EventEncoder()
    run_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id))

    config = {"configurable": {"thread_id": thread_id}}

    existing_state = planner_graph.get_state(config)
    existing_history: list = []
    if existing_state and existing_state.values:
        existing_history = list(existing_state.values.get("conversation_history") or [])

    new_history = existing_history + [HumanMessage(content=message)]

    now = datetime.now(UTC).isoformat()
    planner_thread_store.append_message(
        thread_id,
        {
            "message_id": str(uuid.uuid4()),
            "role": "user",
            "content": message,
            "created_at": now,
        },
    )

    try:
        await asyncio.to_thread(
            planner_graph.invoke,
            {"conversation_history": new_history},
            config,
        )
    except Exception as exc:
        logger.error("Planner graph invocation failed: %s", exc)
        yield encoder.encode(RunErrorEvent(message="Graph execution failed."))
        yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
        return

    result_state = planner_graph.get_state(config)
    if not result_state or not result_state.values:
        yield encoder.encode(RunErrorEvent(message="No state returned from graph."))
        yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
        return

    state_values = result_state.values
    error = state_values.get("error")
    clarification_question = state_values.get("clarification_question")
    final_plan = state_values.get("final_plan")

    if error:
        ai_content = "I encountered an error while processing your request. Please try again."
        yield encoder.encode(TextMessageStartEvent(message_id=message_id, role="assistant"))
        yield encoder.encode(TextMessageChunkEvent(message_id=message_id, delta=ai_content))
        yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": ai_content,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        yield encoder.encode(RunErrorEvent(message=error))
        yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
        return

    ai_history_content: str | None = None

    if final_plan is not None:
        markdown = final_plan.markdown
        chunk_size = 200
        yield encoder.encode(TextMessageStartEvent(message_id=message_id, role="assistant"))
        for i in range(0, len(markdown), chunk_size):
            yield encoder.encode(
                TextMessageChunkEvent(message_id=message_id, delta=markdown[i : i + chunk_size])
            )
        yield encoder.encode(TextMessageEndEvent(message_id=message_id))

        plan_payload = {
            "type": "plan",
            "plan": final_plan.plan.model_dump(mode="json"),
            "markdown": final_plan.markdown,
            "suggested_filename": final_plan.suggested_filename,
            "planning_brief": final_plan.planning_brief.model_dump(mode="json"),
        }
        yield encoder.encode(CustomEvent(name="plan", value=plan_payload))

        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": markdown,
                "plan_event": plan_payload,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

        try:
            await save_prd(user_id, message, final_plan)
        except Exception as persist_exc:
            logger.warning("Failed to persist planner output: %s", persist_exc)

        ai_history_content = (
            "I have generated a complete PRD based on our discussion. "
            "If you'd like to refine it, describe what you'd like to change."
        )

    elif clarification_question:
        yield encoder.encode(TextMessageStartEvent(message_id=message_id, role="assistant"))
        yield encoder.encode(
            TextMessageChunkEvent(message_id=message_id, delta=clarification_question)
        )
        yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": clarification_question,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        ai_history_content = clarification_question
    else:
        msg = "I'm ready to generate your plan. What else would you like to clarify?"
        yield encoder.encode(TextMessageStartEvent(message_id=message_id, role="assistant"))
        yield encoder.encode(TextMessageChunkEvent(message_id=message_id, delta=msg))
        yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": msg,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        ai_history_content = msg

    if ai_history_content:
        updated_history = new_history + [AIMessage(content=ai_history_content)]
        try:
            await asyncio.to_thread(
                planner_graph.update_state,
                config,
                {"conversation_history": updated_history},
            )
        except Exception as update_exc:
            logger.warning("Failed to update planner graph state: %s", update_exc)

    yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
```

- [ ] **Step 4: Run planner chat tests — confirm GREEN**

```bash
uv run pytest tests/test_planner_chat_api.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Run broader test suite**

```bash
uv run pytest --ignore=tests/test_planner_graph.py -v
```
Expected: all PASSED (skipping `test_planner_graph.py` which invokes real LLM APIs).

- [ ] **Step 6: Commit**

```bash
git add src/api/planner_chat.py
git commit -m "feat(planner-chat): emit AG-UI protocol events via EventEncoder"
```

---

## Task 4: Add @ag-ui/core to frontend

**Files:**
- Modify: `ui/package.json`, `ui/package-lock.json`

- [ ] **Step 1: Install**

```bash
cd ui && npm install @ag-ui/core
```
Expected: installs without peer-dependency conflicts.

- [ ] **Step 2: Confirm TypeScript resolves the package**

```bash
cd ui && npx tsc --noEmit
```
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add ui/package.json ui/package-lock.json
git commit -m "chore(ui): add @ag-ui/core dependency"
```

---

## Task 5: Update frontend SSE parser (TDD)

**Files:**
- Create: `ui/src/api/plannerChatClient.test.ts`
- Modify: `ui/src/api/plannerChatClient.ts`

The `PlannerStreamCallbacks` interface and the three exported function signatures (`streamPlannerChat`, `getPlannerChatMessages`, `deletePlannerChatLastExchange`) are unchanged. Only the internal `handleEvent` parsing logic changes.

- [ ] **Step 1: Create `ui/src/api/plannerChatClient.test.ts`**

```typescript
import { describe, it, expect, vi } from 'vitest'
import { streamPlannerChat } from './plannerChatClient'

function makeStream(lines: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const line of lines) controller.enqueue(enc.encode(line))
      controller.close()
    },
  })
}

function mockFetch(lines: string[]) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    body: makeStream(lines),
  } as unknown as Response)
}

describe('streamPlannerChat — AG-UI event parsing', () => {
  it('calls onSession with threadId from RUN_STARTED', async () => {
    mockFetch([
      'data: {"type":"RUN_STARTED","threadId":"thread-1","runId":"run-1"}\n\n',
      'data: {"type":"RUN_FINISHED","threadId":"thread-1","runId":"run-1"}\n\n',
    ])
    const onSession = vi.fn()
    const onDone = vi.fn()
    await streamPlannerChat('hello', null, 'tok', {
      onSession, onChunk: vi.fn(), onPlan: vi.fn(), onDone,
    })
    expect(onSession).toHaveBeenCalledWith('thread-1')
    expect(onDone).toHaveBeenCalled()
  })

  it('calls onChunk with delta from TEXT_MESSAGE_CHUNK', async () => {
    mockFetch([
      'data: {"type":"RUN_STARTED","threadId":"t","runId":"r"}\n\n',
      'data: {"type":"TEXT_MESSAGE_START","messageId":"m1","role":"assistant"}\n\n',
      'data: {"type":"TEXT_MESSAGE_CHUNK","messageId":"m1","delta":"Hello world"}\n\n',
      'data: {"type":"TEXT_MESSAGE_END","messageId":"m1"}\n\n',
      'data: {"type":"RUN_FINISHED","threadId":"t","runId":"r"}\n\n',
    ])
    const onChunk = vi.fn()
    await streamPlannerChat('hello', null, 'tok', {
      onSession: vi.fn(), onChunk, onPlan: vi.fn(), onDone: vi.fn(),
    })
    expect(onChunk).toHaveBeenCalledWith('Hello world')
  })

  it('calls onPlan with plan payload from CUSTOM plan event', async () => {
    const planPayload = {
      type: 'plan' as const,
      plan: { title: 'My App' },
      markdown: '# My App',
      suggested_filename: 'my-app.md',
      planning_brief: {},
    }
    mockFetch([
      'data: {"type":"RUN_STARTED","threadId":"t","runId":"r"}\n\n',
      `data: ${JSON.stringify({ type: 'CUSTOM', name: 'plan', value: planPayload })}\n\n`,
      'data: {"type":"RUN_FINISHED","threadId":"t","runId":"r"}\n\n',
    ])
    const onPlan = vi.fn()
    await streamPlannerChat('hello', null, 'tok', {
      onSession: vi.fn(), onChunk: vi.fn(), onPlan, onDone: vi.fn(),
    })
    expect(onPlan).toHaveBeenCalledWith(planPayload)
  })

  it('calls onError with message from RUN_ERROR', async () => {
    mockFetch([
      'data: {"type":"RUN_STARTED","threadId":"t","runId":"r"}\n\n',
      'data: {"type":"RUN_ERROR","message":"Something went wrong"}\n\n',
    ])
    const onError = vi.fn()
    await streamPlannerChat('hello', null, 'tok', {
      onSession: vi.fn(), onChunk: vi.fn(), onPlan: vi.fn(), onDone: vi.fn(), onError,
    })
    expect(onError).toHaveBeenCalledWith('Something went wrong')
  })
})
```

- [ ] **Step 2: Run tests — confirm RED**

```bash
cd ui && npx vitest run src/api/plannerChatClient.test.ts
```
Expected: 4 FAILED (current code looks for `session`/`chunk`/`plan`/`done`/`error`; test sends AG-UI types).

- [ ] **Step 3: Rewrite `handleEvent` in `plannerChatClient.ts`**

Locate the `handleEvent` function (currently typed as `(parsed: PlannerChatStreamEvent): boolean`).

Replace it with:
```typescript
const handleEvent = (parsed: Record<string, unknown>): boolean => {
  if (parsed.type === 'RUN_STARTED') {
    callbacks.onSession(parsed.threadId as string)
    return false
  }
  if (parsed.type === 'TEXT_MESSAGE_CHUNK') {
    callbacks.onChunk(parsed.delta as string)
    return false
  }
  if (parsed.type === 'CUSTOM' && parsed.name === 'plan') {
    callbacks.onPlan(parsed.value as PlannerChatStreamEvent & { type: 'plan' })
    return false
  }
  if (parsed.type === 'RUN_FINISHED') {
    callbacks.onDone()
    return true
  }
  if (parsed.type === 'RUN_ERROR') {
    callbacks.onError?.(parsed.message as string)
    return true
  }
  return false
}
```

Also update the `parsed` variable declaration inside the SSE parsing loop. Find:
```typescript
let parsed: PlannerChatStreamEvent
try {
  parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as PlannerChatStreamEvent
```
Replace with:
```typescript
let parsed: Record<string, unknown>
try {
  parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as Record<string, unknown>
```
Apply the same change to the trailing-buffer parse block at the bottom of the function.

- [ ] **Step 4: Run tests — confirm GREEN**

```bash
cd ui && npx vitest run src/api/plannerChatClient.test.ts
```
Expected: 4 PASSED.

- [ ] **Step 5: Typecheck**

```bash
cd ui && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add ui/src/api/plannerChatClient.ts ui/src/api/plannerChatClient.test.ts
git commit -m "feat(planner-chat): consume AG-UI events in frontend SSE parser"
```

---

## Task 6: End-to-end smoke test

No code changes. Manual verification only.

- [ ] **Step 1: Start the backend**

```bash
uv run uvicorn src.api.endpoints:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend**

```bash
cd ui && npm run dev
```

- [ ] **Step 3: Test clarification flow**

Open Planner Chat. Send a message that triggers a clarification question. Confirm:
- Clarification text streams character-by-character ✓
- No console errors ✓

- [ ] **Step 4: Test plan generation flow**

Continue the conversation until a full plan is produced. Confirm:
- Plan markdown streams incrementally ✓
- Plan component renders correctly (plan event fires `onPlan`) ✓
- Save / export actions still work ✓

- [ ] **Step 5: Inspect the wire format in DevTools**

Open Network tab → find the `POST /api/planner/chat` request → EventStream tab. Confirm the first event is `RUN_STARTED` and the last is `RUN_FINISHED`.

---

## Self-review

**Spec coverage:**
- ✅ Backend emits AG-UI events — Task 3
- ✅ Frontend consumes AG-UI events — Task 5
- ✅ `plan` mapped to `CUSTOM(name="plan", value={…})` — Task 3 step 3, Task 5 step 3
- ✅ `PlannerStreamCallbacks` interface unchanged — Task 5 step 3 (same signature)
- ✅ React components untouched — no component files appear in any task
- ✅ `planner_graph` untouched — not referenced in any modification step
- ✅ Thread store `plan_event` shape unchanged — Task 3 step 3 stores same `plan_payload` dict
- ✅ Other streaming endpoints untouched — not in scope

**Placeholder scan:** None found. All steps contain exact code or exact commands.

**Type consistency:**
- `threadId` (camelCase) used in backend `RunStartedEvent` constructor, test assertions, and frontend parsing — consistent throughout (AG-UI SDK uses `to_camel` alias generator)
- `PlannerChatStreamEvent & { type: 'plan' }` used in `onPlan` callback in Task 5 — matches existing type in `plannerChatClient.ts`; no type.ts change required
