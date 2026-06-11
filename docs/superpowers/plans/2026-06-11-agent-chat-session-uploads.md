# Agent Chat Session Uploads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-scoped file uploads to agent chat so files uploaded with a turn are ingested immediately, used in that same turn, remain available in later turns of the same session, and are hard-deleted with the session.

**Architecture:** Introduce a dedicated session-attachment model parallel to workspace RAG resources, then thread attachment IDs into `prepare_agent_rag_chat` so retrieval uses both agent-linked resources and session attachments. Keep the upload path synchronous for turns with files by accepting multipart requests in agent chat endpoints and waiting for ingestion before starting answer generation.

**Tech Stack:** FastAPI + Pydantic + `UploadFile`, existing RAG ingestion/storage helpers in `src/rag.py`, Supabase REST store in `src/db/supabase_store.py`, React + TypeScript + existing chat transport/client stack, Vitest + pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/rag.py` | Modify | Add session-attachment model, creation/list/delete helpers, and inline ingestion orchestration |
| `src/db/supabase_store.py` | Modify | Add persistence methods for session attachments and delete-cascade support |
| `src/api/rag_chat_helpers.py` | Modify | Merge ready session attachment IDs into agent-chat retrieval preparation |
| `src/api/endpoints.py` | Modify | Add multipart parsing for agent chat, attachment listing endpoint, and session-delete cleanup |
| `ui/src/types.ts` | Modify | Add session-attachment types |
| `ui/src/api/client.ts` | Modify | Add attachment listing API and multipart agent-chat streaming support |
| `ui/src/components/chat/transports.ts` | Modify | Thread files into agent chat transport only |
| `ui/src/components/chat/ChatThreadContainer.tsx` | Modify | Add file picker, attachment state, and send-flow integration |
| `ui/src/components/agents/AgentChat.tsx` | Modify | Enable session attachment display for agent chat |
| `tests/test_sessions_store.py` | Modify | Add store coverage for attachment CRUD/list/delete |
| `tests/test_rag_chat_helpers.py` | Modify | Add merged resource-ID preparation coverage |
| `tests/test_api.py` | Modify | Add multipart upload, same-turn retrieval, listing, and delete-cascade coverage |

---

### Task 1: Persistence + domain model for session attachments

**Files:**
- Modify: `src/rag.py`
- Modify: `src/db/supabase_store.py`
- Test: `tests/test_sessions_store.py`

- [ ] **Step 1: Write the failing store tests**

```python
async def test_store_lists_ready_session_attachment_resource_ids():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [
        {"id": "att-1", "resource_id": "res-a"},
        {"id": "att-2", "resource_id": "res-b"},
    ]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    ready_ids = await store.list_ready_rag_chat_session_attachment_resource_ids(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert ready_ids == ["res-a", "res-b"]


async def test_store_deletes_session_attachments_for_chat_session():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [{"id": "att-1"}, {"id": "att-2"}]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    deleted = await store.delete_rag_chat_session_attachments(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert deleted == [
        {"attachment_id": "att-1"},
        {"attachment_id": "att-2"},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_sessions_store.py::test_store_lists_ready_session_attachment_resource_ids tests/test_sessions_store.py::test_store_deletes_session_attachments_for_chat_session`

Expected: FAIL with `AttributeError` because the new store methods do not exist yet.

- [ ] **Step 3: Add the minimal store methods in `src/db/supabase_store.py`**

```python
async def list_ready_rag_chat_session_attachment_resource_ids(
    self,
    *,
    session_id: str,
    owner_id: str,
    agent_id: str,
) -> list[str]:
    response = await self._request(
        "GET",
        "rag_chat_session_attachments",
        params={
            "select": "id,resource_id",
            "session_id": f"eq.{session_id}",
            "owner_id": f"eq.{owner_id}",
            "agent_id": f"eq.{agent_id}",
            "state": "eq.ready",
            "order": "created_at.asc",
        },
    )
    return [
        row["resource_id"]
        for row in response.json()
        if isinstance(row.get("resource_id"), str) and row["resource_id"]
    ]


async def delete_rag_chat_session_attachments(
    self,
    *,
    session_id: str,
    owner_id: str,
    agent_id: str,
) -> list[dict[str, str]]:
    response = await self._request(
        "DELETE",
        "rag_chat_session_attachments",
        params={
            "session_id": f"eq.{session_id}",
            "owner_id": f"eq.{owner_id}",
            "agent_id": f"eq.{agent_id}",
        },
        extra_headers={"Prefer": "return=representation"},
    )
    return [
        {"attachment_id": row["id"]}
        for row in response.json()
        if isinstance(row.get("id"), str)
    ]
```

Then add the domain model and orchestration scaffold in `src/rag.py`:

```python
@dataclass
class RagSessionAttachment:
    attachment_id: str
    session_id: str
    agent_id: str
    owner_id: str
    workspace_id: str
    resource_id: str
    filename: str
    mime_type: str
    byte_size: int
    storage_uri: str
    state: str = "uploaded"
    error_details: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "attachment_id": self.attachment_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "resource_id": self.resource_id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "byte_size": self.byte_size,
            "storage_uri": self.storage_uri,
            "state": self.state,
            "error_details": self.error_details,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_sessions_store.py::test_store_lists_ready_session_attachment_resource_ids tests/test_sessions_store.py::test_store_deletes_session_attachments_for_chat_session`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag.py src/db/supabase_store.py tests/test_sessions_store.py
git commit -m "feat(rag): add session attachment persistence primitives"
```

---

### Task 2: Retrieval preparation merges session attachments with agent resources

**Files:**
- Modify: `src/api/rag_chat_helpers.py`
- Modify: `src/rag.py`
- Test: `tests/test_rag_chat_helpers.py`

- [ ] **Step 1: Write the failing helper test**

```python
@pytest.mark.asyncio
async def test_prepare_agent_rag_chat_merges_session_attachment_resource_ids():
    from src.api.rag_chat_helpers import prepare_agent_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from unittest.mock import AsyncMock, MagicMock, patch

    agent = MagicMock(system_instructions="Use docs.")

    with patch(
        "src.api.rag_chat_helpers.get_agent_for_chat",
        new_callable=AsyncMock,
        return_value=(agent, ["agent-res-1"]),
    ), patch(
        "src.api.rag_chat_helpers.create_or_get_chat_session",
        new_callable=AsyncMock,
        return_value="chat-1",
    ), patch(
        "src.api.rag_chat_helpers.list_ready_rag_chat_session_attachment_resource_ids",
        new_callable=AsyncMock,
        return_value=["session-res-1", "session-res-2"],
    ), patch(
        "src.api.rag_chat_helpers.retrieve_context_for_query",
        new_callable=AsyncMock,
        return_value=MagicMock(context="", chunks=[]),
    ), patch(
        "src.api.rag_chat_helpers.get_user_memory_prompt_block",
        new_callable=AsyncMock,
        return_value="",
    ), patch(
        "src.api.rag_chat_helpers.list_rag_chat_messages",
        new_callable=AsyncMock,
        return_value=[],
    ), patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr:
        mock_mgr.return_value.get_connected_app_names.return_value = []

        prepared = await prepare_agent_rag_chat(
            agent_id="agent-1",
            user_id="user-1",
            normalized_message="Summarize the uploaded PDF",
            session_id=None,
            timings=RagChatTimings(),
        )

    assert prepared is not None
    assert prepared.resource_ids == ["agent-res-1", "session-res-1", "session-res-2"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/test_rag_chat_helpers.py::test_prepare_agent_rag_chat_merges_session_attachment_resource_ids`

Expected: FAIL because `prepare_agent_rag_chat` does not yet load session attachment IDs.

- [ ] **Step 3: Implement the minimal merge in `src/api/rag_chat_helpers.py`**

Add the import from `src.rag`:

```python
from src.rag import (
    RagChatMessage,
    append_chat_message,
    create_or_get_chat_session,
    create_or_get_workspace_chat_session,
    get_agent_for_chat,
    list_chat_messages as list_rag_chat_messages,
    list_ready_rag_chat_session_attachment_resource_ids,
    list_workspace_ready_resource_ids,
    retrieve_context_for_query,
)
```

Then update `prepare_agent_rag_chat`:

```python
session_resource_ids_task = list_ready_rag_chat_session_attachment_resource_ids(
    session_id=chat_session_id,
    owner_id=user_id,
    agent_id=agent_id,
)
memory_task = get_user_memory_prompt_block(user_id, normalized_message)
history_task = list_rag_chat_messages(chat_session_id, user_id)
session_resource_ids, user_memory_context, history = await asyncio.gather(
    session_resource_ids_task,
    memory_task,
    history_task,
)
resource_ids = list(dict.fromkeys([*resource_ids, *session_resource_ids]))
rag_context = await retrieve_context_for_query(
    user_id=user_id,
    resource_ids=resource_ids,
    question=normalized_message,
)
```

Add a small pass-through helper in `src/rag.py` if needed:

```python
async def list_ready_rag_chat_session_attachment_resource_ids(
    *,
    session_id: str,
    owner_id: str,
    agent_id: str,
) -> list[str]:
    return await get_session_store().list_ready_rag_chat_session_attachment_resource_ids(
        session_id=session_id,
        owner_id=owner_id,
        agent_id=agent_id,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/test_rag_chat_helpers.py::test_prepare_agent_rag_chat_merges_session_attachment_resource_ids`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/rag_chat_helpers.py src/rag.py tests/test_rag_chat_helpers.py
git commit -m "feat(rag_chat): merge session attachments into agent retrieval"
```

---

### Task 3: Multipart agent-chat API + same-turn ingestion

**Files:**
- Modify: `src/api/endpoints.py`
- Modify: `src/rag.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing API tests**

```python
def test_rag_agent_chat_with_upload_rejects_invalid_file_type(client, auth_headers):
    response = client.post(
        "/api/rag/agents/agent-1/chat",
        headers=auth_headers,
        data={"message": "Use this file"},
        files={"files": ("payload.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_rag_agent_chat_stream_uploads_files_before_running_loop(client, auth_headers):
    async def _record_ingest(**_kwargs):
        call_order.append("ingest")
        return []

    with patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()), \
         patch("src.api.endpoints.prepare_agent_rag_chat", new_callable=AsyncMock) as mock_prepare, \
         patch("src.api.endpoints.ingest_agent_chat_session_uploads", new_callable=AsyncMock) as mock_ingest, \
         patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=_fake_agent_loop_result("done"))):
        call_order: list[str] = []
        mock_ingest.side_effect = _record_ingest
        mock_prepare.side_effect = [
            _fake_prepared_chat(),
            _fake_prepared_chat(resource_ids=["agent-res-1", "session-res-1"]),
        ]

        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            headers=auth_headers,
            data={"message": "Use this PDF"},
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    assert mock_ingest.await_count == 1
    assert mock_prepare.await_count == 2
    assert call_order == ["ingest"]
```

If the test suite already has helper fixtures for authenticated clients and fake prepared chats, reuse them instead of introducing new ones.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_api.py::test_rag_agent_chat_with_upload_rejects_invalid_file_type tests/test_api.py::test_rag_agent_chat_stream_uploads_files_before_running_loop`

Expected: FAIL because the agent-chat endpoints only accept JSON and no upload helper exists yet.

- [ ] **Step 3: Add the minimal upload parsing and ingestion helper**

In `src/rag.py`, add:

```python
async def ingest_agent_chat_session_uploads(
    *,
    session_id: str,
    agent_id: str,
    user_id: str,
    files: list[UploadFile],
) -> list[RagSessionAttachment]:
    attachments: list[RagSessionAttachment] = []
    for file in files:
        content = await file.read()
        _validate_upload(file, content)
        resource = await create_resource_and_ingest(file=file, user_id=user_id)
        attachment = RagSessionAttachment(
            attachment_id=str(uuid.uuid4()),
            session_id=session_id,
            agent_id=agent_id,
            owner_id=user_id,
            workspace_id=_workspace_id_for_user(user_id),
            resource_id=resource.resource_id,
            filename=resource.filename,
            mime_type=resource.mime_type,
            byte_size=resource.byte_size,
            storage_uri=resource.storage_uri,
            state=resource.state,
            error_details=resource.error_details,
        )
        await get_session_store().create_rag_chat_session_attachment(attachment.to_dict())
        attachments.append(attachment)
    return attachments
```

In `src/api/endpoints.py`, add a multipart-friendly dependency/parser and use it in both agent-chat routes:

```python
from fastapi import File, Form, UploadFile


async def _parse_rag_chat_form(
    message: str = Form(...),
    session_id: str | None = Form(default=None),
    tools: str | None = Form(default=None),
    files: list[UploadFile] = File(default_factory=list),
) -> tuple[RagChatRequest, list[UploadFile]]:
    payload = RagChatRequest(
        message=message,
        session_id=session_id,
        tools=RagChatTools.model_validate_json(tools) if tools else RagChatTools(),
    )
    return payload, files
```

Then, in each agent-chat endpoint:

```python
body, files = await _parse_rag_chat_form(...)  # or equivalent dependency result
prepared = await prepare_agent_rag_chat(...)
if files:
    await ingest_agent_chat_session_uploads(
        session_id=prepared.chat_session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
        files=files,
    )
    prepared = await prepare_agent_rag_chat(...)
```

Keep the non-file JSON path working by branching on `request.headers.get("content-type")`.

- [ ] **Step 4: Add the attachment-listing endpoint and delete cascade**

Add to `src/api/endpoints.py`:

```python
@app.get("/api/rag/agents/{agent_id}/chat/sessions/{session_id}/attachments", tags=["RAG"])
async def list_rag_agent_chat_session_attachments(
    agent_id: str,
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    attachments = await list_rag_chat_session_attachments(
        session_id=session_id,
        owner_id=current_user.user_id,
        agent_id=agent_id,
    )
    return {"attachments": [attachment.to_dict() for attachment in attachments]}
```

In the existing session delete path, call the new cleanup helper before deleting the chat session record:

```python
await delete_rag_chat_session_attachments_and_artifacts(
    session_id=session_id,
    owner_id=current_user.user_id,
    agent_id=agent_id,
)
deleted = await delete_rag_chat_session(...)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_api.py::test_rag_agent_chat_with_upload_rejects_invalid_file_type tests/test_api.py::test_rag_agent_chat_stream_uploads_files_before_running_loop`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/endpoints.py src/rag.py tests/test_api.py
git commit -m "feat(api): add multipart agent chat uploads"
```

---

### Task 4: Frontend types + API client + transport support for attachments

**Files:**
- Modify: `ui/src/types.ts`
- Modify: `ui/src/api/client.ts`
- Modify: `ui/src/components/chat/transports.ts`

- [ ] **Step 1: Write the failing client/type tests**

If the UI already has a client test file, add:

```typescript
it('builds multipart FormData for agent chat when files are present', async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response('data: {"type":"done"}\n\n', { status: 200 })
  )
  vi.stubGlobal('fetch', fetchMock)

  await streamRagAgentChat(
    'agent-1',
    'Use these notes',
    null,
    'token',
    {
      onSession: vi.fn(),
      onChunk: vi.fn(),
      onCitations: vi.fn(),
      onDone: vi.fn(),
    },
    undefined,
    [new File(['hello'], 'notes.txt', { type: 'text/plain' })],
  )

  const request = fetchMock.mock.calls[0][1] as RequestInit
  expect(request.body).toBeInstanceOf(FormData)
})
```

If no UI client test file exists yet, create one at `ui/src/api/client.test.ts`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui && npm test -- --run ui/src/api/client.test.ts`

Expected: FAIL because `streamRagAgentChat` does not accept files yet.

- [ ] **Step 3: Add the minimal types and client support**

In `ui/src/types.ts`:

```typescript
export type RagSessionAttachment = {
  attachment_id: string
  session_id: string
  agent_id: string
  owner_id: string
  resource_id: string
  filename: string
  mime_type: string
  byte_size: number
  state: RagResourceState
  error_details?: string | null
  created_at: string
  updated_at: string
}
```

In `ui/src/api/client.ts`, update the signature and request building:

```typescript
export async function streamRagAgentChat(
  agentId: string,
  message: string,
  sessionId: string | null,
  accessToken: string | null,
  options: RagAgentChatStreamOptions,
  tools?: Record<string, boolean>,
  files: File[] = [],
): Promise<void> {
  const headers: HeadersInit = {
    Accept: 'text/event-stream',
    ...authHeaders(accessToken),
  }

  let body: BodyInit
  if (files.length > 0) {
    const form = new FormData()
    form.append('message', message)
    if (sessionId) form.append('session_id', sessionId)
    if (tools) form.append('tools', JSON.stringify(tools))
    files.forEach((file) => form.append('files', file))
    body = form
  } else {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify({ message, session_id: sessionId, tools })
  }

  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/stream`, {
    method: 'POST',
    headers,
    body,
    signal: options.signal,
  })
```

Add:

```typescript
export async function listRagAgentChatSessionAttachments(
  agentId: string,
  sessionId: string,
  accessToken: string | null,
): Promise<{ attachments: RagSessionAttachment[] }> {
  const response = await fetch(
    `${API_BASE}/api/rag/agents/${agentId}/chat/sessions/${sessionId}/attachments`,
    { headers: authHeaders(accessToken) },
  )
  if (!response.ok) {
    throw new Error(`Failed to load attachments: ${response.status}`)
  }
  return (await response.json()) as { attachments: RagSessionAttachment[] }
}
```

In `ui/src/components/chat/transports.ts`, extend `streamMessage`:

```typescript
streamMessage: (
  message: string,
  sessionId: string | null,
  accessToken: string,
  callbacks: StreamCallbacks,
  tools?: ToolConfig,
  files?: File[],
) => Promise<void>
```

Also add an optional attachment loader so `ChatThreadContainer` stays generic and does not need direct access to `agentId`:

```typescript
listSessionAttachments?: (
  sessionId: string,
  accessToken: string,
) => Promise<RagSessionAttachment[]>
```

Implement it for agent chat:

```typescript
listSessionAttachments: async (sessionId, accessToken) => {
  const res = await listRagAgentChatSessionAttachments(agentId, sessionId, accessToken)
  return res.attachments
},
```

Leave it undefined for workspace chat.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui && npm test -- --run ui/src/api/client.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/types.ts ui/src/api/client.ts ui/src/components/chat/transports.ts ui/src/api/client.test.ts
git commit -m "feat(ui): add agent chat attachment client support"
```

---

### Task 5: Chat UI for picking files, showing attachments, and sending them

**Files:**
- Modify: `ui/src/components/chat/ChatThreadContainer.tsx`
- Modify: `ui/src/components/agents/AgentChat.tsx`
- Test: `ui/src/components/agents/AgentChat.test.tsx` or `ui/src/components/chat/ChatThreadContainer.test.tsx`

- [ ] **Step 1: Write the failing UI test**

```typescript
it('passes selected files into the agent chat transport when sending', async () => {
  const streamMessage = vi.fn().mockResolvedValue(undefined)
  const transport: ChatTransport = {
    key: 'agent:1',
    listSessions: vi.fn().mockResolvedValue([]),
    loadSessionMessages: vi.fn(),
    streamMessage,
    deleteLastExchange: vi.fn(),
  }

  render(
    <ChatThreadContainer
      transport={transport}
      accessToken="token"
      activeSessionId={null}
      onSessionActivated={vi.fn()}
      onSessionsChanged={vi.fn()}
      title="Agent"
      emptyState="Ask Agent"
    />
  )

  await userEvent.upload(screen.getByLabelText(/attach files/i), [
    new File(['hello'], 'notes.txt', { type: 'text/plain' }),
  ])
  await userEvent.type(screen.getByRole('textbox'), 'Use this note')
  await userEvent.click(screen.getByRole('button', { name: /send/i }))

  expect(streamMessage).toHaveBeenCalledWith(
    'Use this note',
    null,
    'token',
    expect.any(Object),
    expect.any(Object),
    [expect.any(File)],
  )
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui && npm test -- --run ui/src/components/chat/ChatThreadContainer.test.tsx`

Expected: FAIL because there is no file picker and `streamMessage` does not receive files.

- [ ] **Step 3: Add the minimal file-picker and attachment state**

In `ChatThreadContainer.tsx`, add state:

```typescript
const [pendingFiles, setPendingFiles] = useState<File[]>([])
const [sessionAttachments, setSessionAttachments] = useState<RagSessionAttachment[]>([])
```

Add an input near the existing tool button:

```tsx
<label className="inline-flex cursor-pointer items-center">
  <input
    aria-label="Attach files"
    type="file"
    multiple
    className="sr-only"
    onChange={(event) => {
      const nextFiles = Array.from(event.target.files ?? [])
      setPendingFiles((prev) => [...prev, ...nextFiles])
      event.currentTarget.value = ''
    }}
  />
  <Button type="button" variant="outline" size="icon">
    <Paperclip className="h-4 w-4" />
  </Button>
</label>
```

Update the send call:

```typescript
await transport.streamMessage(
  question,
  streamedSessionId,
  accessToken,
  callbacks,
  toolConfig,
  pendingFiles,
)
setPendingFiles([])
```

After `onSession`, fetch attachments for the active session:

```typescript
const attachments = await transport.listSessionAttachments?.(nextSessionId, accessToken)
setSessionAttachments(attachments ?? [])
```

Render the attachment chips above the input or below the title:

```tsx
{sessionAttachments.length > 0 && (
  <div className="flex flex-wrap gap-2">
    {sessionAttachments.map((attachment) => (
      <Badge key={attachment.attachment_id} variant="secondary">
        {attachment.filename} · {attachment.state}
      </Badge>
    ))}
  </div>
)}
```

- [ ] **Step 4: Run the UI test to verify it passes**

Run: `cd ui && npm test -- --run ui/src/components/chat/ChatThreadContainer.test.tsx`

Expected: PASS

- [ ] **Step 5: Run focused regression tests**

Run: `uv run pytest -q tests/test_sessions_store.py tests/test_rag_chat_helpers.py tests/test_api.py`

Run: `cd ui && npm test -- --run ui/src/api/client.test.ts ui/src/components/chat/ChatThreadContainer.test.tsx`

Expected: PASS on both commands

- [ ] **Step 6: Commit**

```bash
git add ui/src/components/chat/ChatThreadContainer.tsx ui/src/components/agents/AgentChat.tsx
git add ui/src/types.ts ui/src/api/client.ts ui/src/components/chat/transports.ts
git add ui/src/components/chat/ChatThreadContainer.test.tsx ui/src/api/client.test.ts
git commit -m "feat(agent-chat): add session-scoped file uploads"
```

---

## Self-Review

- Spec coverage:
  - Session-scoped attachment model: Task 1
  - Same-turn ingestion and retrieval: Tasks 2-3
  - Attachment listing and UI status: Tasks 3-5
  - Hard delete on session removal: Task 3
  - Regression safety for plain text chat: Tasks 3 and 5
- Placeholder scan:
  - No `TBD`, `TODO`, or “similar to above” references remain
  - Each code-changing step includes concrete code or signatures
- Type consistency:
  - Backend uses `resource_id` on attachments consistently
  - Frontend uses `RagSessionAttachment` consistently
  - `streamMessage(..., tools?, files?)` is threaded in one direction only
