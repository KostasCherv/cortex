# Agent Chat Session Uploads — Design Spec

**Date:** 2026-06-11  
**Status:** Approved

---

## Overview

Add file uploads to agent conversations as session-scoped RAG attachments. Users can upload files on any turn, and files uploaded with a message must affect that same turn. Attachments remain available for later turns in the same chat session and are deleted, along with their indexed artifacts, when the session is deleted.

The design reuses the existing storage and ingestion pipeline, but introduces a dedicated session-attachment model instead of overloading workspace RAG resources.

---

## Goals

- Allow file uploads directly from agent chat
- Make new files available in the same turn they are uploaded with
- Keep uploaded files available across later turns in the same session
- Reuse existing RAG ingestion and retrieval infrastructure where possible
- Hard-delete attached files and derived artifacts when the chat session is deleted

## Non-goals

- Reusing session uploads outside the originating chat session
- Background ingestion that completes after the assistant has already answered
- Soft-delete, archival, or audit retention for session attachments
- File uploads for planner chat or other non-agent conversation surfaces

---

## Architecture

Introduce a new backend concept: session attachments. A session attachment is a stored and indexed file owned by a user and bound to a single agent chat `session_id`.

Agent chat retrieval becomes the union of:

- The agent's existing linked resource IDs
- The chat session's ready attachment resource IDs

Uploads are processed inline with the chat request:

1. Resolve or create the target chat session
2. Validate and persist uploaded files as session attachments
3. Ingest and index those files synchronously for the request
4. Run the chat turn against the merged retrieval set
5. Persist messages as normal

This keeps the behavior exact: files uploaded with a turn are guaranteed to influence that turn's answer.

---

## Data Model

### Backend domain model

Add a session attachment model parallel to `RagResource`.

```python
@dataclass
class RagSessionAttachment:
    attachment_id: str
    session_id: str
    agent_id: str
    owner_id: str
    workspace_id: str
    filename: str
    mime_type: str
    byte_size: int
    storage_uri: str
    state: str = "uploaded"   # uploaded | processing | ready | failed
    error_details: str | None = None
    created_at: str = ...
    updated_at: str = ...
```

The attachment state machine matches the existing resource ingestion lifecycle so current ingestion behavior can be reused with minimal branching.

### Persistence

Add store methods for:

- Creating a session attachment
- Updating attachment state and error details
- Listing attachments for a session
- Listing ready attachment IDs for a session
- Deleting all attachments for a session

Session attachments must stay separate from workspace RAG resources so session lifecycle and visibility rules remain unambiguous.

---

## API Design

### Agent chat request format

Extend the existing agent chat endpoints to accept `multipart/form-data` instead of JSON-only when files are present.

Endpoints:

- `POST /api/rag/agents/{agent_id}/chat`
- `POST /api/rag/agents/{agent_id}/chat/stream`

Form fields:

- `message`
- `session_id` optional
- `tools` optional, encoded the same way current agent chat settings are represented
- `files` repeated

Behavior:

- Requests without files continue to work as plain text agent chat
- Requests with files create session attachments before running retrieval
- If `session_id` is omitted, the backend creates the session first, then binds attachments to it

### Attachment listing

Add a session attachment read endpoint for the UI:

- `GET /api/rag/agents/{agent_id}/chat/sessions/{session_id}/attachments`

This returns filename, size, state, and timestamps so the UI can show the current session file set and any failures.

---

## Retrieval Flow

`prepare_agent_rag_chat` should be extended to merge resource IDs from:

- `agent.linked_resource_ids`
- Ready session attachment IDs for `chat_session_id`

The merged IDs are passed into the existing retrieval path so downstream prompt construction and citation handling remain unchanged.

This keeps the new behavior localized to session preparation instead of scattering attachment logic across the agent loop.

---

## UI Changes

Update the agent chat UI to support:

- Selecting files before sending a message
- Uploading files on any turn
- Showing all current session attachments in the conversation surface
- Showing attachment states: `uploading`, `processing`, `ready`, `failed`
- Re-uploading failed files on a later turn

`AgentChat` and the shared chat thread container should allow a send action that includes:

- Text message
- Current `session_id`
- Zero or more files

Streaming behavior must not emit assistant answer chunks until newly uploaded files have completed ingestion for that turn.

---

## Deletion Semantics

Deleting an agent chat session must cascade through:

- Session attachment records
- Stored file blobs
- Indexed ingestion artifacts for those attachments

This is a hard delete. Once the session is deleted, its uploaded files are no longer recoverable or visible.

---

## Error Handling

- If any uploaded file fails validation, return `400` with a per-file error and do not run the chat turn
- If storage or ingestion fails for any uploaded file, mark that attachment as `failed`, return an error payload for the turn, and do not generate an assistant response
- Existing ready attachments for the session remain usable on later turns even if a new upload fails
- Plain text turns without uploads continue to behave exactly as they do today

This design favors correctness over partial success. A turn that includes new files should either use all of them successfully or fail before answer generation.

---

## Testing

### Backend

- Validation tests for supported file types and size limits
- Store tests for attachment CRUD and ready-only lookups
- Retrieval preparation tests proving session attachment IDs are merged with agent-linked resource IDs
- Deletion tests proving session delete cascades to attachments and artifacts

### API

- Multipart agent chat request succeeds with uploaded files
- Uploaded files affect the same turn they were sent with
- Existing session attachments remain available on later turns
- Upload failure blocks assistant response for that turn
- Plain text agent chat remains backward compatible

### UI

- File selection and send flow
- Attachment status rendering for `uploading`, `processing`, `ready`, `failed`
- Session attachment list persists across later turns in the same chat
- Failed upload can be retried by re-uploading the file

---

## Files Expected To Change

| File | Change |
|------|--------|
| `src/rag.py` | Add session attachment model and lifecycle helpers |
| `src/api/endpoints.py` | Accept multipart agent chat requests and add attachment listing endpoint |
| `src/api/rag_chat_helpers.py` | Merge session attachment IDs into retrieval preparation |
| `src/db/supabase_store.py` | Add persistence methods for session attachments and delete cascade support |
| `ui/src/components/agents/AgentChat.tsx` | Enable file selection and attachment display |
| `ui/src/components/chat/ChatThreadContainer.tsx` | Support sending files with a chat turn |
| `ui/src/components/chat/transports.ts` | Send multipart requests for agent chat when files are present |
| `tests/test_api.py` | Add multipart upload and same-turn retrieval coverage |
| `tests/test_sessions_store.py` | Add attachment persistence coverage |
| `tests/test_rag_chat_helpers.py` | Add merged resource selection coverage |

---

## Tradeoffs

- Inline ingestion increases latency for turns that include uploads, but preserves the required same-turn guarantee
- A separate session-attachment model adds persistence surface area, but keeps workspace resources and session-only files clearly separated
- Blocking answer generation on upload failure is stricter than partial success, but avoids ambiguous assistant behavior when the user expects a file to be included
