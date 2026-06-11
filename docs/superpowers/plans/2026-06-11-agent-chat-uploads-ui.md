# Plan: Agent Chat File Upload UI Integration

**Date:** 2026-06-11  
**Scope:** `ui/src/` only — no backend changes needed.

---

## Context

The backend already accepts `multipart/form-data` on `POST /api/rag/agents/{id}/chat/stream` with a `files` field alongside `message`, `session_id`, and `tools`. Workspace chat does not support uploads. The UI stack is React 19 + Vite + Shadcn.

---

## Tasks

### Task 1 — Extend `ChatTransport` and `streamRagAgentChat`

**Files:** `ui/src/components/chat/transports.ts`, `ui/src/api/client.ts`

1. Add `supportsFileUpload?: boolean` to the `ChatTransport` type.
2. Add `files?: File[]` to `streamMessage` signature on `ChatTransport`.
3. Set `supportsFileUpload: true` on `createAgentChatTransport`; leave workspace transport without it.
4. In `streamRagAgentChat` (`client.ts`), accept `files?: File[]`:
   - If `files` is non-empty: build a `FormData` with `message`, `session_id` (if set), `tools` (JSON string), and one `files` entry per file. Drop the `Content-Type` header so the browser sets the multipart boundary.
   - If no files: keep the existing `JSON.stringify` path unchanged.

---

### Task 2 — Add file attachment state to `ChatThreadContainer`

**File:** `ui/src/components/chat/ChatThreadContainer.tsx`

1. Add state: `pendingFiles: File[]` (default `[]`).
2. Add a hidden `<input type="file" multiple accept=".pdf,.txt,..." ref={fileInputRef} />` (accepted MIME types should match backend: `application/pdf`, `text/plain`, etc.).
3. Add a paperclip `Button` (icon-only, from `lucide-react`) to the left of the `Textarea` — only rendered when `transport.supportsFileUpload === true`.
4. Below the textarea row (above the input bar border), show a horizontal list of file chips when `pendingFiles.length > 0`:
   - Each chip: filename (truncated to ~20 chars) + file size + `×` remove button.
   - Use existing `Badge` or a small inline `div` — no new UI primitives needed.
5. Reset `pendingFiles` to `[]` when transport changes (add to the existing transport-change `useEffect`).

---

### Task 3 — Thread files through `send()`

**File:** `ui/src/components/chat/ChatThreadContainer.tsx`

1. Capture `pendingFiles` at the top of `send()` before any state mutation.
2. Clear `setPendingFiles([])` alongside `setInput('')`.
3. Show attached filenames in the optimistic user message bubble: append a small "📎 filename, filename2" line below `content` — or add a `attachmentNames?: string[]` field to the optimistic message and render it conditionally. (Keep `RagChatMessage` type unchanged; use a local-only display approach.)
4. Pass captured files to `transport.streamMessage(question, sessionId, accessToken, callbacks, toolConfig, files)`.
5. In `createAgentChatTransport.streamMessage`, forward `files` to `streamRagAgentChat`.

---

### Task 4 — Add `listRagAgentChatSessionAttachments` to `client.ts`

**File:** `ui/src/api/client.ts`

Add a typed API function:
```ts
export async function listRagAgentChatSessionAttachments(
  agentId: string,
  sessionId: string,
  accessToken: string | null,
): Promise<{ attachments: SessionAttachment[] }>
```
targeting `GET /api/rag/agents/{agentId}/chat/sessions/{sessionId}/attachments`.

Add `SessionAttachment` to `ui/src/types.ts`:
```ts
export type SessionAttachment = {
  attachment_id: string
  filename: string
  mime_type: string
  byte_size: number
  state: 'uploaded' | 'processing' | 'ready' | 'failed'
  error_details?: string | null
  created_at: string
}
```

*(This function is wired in Task 5; not needed before then.)*

---

### Task 5 — Show session attachments when loading a session

**File:** `ui/src/components/chat/ChatThreadContainer.tsx`

When `openSession` resolves, if `transport.supportsFileUpload`, call `listRagAgentChatSessionAttachments` and store the result in `sessionAttachments: SessionAttachment[]` state. Display a compact attachment shelf at the top of the message thread (or just below the header) listing filename + state badge (`ready` = green, `failed` = red, `processing` = muted spinner). Clear `sessionAttachments` on transport change.

This is a read-only view — no re-upload from this list in scope.

---

## File change summary

| File | Change |
|---|---|
| `ui/src/api/client.ts` | `streamRagAgentChat` accepts `files`; multipart path; new `listRagAgentChatSessionAttachments` |
| `ui/src/types.ts` | Add `SessionAttachment` type |
| `ui/src/components/chat/transports.ts` | `ChatTransport.supportsFileUpload?`; `streamMessage` accepts `files?`; agent transport sets flag + forwards files |
| `ui/src/components/chat/ChatThreadContainer.tsx` | File picker button, pending-files chips, `send()` threading, session attachment shelf |

---

## Out of scope

- Re-uploading a failed attachment from the UI (backend supports it but not planned here)
- Drag-and-drop file upload
- Workspace chat uploads (backend doesn't support it)
- Progress bar during upload (streaming response starts only after upload completes server-side)
