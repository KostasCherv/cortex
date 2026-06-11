# Code Review — Session-Scoped File Uploads (`agent chat`)

**Date:** 2026-06-11
**Scope:** `src/api/endpoints.py`, `src/api/rag_chat_helpers.py`, `src/db/supabase_store.py`, `src/rag.py`, `tests/test_api.py`, `tests/test_rag_chat_helpers.py`, `tests/test_sessions_store.py`

---

## Overview

The changeset adds session-scoped file uploads to the agent chat endpoints. Files are ingested synchronously before the answer loop runs. A new `rag_chat_session_attachments` table stores attachment state, and session attachment resource IDs are merged into the retrieval context.

---

## CRITICAL

### 1. `RagValidationError` from upload ingestion not caught in non-stream handler

**File:** `src/api/endpoints.py` — `rag_chat_with_agent`

The `if files: await ingest_agent_chat_session_uploads(...)` block sits **outside** the `try/except` block that handles `RagValidationError`. If `_validate_upload` raises `RagValidationError`, it propagates uncaught past the inner except block (which only fires during the agent loop) and returns a 500 instead of a 400. The stream handler has the same structural problem.

**Fix:** Move the `if files:` ingestion call inside the `try:` block, or add a dedicated `except RagValidationError` guard around the ingestion call for both endpoints.

---

### 2. Silent exception swallowing in `delete_rag_chat_session_attachments_and_artifacts`

**File:** `src/rag.py:666-667, 671-672`

Two `except Exception: pass` blocks. Per project guidelines, exceptions must never be silently swallowed. Failures to delete storage objects or resource artifacts during session deletion will be invisible in logs and monitoring.

**Fix:** Log the error at `WARNING` level instead of `pass`:
```python
except Exception as exc:
    logger.warning("Failed to delete resource artifact %s: %s", resource_id, exc)
```

---

### 3. `RagValidationError` handling breaks `end_workflow_run` telemetry

**File:** `src/api/endpoints.py` — `rag_chat_with_agent` except block

```python
except Exception as exc:
    if isinstance(exc, RagValidationError):
        _raise_rag_validation_error(exc)   # raises HTTPException — exits here
    end_workflow_run(trace_ctx, status="error", ...)  # never reached for RagValidationError
    raise
```

When `_raise_rag_validation_error` is called it raises an `HTTPException`, exiting the except block before `end_workflow_run` runs. Workflow runs are not properly closed in the error telemetry path.

**Fix:** Call `end_workflow_run` before re-raising:
```python
except Exception as exc:
    end_workflow_run(trace_ctx, status="error", error=_workflow_error_text(exc))
    if isinstance(exc, RagValidationError):
        _raise_rag_validation_error(exc)
    raise
```

---

## HIGH

### 4. Private `_request` method accessed directly from module-level functions

**File:** `src/rag.py:463, 495, 608`

`_create_rag_chat_session_attachment`, `_update_rag_chat_session_attachment`, and `list_rag_chat_session_attachments` all call `get_session_store()._request(...)` with `# type: ignore[attr-defined]`. This breaks the store abstraction. The two new `SupabaseSessionStore` methods in `supabase_store.py` (`list_ready_...` and `delete_...`) correctly follow the pattern — the three private-access callers should also become proper public methods on the store.

**Fix:** Add `create_rag_chat_session_attachment`, `update_rag_chat_session_attachment`, and `list_rag_chat_session_attachments` as public methods on `SupabaseSessionStore`, and call them through the store interface.

---

### 5. Partial failure leaves orphaned ready attachments

**File:** `src/rag.py` — `ingest_agent_chat_session_uploads`

When processing N files in a loop, if file K fails, files 1..K-1 are already in `state="ready"` with ingested resources. The function raises immediately without rolling back the already-processed attachments. The session ends up with an incomplete and inconsistent attachment set.

**Fix:** Collect successfully processed attachments and, on failure, clean up those attachments (delete their resources and storage objects) before re-raising.

---

### 6. Missing `@pytest.mark.asyncio` on two store tests

**File:** `tests/test_sessions_store.py:1093, 1122`

`test_store_lists_ready_rag_chat_session_attachment_resource_ids` and `test_store_deletes_session_attachments_for_chat_session` are `async def` functions without `@pytest.mark.asyncio`. Depending on pytest-asyncio mode, these tests may silently not run as coroutines.

**Fix:** Add `@pytest.mark.asyncio` to both.

---

## MEDIUM

### 7. `RagChatTools` JSON parse in multipart path has no error handling

**File:** `src/api/endpoints.py` — `_parse_rag_chat_request`

```python
tools = RagChatTools.model_validate_json(form["tools"])
```

A malformed `tools` JSON value in a multipart request raises `pydantic.ValidationError` uncaught, returning a 500. The JSON path already wraps its error in `RequestValidationError(exc.errors())`.

**Fix:**
```python
try:
    tools = RagChatTools.model_validate_json(form["tools"])
except ValidationError as exc:
    raise RequestValidationError(exc.errors()) from exc
```

---

### 8. `RagSessionAttachment` is mutable; state is mutated in-place

**File:** `src/rag.py:404-438`

Project guidelines require `@dataclass(frozen=True)` for immutable data. The attachment dataclass is mutable, and `ingest_agent_chat_session_uploads` mutates `attachment.state`, `attachment.error_details`, and `attachment.updated_at` after creation.

**Fix:** Use a local dict or replace-pattern when tracking state transitions during ingestion, keeping `RagSessionAttachment` as `frozen=True`.

---

### 9. Missing test for file uploads on the non-stream endpoint

**File:** `tests/test_api.py`

`test_rag_agent_chat_stream_uploads_files_before_running_loop` covers the stream path only. The non-stream `POST /api/rag/agents/{agent_id}/chat` with file uploads has no equivalent happy-path test. This gap directly enabled Issue #1 to go undetected.

**Fix:** Add a parallel test for the non-stream endpoint covering:
- Files are ingested before the loop
- `prepare_agent_rag_chat` is called twice
- `RagValidationError` during ingestion returns 400 not 500

---

## LOW

### 10. `form.get("tools")` called twice

**File:** `src/api/endpoints.py` — `_parse_rag_chat_request`

```python
if isinstance(form.get("tools"), str) and form.get("tools")
```

**Fix:** Extract to `tools_raw = form.get("tools")` and use the variable.

---

### 11. `owner_id` exposed in public API response

**File:** `src/rag.py:430`, `src/api/endpoints.py` — list attachments endpoint

`to_dict()` includes `owner_id` which is serialized in the attachments list response. Clients already know their own user ID.

**Fix:** Omit `owner_id` from `to_dict()` or strip it in the endpoint response.

---

### 12. Extra blank line in test file

**File:** `tests/test_api.py:713`

Double blank line between `_fake_prepared_chat` helper and `test_rag_chat_calls_agent_loop`. PEP 8 convention is two blank lines between top-level definitions.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | CRITICAL | `endpoints.py` | `RagValidationError` from ingestion not caught in non-stream handler → 500 |
| 2 | CRITICAL | `rag.py` | Silent `except: pass` in delete cleanup — violates project guidelines |
| 3 | CRITICAL | `endpoints.py` | `end_workflow_run` skipped when `RagValidationError` re-raised |
| 4 | HIGH | `rag.py` | `_request` accessed directly, bypassing store abstraction |
| 5 | HIGH | `rag.py` | Partial multi-file failure leaves orphaned ready attachments |
| 6 | HIGH | `test_sessions_store.py` | Missing `@pytest.mark.asyncio` on two async tests |
| 7 | MEDIUM | `endpoints.py` | Multipart `tools` JSON parse missing error handling → 500 on bad input |
| 8 | MEDIUM | `rag.py` | Mutable dataclass + in-place state mutation violates immutability guideline |
| 9 | MEDIUM | `test_api.py` | No upload test for non-stream endpoint (gap that hid Issue #1) |
| 10 | LOW | `endpoints.py` | `form.get("tools")` called twice |
| 11 | LOW | `rag.py` / `endpoints.py` | `owner_id` exposed in public API response |
| 12 | LOW | `test_api.py` | Extra blank line (style) |
