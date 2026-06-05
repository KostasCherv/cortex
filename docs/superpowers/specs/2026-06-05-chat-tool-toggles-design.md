# Chat Tool Toggles — Design Spec

**Date:** 2026-06-05  
**Status:** Approved

---

## Overview

Add a `+` button to the chat input area that opens a tool-selection popover. Users can enable or disable individual tools (web search, Composio integrations) per chat session. Web search is on by default; Composio is off by default. The design is intentionally extensible — adding a new tool requires one entry in a registry plus a matching backend field.

---

## Goals

- Faster, cleaner responses when users don't need external tools
- Explicit user control instead of opaque keyword-based auto-detection
- Architecture that makes adding new tools trivial

## Non-goals

- Persisting tool preferences across sessions (localStorage / DB) — per-session only
- Per-message tool selection
- Exposing RAG / document retrieval as a toggle (it's always on)
- Per-app Composio toggles (single "Connected apps" switch covers all)

---

## Data Model

### Frontend — Tool Registry (`ui/src/components/chat/toolConfig.ts`)

```typescript
import type { LucideIcon } from 'lucide-react'

export type ToolDefinition = {
  id: string          // must match backend key exactly
  label: string
  icon: LucideIcon
  defaultEnabled: boolean
}

export type ToolConfig = Record<string, boolean>

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  { id: 'web_search',  label: 'Web search',     icon: Globe,  defaultEnabled: true  },
  { id: 'composio',    label: 'Connected apps',  icon: Puzzle, defaultEnabled: false },
]

export function defaultToolConfig(): ToolConfig {
  return Object.fromEntries(TOOL_DEFINITIONS.map((t) => [t.id, t.defaultEnabled]))
}
```

**To add a new tool:** append one entry to `TOOL_DEFINITIONS`. The UI renders it automatically.

### Backend — Request Model (`src/api/endpoints.py`)

```python
class RagChatTools(BaseModel):
    web_search: bool = True
    composio: bool = False

class RagChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    tools: RagChatTools = Field(default_factory=RagChatTools)
```

Applies to both workspace chat (`/api/rag/chat/stream`) and agent chat (`/api/rag/agents/{id}/chat/stream`).

---

## UI Components

### `ToolMenuButton` (`ui/src/components/chat/ToolMenuButton.tsx`)

Props:
```typescript
type Props = {
  toolConfig: ToolConfig
  onToggle: (id: string, enabled: boolean) => void
  disabled?: boolean
}
```

Renders a `+` icon button. When any non-default tool state is active, shows a small blue dot indicator on the button. Clicking opens a `Popover` listing all entries in `TOOL_DEFINITIONS`, each with icon, label, and a `Switch`.

Uses existing shadcn/ui primitives: `Popover`, `Switch`, `Button`.

### `ChatThreadContainer` changes

- Add state: `const [toolConfig, setToolConfig] = useState<ToolConfig>(defaultToolConfig)`
- Reset `toolConfig` to defaults when `transport.key` changes (effect already exists for this)
- Render `<ToolMenuButton>` to the left of the `Textarea`, inside the existing input row
- Pass `toolConfig` to `transport.streamMessage(..., toolConfig)`

---

## Data Flow

```
ToolMenuButton (toggle)
      ↓
toolConfig state in ChatThreadContainer
      ↓
transport.streamMessage(message, sessionId, token, callbacks, toolConfig)
      ↓
streamRagWorkspaceChat / streamRagAgentChat — adds tools to request body
      ↓
POST { message, session_id, tools: { web_search, composio } }
      ↓
RagChatRequest.tools parsed by FastAPI
      ↓
prepare_*_rag_chat — hard overrides (bypass should_bind_composio_tools)
  · bind_tools = request.tools.composio and settings.composio_enabled
  · web_search allowed = request.tools.web_search
      ↓
Agent graph / LLM execution respects flags
```

### Web search override

The `tools.web_search` flag is passed into the agent graph. When `False`, a prompt-level instruction tells the LLM not to invoke the `web_search` action. This is the least-invasive hook given the current graph architecture.

### Composio override

`bind_tools` in `RagChatPrepared` is set directly from `request.tools.composio`. The existing `should_bind_composio_tools` auto-detection is **skipped** — user intent wins.

---

## Transport Interface Change

```typescript
// transports.ts
export type ChatTransport = {
  ...
  streamMessage: (
    message: string,
    sessionId: string | null,
    accessToken: string,
    callbacks: StreamCallbacks,
    tools?: ToolConfig,
  ) => Promise<void>
}
```

The `tools` param is optional with the existing default (`defaultToolConfig()`) so existing callers (e.g. AgentChat) continue to work unchanged until updated.

---

## Files Changed

| File | Change |
|------|--------|
| `ui/src/components/chat/toolConfig.ts` | New — tool registry and helpers |
| `ui/src/components/chat/ToolMenuButton.tsx` | New — `+` button + popover |
| `ui/src/components/chat/ChatThreadContainer.tsx` | Add tool state, render button, pass to transport |
| `ui/src/components/chat/transports.ts` | Add `tools?` param to `streamMessage` |
| `ui/src/components/agents/AgentChat.tsx` | Update transport call if needed |
| `ui/src/api/client.ts` | Pass `tools` in request body for both stream functions |
| `src/api/endpoints.py` | Add `RagChatTools`, extend `RagChatRequest` |
| `src/api/rag_chat_helpers.py` | Accept `tools` in `prepare_*` functions, apply overrides |

---

## Testing

- Unit: `defaultToolConfig()` returns correct defaults; `ToolMenuButton` toggles fire correct callbacks
- Integration: POST with `tools.composio=false` does not bind tools even when composio is enabled
- Integration: POST with `tools.web_search=false` does not trigger web search on queries containing external-intent markers
- E2E: toggle web search off → send a URL-containing message → `web_used` badge does not appear
