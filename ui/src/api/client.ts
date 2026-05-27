import type {
  Citation,
  FollowupStreamEvent,
  HealthResponse,
  RagAgent,
  BillingUsageSummary,
  RagCitation,
  RagChatStreamEvent,
  RagAgentDraft,
  RagChatMessage,
  RagChatSessionSummary,
  RagResource,
  ResearchRequest,
  RunFeedbackRequest,
  SessionRunStreamEvent,
  SessionDetail,
  SessionSummary,
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

type SessionRunStreamOptions = {
  signal?: AbortSignal
  onEvent: (event: SessionRunStreamEvent) => void
  onDone?: () => void
}

function authHeaders(accessToken: string | null): HeadersInit {
  if (!accessToken) {
    return {}
  }
  return { Authorization: `Bearer ${accessToken}` }
}

async function parseApiError(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?:
        | string
        | {
            code?: string
            message?: string
            limit_type?: string
            limit?: number
            used?: number
          }
    }
    if (typeof body.detail === 'string' && body.detail.trim()) return body.detail
    if (body.detail && typeof body.detail === 'object') {
      if (body.detail.code === 'quota_exceeded') {
        const limitType =
          body.detail.limit_type === 'research_daily' ? 'daily research queries' : 'daily questions'
        const used = typeof body.detail.used === 'number' ? body.detail.used : 0
        const limit = typeof body.detail.limit === 'number' ? body.detail.limit : 0
        return `Quota reached for ${limitType} (${used}/${limit}). Upgrade to Pro or wait for reset.`
      }
      if (typeof body.detail.message === 'string' && body.detail.message.trim()) return body.detail.message
    }
  } catch {
    // Keep fallback.
  }
  return fallback
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`)
  }
  return (await response.json()) as HealthResponse
}

// ---------------------------------------------------------------------------
// Session API
// ---------------------------------------------------------------------------

export async function createSession(
  accessToken: string | null,
  query?: string,
): Promise<{ session_id: string; title: string; created_at: string }> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ query: query ?? null }),
  })
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; title: string; created_at: string }
}

export async function getSession(sessionId: string, accessToken: string | null): Promise<SessionDetail> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Session not found: ${response.status}`)
  }
  return (await response.json()) as SessionDetail
}

export async function listSessions(
  accessToken: string | null,
): Promise<{ sessions: SessionSummary[] }> {
  const response = await fetch(`${API_BASE}/sessions`, {
    headers: authHeaders(accessToken),
  })
  // Backward compatibility while backend is restarting/updating.
  // 401 is treated as empty so optional sidebar loading never hard-fails the UI.
  if (response.status === 401 || response.status === 404 || response.status === 405) {
    return { sessions: [] }
  }
  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.status}`)
  }
  return (await response.json()) as { sessions: SessionSummary[] }
}

export async function updateSessionTitle(
  sessionId: string,
  title: string,
  accessToken: string | null,
): Promise<{ session_id: string; title: string }> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) {
    throw new Error(`Failed to rename session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; title: string }
}

export async function deleteSession(
  sessionId: string,
  accessToken: string | null,
): Promise<{ session_id: string; deleted: boolean }> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; deleted: boolean }
}

type FollowupOptions = {
  signal?: AbortSignal
  onChunk: (text: string) => void
  onCitations: (citations: Citation[]) => void
  onSuggestions?: (suggestions: string[]) => void
  onDone: () => void
  onError?: (error: string) => void
}

export async function streamFollowup(
  sessionId: string,
  question: string,
  runId: string | null,
  accessToken: string | null,
  options: FollowupOptions,
): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/followup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ question, run_id: runId }),
    signal: options.signal,
  })

  if (!response.ok) {
    throw new Error(`Followup request failed: ${response.status}`)
  }
  if (!response.body) {
    throw new Error('Streaming not supported.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const handleFollowupEvent = (parsed: FollowupStreamEvent): boolean => {
    if (parsed.type === 'chunk') {
      options.onChunk(parsed.text)
      return false
    }
    if (parsed.type === 'citations') {
      options.onCitations(parsed.citations)
      return false
    }
    if (parsed.type === 'suggestions') {
      options.onSuggestions?.(parsed.suggestions)
      return false
    }
    if (parsed.type === 'done') {
      options.onDone()
      return true
    }
    if (parsed.type === 'error') {
      options.onError?.(parsed.error)
      return true
    }
    return false
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''

    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      let parsed: FollowupStreamEvent
      try {
        parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as FollowupStreamEvent
      } catch {
        continue
      }
      if (handleFollowupEvent(parsed)) {
        return
      }
    }
  }

  if (buffer.trim()) {
    const dataLine = buffer
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line.startsWith('data:'))
    if (dataLine) {
      try {
        const parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as FollowupStreamEvent
        if (handleFollowupEvent(parsed)) {
          return
        }
      } catch {
        // Ignore trailing partial event and report abnormal termination below.
      }
    }
  }

  options.onError?.('Follow-up stream ended before a terminal event was received.')
}

export async function startSessionResearch(
  sessionId: string,
  payload: ResearchRequest,
  accessToken: string | null,
): Promise<{ run_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/research`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(await parseApiError(response, `Session research failed: ${response.status}`))
  }
  return (await response.json()) as { run_id: string; status: string }
}

export async function streamSessionRun(
  sessionId: string,
  runId: string,
  accessToken: string | null,
  options: SessionRunStreamOptions,
): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/runs/${runId}/stream`, {
    method: 'GET',
    headers: {
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    signal: options.signal,
  })
  if (!response.ok) {
    throw new Error(`Session run stream failed: ${response.status}`)
  }
  if (!response.body) {
    throw new Error('Streaming not supported by this browser response.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''

    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      let parsed: SessionRunStreamEvent
      try {
        parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as SessionRunStreamEvent
      } catch {
        continue
      }
      options.onEvent(parsed)
      if (parsed.type === 'done' || parsed.type === 'error') {
        options.onDone?.()
        return
      }
    }
  }
  options.onDone?.()
}

export async function submitRunFeedback(
  sessionId: string,
  runId: string,
  payload: RunFeedbackRequest,
  accessToken: string | null,
): Promise<{ session_id: string; run_id: string; feedback_submitted_at: string; feedback_helpful: boolean }> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/runs/${runId}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    let detail = `Failed to submit feedback: ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) {
        detail = body.detail
      }
    } catch {
      // Keep the default error message if the response is not JSON.
    }
    throw new Error(detail)
  }

  return (await response.json()) as {
    session_id: string
    run_id: string
    feedback_submitted_at: string
    feedback_helpful: boolean
  }
}

// ---------------------------------------------------------------------------
// RAG Agent API
// ---------------------------------------------------------------------------

export async function uploadRagResource(
  file: File,
  accessToken: string | null,
): Promise<{ resource: RagResource }> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE}/api/rag/resources/upload`, {
    method: 'POST',
    headers: authHeaders(accessToken),
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`Failed to upload resource: ${response.status}`)
  }
  return (await response.json()) as { resource: RagResource }
}

export async function listRagResources(
  accessToken: string | null,
): Promise<{ resources: RagResource[] }> {
  const response = await fetch(`${API_BASE}/api/rag/resources`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to load resources: ${response.status}`)
  }
  return (await response.json()) as { resources: RagResource[] }
}

export async function getRagResourceStatus(
  resourceId: string,
  accessToken: string | null,
): Promise<{ resource: RagResource }> {
  const response = await fetch(`${API_BASE}/api/rag/resources/${resourceId}/status`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to fetch resource status: ${response.status}`)
  }
  return (await response.json()) as { resource: RagResource }
}

export async function deleteRagResource(
  resourceId: string,
  accessToken: string | null,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/resources/${resourceId}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete resource: ${response.status}`)
  }
}

type RagAgentPayload = {
  name: string
  description: string
  system_instructions: string
  linked_resource_ids: string[]
}

export async function generateRagAgentDraft(
  prompt: string,
  accessToken: string | null,
): Promise<{ draft: RagAgentDraft }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/draft`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ prompt }),
  })
  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to generate agent draft: ${response.status}`))
  }
  return (await response.json()) as { draft: RagAgentDraft }
}

export async function createRagAgent(
  payload: RagAgentPayload,
  accessToken: string | null,
): Promise<{ agent: RagAgent }> {
  const response = await fetch(`${API_BASE}/api/rag/agents`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error(`Failed to create agent: ${response.status}`)
  }
  return (await response.json()) as { agent: RagAgent }
}

export async function deleteRagAgent(
  agentId: string,
  accessToken: string | null,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete agent: ${response.status}`)
  }
}

export async function listRagAgents(
  accessToken: string | null,
): Promise<{ agents: RagAgent[] }> {
  const response = await fetch(`${API_BASE}/api/rag/agents`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to list agents: ${response.status}`)
  }
  return (await response.json()) as { agents: RagAgent[] }
}

export async function updateRagAgent(
  agentId: string,
  payload: Partial<RagAgentPayload>,
  accessToken: string | null,
): Promise<{ agent: RagAgent }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error(`Failed to update agent: ${response.status}`)
  }
  return (await response.json()) as { agent: RagAgent }
}

export async function linkRagAgentResources(
  agentId: string,
  resourceIds: string[],
  accessToken: string | null,
): Promise<{ agent: RagAgent }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/resources:link`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ resource_ids: resourceIds }),
  })
  if (!response.ok) {
    throw new Error(`Failed to link resources: ${response.status}`)
  }
  return (await response.json()) as { agent: RagAgent }
}

export async function chatWithRagAgent(
  agentId: string,
  message: string,
  sessionId: string | null,
  accessToken: string | null,
): Promise<{ session_id: string; messages: RagChatMessage[] }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to chat with RAG agent: ${response.status}`))
  }
  const parsed = (await response.json()) as {
    session_id: string
    messages: Array<Omit<RagChatMessage, 'citations' | 'suggestions'> & { citations?: unknown; suggestions?: unknown }>
  }
  return {
    session_id: parsed.session_id,
    messages: parsed.messages.map((message) => ({
      ...message,
      citations: normalizeRagCitations(message.citations),
      suggestions: normalizeSuggestions(message.suggestions),
    })),
  }
}

type RagAgentChatStreamOptions = {
  signal?: AbortSignal
  onSession: (sessionId: string, webUsed?: boolean, webProvider?: string | null) => void
  onChunk: (text: string) => void
  onCitations: (citations: RagCitation[]) => void
  onSuggestions?: (suggestions: string[]) => void
  onDone: () => void
  onError?: (error: string) => void
}

function normalizeRagCitation(citation: Partial<RagCitation> | null | undefined): RagCitation | null {
  if (
    typeof citation?.source_title !== 'string' ||
    typeof citation?.source_url !== 'string' ||
    typeof citation?.chunk_id !== 'string' ||
    typeof citation?.text !== 'string'
  ) {
    return null
  }
  return {
    source_title: citation.source_title,
    source_url: citation.source_url,
    chunk_id: citation.chunk_id,
    text: citation.text,
  }
}

function normalizeRagCitations(citations: unknown): RagCitation[] {
  if (!Array.isArray(citations)) {
    return []
  }
  return citations.flatMap((citation) => {
    const normalized = normalizeRagCitation(citation as Partial<RagCitation>)
    return normalized ? [normalized] : []
  })
}

function normalizeSuggestions(suggestions: unknown): string[] {
  if (!Array.isArray(suggestions)) {
    return []
  }
  return suggestions.filter((value): value is string => typeof value === 'string')
}

export async function streamRagAgentChat(
  agentId: string,
  message: string,
  sessionId: string | null,
  accessToken: string | null,
  options: RagAgentChatStreamOptions,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal: options.signal,
  })

  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to stream RAG agent chat: ${response.status}`))
  }
  if (!response.body) {
    throw new Error('Streaming not supported.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const handleEvent = (parsed: RagChatStreamEvent): boolean => {
    if (parsed.type === 'session') {
      options.onSession(parsed.session_id, parsed.web_used, parsed.web_provider)
      return false
    }
    if (parsed.type === 'chunk') {
      options.onChunk(parsed.text)
      return false
    }
    if (parsed.type === 'citations') {
      options.onCitations(normalizeRagCitations(parsed.citations))
      return false
    }
    if (parsed.type === 'suggestions') {
      options.onSuggestions?.(parsed.suggestions)
      return false
    }
    if (parsed.type === 'done') {
      options.onDone()
      return true
    }
    if (parsed.type === 'error') {
      options.onError?.(parsed.error)
      return true
    }
    return false
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''

    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      let parsed: RagChatStreamEvent
      try {
        parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as RagChatStreamEvent
      } catch {
        continue
      }
      if (handleEvent(parsed)) return
    }
  }

  if (buffer.trim()) {
    const dataLine = buffer
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line.startsWith('data:'))
    if (dataLine) {
      try {
        const parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as RagChatStreamEvent
        if (handleEvent(parsed)) return
      } catch {
        // Ignore trailing partial event and report abnormal termination below.
      }
    }
  }

  options.onError?.('Chat stream ended before a terminal event was received.')
}

export async function streamRagWorkspaceChat(
  message: string,
  sessionId: string | null,
  accessToken: string | null,
  options: RagAgentChatStreamOptions,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal: options.signal,
  })

  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to stream workspace chat: ${response.status}`))
  }
  if (!response.body) {
    throw new Error('Streaming not supported.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const handleEvent = (parsed: RagChatStreamEvent): boolean => {
    if (parsed.type === 'session') {
      options.onSession(parsed.session_id, parsed.web_used, parsed.web_provider)
      return false
    }
    if (parsed.type === 'chunk') {
      options.onChunk(parsed.text)
      return false
    }
    if (parsed.type === 'citations') {
      options.onCitations(normalizeRagCitations(parsed.citations))
      return false
    }
    if (parsed.type === 'suggestions') {
      options.onSuggestions?.(parsed.suggestions)
      return false
    }
    if (parsed.type === 'done') {
      options.onDone()
      return true
    }
    if (parsed.type === 'error') {
      options.onError?.(parsed.error)
      return true
    }
    return false
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''
    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      let parsed: RagChatStreamEvent
      try {
        parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as RagChatStreamEvent
      } catch {
        continue
      }
      if (handleEvent(parsed)) return
    }
  }
  options.onError?.('Chat stream ended before a terminal event was received.')
}

export async function listRagAgentChatSessions(
  agentId: string,
  accessToken: string | null,
): Promise<{ sessions: RagChatSessionSummary[] }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/sessions`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to list RAG agent chat sessions: ${response.status}`)
  }
  return (await response.json()) as { sessions: RagChatSessionSummary[] }
}

export async function listRagWorkspaceChatSessions(
  accessToken: string | null,
): Promise<{ sessions: RagChatSessionSummary[] }> {
  const response = await fetch(`${API_BASE}/api/rag/chat/sessions`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to list workspace chat sessions: ${response.status}`)
  }
  return (await response.json()) as { sessions: RagChatSessionSummary[] }
}

export async function getRagAgentChatSessionMessages(
  agentId: string,
  sessionId: string,
  accessToken: string | null,
): Promise<{ session_id: string; agent_id: string; messages: RagChatMessage[] }> {
  const response = await fetch(
    `${API_BASE}/api/rag/agents/${agentId}/chat/sessions/${sessionId}/messages`,
    {
      headers: authHeaders(accessToken),
    },
  )
  if (!response.ok) {
    throw new Error(`Failed to load RAG agent chat session: ${response.status}`)
  }
  const parsed = (await response.json()) as {
    session_id: string
    agent_id: string
    messages: Array<Omit<RagChatMessage, 'citations'> & { citations?: unknown }>
  }
  return {
    session_id: parsed.session_id,
    agent_id: parsed.agent_id,
    messages: parsed.messages.map((message) => ({
      ...message,
      citations: normalizeRagCitations(message.citations),
      suggestions: normalizeSuggestions(message.suggestions),
    })),
  }
}

export async function getRagWorkspaceChatSessionMessages(
  sessionId: string,
  accessToken: string | null,
): Promise<{ session_id: string; agent_id: string | null; messages: RagChatMessage[] }> {
  const response = await fetch(`${API_BASE}/api/rag/chat/sessions/${sessionId}/messages`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to load workspace chat session: ${response.status}`)
  }
  const parsed = (await response.json()) as {
    session_id: string
    agent_id: string | null
    messages: Array<Omit<RagChatMessage, 'citations'> & { citations?: unknown }>
  }
  return {
    session_id: parsed.session_id,
    agent_id: parsed.agent_id,
    messages: parsed.messages.map((message) => ({
      ...message,
      citations: normalizeRagCitations(message.citations),
      suggestions: normalizeSuggestions(message.suggestions),
    })),
  }
}

export async function updateRagAgentChatSessionTitle(
  agentId: string,
  sessionId: string,
  title: string,
  accessToken: string | null,
): Promise<{ session_id: string; title: string }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) {
    throw new Error(`Failed to rename RAG chat session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; title: string }
}

export async function deleteRagAgentChatSession(
  agentId: string,
  sessionId: string,
  accessToken: string | null,
): Promise<{ session_id: string; deleted: boolean }> {
  const response = await fetch(`${API_BASE}/api/rag/agents/${agentId}/chat/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete RAG chat session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; deleted: boolean }
}

export async function updateRagWorkspaceChatSessionTitle(
  sessionId: string,
  title: string,
  accessToken: string | null,
): Promise<{ session_id: string; title: string }> {
  const response = await fetch(`${API_BASE}/api/rag/chat/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) {
    throw new Error(`Failed to rename workspace chat session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; title: string }
}

export async function deleteRagWorkspaceChatSession(
  sessionId: string,
  accessToken: string | null,
): Promise<{ session_id: string; deleted: boolean }> {
  const response = await fetch(`${API_BASE}/api/rag/chat/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete workspace chat session: ${response.status}`)
  }
  return (await response.json()) as { session_id: string; deleted: boolean }
}

export async function getBillingUsage(accessToken: string | null): Promise<BillingUsageSummary> {
  const response = await fetch(`${API_BASE}/api/billing/usage`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to load billing usage: ${response.status}`))
  }
  return (await response.json()) as BillingUsageSummary
}

export async function createCheckoutSession(accessToken: string | null): Promise<{ url: string }> {
  const response = await fetch(`${API_BASE}/api/billing/checkout-session`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({}),
  })
  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to create checkout session: ${response.status}`))
  }
  return (await response.json()) as { url: string }
}

export async function createPortalSession(accessToken: string | null): Promise<{ url: string }> {
  const response = await fetch(`${API_BASE}/api/billing/portal-session`, {
    method: 'POST',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(await parseApiError(response, `Failed to create portal session: ${response.status}`))
  }
  return (await response.json()) as { url: string }
}

export async function deleteRagWorkspaceChatLastExchange(
  sessionId: string,
  accessToken: string | null,
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/rag/chat/sessions/${sessionId}/last-exchange`,
    { method: 'DELETE', headers: authHeaders(accessToken) },
  )
  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Failed to delete last exchange: ${response.status}`),
    )
  }
}

export async function deleteRagAgentChatLastExchange(
  agentId: string,
  sessionId: string,
  accessToken: string | null,
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/rag/agents/${agentId}/chat/sessions/${sessionId}/last-exchange`,
    { method: 'DELETE', headers: authHeaders(accessToken) },
  )
  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Failed to delete last exchange: ${response.status}`),
    )
  }
}
