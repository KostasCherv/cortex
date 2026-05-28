import type { PlannerChatMessage, PlannerChatStreamEvent } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

function authHeaders(accessToken: string): HeadersInit {
  return { Authorization: `Bearer ${accessToken}` }
}

export type PlannerStreamCallbacks = {
  signal?: AbortSignal
  onSession: (threadId: string) => void
  onChunk: (text: string) => void
  onPlan: (event: PlannerChatStreamEvent & { type: 'plan' }) => void
  onDone: () => void
  onError?: (error: string) => void
}

export async function streamPlannerChat(
  message: string,
  threadId: string | null,
  accessToken: string,
  callbacks: PlannerStreamCallbacks,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/planner/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ message, thread_id: threadId }),
    signal: callbacks.signal,
  })

  if (!response.ok) {
    throw new Error(`Planner chat request failed: ${response.status}`)
  }
  if (!response.body) {
    throw new Error('Streaming not supported.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const handleEvent = (parsed: PlannerChatStreamEvent): boolean => {
    if (parsed.type === 'session') {
      callbacks.onSession(parsed.thread_id)
      return false
    }
    if (parsed.type === 'chunk') {
      callbacks.onChunk(parsed.text)
      return false
    }
    if (parsed.type === 'plan') {
      callbacks.onPlan(parsed)
      return false
    }
    if (parsed.type === 'done') {
      callbacks.onDone()
      return true
    }
    if (parsed.type === 'error') {
      callbacks.onError?.(parsed.error)
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
      let parsed: PlannerChatStreamEvent
      try {
        parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as PlannerChatStreamEvent
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
        const parsed = JSON.parse(dataLine.replace(/^data:\s?/, '')) as PlannerChatStreamEvent
        if (handleEvent(parsed)) return
      } catch {
        // Ignore trailing partial event.
      }
    }
  }

  callbacks.onError?.('Planner stream ended before a terminal event was received.')
}

export async function getPlannerChatMessages(
  threadId: string,
  accessToken: string,
): Promise<{ thread_id: string; messages: PlannerChatMessage[] }> {
  const response = await fetch(`${API_BASE}/api/planner/chat/${threadId}/messages`, {
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to load planner chat messages: ${response.status}`)
  }
  return (await response.json()) as { thread_id: string; messages: PlannerChatMessage[] }
}

export async function deletePlannerChatLastExchange(
  threadId: string,
  accessToken: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/planner/chat/${threadId}/last`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    throw new Error(`Failed to delete planner chat last exchange: ${response.status}`)
  }
}
