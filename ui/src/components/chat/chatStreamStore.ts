import { useSyncExternalStore } from 'react'
import type { RagChatMessage } from '@/types'
import type { ChatTransport } from './transports'
import type { ToolConfig } from './toolConfig'

/** Browsers cap concurrent connections per origin; parallel streams also burn quota faster. */
export const MAX_CONCURRENT_STREAMS = 3

export type ChatStreamState = {
  key: string
  transportKey: string
  status: 'streaming' | 'done' | 'error'
  /** Raw question text, used to restore the composer on stop. */
  question: string
  /** Question as rendered in the user bubble (may include attachment names). */
  displayQuestion: string
  streamingText: string
  streamingStatus: string | null
  citations: RagChatMessage['citations']
  suggestions: string[]
  webUsed: boolean
  error: string | null
  resolvedSessionId: string | null
  abortController: AbortController
}

const streams = new Map<string, ChatStreamState>()
// A stream started without a session id lives under a pending key until the
// server assigns one; the alias keeps subscribers of the pending key attached.
const aliases = new Map<string, string>()
const listeners = new Set<() => void>()
let streamingSessionIdsSnapshot: string[] = []

function resolveKey(key: string): string {
  return aliases.get(key) ?? key
}

function emit() {
  const ids: string[] = []
  for (const stream of streams.values()) {
    if (stream.status === 'streaming' && stream.resolvedSessionId) {
      ids.push(stream.resolvedSessionId)
    }
  }
  const unchanged =
    ids.length === streamingSessionIdsSnapshot.length &&
    ids.every((id, index) => id === streamingSessionIdsSnapshot[index])
  if (!unchanged) streamingSessionIdsSnapshot = ids
  for (const listener of listeners) listener()
}

function patch(key: string, updates: Partial<ChatStreamState>) {
  const current = streams.get(key)
  if (!current) return
  streams.set(key, { ...current, ...updates })
  emit()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function sessionStreamKey(transportKey: string, sessionId: string): string {
  return `${transportKey}:${sessionId}`
}

export function peekStream(key: string): ChatStreamState | undefined {
  return streams.get(resolveKey(key))
}

const PENDING_KEY_MARKER = ':pending-'

/** Returns a user-facing reason the stream cannot start, or null if it can. */
export function getStreamStartBlocker(key: string | null): string | null {
  if (key && peekStream(key)?.status === 'streaming') {
    return 'A reply is already streaming in this chat. Stop it or wait for it to finish.'
  }
  // Only one session-less ("pending") stream per transport: a second one would
  // orphan the first and fight over which new session the view adopts.
  if (key?.includes(PENDING_KEY_MARKER)) {
    const transportKey = key.slice(0, key.indexOf(PENDING_KEY_MARKER))
    for (const stream of streams.values()) {
      if (
        stream.status === 'streaming' &&
        stream.transportKey === transportKey &&
        stream.key.includes(PENDING_KEY_MARKER)
      ) {
        return 'A reply is already streaming in this chat. Stop it or wait for it to finish.'
      }
    }
  }
  let active = 0
  for (const stream of streams.values()) {
    if (stream.status === 'streaming') active += 1
  }
  if (active >= MAX_CONCURRENT_STREAMS) {
    return `Up to ${MAX_CONCURRENT_STREAMS} chats can stream at once. Wait for one to finish.`
  }
  return null
}

export function consumeStream(key: string) {
  const canonical = resolveKey(key)
  if (!streams.delete(canonical)) return
  for (const [alias, target] of aliases) {
    if (target === canonical) aliases.delete(alias)
  }
  emit()
}

export function stopChatStream(key: string) {
  peekStream(key)?.abortController.abort()
  consumeStream(key)
}

export function useChatStream(key: string | null): ChatStreamState | undefined {
  return useSyncExternalStore(subscribe, () => (key ? peekStream(key) : undefined))
}

export function useStreamingSessionIds(): string[] {
  return useSyncExternalStore(subscribe, () => streamingSessionIdsSnapshot)
}

export function startChatStream(opts: {
  transport: ChatTransport
  sessionId: string | null
  question: string
  displayQuestion?: string
  accessToken: string
  toolConfig?: ToolConfig
  onSessionsChanged?: () => void
}): string {
  const { transport, sessionId, question, accessToken } = opts
  // Errored streams that never got a session id have no key any view can reach
  // again; sweep them here so they can't accumulate for the app lifetime.
  let evicted = false
  for (const [entryKey, entry] of streams) {
    if (entry.status === 'error' && entry.resolvedSessionId === null) {
      streams.delete(entryKey)
      evicted = true
    }
  }
  if (evicted) emit()

  const key = sessionId
    ? sessionStreamKey(transport.key, sessionId)
    : `${transport.key}${PENDING_KEY_MARKER}${crypto.randomUUID()}`
  const blocker = getStreamStartBlocker(key)
  if (blocker) throw new Error(blocker)

  const controller = new AbortController()
  streams.set(key, {
    key,
    transportKey: transport.key,
    status: 'streaming',
    question,
    displayQuestion: opts.displayQuestion ?? question,
    streamingText: '',
    streamingStatus: null,
    citations: [],
    suggestions: [],
    webUsed: false,
    error: null,
    resolvedSessionId: sessionId,
    abortController: controller,
  })
  emit()

  // The entry may move from a pending key to a session key mid-stream.
  let currentKey = key
  const update = (updates: Partial<ChatStreamState>) => {
    if (!controller.signal.aborted) patch(currentKey, updates)
  }
  const finish = (updates: Partial<ChatStreamState>) => {
    update(updates)
    if (!controller.signal.aborted) opts.onSessionsChanged?.()
  }

  void transport
    .streamMessage(
      question,
      sessionId,
      accessToken,
      {
        signal: controller.signal,
        onSession: (nextSessionId) => {
          if (controller.signal.aborted) return
          const nextKey = sessionStreamKey(transport.key, nextSessionId)
          const entry = streams.get(currentKey)
          if (entry && nextKey !== currentKey) {
            streams.delete(currentKey)
            aliases.set(currentKey, nextKey)
            streams.set(nextKey, { ...entry, key: nextKey, resolvedSessionId: nextSessionId })
            currentKey = nextKey
            emit()
          } else {
            update({ resolvedSessionId: nextSessionId })
          }
        },
        onStatus: (message) => update({ streamingStatus: message }),
        onChunk: (text) => {
          const entry = streams.get(currentKey)
          if (!entry || controller.signal.aborted) return
          patch(currentKey, { streamingText: entry.streamingText + text, streamingStatus: null })
        },
        onCitations: (citations) => update({ citations }),
        onSuggestions: (suggestions) => update({ suggestions }),
        onWebUsed: () => update({ webUsed: true }),
        onDone: () => finish({ status: 'done', streamingStatus: null }),
        onError: (error) => finish({ status: 'error', error, streamingStatus: null }),
      },
      opts.toolConfig,
    )
    .catch((err) => {
      if (streams.get(currentKey)?.status !== 'streaming') return
      finish({
        status: 'error',
        error: err instanceof Error ? err.message : 'Chat failed.',
        streamingStatus: null,
      })
    })

  return key
}

/** Test helper: abort everything and clear the store. */
export function resetChatStreamStore() {
  for (const stream of streams.values()) stream.abortController.abort()
  streams.clear()
  aliases.clear()
  streamingSessionIdsSnapshot = []
  emit()
}
