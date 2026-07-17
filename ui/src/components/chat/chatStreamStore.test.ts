import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatTransport, StreamCallbacks } from './transports'
import {
  MAX_CONCURRENT_STREAMS,
  consumeStream,
  getStreamStartBlocker,
  peekStream,
  resetChatStreamStore,
  sessionStreamKey,
  startChatStream,
  stopChatStream,
} from './chatStreamStore'

type StreamControls = {
  callbacks: StreamCallbacks
  resolve: () => void
  reject: (err: Error) => void
}

function makeTransport(key: string): { transport: ChatTransport; streams: StreamControls[] } {
  const streams: StreamControls[] = []
  const transport = {
    key,
    listSessions: vi.fn(),
    loadSessionMessages: vi.fn(),
    deleteLastExchange: vi.fn(),
    streamMessage: vi.fn(
      (_message: string, _sessionId: string | null, _token: string, callbacks: StreamCallbacks) =>
        new Promise<void>((resolve, reject) => {
          streams.push({ callbacks, resolve, reject: reject as (err: Error) => void })
        }),
    ),
  } as unknown as ChatTransport
  return { transport, streams }
}

function start(transport: ChatTransport, sessionId: string | null, extra: Partial<Parameters<typeof startChatStream>[0]> = {}) {
  return startChatStream({
    transport,
    sessionId,
    question: 'hello',
    displayQuestion: 'hello',
    accessToken: 'token',
    ...extra,
  })
}

describe('chatStreamStore', () => {
  beforeEach(() => {
    resetChatStreamStore()
  })

  it('accumulates two concurrent streams independently', () => {
    const a = makeTransport('workspace')
    const b = makeTransport('agent:1')
    const keyA = start(a.transport, 'session-a')
    const keyB = start(b.transport, 'session-b')

    a.streams[0].callbacks.onChunk('alpha ')
    b.streams[0].callbacks.onChunk('beta')
    a.streams[0].callbacks.onChunk('one')

    expect(peekStream(keyA)?.streamingText).toBe('alpha one')
    expect(peekStream(keyB)?.streamingText).toBe('beta')
  })

  it('rekeys a pending stream when the session id arrives', () => {
    const { transport, streams } = makeTransport('workspace')
    const pendingKey = start(transport, null)
    expect(pendingKey).toContain('pending-')

    streams[0].callbacks.onSession('real-id')
    streams[0].callbacks.onChunk('answer')

    const realKey = sessionStreamKey('workspace', 'real-id')
    expect(peekStream(realKey)?.streamingText).toBe('answer')
    // the pending key still resolves to the same entry for existing subscribers
    expect(peekStream(pendingKey)).toBe(peekStream(realKey))
    expect(peekStream(realKey)?.resolvedSessionId).toBe('real-id')
  })

  it('rejects a second stream on the same session', () => {
    const { transport } = makeTransport('workspace')
    start(transport, 'session-a')
    expect(() => start(transport, 'session-a')).toThrow(/already/i)
    expect(getStreamStartBlocker(sessionStreamKey('workspace', 'session-a'))).toMatch(/already/i)
  })

  it('caps the number of concurrent streams', () => {
    const { transport } = makeTransport('workspace')
    for (let i = 0; i < MAX_CONCURRENT_STREAMS; i += 1) {
      start(transport, `session-${i}`)
    }
    expect(() => start(transport, 'session-overflow')).toThrow(/at once/i)
    expect(getStreamStartBlocker(sessionStreamKey('workspace', 'session-overflow'))).toMatch(/at once/i)
  })

  it('stop aborts the stream and removes the entry', () => {
    const { transport, streams } = makeTransport('workspace')
    const key = start(transport, 'session-a')
    stopChatStream(key)

    expect(streams[0].callbacks.signal?.aborted).toBe(true)
    expect(peekStream(key)).toBeUndefined()

    // late chunks after abort are ignored
    streams[0].callbacks.onChunk('late')
    expect(peekStream(key)).toBeUndefined()
  })

  it('marks the stream done, notifies, and can be consumed later', () => {
    const onSessionsChanged = vi.fn()
    const { transport, streams } = makeTransport('workspace')
    const key = start(transport, 'session-a', { onSessionsChanged })

    streams[0].callbacks.onChunk('final answer')
    streams[0].callbacks.onSuggestions?.(['follow up'])
    streams[0].callbacks.onDone()

    const finished = peekStream(key)
    expect(finished?.status).toBe('done')
    expect(finished?.streamingText).toBe('final answer')
    expect(finished?.suggestions).toEqual(['follow up'])
    expect(onSessionsChanged).toHaveBeenCalledTimes(1)

    consumeStream(key)
    expect(peekStream(key)).toBeUndefined()
  })

  it('records stream errors', async () => {
    const { transport, streams } = makeTransport('workspace')
    const key = start(transport, 'session-a')
    streams[0].callbacks.onError?.('boom')
    expect(peekStream(key)?.status).toBe('error')
    expect(peekStream(key)?.error).toBe('boom')
  })

  it('allows only one session-less pending stream per transport', () => {
    const { transport } = makeTransport('workspace')
    start(transport, null)
    expect(() => start(transport, null)).toThrow(/already/i)
    // a different transport is unaffected
    const other = makeTransport('agent:1')
    expect(() => start(other.transport, null)).not.toThrow()
  })

  it('evicts errored streams that never resolved a session id on the next start', () => {
    const { transport, streams } = makeTransport('workspace')
    const orphanKey = start(transport, null)
    streams[0].callbacks.onError?.('boom')
    expect(peekStream(orphanKey)?.status).toBe('error')

    start(transport, 'session-a')
    expect(peekStream(orphanKey)).toBeUndefined()
  })

  it('records transport rejections as errors', async () => {
    const { transport, streams } = makeTransport('workspace')
    const key = start(transport, 'session-a')
    streams[0].reject(new Error('network down'))
    await vi.waitFor(() => {
      expect(peekStream(key)?.status).toBe('error')
    })
    expect(peekStream(key)?.error).toBe('network down')
  })
})
