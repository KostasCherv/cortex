import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { RagChatMessage } from '@/types'
import { ChatThreadContainer } from './ChatThreadContainer'
import { resetChatStreamStore } from './chatStreamStore'
import type { ChatTransport, StreamCallbacks } from './transports'

type CapturedStream = {
  message: string
  sessionId: string | null
  callbacks: StreamCallbacks
}

function makeTransport(messagesBySession: Record<string, RagChatMessage[]> = {}) {
  const captured: CapturedStream[] = []
  const loadSessionMessages = vi.fn(async (sessionId: string) => ({
    session_id: sessionId,
    messages: messagesBySession[sessionId] ?? [],
  }))
  const transport = {
    key: 'workspace',
    listSessions: vi.fn(async () => []),
    loadSessionMessages,
    deleteLastExchange: vi.fn(async () => {}),
    streamMessage: vi.fn(
      (message: string, sessionId: string | null, _token: string, callbacks: StreamCallbacks) => {
        captured.push({ message, sessionId, callbacks })
        return new Promise<void>(() => {})
      },
    ),
  } as unknown as ChatTransport
  return { transport, captured, loadSessionMessages }
}

function renderContainer(transport: ChatTransport, activeSessionId: string | null) {
  const props = {
    transport,
    accessToken: 'token',
    onSessionActivated: vi.fn(),
    onSessionsChanged: vi.fn(),
    title: 'Chat',
    emptyState: 'Ask anything.',
  }
  const view = render(<ChatThreadContainer {...props} activeSessionId={activeSessionId} />)
  return {
    ...props,
    rerenderWith: (nextSessionId: string | null) =>
      view.rerender(<ChatThreadContainer {...props} activeSessionId={nextSessionId} />),
  }
}

describe('ChatThreadContainer parallel streams', () => {
  beforeEach(() => {
    resetChatStreamStore()
  })

  it('shows a loading indicator instead of the empty state while session messages are being fetched', async () => {
    let resolveMessages: (value: { session_id: string; messages: RagChatMessage[] }) => void = () => {}
    const transport = {
      key: 'workspace',
      listSessions: vi.fn(async () => []),
      loadSessionMessages: vi.fn(() => new Promise((resolve) => { resolveMessages = resolve })),
      deleteLastExchange: vi.fn(async () => {}),
      streamMessage: vi.fn(() => new Promise<void>(() => {})),
    } as unknown as ChatTransport

    renderContainer(transport, 'session-a')

    expect(screen.getByText(/loading discussion/i)).toBeInTheDocument()
    expect(screen.queryByText('Ask anything.')).not.toBeInTheDocument()

    resolveMessages({ session_id: 'session-a', messages: [] })
    await waitFor(() => expect(screen.getByText('Ask anything.')).toBeInTheDocument())
    expect(screen.queryByText(/loading discussion/i)).not.toBeInTheDocument()
  })

  it('hides the previous session messages behind the loader instead of leaving them visible while switching', async () => {
    let resolveSessionB: (value: { session_id: string; messages: RagChatMessage[] }) => void = () => {}
    const sessionAMessage: RagChatMessage = {
      message_id: 'msg-a',
      session_id: 'session-a',
      agent_id: null,
      owner_id: '',
      role: 'assistant',
      content: 'reply from session a',
      citations: [],
      created_at: '2026-01-01T00:00:00Z',
    }
    const loadSessionMessages = vi.fn((sid: string) => {
      if (sid === 'session-a') return Promise.resolve({ session_id: 'session-a', messages: [sessionAMessage] })
      return new Promise((resolve) => { resolveSessionB = resolve })
    })
    const transport = {
      key: 'workspace',
      listSessions: vi.fn(async () => []),
      loadSessionMessages,
      deleteLastExchange: vi.fn(async () => {}),
      streamMessage: vi.fn(() => new Promise<void>(() => {})),
    } as unknown as ChatTransport

    const { rerenderWith } = renderContainer(transport, 'session-a')
    expect(await screen.findByText('reply from session a')).toBeInTheDocument()

    rerenderWith('session-b')
    expect(screen.getByText(/loading discussion/i)).toBeInTheDocument()
    expect(screen.queryByText('reply from session a')).not.toBeInTheDocument()

    resolveSessionB({ session_id: 'session-b', messages: [] })
    await waitFor(() => expect(screen.getByText('Ask anything.')).toBeInTheDocument())
    expect(screen.queryByText('reply from session a')).not.toBeInTheDocument()
  })

  it('keeps a stream alive when switching to another session and folds it in on return', async () => {
    const { transport, captured, loadSessionMessages } = makeTransport({ 'session-a': [], 'session-b': [] })
    const { rerenderWith } = renderContainer(transport, 'session-a')
    await waitFor(() => expect(loadSessionMessages).toHaveBeenCalledWith('session-a', 'token'))

    fireEvent.change(screen.getByPlaceholderText('Ask a question...'), { target: { value: 'question for a' } })
    fireEvent.click(screen.getByLabelText('Send message'))
    expect(captured).toHaveLength(1)

    // switch to session B: the stream for A must NOT be aborted
    rerenderWith('session-b')
    await waitFor(() => expect(loadSessionMessages).toHaveBeenCalledWith('session-b', 'token'))
    expect(captured[0].callbacks.signal?.aborted).toBe(false)

    // A completes in the background
    captured[0].callbacks.onChunk('answer for a')
    captured[0].callbacks.onDone()

    // returning to A refetches persisted messages instead of trusting stale state
    const callsBefore = loadSessionMessages.mock.calls.filter(([id]) => id === 'session-a').length
    rerenderWith('session-a')
    await waitFor(() => {
      const callsAfter = loadSessionMessages.mock.calls.filter(([id]) => id === 'session-a').length
      expect(callsAfter).toBe(callsBefore + 1)
    })
  })

  it('streams in a second session while the first is still running', async () => {
    const { transport, captured } = makeTransport({ 'session-a': [], 'session-b': [] })
    const { rerenderWith } = renderContainer(transport, 'session-a')

    fireEvent.change(await screen.findByPlaceholderText('Ask a question...'), { target: { value: 'first question' } })
    fireEvent.click(screen.getByLabelText('Send message'))

    rerenderWith('session-b')
    const composer = await screen.findByPlaceholderText('Ask a question...')
    await waitFor(() => expect(composer).not.toBeDisabled())
    fireEvent.change(composer, { target: { value: 'second question' } })
    fireEvent.click(screen.getByLabelText('Send message'))

    expect(captured).toHaveLength(2)
    expect(captured[0].callbacks.signal?.aborted).toBe(false)
    expect(captured[1].sessionId).toBe('session-b')

    // both streams render independently in their own session view
    captured[1].callbacks.onChunk('b partial')
    expect(await screen.findByText('b partial')).toBeInTheDocument()
  })

  it('surfaces a background stream error when returning to the session', async () => {
    const { transport, captured, loadSessionMessages } = makeTransport({ 'session-a': [], 'session-b': [] })
    const { rerenderWith } = renderContainer(transport, 'session-a')
    await waitFor(() => expect(loadSessionMessages).toHaveBeenCalledWith('session-a', 'token'))

    fireEvent.change(screen.getByPlaceholderText('Ask a question...'), { target: { value: 'doomed question' } })
    fireEvent.click(screen.getByLabelText('Send message'))

    rerenderWith('session-b')
    await waitFor(() => expect(loadSessionMessages).toHaveBeenCalledWith('session-b', 'token'))
    captured[0].callbacks.onError?.('backend exploded')

    rerenderWith('session-a')
    expect(await screen.findByRole('alert')).toHaveTextContent('backend exploded')
    expect(screen.getByPlaceholderText('Ask a question...')).toHaveValue('doomed question')
  })

  it('disables the composer only while the viewed session is streaming', async () => {
    const { transport, captured } = makeTransport({ 'session-a': [] })
    renderContainer(transport, 'session-a')

    const composer = await screen.findByPlaceholderText('Ask a question...')
    fireEvent.change(composer, { target: { value: 'question' } })
    fireEvent.click(screen.getByLabelText('Send message'))
    expect(composer).toBeDisabled()

    captured[0].callbacks.onChunk('done text')
    captured[0].callbacks.onDone()
    await waitFor(() => expect(composer).not.toBeDisabled())
    expect(screen.getByText('done text')).toBeInTheDocument()
  })
})
