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
