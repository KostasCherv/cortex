import { describe, expect, it } from 'vitest'
import type { RagChatMessage } from '@/types'
import { removeLastExchange } from './chatThreadState'

function msg(role: 'user' | 'assistant', content: string): RagChatMessage {
  return {
    message_id: `${role}-${content}`,
    session_id: 's',
    agent_id: null,
    owner_id: '',
    role,
    content,
    citations: [],
    created_at: new Date().toISOString(),
  }
}

describe('removeLastExchange', () => {
  it('removes the trailing assistant messages and their user message', () => {
    const messages = [msg('user', 'a'), msg('assistant', 'b'), msg('user', 'c'), msg('assistant', 'd')]
    expect(removeLastExchange(messages).map((m) => m.content)).toEqual(['a', 'b'])
  })

  it('removes a dangling trailing user message', () => {
    const messages = [msg('user', 'a'), msg('assistant', 'b'), msg('user', 'c')]
    expect(removeLastExchange(messages).map((m) => m.content)).toEqual(['a', 'b'])
  })

  it('returns an empty list unchanged', () => {
    expect(removeLastExchange([])).toEqual([])
  })
})
