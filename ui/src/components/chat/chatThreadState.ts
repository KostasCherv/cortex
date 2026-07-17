import type { RagChatMessage } from '@/types'

/** Drops the last user+assistant exchange (trailing assistant replies plus the user message that prompted them). */
export function removeLastExchange(messages: RagChatMessage[]): RagChatMessage[] {
  const next = [...messages]
  while (next.length && next[next.length - 1].role === 'assistant') {
    next.pop()
  }
  if (next.length && next[next.length - 1].role === 'user') {
    next.pop()
  }
  return next
}
