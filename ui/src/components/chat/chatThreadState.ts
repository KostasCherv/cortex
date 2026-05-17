import type { RagChatMessage } from '@/types'

export function getStopEditState(
  messages: RagChatMessage[],
): { draft: string; editingMessageId: string } | null {
  const lastUser = [...messages].reverse().find((message) => message.role === 'user')
  if (!lastUser) {
    return null
  }
  return {
    draft: lastUser.content,
    editingMessageId: lastUser.message_id,
  }
}

export function replaceLastEditableUserMessage(
  messages: RagChatMessage[],
  content: string,
): RagChatMessage[] {
  const next = [...messages]
  while (next.length && next[next.length - 1].role === 'assistant') {
    next.pop()
  }
  if (next.length && next[next.length - 1].role === 'user') {
    next[next.length - 1] = { ...next[next.length - 1], content }
  }
  return next
}
