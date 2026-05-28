import {
  deletePlannerChatLastExchange,
  getPlannerChatMessages,
  streamPlannerChat,
} from '@/api/plannerChatClient'
import type { PlannerChatStreamEvent } from '@/types'
import type { RagChatMessage } from '@/types'
import type { ChatTransport, StreamCallbacks } from './transports'

function plannerMessageToRagMessage(
  msg: { message_id: string; role: 'user' | 'assistant'; content: string; created_at: string },
  threadId: string,
): RagChatMessage {
  return {
    message_id: msg.message_id,
    session_id: threadId,
    agent_id: null,
    owner_id: '',
    role: msg.role,
    content: msg.content,
    citations: [],
    created_at: msg.created_at,
  }
}

export function createPlannerChatTransport(
  onPlan: (event: PlannerChatStreamEvent & { type: 'plan' }) => void,
): ChatTransport {
  return {
    key: 'planner-interactive',
    listSessions: async () => [],
    loadSessionMessages: async (threadId, accessToken) => {
      const res = await getPlannerChatMessages(threadId, accessToken)
      return {
        session_id: res.thread_id,
        messages: res.messages.map((m) => plannerMessageToRagMessage(m, res.thread_id)),
      }
    },
    streamMessage: async (message, threadId, accessToken, callbacks: StreamCallbacks) => {
      await streamPlannerChat(message, threadId, accessToken, {
        signal: callbacks.signal,
        onSession: (newThreadId) => callbacks.onSession(newThreadId),
        onChunk: (text) => callbacks.onChunk(text),
        onPlan: (planEvent) => {
          onPlan(planEvent)
          // Don't emit a separate chunk; the streaming chunks already cover the content.
        },
        onDone: () => callbacks.onDone(),
        onError: (error) => callbacks.onError?.(error),
      })
    },
    deleteLastExchange: async (threadId, accessToken) => {
      await deletePlannerChatLastExchange(threadId, accessToken)
    },
  }
}
