import {
  getRagAgentChatSessionMessages,
  getRagWorkspaceChatSessionMessages,
  listRagAgentChatSessions,
  listRagWorkspaceChatSessions,
  streamRagAgentChat,
  streamRagWorkspaceChat,
} from '@/api/client'
import type { RagChatMessage, RagChatSessionSummary } from '@/types'

export type StreamCallbacks = {
  signal?: AbortSignal
  onSession: (sessionId: string, webUsed?: boolean, webProvider?: string | null) => void
  onChunk: (text: string) => void
  onCitations: (citations: RagChatMessage['citations']) => void
  onSuggestions?: (suggestions: string[]) => void
  onDone: () => void
  onError?: (error: string) => void
}

export type ChatTransport = {
  key: string
  listSessions: (accessToken: string) => Promise<RagChatSessionSummary[]>
  loadSessionMessages: (
    sessionId: string,
    accessToken: string,
  ) => Promise<{ session_id: string; web_search_enabled?: boolean; messages: RagChatMessage[] }>
  streamMessage: (
    message: string,
    sessionId: string | null,
    accessToken: string,
    callbacks: StreamCallbacks,
  ) => Promise<void>
}

export function createAgentChatTransport(agentId: string): ChatTransport {
  return {
    key: `agent:${agentId}`,
    listSessions: async (accessToken) => {
      const res = await listRagAgentChatSessions(agentId, accessToken)
      return res.sessions
    },
    loadSessionMessages: async (sessionId, accessToken) => {
      const res = await getRagAgentChatSessionMessages(agentId, sessionId, accessToken)
      return {
        session_id: res.session_id,
        web_search_enabled: res.web_search_enabled,
        messages: res.messages,
      }
    },
    streamMessage: async (message, sessionId, accessToken, callbacks) => {
      await streamRagAgentChat(agentId, message, sessionId, accessToken, callbacks)
    },
  }
}

export const workspaceChatTransport: ChatTransport = {
  key: 'workspace',
  listSessions: async (accessToken) => {
    const res = await listRagWorkspaceChatSessions(accessToken)
    return res.sessions
  },
  loadSessionMessages: async (sessionId, accessToken) => {
    const res = await getRagWorkspaceChatSessionMessages(sessionId, accessToken)
    return {
      session_id: res.session_id,
      web_search_enabled: res.web_search_enabled,
      messages: res.messages,
    }
  },
  streamMessage: async (message, sessionId, accessToken, callbacks) => {
    await streamRagWorkspaceChat(message, sessionId, accessToken, callbacks)
  },
}
