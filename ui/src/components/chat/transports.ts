import {
  deleteRagAgentChatLastExchange,
  deleteRagWorkspaceChatLastExchange,
  getRagAgentChatSessionMessages,
  getRagWorkspaceChatSessionMessages,
  listRagAgentChatSessions,
  listRagWorkspaceChatSessions,
  streamRagAgentChat,
  streamRagWorkspaceChat,
} from '@/api/client'
import type { RagChatMessage, RagChatSessionSummary } from '@/types'
import type { ToolConfig } from './toolConfig'

export type StreamCallbacks = {
  signal?: AbortSignal
  onSession: (sessionId: string) => void
  onChunk: (text: string) => void
  onCitations: (citations: RagChatMessage['citations']) => void
  onSuggestions?: (suggestions: string[]) => void
  onWebUsed?: () => void
  onDone: () => void
  onError?: (error: string) => void
}

export type ChatTransport = {
  key: string
  listSessions: (accessToken: string) => Promise<RagChatSessionSummary[]>
  loadSessionMessages: (
    sessionId: string,
    accessToken: string,
  ) => Promise<{ session_id: string; messages: RagChatMessage[] }>
  streamMessage: (
    message: string,
    sessionId: string | null,
    accessToken: string,
    callbacks: StreamCallbacks,
    tools?: ToolConfig,
  ) => Promise<void>
  deleteLastExchange: (sessionId: string, accessToken: string) => Promise<void>
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
        messages: res.messages,
      }
    },
    streamMessage: async (message, sessionId, accessToken, callbacks, tools) => {
      await streamRagAgentChat(agentId, message, sessionId, accessToken, {
        signal: callbacks.signal,
        onSession: callbacks.onSession,
        onChunk: callbacks.onChunk,
        onCitations: callbacks.onCitations,
        onSuggestions: callbacks.onSuggestions,
        onWebUsed: callbacks.onWebUsed,
        onDone: callbacks.onDone,
        onError: callbacks.onError,
      }, tools)
    },
    deleteLastExchange: async (sessionId, accessToken) => {
      await deleteRagAgentChatLastExchange(agentId, sessionId, accessToken)
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
      messages: res.messages,
    }
  },
  streamMessage: async (message, sessionId, accessToken, callbacks, tools) => {
    await streamRagWorkspaceChat(message, sessionId, accessToken, {
      signal: callbacks.signal,
      onSession: callbacks.onSession,
      onChunk: callbacks.onChunk,
      onCitations: callbacks.onCitations,
      onSuggestions: callbacks.onSuggestions,
      onWebUsed: callbacks.onWebUsed,
      onDone: callbacks.onDone,
      onError: callbacks.onError,
    }, tools)
  },
  deleteLastExchange: async (sessionId, accessToken) => {
    await deleteRagWorkspaceChatLastExchange(sessionId, accessToken)
  },
}
