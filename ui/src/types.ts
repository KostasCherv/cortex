export type HealthResponse = {
  status: string
  version: string
}

export type ResearchRequest = {
  query: string
  use_vector_store: boolean
}

export type ResearchStreamEvent = {
  node: string
  data: {
    error?: string
    report?: string
    [key: string]: unknown
  }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export type SessionRun = {
  run_id: string
  query: string
  source_urls: string[]
  report: string
  status: 'running' | 'completed' | 'failed'
  error_details?: string | null
  latest_node?: string | null
  latest_event_at?: string | null
  partial_report?: string
  created_at: string
}

export type Citation = {
  source_url: string
  source_title: string
}

export type RagCitation = Citation & {
  chunk_id: string
  text: string
}

export type ConversationTurn = {
  role: 'user' | 'assistant'
  content: string
  run_id: string | null
  citations: Citation[]
  created_at: string
  suggestions?: string[]
}

export type SessionDetail = {
  session_id: string
  title: string
  runs: SessionRun[]
  conversation: ConversationTurn[]
  created_at: string
}

export type SessionSummary = {
  session_id: string
  title: string
  created_at: string
  latest_run_status?: 'running' | 'completed' | 'failed' | null
}

export type FollowupStreamEvent =
  | { type: 'chunk'; text: string }
  | { type: 'citations'; citations: Citation[] }
  | { type: 'suggestions'; suggestions: string[] }
  | { type: 'done' }
  | { type: 'error'; error: string }

export type SessionRunStreamEvent =
  | { type: 'progress'; node: string | null; status: 'running' | 'completed' | 'failed'; updated_at: string | null }
  | { type: 'report_chunk'; text: string }
  | { type: 'done' }
  | { type: 'error'; error: string }

export type RagResourceState = 'uploaded' | 'processing' | 'ready' | 'failed'

export type RagResource = {
  resource_id: string
  owner_id: string
  workspace_id: string
  filename: string
  mime_type: string
  byte_size: number
  storage_uri: string
  state: RagResourceState
  error_details?: string | null
  created_at: string
  updated_at: string
}

export type RagAgent = {
  agent_id: string
  owner_id: string
  workspace_id: string
  name: string
  description: string
  system_instructions: string
  linked_resource_ids: string[]
  created_at: string
  updated_at: string
}

export type RagChatMessage = {
  message_id: string
  session_id: string
  agent_id: string
  owner_id: string
  role: 'user' | 'assistant'
  content: string
  citations: RagCitation[]
  created_at: string
}

export type RagChatSessionSummary = {
  session_id: string
  agent_id: string
  owner_id: string
  title: string
  web_search_enabled?: boolean
  created_at: string
  last_message_at: string | null
  last_message_preview: string
}

export type RagChatStreamEvent =
  | {
      type: 'session'
      session_id: string
      web_search_enabled?: boolean
      web_used?: boolean
      web_provider?: string | null
    }
  | { type: 'chunk'; text: string }
  | { type: 'citations'; citations: RagCitation[] }
  | { type: 'done' }
  | { type: 'error'; error: string }
