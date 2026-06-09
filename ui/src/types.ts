export type HealthResponse = {
  status: string
  version: string
}

export type ResearchRequest = {
  query: string
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
  langfuse_trace_id?: string | null
  langfuse_observation_id?: string | null
  feedback_submitted_at?: string | null
  feedback_helpful?: boolean | null
  created_at: string
}

export type RunFeedbackRequest = {
  helpful: boolean
  comment?: string | null
}

export type Citation = {
  source_url: string | null
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
  | { type: 'web_used'; provider: string }
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

export type RagAgentDraft = {
  name: string
  description: string
  system_instructions: string
}

export type RagChatMessage = {
  message_id: string
  session_id: string
  agent_id: string | null
  owner_id: string
  role: 'user' | 'assistant'
  content: string
  citations: RagCitation[]
  suggestions?: string[]
  created_at: string
}

export type RagChatSessionSummary = {
  session_id: string
  agent_id: string | null
  owner_id: string
  title: string
  created_at: string
  last_message_at: string | null
  last_message_preview: string
}

export type UserMemory = {
  content: string
  updated_at: string | null
  last_refreshed_at: string | null
}

export type RagChatStreamEvent =
  | {
      type: 'session'
      session_id: string
      web_used?: boolean
      web_provider?: string | null
    }
  | { type: 'chunk'; text: string }
  | { type: 'citations'; citations: RagCitation[] }
  | { type: 'suggestions'; suggestions: string[] }
  | { type: 'web_used'; provider: string }
  | { type: 'done' }
  | { type: 'error'; error: string }

export type PRDPlanningBrief = {
  problem_statement: string
  desired_outcome: string
  constraints: string[]
  assumptions: string[]
  open_questions: string[]
}

export type PRDRequirement = {
  id: string
  description: string
  priority: 'Must Have' | 'Should Have' | 'Could Have' | "Won't Have"
  rationale: string
}

export type PRDMilestone = {
  id: string
  title: string
  description: string
  deliverables: string[]
}

export type PRDPlan = {
  title: string
  executive_summary: string
  problem_statement: string
  goals: string[]
  non_goals: string[]
  target_users: string[]
  user_stories: string[]
  requirements: PRDRequirement[]
  success_metrics: string[]
  milestones: PRDMilestone[]
  out_of_scope: string[]
  risks: string[]
  assumptions: string[]
  open_questions: string[]
}

export type PRDPlanResponse = {
  plan: PRDPlan
  markdown: string
  suggested_filename: string
  planning_brief: PRDPlanningBrief
}

export type SavedPRDSummary = {
  plan_id: string
  title: string
  summary: string
  prompt_preview: string
  created_at: string
  updated_at: string
}

export type SavedPRD = PRDPlanResponse & {
  plan_id: string
  prompt: string
  prompt_preview: string
  created_at: string
  updated_at: string
}

export type SavedPRDListResponse = {
  plans: SavedPRDSummary[]
}

export type PlannerChatStreamEvent =
  | { type: 'session'; thread_id: string }
  | { type: 'chunk'; text: string }
  | {
      type: 'plan'
      plan: PRDPlan
      markdown: string
      suggested_filename: string
      planning_brief: PRDPlanningBrief
    }
  | { type: 'done' }
  | { type: 'error'; error: string }

export type PlannerChatMessage = {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  plan_event?: PlannerChatStreamEvent & { type: 'plan' }
  created_at: string
}

export type TravelRequirements = {
  destination?: string | null
  start_date?: string | null
  end_date?: string | null
  trip_length_days?: number | null
  traveler_count?: number | null
  party_type?: string | null
  budget_band?: string | null
  interests: string[]
  constraints: string[]
  pace?: string | null
}

export type ItineraryDay = {
  day_number: number
  title: string
  morning: string[]
  afternoon: string[]
  evening: string[]
  notes: string[]
}

export type RecommendedArea = {
  name: string
  why: string
  vibe: string
}

export type GeneratedItinerary = {
  title: string
  summary: string
  destination: string
  budget_band: string
  days: ItineraryDay[]
  tips: string[]
  recommended_areas: RecommendedArea[]
  getting_there: string[]
  getting_around: string[]
  must_do_highlights: string[]
  booking_advice: string[]
  revision_summary?: string | null
}

export type ItinerarySessionMessage = {
  message_id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  metadata: Record<string, unknown>
  created_at: string
}

export type ItineraryVersion = {
  version_id: string
  session_id: string
  version_number: number
  revision_summary: string
  markdown: string
  itinerary: GeneratedItinerary
  created_at: string
}

export type ItinerarySessionSummary = {
  session_id: string
  owner_id: string
  workspace_id: string
  title: string
  status: string
  current_version_id?: string | null
  prompt_preview: string
  last_message_preview: string
  created_at: string
  updated_at: string
}

export type ItinerarySessionDetail = ItinerarySessionSummary & {
  requirements: TravelRequirements
  messages: ItinerarySessionMessage[]
  versions: ItineraryVersion[]
  current_version?: ItineraryVersion | null
}

export type ItinerarySessionListResponse = {
  sessions: ItinerarySessionSummary[]
}

export type ItineraryPlannerResponse = {
  session: ItinerarySessionDetail
  assistant_message: ItinerarySessionMessage
  current_itinerary?: GeneratedItinerary | null
  new_version?: ItineraryVersion | null
  created_new_version: boolean
  missing_fields: string[]
}

export type BillingUsageSummary = {
  plan: 'free' | 'pro'
  date: string
  limits: {
    research_queries_daily: number
    total_questions_daily: number
  }
  usage: {
    research_queries_count: number
    total_questions_count: number
  }
  resets_at: string
  subscription: {
    status: string
    current_period_end: string | null
    cancel_at_period_end: boolean | null
    cancel_at: string | null
    canceled_at: string | null
  } | null
}
