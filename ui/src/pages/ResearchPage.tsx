import { useCallback, useEffect, useRef, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import {
  createSession,
  getSession,
  startSessionResearch,
} from '@/api/client'
import { FollowupChat } from '@/components/research/FollowupChat'
import { InlineProgress } from '@/components/research/InlineProgress'
import { QueryComposer } from '@/components/research/QueryComposer'
import { ReportViewer } from '@/components/research/ReportViewer'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { ConversationTurn, SessionDetail } from '@/types'

type Props = {
  authSession: Session | null
  activeSessionId: string | null
  onSessionActivated: (id: string | null) => void
  onSessionsChanged: () => void
}

export function ResearchPage({ authSession, activeSessionId, onSessionActivated, onSessionsChanged }: Props) {
  const [runStatus, setRunStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle')
  const [report, setReport] = useState('')
  const [lastQuery, setLastQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [conversation, setConversation] = useState<ConversationTurn[]>([])

  const loadedSessionRef = useRef<string | null>(null)
  const pollTimerRef = useRef<number | null>(null)

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const resetViewState = useCallback(() => {
    setSessionId(null)
    setRunId(null)
    setConversation([])
    setReport('')
    setLastQuery('')
    setRunStatus('idle')
    setError(null)
  }, [])

  const syncFromSessionDetail = useCallback(
    (detail: SessionDetail) => {
      const latestRun = detail.runs.at(-1) ?? null
      const nextStatus = latestRun?.status ?? 'idle'
      loadedSessionRef.current = detail.session_id
      setSessionId(detail.session_id)
      setRunId(latestRun?.run_id ?? null)
      setConversation(detail.conversation)
      setLastQuery(latestRun?.query ?? '')
      setReport(latestRun?.report ?? '')
      setRunStatus(nextStatus)
      setError(nextStatus === 'failed' ? latestRun?.error_details ?? 'Research failed.' : null)
      if (nextStatus !== 'running') {
        stopPolling()
      }
    },
    [stopPolling],
  )

  const startPolling = useCallback(
    (nextSessionId: string) => {
      if (!authSession?.access_token) return
      stopPolling()
      pollTimerRef.current = window.setInterval(() => {
        void getSession(nextSessionId, authSession.access_token)
          .then((detail) => syncFromSessionDetail(detail))
          .catch((pollError) => {
            setError(pollError instanceof Error ? pollError.message : 'Failed to refresh session.')
            setRunStatus('failed')
            stopPolling()
          })
      }, 3000)
    },
    [authSession, stopPolling, syncFromSessionDetail],
  )

  useEffect(() => {
    return () => {
      stopPolling()
    }
  }, [stopPolling])

  const openSession = useCallback(
    async (selectedSessionId: string) => {
      if (!authSession?.access_token) return
      try {
        const detail = await getSession(selectedSessionId, authSession.access_token)
        syncFromSessionDetail(detail)
        if (detail.runs.at(-1)?.status === 'running') {
          startPolling(detail.session_id)
        }
      } catch (sessionError) {
        setError(sessionError instanceof Error ? sessionError.message : 'Failed to load session.')
      }
    },
    [authSession, startPolling, syncFromSessionDetail],
  )

  // Respond to session selection from the rail
  useEffect(() => {
    if (!activeSessionId) {
      if (activeSessionId === null && loadedSessionRef.current !== null) {
        loadedSessionRef.current = null
        queueMicrotask(resetViewState)
        stopPolling()
      }
      return
    }
    if (activeSessionId === loadedSessionRef.current) return
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void openSession(activeSessionId)
  }, [activeSessionId, openSession, resetViewState, stopPolling])

  const handleConversationUpdate = useCallback((turn: ConversationTurn) => {
    setConversation((prev) => [...prev, turn])
  }, [])

  const handleSubmit = useCallback(
    async (query: string, useVectorStore: boolean) => {
      if (!query.trim()) {
        setError('Please enter a research query.')
        return
      }
      if (!authSession?.access_token) {
        setError('Please sign in with Google to create and use sessions.')
        return
      }
      const normalizedQuery = query.trim()

      setError(null)
      setReport('')
      setLastQuery(normalizedQuery)
      setConversation([])
      setRunId(null)
      setRunStatus('running')

      let currentSessionId: string
      try {
        const { session_id } = await createSession(authSession.access_token, normalizedQuery)
        currentSessionId = session_id
        loadedSessionRef.current = session_id
        setSessionId(session_id)
        onSessionActivated(session_id)
        onSessionsChanged()
      } catch (sessionError) {
        setError(sessionError instanceof Error ? sessionError.message : 'Failed to create session.')
        setRunStatus('failed')
        return
      }

      try {
        const started = await startSessionResearch(
          currentSessionId,
          { query: normalizedQuery, use_vector_store: useVectorStore },
          authSession.access_token,
        )
        setRunId(started.run_id)
        setRunStatus('running')
        onSessionsChanged()
        startPolling(currentSessionId)
      } catch (streamError) {
        const message =
          streamError instanceof Error ? streamError.message : 'Unable to start background research run.'
        setError(message)
        setRunStatus('failed')
      }
    },
    [authSession, onSessionActivated, onSessionsChanged, startPolling],
  )

  const hasContent = !!(report || runStatus === 'running' || runStatus === 'failed' || error)

  return (
    <div className="flex h-dvh flex-col max-md:h-full">
      {!hasContent ? (
        // Empty state: centered composer
        <div className="flex flex-1 flex-col items-center justify-center px-6 py-8">
          <div className="w-full max-w-2xl space-y-2">
            <p className="text-sm font-medium text-foreground mb-3">Research</p>
            {!authSession && (
              <p className="text-sm text-muted-foreground mb-4">
                Sign in to save and revisit your research sessions.
              </p>
            )}
            <QueryComposer
              onSubmit={handleSubmit}
              disabled={!authSession}
              isStreaming={false}
            />
          </div>
        </div>
      ) : (
        // Active state: scrollable content
        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-6 py-8">
            <div className="mx-auto max-w-2xl px-6 max-md:px-4">
              <InlineProgress status={runStatus} error={error} />
            </div>
            <div className="mx-auto max-w-2xl px-6 max-md:px-4">
              <ReportViewer report={report} query={lastQuery} isStreaming={runStatus === 'running'} error={error} />
            </div>
            {report && sessionId && (
              <div className="w-full px-6 max-md:px-4">
                <FollowupChat
                  key={sessionId}
                  sessionId={sessionId}
                  runId={runId}
                  accessToken={authSession?.access_token ?? null}
                  conversation={conversation}
                  onConversationUpdate={handleConversationUpdate}
                />
              </div>
            )}
          </div>
        </ScrollArea>
      )}

    </div>
  )
}
