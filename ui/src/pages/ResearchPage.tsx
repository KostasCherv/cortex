import { useCallback, useEffect, useRef, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import {
  createSession,
  getSession,
  startSessionResearch,
  streamSessionRun,
  submitRunFeedback,
} from '@/api/client'
import { FollowupChat } from '@/components/research/FollowupChat'
import { InlineProgress } from '@/components/research/InlineProgress'
import { QueryComposer } from '@/components/research/QueryComposer'
import { ResearchProgress } from '@/components/research/ResearchProgress'
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
  const [latestNode, setLatestNode] = useState<string | null>(null)
  const [streamingReport, setStreamingReport] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [conversation, setConversation] = useState<ConversationTurn[]>([])
  const [feedbackSubmittedAt, setFeedbackSubmittedAt] = useState<string | null>(null)
  const [feedbackHelpful, setFeedbackHelpful] = useState<boolean | null>(null)
  const [feedbackPending, setFeedbackPending] = useState(false)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)
  const [feedbackAvailable, setFeedbackAvailable] = useState(false)

  const loadedSessionRef = useRef<string | null>(null)
  const pollTimerRef = useRef<number | null>(null)
  const runStreamAbortRef = useRef<AbortController | null>(null)

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const stopRunStream = useCallback(() => {
    runStreamAbortRef.current?.abort()
    runStreamAbortRef.current = null
  }, [])

  const resetViewState = useCallback(() => {
    setSessionId(null)
    setRunId(null)
    setConversation([])
    setReport('')
    setStreamingReport('')
    setLatestNode(null)
    setLastQuery('')
      setRunStatus('idle')
      setError(null)
      setFeedbackSubmittedAt(null)
    setFeedbackHelpful(null)
    setFeedbackPending(false)
    setFeedbackError(null)
    setFeedbackAvailable(false)
    stopRunStream()
  }, [stopRunStream])

  const syncFromSessionDetail = useCallback(
    (detail: SessionDetail) => {
      const latestRun = detail.runs.at(-1) ?? null
      const nextStatus = latestRun?.status ?? 'idle'
      loadedSessionRef.current = detail.session_id
      setSessionId(detail.session_id)
      setRunId(latestRun?.run_id ?? null)
      setConversation(detail.conversation)
      setLastQuery(latestRun?.query ?? '')
      setReport((prev) => {
        const nextReport = latestRun?.report ?? ''
        // Never replace a non-empty rendered report with an empty polling snapshot.
        if (!nextReport && prev) return prev
        return nextReport
      })
      setStreamingReport((prev) => {
        const nextPartial = latestRun?.partial_report ?? ''
        if (!nextPartial) return prev
        // Keep the longest/latest partial text to avoid visual rewinds.
        return nextPartial.length >= prev.length ? nextPartial : prev
      })
      setLatestNode(latestRun?.latest_node ?? null)
      setRunStatus(nextStatus)
      setError(nextStatus === 'failed' ? latestRun?.error_details ?? 'Research failed.' : null)
      setFeedbackSubmittedAt(latestRun?.feedback_submitted_at ?? null)
      setFeedbackHelpful(latestRun?.feedback_helpful ?? null)
      setFeedbackPending(false)
      setFeedbackError(null)
      setFeedbackAvailable(Boolean(latestRun && latestRun.status === 'completed'))
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

  const startRunStream = useCallback(
    (nextSessionId: string, nextRunId: string) => {
      if (!authSession?.access_token) return
      stopRunStream()
      const controller = new AbortController()
      runStreamAbortRef.current = controller
      void streamSessionRun(nextSessionId, nextRunId, authSession.access_token, {
        signal: controller.signal,
        onEvent: (event) => {
          if (event.type === 'progress') {
            setLatestNode(event.node ?? null)
            if (event.status === 'completed') setRunStatus('completed')
            if (event.status === 'failed') setRunStatus('failed')
            return
          }
          if (event.type === 'report_chunk') {
            setStreamingReport((prev) => prev + event.text)
            return
          }
          if (event.type === 'done') {
            setRunStatus('completed')
            void getSession(nextSessionId, authSession.access_token)
              .then((detail) => syncFromSessionDetail(detail))
              .catch(() => {})
            return
          }
          if (event.type === 'error') {
            setError(event.error)
            setRunStatus('failed')
          }
        },
        onDone: () => {
          runStreamAbortRef.current = null
          void getSession(nextSessionId, authSession.access_token)
            .then((detail) => syncFromSessionDetail(detail))
            .catch(() => startPolling(nextSessionId))
        },
      }).catch(() => {
        if (runStreamAbortRef.current === controller) {
          runStreamAbortRef.current = null
          startPolling(nextSessionId)
        }
      })
    },
    [authSession, startPolling, stopRunStream, syncFromSessionDetail],
  )

  useEffect(() => {
    return () => {
      stopPolling()
      stopRunStream()
    }
  }, [stopPolling, stopRunStream])

  const openSession = useCallback(
    async (selectedSessionId: string) => {
      if (!authSession?.access_token) return
      setStreamingReport('')
      setLatestNode(null)
      setReport('')
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
      setLatestNode('queued')
      setStreamingReport('')
      setFeedbackSubmittedAt(null)
      setFeedbackHelpful(null)
      setFeedbackPending(false)
      setFeedbackError(null)
      setFeedbackAvailable(false)
      stopRunStream()
      stopPolling()

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
        startRunStream(currentSessionId, started.run_id)
      } catch (streamError) {
        const message =
          streamError instanceof Error ? streamError.message : 'Unable to start background research run.'
        setError(message)
        setRunStatus('failed')
      }
    },
    [authSession, onSessionActivated, onSessionsChanged, startRunStream, stopPolling, stopRunStream],
  )

  const handleFeedbackSubmit = useCallback(
    async (helpful: boolean, comment: string | null) => {
      if (!sessionId || !runId || !authSession?.access_token) return
      setFeedbackPending(true)
      setFeedbackError(null)
      try {
        const result = await submitRunFeedback(
          sessionId,
          runId,
          { helpful, comment },
          authSession.access_token,
        )
        setFeedbackSubmittedAt(result.feedback_submitted_at)
        setFeedbackHelpful(result.feedback_helpful)
        onSessionsChanged()
      } catch (submitError) {
        setFeedbackError(
          submitError instanceof Error ? submitError.message : 'Failed to submit feedback.',
        )
      } finally {
        setFeedbackPending(false)
      }
    },
    [authSession, onSessionsChanged, runId, sessionId],
  )

  const awaitingFinalReport = runStatus === 'completed' && !report && !error
  const hasContent = !!(report || runStatus === 'running' || runStatus === 'failed' || error || awaitingFinalReport)
  const feedbackUnavailableReason = report && runStatus === 'completed' && !feedbackAvailable
    ? 'Feedback is not available for this run.'
    : null

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
              <ResearchProgress
                latestNode={latestNode}
                status={runStatus}
                isStreaming={runStatus === 'running' || awaitingFinalReport}
              />
            </div>
            <div className="mx-auto max-w-2xl px-6 max-md:px-4">
              <ReportViewer
                report={report}
                streamingReport={streamingReport}
                query={lastQuery}
                isStreaming={runStatus === 'running' || awaitingFinalReport}
                error={error}
                feedbackEnabled={Boolean(report && sessionId && runId && runStatus === 'completed' && feedbackAvailable)}
                feedbackUnavailableReason={feedbackUnavailableReason}
                feedbackSubmittedAt={feedbackSubmittedAt}
                feedbackHelpful={feedbackHelpful}
                feedbackPending={feedbackPending}
                feedbackError={feedbackError}
                onFeedbackSubmit={handleFeedbackSubmit}
              />
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
