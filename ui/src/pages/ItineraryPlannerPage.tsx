import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Loader2, SendHorizontal } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { createItinerarySession, getItinerarySession, postItinerarySessionMessage } from '@/api/client'
import { assistantAvatarClassName, assistantBubbleClassName, ChatMarkdown } from '@/components/chat/ChatMarkdown'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type {
  GeneratedItinerary,
  ItinerarySessionDetail,
  ItinerarySessionMessage,
} from '@/types'

function RequirementsSummary({ session }: { session: ItinerarySessionDetail }) {
  const { requirements } = session
  const chips = [
    requirements.destination,
    requirements.start_date && requirements.end_date ? `${requirements.start_date} to ${requirements.end_date}` : null,
    requirements.traveler_count ? `${requirements.traveler_count} travelers` : null,
    requirements.party_type,
    requirements.budget_band,
    requirements.pace,
    ...(requirements.interests ?? []),
  ].filter(Boolean)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Trip requirements</CardTitle>
        <CardDescription>Live summary of the travel constraints collected so far.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {chips.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {chips.map((chip) => (
              <span
                key={chip}
                className="rounded-full border border-border/70 bg-muted/30 px-2.5 py-1 text-xs text-foreground"
              >
                {chip}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No structured trip requirements yet.</p>
        )}
        {requirements.constraints?.length ? (
          <div>
            <h3 className="text-sm font-medium">Constraints</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {requirements.constraints.map((constraint) => (
                <li key={constraint}>{constraint}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function ItineraryView({ itinerary }: { itinerary: GeneratedItinerary | null }) {
  if (!itinerary) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Itinerary draft</CardTitle>
          <CardDescription>The structured itinerary appears here after the interview is complete.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{itinerary.title}</CardTitle>
        <CardDescription>{itinerary.summary}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full border border-border/70 bg-muted/30 px-2.5 py-1 text-xs">{itinerary.destination}</span>
          <span className="rounded-full border border-border/70 bg-muted/30 px-2.5 py-1 text-xs">{itinerary.budget_band}</span>
        </div>
        {itinerary.recommended_areas.length ? (
          <div>
            <h3 className="text-sm font-medium">Recommended areas</h3>
            <div className="mt-2 space-y-2">
              {itinerary.recommended_areas.map((area) => (
                <div key={`${area.name}-${area.vibe}`} className="rounded-lg border border-border/70 bg-muted/10 p-3">
                  <p className="text-sm font-medium text-foreground">{area.name}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{area.why}</p>
                  <p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">{area.vibe}</p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
        {itinerary.getting_there.length ? (
          <div>
            <h3 className="text-sm font-medium">Getting there</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {itinerary.getting_there.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {itinerary.getting_around.length ? (
          <div>
            <h3 className="text-sm font-medium">Getting around</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {itinerary.getting_around.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="space-y-3">
          {itinerary.days.map((day) => (
            <section key={`${day.day_number}-${day.title}`} className="rounded-lg border border-border/70 bg-background/70 p-4">
              <h3 className="font-medium">Day {day.day_number}: {day.title}</h3>
              <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                {day.morning.length ? <p><strong className="text-foreground">Morning:</strong> {day.morning.join('; ')}</p> : null}
                {day.afternoon.length ? <p><strong className="text-foreground">Afternoon:</strong> {day.afternoon.join('; ')}</p> : null}
                {day.evening.length ? <p><strong className="text-foreground">Evening:</strong> {day.evening.join('; ')}</p> : null}
                {day.notes.length ? <p><strong className="text-foreground">Notes:</strong> {day.notes.join('; ')}</p> : null}
              </div>
            </section>
          ))}
        </div>
        {itinerary.tips.length ? (
          <div>
            <h3 className="text-sm font-medium">Tips</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {itinerary.tips.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {itinerary.must_do_highlights.length ? (
          <div>
            <h3 className="text-sm font-medium">Must-do highlights</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {itinerary.must_do_highlights.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {itinerary.booking_advice.length ? (
          <div>
            <h3 className="text-sm font-medium">Booking advice</h3>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {itinerary.booking_advice.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

export function ItineraryPlannerPage({
  authSession,
  activeSessionId,
  onSessionActivated,
  onSessionsChanged,
}: {
  authSession: Session | null
  activeSessionId?: string | null
  onSessionActivated?: (sessionId: string | null) => void
  onSessionsChanged?: () => void
}) {
  const [draft, setDraft] = useState('')
  const [session, setSession] = useState<ItinerarySessionDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingMessage, setPendingMessage] = useState<ItinerarySessionMessage | null>(null)
  const [dismissedSessionId, setDismissedSessionId] = useState<string | null>(null)
  const loadRequestIdRef = useRef(0)
  const submitRequestIdRef = useRef(0)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const restoreFocusRef = useRef(false)

  const loadSession = useCallback(
    async (sessionId: string) => {
      if (!authSession?.access_token) return
      const requestId = loadRequestIdRef.current + 1
      loadRequestIdRef.current = requestId
      setDetailLoading(true)
      setError(null)
      try {
        const detail = await getItinerarySession(sessionId, authSession.access_token)
        if (loadRequestIdRef.current !== requestId) return
        setSession(detail)
        setDismissedSessionId(null)
      } catch (loadError) {
        if (loadRequestIdRef.current !== requestId) return
        setError(loadError instanceof Error ? loadError.message : 'Failed to load itinerary session.')
      } finally {
        if (loadRequestIdRef.current === requestId) {
          setDetailLoading(false)
        }
      }
    },
    [authSession?.access_token],
  )

  useEffect(() => {
    if (!authSession?.access_token) {
      loadRequestIdRef.current += 1
      submitRequestIdRef.current += 1
      setSession(null)
      setDraft('')
      setError(null)
      setPendingMessage(null)
      return
    }
    if (activeSessionId === undefined) {
      return
    }
    if (!activeSessionId) {
      loadRequestIdRef.current += 1
      setSession(null)
      setPendingMessage(null)
      setDetailLoading(false)
      setDismissedSessionId(null)
      return
    }
    if (activeSessionId === dismissedSessionId) {
      return
    }
    if (session?.session_id === activeSessionId) return
    void loadSession(activeSessionId)
  }, [activeSessionId, authSession?.access_token, dismissedSessionId, loadSession, session?.session_id])

  const displayedMessages = useMemo(() => {
    if (!pendingMessage) return session?.messages ?? []
    return [...(session?.messages ?? []), pendingMessage]
  }, [pendingMessage, session?.messages])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [displayedMessages.length, sending, session?.session_id])

  useEffect(() => {
    if (sending || !restoreFocusRef.current) return
    restoreFocusRef.current = false
    inputRef.current?.focus()
  }, [sending])

  const resetConversation = useCallback(() => {
    loadRequestIdRef.current += 1
    submitRequestIdRef.current += 1
    setDismissedSessionId(activeSessionId ?? session?.session_id ?? null)
    setSession(null)
    setDraft('')
    setError(null)
    setSending(false)
    setDetailLoading(false)
    setPendingMessage(null)
    restoreFocusRef.current = true
    onSessionActivated?.(null)
  }, [activeSessionId, onSessionActivated, session?.session_id])

  const handleSubmit = useCallback(async () => {
    const normalized = draft.trim()
    if (!normalized) {
      setError('Describe your trip first.')
      return
    }
    if (!authSession?.access_token) {
      setError('Sign in to start an itinerary session.')
      return
    }

    const requestId = submitRequestIdRef.current + 1
    submitRequestIdRef.current = requestId
    restoreFocusRef.current = true
    setPendingMessage({
      message_id: `pending-${requestId}`,
      session_id: session?.session_id ?? activeSessionId ?? 'pending',
      role: 'user',
      content: normalized,
      metadata: {},
      created_at: new Date().toISOString(),
    })
    setSending(true)
    setError(null)
    setDraft('')

    try {
      let sessionId = session?.session_id ?? activeSessionId ?? null
      if (!sessionId) {
        const created = await createItinerarySession(authSession.access_token)
        if ('session' in created) {
          setSession(created.session)
          sessionId = created.session.session_id
        } else {
          sessionId = created.session_id
        }
        onSessionActivated?.(sessionId)
        onSessionsChanged?.()
        setDismissedSessionId(null)
      }
      if (!sessionId) throw new Error('Failed to initialize itinerary session.')
      const response = await postItinerarySessionMessage(sessionId, normalized, authSession.access_token)
      if (submitRequestIdRef.current !== requestId) return
      setSession(response.session)
      setPendingMessage(null)
      setDismissedSessionId(null)
      onSessionActivated?.(response.session.session_id)
      onSessionsChanged?.()
    } catch (submitError) {
      if (submitRequestIdRef.current !== requestId) return
      setPendingMessage(null)
      setDraft(normalized)
      setError(submitError instanceof Error ? submitError.message : 'Failed to update itinerary.')
    } finally {
      if (submitRequestIdRef.current === requestId) {
        setSending(false)
      }
    }
  }, [activeSessionId, authSession?.access_token, draft, onSessionActivated, onSessionsChanged, session?.session_id])

  const currentItinerary = session?.current_version?.itinerary ?? null
  const showNewChat = Boolean(session || pendingMessage || activeSessionId)

  return (
    <main className="h-full overflow-y-auto bg-background px-4 py-6">
      <div className="mx-auto flex max-w-5xl flex-col gap-4">
        <Card>
          <CardHeader className="flex flex-row items-start justify-between gap-2">
            <div>
              <CardTitle className="text-xl">Itinerary planner</CardTitle>
              <CardDescription>
                Chat through trip details, generate a structured plan, and refine it in place.
              </CardDescription>
            </div>
            {showNewChat ? (
              <Button type="button" variant="outline" size="sm" onClick={resetConversation} disabled={sending}>
                New chat
              </Button>
            ) : null}
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[440px] px-6 py-4">
              <div className="space-y-4">
                {displayedMessages.length ? (
                  displayedMessages.map((message) =>
                    message.role === 'user' ? (
                      <div key={message.message_id} className="flex flex-col items-end">
                        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
                          {message.content}
                        </div>
                      </div>
                    ) : (
                      <div key={message.message_id} className="flex items-start gap-2">
                        <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                        <div className={cn('max-w-[75%]', assistantBubbleClassName)}>
                          <ChatMarkdown content={message.content} />
                        </div>
                      </div>
                    ),
                  )
                ) : detailLoading ? (
                  <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
                    <Loader2 size={14} className="animate-spin" />
                    Loading itinerary session...
                  </div>
                ) : (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    Start a new itinerary chat or choose one from the sidebar.
                  </p>
                )}

                {sending ? (
                  <div className="flex items-start gap-2">
                    <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                    <div className={cn('max-w-[75%]', assistantBubbleClassName)}>
                      <ChatMarkdown content="Planning your next step..." />
                    </div>
                  </div>
                ) : null}
                <div ref={bottomRef} aria-hidden="true" />
              </div>
            </ScrollArea>

            {error ? (
              <p
                role="alert"
                className="mx-6 mb-2 rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-xs text-destructive"
              >
                {error}
              </p>
            ) : null}

            <div className="border-t px-6 py-4">
              <div className="flex items-end gap-2">
                <label htmlFor="itinerary-planner-input" className="sr-only">
                  Describe your trip
                </label>
                <Textarea
                  ref={inputRef}
                  id="itinerary-planner-input"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void handleSubmit()
                    }
                  }}
                  placeholder={
                    currentItinerary
                      ? 'Describe what to change or add to the itinerary...'
                      : 'Tell me where you want to go, for how long, your budget, and what kind of trip you want...'
                  }
                  className="min-h-[120px] max-h-56 resize-y text-sm"
                  rows={4}
                  disabled={sending}
                />
                <Button
                  type="button"
                  size="icon"
                  onClick={() => void handleSubmit()}
                  disabled={sending || !authSession || !draft.trim()}
                  aria-label="Send message"
                >
                  <SendHorizontal size={15} />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {session ? <RequirementsSummary session={session} /> : null}
        <ItineraryView itinerary={currentItinerary} />
      </div>
    </main>
  )
}
