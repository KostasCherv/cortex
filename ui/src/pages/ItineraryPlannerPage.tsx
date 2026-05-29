import { useCallback, useEffect, useRef, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { createItinerarySession, getItinerarySession, postItinerarySessionMessage } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import type {
  GeneratedItinerary,
  ItinerarySessionDetail,
} from '@/types'

function formatDateLabel(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

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

function ItineraryView({ itinerary, versions }: { itinerary: GeneratedItinerary | null; versions: ItinerarySessionDetail['versions'] }) {
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
    <div className="space-y-4">
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
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Version history</CardTitle>
          <CardDescription>Saved revisions for this itinerary session.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {versions.length ? (
            versions.slice().reverse().map((version) => (
              <div key={version.version_id} className="rounded-md border border-border/70 bg-muted/15 px-3 py-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="font-medium">Version {version.version_number}</span>
                  <span className="text-xs text-muted-foreground">{formatDateLabel(version.created_at)}</span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{version.revision_summary}</p>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">No saved versions yet.</p>
          )}
        </CardContent>
      </Card>
    </div>
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
  const requestIdRef = useRef(0)

  const loadSession = useCallback(
    async (sessionId: string) => {
      if (!authSession?.access_token) return
      const requestId = requestIdRef.current + 1
      requestIdRef.current = requestId
      setDetailLoading(true)
      setError(null)
      try {
        const detail = await getItinerarySession(sessionId, authSession.access_token)
        if (requestIdRef.current !== requestId) return
        setSession(detail)
      } catch (loadError) {
        if (requestIdRef.current !== requestId) return
        setError(loadError instanceof Error ? loadError.message : 'Failed to load itinerary session.')
      } finally {
        if (requestIdRef.current === requestId) {
          setDetailLoading(false)
        }
      }
    },
    [authSession?.access_token],
  )

  useEffect(() => {
    if (!authSession?.access_token) {
      setSession(null)
      setDraft('')
      setError(null)
      return
    }
    if (activeSessionId === undefined) {
      return
    }
    if (!activeSessionId) {
      requestIdRef.current += 1
      setSession(null)
      setDetailLoading(false)
      return
    }
    if (session?.session_id === activeSessionId) return
    void loadSession(activeSessionId)
  }, [activeSessionId, authSession?.access_token, loadSession, session?.session_id])

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

    setSending(true)
    setError(null)
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
      }
      if (!sessionId) throw new Error('Failed to initialize itinerary session.')
      const response = await postItinerarySessionMessage(sessionId, normalized, authSession.access_token)
      setSession(response.session)
      setDraft('')
      onSessionActivated?.(response.session.session_id)
      onSessionsChanged?.()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Failed to update itinerary.')
    } finally {
      setSending(false)
    }
  }, [activeSessionId, authSession?.access_token, draft, onSessionActivated, onSessionsChanged, session?.session_id])

  const currentItinerary = session?.current_version?.itinerary ?? null

  return (
    <main className="h-full overflow-y-auto bg-background px-4 py-6">
      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
        <Card className="min-h-[70vh]">
          <CardHeader>
            <CardTitle className="text-xl">Itinerary planner</CardTitle>
            <CardDescription>
              Chat through trip details, generate a structured plan, and refine it in place.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex h-[calc(70vh-5rem)] flex-col gap-4">
            {detailLoading ? (
              <div className="flex-1 rounded-lg border border-dashed border-border/70 bg-muted/20 p-4 text-sm text-muted-foreground">
                Loading itinerary session...
              </div>
            ) : session ? (
              <ScrollArea className="flex-1 rounded-lg border border-border/70 bg-muted/10 p-4">
                <div className="space-y-4">
                  {session.messages.map((message) => (
                    <div
                      key={message.message_id}
                      className={message.role === 'user' ? 'ml-auto max-w-[85%]' : 'max-w-[85%]'}
                    >
                      <div
                        className={
                          message.role === 'user'
                            ? 'rounded-2xl rounded-br-md bg-primary px-4 py-3 text-sm text-primary-foreground'
                            : 'rounded-2xl rounded-bl-md border border-border/70 bg-background px-4 py-3 text-sm text-foreground'
                        }
                      >
                        {message.content}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="flex-1 rounded-lg border border-dashed border-border/70 bg-muted/20 p-4 text-sm text-muted-foreground">
                Start a new itinerary chat or choose one from the sidebar.
              </div>
            )}

            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <div className="space-y-2">
              <label htmlFor="itinerary-planner-input" className="text-sm font-medium">
                Describe your trip
              </label>
              <Textarea
                id="itinerary-planner-input"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Tell me where you want to go, for how long, your budget, and what kind of trip you want."
                className="min-h-[120px]"
                disabled={sending}
              />
            </div>
            <div className="flex justify-end">
              <Button type="button" onClick={() => void handleSubmit()} disabled={sending || !authSession}>
                {sending ? 'Sending...' : 'Send'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {session ? <RequirementsSummary session={session} /> : null}
          <ItineraryView itinerary={currentItinerary} versions={session?.versions ?? []} />
        </div>
      </div>
    </main>
  )
}
