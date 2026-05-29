import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Download, Loader2, MessageSquare, SendHorizontal, Square, Zap } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { generatePRD, getPRD } from '@/api/client'
import { getPlannerChatMessages, streamPlannerChat } from '@/api/plannerChatClient'
import { SavedPlanDetailView } from '@/components/planner/SavedPlanDetailView'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import type { PlannerChatMessage, PlannerChatStreamEvent, SavedPRD } from '@/types'
import { assistantAvatarClassName, assistantBubbleClassName, ChatMarkdown } from '@/components/chat/ChatMarkdown'
import { cn } from '@/lib/utils'

function downloadMarkdown(markdown: string, filename: string) {
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function planEventToSavedPlan(
  event: PlannerChatStreamEvent & { type: 'plan' },
  prompt: string,
): SavedPRD {
  return {
    plan_id: '',
    plan: event.plan,
    markdown: event.markdown,
    suggested_filename: event.suggested_filename,
    planning_brief: event.planning_brief,
    prompt,
    prompt_preview: prompt.slice(0, 120),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }
}

// ---------------------------------------------------------------------------
// Interactive chat component
// ---------------------------------------------------------------------------

type ChatMessage = PlannerChatMessage & { plan_saved?: SavedPRD }

const THREAD_STORAGE_KEY = 'planner_interactive_thread_id'

function PlannerInteractiveChat({
  accessToken,
  onPlansChanged,
}: {
  accessToken: string
  onPlansChanged?: () => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [threadId, setThreadId] = useState<string | null>(() => sessionStorage.getItem(THREAD_STORAGE_KEY))
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [latestPlan, setLatestPlan] = useState<SavedPRD | null>(null)
  const [restoring, setRestoring] = useState(() => Boolean(sessionStorage.getItem(THREAD_STORAGE_KEY)))
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const requestIdRef = useRef(0)

  // Restore conversation history when mounting with a previously-used thread
  useEffect(() => {
    const storedId = sessionStorage.getItem(THREAD_STORAGE_KEY)
    if (!storedId) return
    setRestoring(true)
    getPlannerChatMessages(storedId, accessToken)
      .then(({ messages: history }) => {
        setMessages(history.map((m) => ({ ...m })))
        setThreadId(storedId)
      })
      .catch(() => {
        // Thread expired on the server — start fresh
        sessionStorage.removeItem(THREAD_STORAGE_KEY)
        setThreadId(null)
        setMessages([])
      })
      .finally(() => setRestoring(false))
  // Only run once on mount — accessToken is stable for the component lifetime
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist thread ID whenever it changes
  useEffect(() => {
    if (threadId) {
      sessionStorage.setItem(THREAD_STORAGE_KEY, threadId)
    } else {
      sessionStorage.removeItem(THREAD_STORAGE_KEY)
    }
  }, [threadId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  const send = async (overrideText?: string) => {
    const text = (overrideText ?? input).trim()
    if (!text || streaming) return

    const requestId = ++requestIdRef.current
    const userMessage: ChatMessage = {
      message_id: `tmp-user-${requestId}`,
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setStreamingText('')
    setStreaming(true)
    setError(null)

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    let accumulated = ''
    let planEvent: (PlannerChatStreamEvent & { type: 'plan' }) | null = null
    let currentThreadId = threadId
    let streamFailed = false

    try {
      await streamPlannerChat(text, threadId, accessToken, {
        signal: controller.signal,
        onSession: (newThreadId) => {
          if (requestIdRef.current !== requestId) return
          currentThreadId = newThreadId
          setThreadId(newThreadId)
        },
        onChunk: (chunk) => {
          if (requestIdRef.current !== requestId) return
          accumulated += chunk
          setStreamingText((prev) => prev + chunk)
        },
        onPlan: (event) => {
          if (requestIdRef.current !== requestId) return
          planEvent = event
        },
        onDone: () => {
          if (requestIdRef.current !== requestId) return
          const savedPlan = planEvent ? planEventToSavedPlan(planEvent, text) : null
          const assistantMessage: ChatMessage = {
            message_id: `tmp-assistant-${requestId}`,
            role: 'assistant',
            content: accumulated.trim(),
            created_at: new Date().toISOString(),
            plan_saved: savedPlan ?? undefined,
          }
          const nextMessages: ChatMessage[] = [assistantMessage]
          if (savedPlan) {
            nextMessages.push({
              message_id: `hint-${requestId}`,
              role: 'assistant',
              content: "_Plan generated. Describe any changes you’d like to make — the AI will produce a refined version._",
              created_at: new Date().toISOString(),
            })
          }
          setMessages((prev) => [...prev, ...nextMessages])
          setStreamingText('')
          if (savedPlan) {
            setLatestPlan(savedPlan)
            onPlansChanged?.()
          }
          void currentThreadId
        },
        onError: (err) => {
          streamFailed = true
          if (requestIdRef.current !== requestId) return
          setError(err)
          setStreamingText('')
        },
      })
    } catch (err) {
      if (controller.signal.aborted) return
      if (requestIdRef.current === requestId) {
        setError(err instanceof Error ? err.message : 'Chat failed.')
        setStreamingText('')
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null
      if (requestIdRef.current === requestId) {
        setStreaming(false)
        if (streamFailed) onPlansChanged?.()
      }
    }
  }

  const lastPlan = useMemo(
    () =>
      latestPlan ??
      [...messages]
        .reverse()
        .map((m) => m.plan_saved)
        .find(Boolean) ??
      null,
    [latestPlan, messages],
  )

  const handleNewChat = () => {
    abortRef.current?.abort()
    setThreadId(null)
    setMessages([])
    setStreamingText('')
    setStreaming(false)
    setError(null)
    setLatestPlan(null)
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-2">
          <div>
            <CardTitle className="text-xl">Interactive planner</CardTitle>
            <CardDescription>
              The AI will ask clarifying questions before generating your PRD.
            </CardDescription>
          </div>
          {(messages.length > 0 || threadId) && (
            <Button type="button" variant="outline" size="sm" onClick={handleNewChat} disabled={streaming}>
              New chat
            </Button>
          )}
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[420px] px-6 py-4">
            <div className="space-y-4">
              {restoring && (
                <div className="flex items-center gap-2 py-8 justify-center text-sm text-muted-foreground">
                  <Loader2 size={14} className="animate-spin" />
                  Restoring conversation...
                </div>
              )}
              {!restoring && messages.length === 0 && (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  Describe a feature or goal to start. The AI will ask a few questions first.
                </p>
              )}
              {messages.map((m) =>
                m.role === 'user' ? (
                  <div key={m.message_id} className="flex flex-col items-end">
                    <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <div key={m.message_id} className="flex gap-2 items-start">
                    <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                    <div className={cn('max-w-[75%]', assistantBubbleClassName)}>
                      <ChatMarkdown content={m.content} />
                    </div>
                  </div>
                ),
              )}
              {streaming && (
                <div className="flex gap-2 items-start">
                  <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                  <div className={cn('max-w-[75%]', assistantBubbleClassName)}>
                    <ChatMarkdown content={streamingText || 'Thinking...'} />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>

          {error && (
            <p
              role="alert"
              className="mx-6 mb-2 rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-xs text-destructive"
            >
              {error}
            </p>
          )}

          <div className="border-t px-6 py-4">
            <div className="flex gap-2 items-end">
              <Textarea
                className="resize-none min-h-10 max-h-32 text-sm"
                placeholder={lastPlan ? 'Describe what to change or add to the PRD…' : 'Describe your product idea, the problem it solves, and who it\'s for…'}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void send()
                  }
                }}
                disabled={streaming}
              />
              {streaming ? (
                <Button
                  size="icon"
                  variant="secondary"
                  onClick={() => {
                    abortRef.current?.abort()
                    setStreamingText('')
                    setStreaming(false)
                    setError(null)
                  }}
                  aria-label="Stop generating"
                >
                  <Square size={15} />
                </Button>
              ) : (
                <Button
                  size="icon"
                  onClick={() => void send()}
                  disabled={!input.trim()}
                  aria-label="Send message"
                >
                  <SendHorizontal size={15} />
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {lastPlan && (
        <>
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">Generated plan</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => downloadMarkdown(lastPlan.markdown, lastPlan.suggested_filename)}
            >
              <Download size={14} />
              Download markdown
            </Button>
          </div>
          <SavedPlanDetailView result={lastPlan} />
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type Mode = 'single-shot' | 'interactive'

export function PRDPlannerPage({
  authSession,
  activePlanId,
  onPlanActivated,
  onPlansChanged,
}: {
  authSession: Session | null
  activePlanId?: string | null
  onPlanActivated?: (planId: string | null) => void
  onPlansChanged?: () => void
}) {
  const [mode, setMode] = useState<Mode>('single-shot')
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState<SavedPRD | null>(null)
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const detailRequestIdRef = useRef(0)

  const signedIn = Boolean(authSession?.access_token)
  const canSubmit = signedIn && prompt.trim().length > 0 && !loading

  const handleSelectPlan = useCallback(
    async (planId: string) => {
      setSelectedPlanId(planId)
      if (!authSession?.access_token) {
        setError('Sign in to load a saved plan.')
        return
      }
      if (result?.plan_id === planId) {
        return
      }

      const requestId = detailRequestIdRef.current + 1
      detailRequestIdRef.current = requestId
      setDetailLoading(true)
      setResult(null)
      setError(null)
      try {
        const savedPlan = await getPRD(planId, authSession.access_token)
        if (detailRequestIdRef.current !== requestId) return
        setResult(savedPlan)
      } catch (detailError) {
        if (detailRequestIdRef.current !== requestId) return
        setError(detailError instanceof Error ? detailError.message : 'Failed to load saved plan.')
      } finally {
        if (detailRequestIdRef.current === requestId) {
          setDetailLoading(false)
        }
      }
    },
    [authSession?.access_token, result?.plan_id],
  )

  useEffect(() => {
    if (!signedIn || activePlanId === undefined) return
    if (!activePlanId) {
      detailRequestIdRef.current += 1
      setSelectedPlanId(null)
      setResult(null)
      setDetailLoading(false)
      return
    }
    if (result?.plan_id === activePlanId) {
      setSelectedPlanId(activePlanId)
      return
    }
    void handleSelectPlan(activePlanId)
  }, [activePlanId, handleSelectPlan, result?.plan_id, signedIn])

  const handleSubmit = useCallback(async () => {
    const normalizedPrompt = prompt.trim()
    if (!normalizedPrompt) {
      setError('Describe the feature or implementation goal first.')
      return
    }
    if (!authSession?.access_token) {
      setError('Sign in to generate a PRD.')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await generatePRD(normalizedPrompt, authSession.access_token)
      detailRequestIdRef.current += 1
      setResult(response)
      setSelectedPlanId(response.plan_id)
      onPlanActivated?.(response.plan_id)
      onPlansChanged?.()
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : 'Failed to generate plan.')
    } finally {
      setLoading(false)
    }
  }, [authSession?.access_token, onPlanActivated, onPlansChanged, prompt])

  useEffect(() => {
    if (signedIn) return
    detailRequestIdRef.current += 1
    setResult(null)
    setSelectedPlanId(null)
    setDetailLoading(false)
    setError(null)
  }, [signedIn])

  return (
    <main className="h-full overflow-y-auto bg-background px-4 py-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-4">
        {/* Mode toggle */}
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant={mode === 'single-shot' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setMode('single-shot')}
          >
            <Zap size={14} />
            Single-shot
          </Button>
          <Button
            type="button"
            variant={mode === 'interactive' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setMode('interactive')}
          >
            <MessageSquare size={14} />
            Interactive
          </Button>
        </div>

        {mode === 'interactive' ? (
          signedIn && authSession?.access_token ? (
            <PlannerInteractiveChat
              accessToken={authSession.access_token}
              onPlansChanged={onPlansChanged}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-xl">Interactive planner</CardTitle>
                <CardDescription>Sign in to use the interactive planning mode.</CardDescription>
              </CardHeader>
            </Card>
          )
        ) : (
          <>
            <Card>
              <CardHeader>
                <CardTitle className="text-xl">PRD Planner</CardTitle>
                <CardDescription>
                  Turn a product idea into a structured Product Requirements Document.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="prd-planner-prompt" className="text-sm font-medium">
                    What product idea should we document?
                  </label>
                  <Textarea
                    id="prd-planner-prompt"
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="Describe the product, feature, or initiative — the problem it solves, who it's for, and any known constraints."
                    className="min-h-[160px]"
                    disabled={loading}
                  />
                </div>
                {!signedIn && (
                  <p className="text-sm text-muted-foreground">
                    Sign in to generate a PRD.
                  </p>
                )}
                {error && (
                  <p role="alert" className="text-sm text-destructive">
                    {error}
                  </p>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  <Button type="button" onClick={() => void handleSubmit()} disabled={!canSubmit}>
                    {loading ? <Loader2 className="animate-spin" size={16} /> : null}
                    Generate PRD
                  </Button>
                  {result && (
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => downloadMarkdown(result.markdown, result.suggested_filename)}
                    >
                      <Download size={16} />
                      Download markdown
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {detailLoading ? (
              <Card>
                <CardContent className="py-12">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="animate-spin" size={16} />
                    Loading saved plan...
                  </div>
                </CardContent>
              </Card>
            ) : result ? (
              <SavedPlanDetailView result={result} />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Select a saved plan</CardTitle>
                  <CardDescription>
                    {selectedPlanId
                      ? 'Loading the selected saved plan from the sidebar...'
                      : 'Generate a new plan or choose one from the sidebar to view its details.'}
                  </CardDescription>
                </CardHeader>
              </Card>
            )}
          </>
        )}
      </div>
    </main>
  )
}
