import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, SendHorizontal } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { RagChatMessage } from '@/types'
import type { ChatTransport } from './transports'

type Props = {
  transport: ChatTransport
  accessToken: string
  activeSessionId: string | null
  onSessionActivated: (id: string | null) => void
  onSessionsChanged: () => void
  title: string
  subtitle?: string
  emptyState: string
  resourceLabel?: string
  defaultWebSearchEnabled?: boolean
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="overflow-x-auto">
      <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-0 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:my-2 prose-code:before:content-none prose-code:after:content-none prose-table:my-2 prose-th:border prose-th:border-border prose-th:px-2 prose-th:py-1 prose-th:text-left prose-td:border prose-td:border-border prose-td:px-2 prose-td:py-1">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  )
}

function CitationMarker({ citation, index }: { citation: RagChatMessage['citations'][number]; index: number }) {
  const triggerRef = useRef<HTMLButtonElement>(null)
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })

  const clearHideTimeout = useCallback(() => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
      hideTimeoutRef.current = null
    }
  }, [])

  const show = useCallback(() => {
    clearHideTimeout()
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    const width = 320
    const padding = 12
    const left = Math.min(rect.left, window.innerWidth - width - padding)
    setPosition({ top: rect.top + 4, left: Math.max(padding, left) })
    setOpen(true)
  }, [clearHideTimeout])

  const scheduleHide = useCallback(() => {
    clearHideTimeout()
    hideTimeoutRef.current = setTimeout(() => setOpen(false), 120)
  }, [clearHideTimeout])

  useEffect(() => clearHideTimeout, [clearHideTimeout])

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="inline-flex h-6 min-w-6 items-center justify-center rounded-full border border-border/70 bg-background px-1.5 text-[11px] font-medium text-muted-foreground transition-colors hover:border-foreground/20 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        aria-label={`Show citation ${index + 1}`}
        onMouseEnter={show}
        onMouseLeave={scheduleHide}
        onFocus={show}
        onBlur={scheduleHide}
      >
        [{index + 1}]
      </button>
      {open &&
        createPortal(
          <div
            role="tooltip"
            className="fixed z-50 w-80"
            style={{ top: position.top, left: position.left, transform: 'translateY(-100%)' }}
            onMouseEnter={show}
            onMouseLeave={scheduleHide}
          >
            <div className="rounded-xl border border-border/80 bg-background/95 p-3 shadow-xl backdrop-blur-sm">
              <div className="mb-2 flex items-center gap-2">
                <Badge variant="outline" className="text-[11px] font-normal">[{index + 1}]</Badge>
                {citation.source_url ? (
                  <a href={citation.source_url} target="_blank" rel="noopener noreferrer" className="truncate text-xs font-medium text-foreground hover:underline">
                    {citation.source_title || 'source'}
                  </a>
                ) : (
                  <span className="truncate text-xs font-medium text-foreground">{citation.source_title || 'source'}</span>
                )}
              </div>
              <div className="max-h-56 overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">{citation.text}</div>
            </div>
            <div aria-hidden className="h-3" />
          </div>,
          document.body,
        )}
    </>
  )
}

function CitationMarkers({ citations }: { citations: RagChatMessage['citations'] }) {
  if (citations.length === 0) return null
  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      {citations.map((citation, index) => {
        const key = citation.chunk_id || citation.source_url || `${citation.source_title}-${index}`
        return <CitationMarker key={key} citation={citation} index={index} />
      })}
    </div>
  )
}

export function ChatThreadContainer({
  transport,
  accessToken,
  activeSessionId,
  onSessionActivated,
  onSessionsChanged,
  title,
  subtitle,
  emptyState,
  resourceLabel,
  defaultWebSearchEnabled = false,
}: Props) {
  const [messages, setMessages] = useState<RagChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [chatting, setChatting] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [webSearchEnabled, setWebSearchEnabled] = useState(defaultWebSearchEnabled)
  const [webUsedLastReply, setWebUsedLastReply] = useState(false)
  const [latestSuggestions, setLatestSuggestions] = useState<string[]>([])

  const messagesRequestRef = useRef(0)
  const loadedSessionRef = useRef<string | null>(null)
  const currentTransportKeyRef = useRef(transport.key)
  const chatAbortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    currentTransportKeyRef.current = transport.key
  }, [transport.key])

  useEffect(() => {
    messagesRequestRef.current += 1
    chatAbortRef.current?.abort()
    chatAbortRef.current = null
    loadedSessionRef.current = null
    setSessionId(null)
    setMessages([])
    setInput('')
    setStreamingText('')
    setError(null)
    setWebSearchEnabled(defaultWebSearchEnabled)
    setWebUsedLastReply(false)
    setLatestSuggestions([])
  }, [transport.key, defaultWebSearchEnabled])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const openSession = useCallback(
    async (nextSessionId: string) => {
      if (loadedSessionRef.current === nextSessionId) return
      const requestId = ++messagesRequestRef.current
      try {
        const res = await transport.loadSessionMessages(nextSessionId, accessToken)
        if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
        loadedSessionRef.current = res.session_id
        setSessionId(res.session_id)
        setMessages(res.messages)
        setWebSearchEnabled(Boolean(res.web_search_enabled))
        setError(null)
      } catch (err) {
        if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
          setError(err instanceof Error ? err.message : 'Failed to load chat session.')
        }
      }
    },
    [accessToken, transport],
  )

  useEffect(() => {
    if (!activeSessionId) {
      if (activeSessionId === null && loadedSessionRef.current !== null) {
        loadedSessionRef.current = null
        setSessionId(null)
        setMessages([])
      }
      return
    }
    void openSession(activeSessionId)
  }, [activeSessionId, openSession])

  const send = async (overrideText?: string) => {
    const text = overrideText ?? input
    if (!text.trim() || chatting) return
    const question = text.trim()
    const requestId = ++messagesRequestRef.current
    const optimisticUserMessage: RagChatMessage = {
      message_id: `tmp-user-${requestId}`,
      session_id: sessionId ?? 'pending',
      agent_id: null,
      owner_id: '',
      role: 'user',
      content: question,
      citations: [],
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, optimisticUserMessage])
    setInput('')
    setStreamingText('')
    setChatting(true)
    setError(null)
    setLatestSuggestions([])

    chatAbortRef.current?.abort()
    const controller = new AbortController()
    chatAbortRef.current = controller

    let streamedSessionId = sessionId
    let accumulated = ''
    let finalCitations: RagChatMessage['citations'] = []
    let pendingSuggestions: string[] = []
    let streamFailed = false

    try {
      await transport.streamMessage(question, sessionId, webSearchEnabled, accessToken, {
        signal: controller.signal,
        onSession: (nextSessionId, nextWebSearchEnabled, webUsed) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          streamedSessionId = nextSessionId
          loadedSessionRef.current = nextSessionId
          setSessionId(nextSessionId)
          if (typeof nextWebSearchEnabled === 'boolean') setWebSearchEnabled(nextWebSearchEnabled)
          setWebUsedLastReply(Boolean(webUsed))
          if (!sessionId) onSessionActivated(nextSessionId)
        },
        onChunk: (textChunk) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          if (controller.signal.aborted) return
          accumulated += textChunk
          setStreamingText((prev) => prev + textChunk)
        },
        onCitations: (citations) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          if (controller.signal.aborted) return
          finalCitations = citations
        },
        onSuggestions: (suggestions) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          if (controller.signal.aborted) return
          pendingSuggestions = suggestions
          setLatestSuggestions(suggestions)
        },
        onDone: () => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          const finalSessionId = streamedSessionId ?? sessionId ?? 'pending'
          const assistantMessage: RagChatMessage = {
            message_id: `tmp-assistant-${requestId}`,
            session_id: finalSessionId,
            agent_id: null,
            owner_id: '',
            role: 'assistant',
            content: accumulated.trim(),
            citations: finalCitations,
            suggestions: pendingSuggestions,
            created_at: new Date().toISOString(),
          }
          loadedSessionRef.current = finalSessionId
          setSessionId(finalSessionId)
          setMessages((prev) => [...prev, assistantMessage])
          setStreamingText('')
          onSessionsChanged()
        },
        onError: (streamError) => {
          streamFailed = true
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          setError(streamError)
          setStreamingText('')
        },
      })
    } catch (err) {
      if (controller.signal.aborted) return
      if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
        setError(err instanceof Error ? err.message : 'Chat failed.')
        setStreamingText('')
      }
    } finally {
      if (chatAbortRef.current === controller) chatAbortRef.current = null
      if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
        setChatting(false)
        if (streamFailed) onSessionsChanged()
      }
    }
  }

  const suggestions = useMemo(
    () =>
      latestSuggestions.length > 0
        ? latestSuggestions
        : ([...messages].reverse().find((m) => m.role === 'assistant')?.suggestions ?? []),
    [latestSuggestions, messages],
  )

  return (
    <div className="flex h-dvh flex-col max-md:h-full">
      <div className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-6 max-md:px-4">
        <div className="min-w-0">
          <p className="font-medium text-sm">{title}</p>
          {subtitle && <p className="text-xs text-muted-foreground truncate max-w-md">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Web search</span>
            <Switch checked={webSearchEnabled} onCheckedChange={setWebSearchEnabled} disabled={chatting} aria-label="Enable web search" />
          </div>
          {resourceLabel && <span className="shrink-0 text-xs text-muted-foreground">{resourceLabel}</span>}
          {webUsedLastReply && <Badge variant="outline">Web used</Badge>}
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1 px-6 py-6 max-md:px-4">
        <div className="space-y-4">
          {messages.length === 0 && <p className="py-8 text-center text-sm text-muted-foreground">{emptyState}</p>}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.message_id} className="flex justify-end">
                <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground max-md:max-w-[86%]">{m.content}</div>
              </div>
            ) : (
              <div key={m.message_id} className="flex flex-col gap-2">
                <div className="flex gap-2 items-start">
                  <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold">AI</div>
                  <div className="max-w-[75%] rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm max-md:max-w-[86%]">
                    <MarkdownMessage content={m.content} />
                    <CitationMarkers citations={m.citations} />
                  </div>
                </div>
                {messages.at(-1)?.message_id === m.message_id && suggestions.length > 0 && (
                  <div className="ml-9 flex max-w-[75%] flex-wrap gap-2 max-md:max-w-[86%]">
                    {suggestions.map((text) => (
                      <Button key={text} type="button" variant="outline" size="sm" className="h-7 text-xs" onClick={() => void send(text)} disabled={chatting}>
                        {text}
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            ),
          )}
          {chatting && (
            <div className="flex gap-2 items-start">
              <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold">AI</div>
              <div className="max-w-[75%] rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm max-md:max-w-[86%]">
                <MarkdownMessage content={streamingText || 'Thinking...'} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {error && <p role="alert" className="mx-6 mb-2 shrink-0 rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-xs text-destructive max-md:mx-4">{error}</p>}

      <div className="shrink-0 border-t bg-background px-6 py-4 max-md:px-4">
        <div className="flex gap-2 items-end">
          <Textarea
            className="resize-none min-h-10 max-h-32 text-sm"
            placeholder="Ask a question..."
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
            disabled={chatting}
          />
          <Button
            size="icon"
            onClick={() => void send()}
            disabled={!input.trim() || chatting}
            className={cn(chatting && 'opacity-50')}
            aria-label={chatting ? 'Sending message' : 'Send message'}
          >
            {chatting ? <Loader2 size={15} className="animate-spin" /> : <SendHorizontal size={15} />}
          </Button>
        </div>
      </div>
    </div>
  )
}
