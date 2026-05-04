import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, SendHorizontal } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getRagAgentChatSessionMessages, streamRagAgentChat } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { RagAgent, RagChatMessage } from '@/types'

type Props = {
  agent: RagAgent
  accessToken: string
  activeSessionId: string | null
  onSessionActivated: (id: string | null) => void
  onSessionsChanged: () => void
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

function CitationMarkers({ citations }: { citations: RagChatMessage['citations'] }) {
  if (citations.length === 0) return null

  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      {citations.map((citation, index) => {
        const key = citation.chunk_id || citation.source_url || `${citation.source_title}-${index}`

        return (
          <span key={key} className="group relative inline-flex">
            <button
              type="button"
              className="inline-flex h-6 min-w-6 items-center justify-center rounded-full border border-border/70 bg-background px-1.5 text-[11px] font-medium text-muted-foreground transition-colors hover:border-foreground/20 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              aria-label={`Show citation ${index + 1}`}
            >
              [{index + 1}]
            </button>
            <div className="absolute left-0 top-full z-20 hidden w-80 pt-2 group-hover:block group-focus-within:block">
              <div className="rounded-xl border border-border/80 bg-background/95 p-3 shadow-xl backdrop-blur-sm">
                <div className="mb-2 flex items-center gap-2">
                  <Badge variant="outline" className="text-[11px] font-normal">
                    [{index + 1}]
                  </Badge>
                  {citation.source_url ? (
                    <a
                      href={citation.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="truncate text-xs font-medium text-foreground hover:underline"
                    >
                      {citation.source_title || 'source'}
                    </a>
                  ) : (
                    <span className="truncate text-xs font-medium text-foreground">
                      {citation.source_title || 'source'}
                    </span>
                  )}
                </div>
                <div className="max-h-56 overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                  {citation.text}
                </div>
              </div>
            </div>
          </span>
        )
      })}
    </div>
  )
}

export function AgentChat({ agent, accessToken, activeSessionId, onSessionActivated, onSessionsChanged }: Props) {
  const [messages, setMessages] = useState<RagChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [chatting, setChatting] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [webSearchEnabled, setWebSearchEnabled] = useState(false)
  const [webUsedLastReply, setWebUsedLastReply] = useState(false)
  const [latestSuggestions, setLatestSuggestions] = useState<string[]>([])

  const messagesRequestRef = useRef(0)
  const loadedSessionRef = useRef<string | null>(null)
  const currentAgentIdRef = useRef(agent.agent_id)
  const chatAbortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    currentAgentIdRef.current = agent.agent_id
  }, [agent.agent_id])

  // Reset when agent changes
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
    setWebSearchEnabled(false)
    setWebUsedLastReply(false)
    setLatestSuggestions([])
  }, [agent.agent_id])

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const openSession = useCallback(
    async (nextSessionId: string) => {
      if (loadedSessionRef.current === nextSessionId) return
      const requestId = ++messagesRequestRef.current
      try {
        const res = await getRagAgentChatSessionMessages(agent.agent_id, nextSessionId, accessToken)
        if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
        loadedSessionRef.current = res.session_id
        setSessionId(res.session_id)
        setMessages(res.messages)
        setWebSearchEnabled(Boolean(res.web_search_enabled))
        setError(null)
      } catch (err) {
        if (requestId === messagesRequestRef.current && currentAgentIdRef.current === agent.agent_id) {
          setError(err instanceof Error ? err.message : 'Failed to load chat session.')
        }
      }
    },
    [accessToken, agent.agent_id],
  )

  // Respond to session selection from the rail
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
      agent_id: agent.agent_id,
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
      await streamRagAgentChat(agent.agent_id, question, sessionId, webSearchEnabled, accessToken, {
        signal: controller.signal,
        onSession: (nextSessionId, nextWebSearchEnabled, webUsed) => {
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          streamedSessionId = nextSessionId
          loadedSessionRef.current = nextSessionId
          setSessionId(nextSessionId)
          if (typeof nextWebSearchEnabled === 'boolean') {
            setWebSearchEnabled(nextWebSearchEnabled)
          }
          setWebUsedLastReply(Boolean(webUsed))
          if (!sessionId) {
            onSessionActivated(nextSessionId)
          }
        },
        onChunk: (textChunk) => {
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          if (controller.signal.aborted) return
          accumulated += textChunk
          setStreamingText((prev) => prev + textChunk)
        },
        onCitations: (citations) => {
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          if (controller.signal.aborted) return
          finalCitations = citations
        },
        onSuggestions: (suggestions) => {
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          if (controller.signal.aborted) return
          pendingSuggestions = suggestions
          setLatestSuggestions(suggestions)
        },
        onDone: () => {
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          const finalSessionId = streamedSessionId ?? sessionId ?? 'pending'
          const assistantMessage: RagChatMessage = {
            message_id: `tmp-assistant-${requestId}`,
            session_id: finalSessionId,
            agent_id: agent.agent_id,
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
          if (requestId !== messagesRequestRef.current || currentAgentIdRef.current !== agent.agent_id) return
          setError(streamError)
          setStreamingText('')
        },
      })
    } catch (err) {
      if (controller.signal.aborted) return
      if (requestId === messagesRequestRef.current && currentAgentIdRef.current === agent.agent_id) {
        setError(err instanceof Error ? err.message : 'Chat failed.')
        setStreamingText('')
      }
    } finally {
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null
      }
      if (requestId === messagesRequestRef.current && currentAgentIdRef.current === agent.agent_id) {
        setChatting(false)
        if (streamFailed) {
          onSessionsChanged()
        }
      }
    }
  }

  const suggestions =
    latestSuggestions.length > 0
      ? latestSuggestions
      : ([...messages].reverse().find((m) => m.role === 'assistant')?.suggestions ?? [])

  return (
    <div className="flex h-dvh flex-col max-md:h-full">
      {/* Agent context header */}
      <div className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-6 max-md:px-4">
        <div className="min-w-0">
          <p className="font-medium text-sm">{agent.name}</p>
          {agent.description && (
            <p className="text-xs text-muted-foreground truncate max-w-md">{agent.description}</p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Web search</span>
            <Switch
              checked={webSearchEnabled}
              onCheckedChange={setWebSearchEnabled}
              disabled={chatting}
              aria-label="Enable web search"
            />
          </div>
          <span className="shrink-0 text-xs text-muted-foreground">
            {agent.linked_resource_ids.length} resources
          </span>
          {webUsedLastReply && <Badge variant="outline">Web used</Badge>}
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="min-h-0 flex-1 px-6 py-6 max-md:px-4">
        <div className="space-y-4">
          {messages.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Ask {agent.name} about its linked resources.
            </p>
          )}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.message_id} className="flex justify-end">
                <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground max-md:max-w-[86%]">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={m.message_id} className="flex flex-col gap-2">
                <div className="flex gap-2 items-start">
                  <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold">
                    AI
                  </div>
                  <div className="max-w-[75%] rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm max-md:max-w-[86%]">
                    <MarkdownMessage content={m.content} />
                    <CitationMarkers citations={m.citations} />
                  </div>
                </div>
                {messages.at(-1)?.message_id === m.message_id && suggestions.length > 0 && (
                  <div className="ml-9 flex max-w-[75%] flex-wrap gap-2 max-md:max-w-[86%]">
                    {suggestions.map((text) => (
                      <Button
                        key={text}
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => void send(text)}
                        disabled={chatting}
                      >
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
              <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold">
                AI
              </div>
              <div className="max-w-[75%] rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm max-md:max-w-[86%]">
                <MarkdownMessage content={streamingText || 'Thinking...'} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {error && (
        <p role="alert" className="mx-6 mb-2 shrink-0 rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-xs text-destructive max-md:mx-4">
          {error}
        </p>
      )}

      {/* Composer */}
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
