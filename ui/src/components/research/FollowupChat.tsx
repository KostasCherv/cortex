import { useCallback, useEffect, useRef, useState } from 'react'
import { SendHorizontal } from 'lucide-react'
import { assistantAvatarClassName, assistantBubbleClassName, ChatMarkdown } from '@/components/chat/ChatMarkdown'
import { streamFollowup } from '@/api/client'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import type { Citation, ConversationTurn } from '@/types'

type FollowupChatProps = {
  sessionId: string
  runId: string | null
  accessToken: string | null
  conversation: ConversationTurn[]
  onConversationUpdate: (turn: ConversationTurn) => void
  onUsageChanged?: () => void
}

function CitationBadges({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {citations.map((c) => (
        <a key={c.source_url} href={c.source_url} target="_blank" rel="noopener noreferrer">
          <Badge variant="outline" className="text-xs hover:bg-muted">
            {c.source_title || 'source'}
          </Badge>
        </a>
      ))}
    </div>
  )
}

export function FollowupChat({
  sessionId,
  runId,
  accessToken,
  conversation,
  onConversationUpdate,
  onUsageChanged,
}: FollowupChatProps) {
  const [question, setQuestion] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [latestSuggestions, setLatestSuggestions] = useState<string[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation, streamingText])

  const handleQuestionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [])

  const submit = useCallback(
    async (overrideText?: string) => {
      const text = overrideText ?? question
      const q = text.trim()
      if (!q || streaming) return

      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      const userTurn: ConversationTurn = {
        role: 'user',
        content: q,
        run_id: runId,
        citations: [],
        created_at: new Date().toISOString(),
      }
      onConversationUpdate(userTurn)
      setQuestion('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
      setStreamingText('')
      setError(null)
      setStreaming(true)
      setLatestSuggestions([])

      let accumulatedAnswer = ''
      let finalCitations: Citation[] = []
      let pendingSuggestions: string[] = []

      try {
        await streamFollowup(sessionId, q, runId, accessToken, {
          signal: controller.signal,
          onChunk: (textChunk) => {
            if (!controller.signal.aborted) {
              accumulatedAnswer += textChunk
              setStreamingText((prev) => prev + textChunk)
            }
          },
          onCitations: (citations) => {
            if (!controller.signal.aborted) {
              finalCitations = citations
            }
          },
          onSuggestions: (suggestions) => {
            if (!controller.signal.aborted) {
              pendingSuggestions = suggestions
              setLatestSuggestions(suggestions)
            }
          },
          onDone: () => {
            const assistantTurn: ConversationTurn = {
              role: 'assistant',
              content: accumulatedAnswer,
              run_id: runId,
              citations: finalCitations,
              suggestions: pendingSuggestions,
              created_at: new Date().toISOString(),
            }
            onConversationUpdate(assistantTurn)
            setStreamingText('')
            setStreaming(false)
          },
          onError: (err) => {
            setError(err)
            setStreaming(false)
            setStreamingText('')
          },
        })
      } catch (err) {
        if (controller.signal.aborted) return
        setError(err instanceof Error ? err.message : 'Follow-up failed.')
        setStreaming(false)
        setStreamingText('')
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
        }
        if (!controller.signal.aborted) {
          onUsageChanged?.()
        }
      }
    },
    [question, streaming, sessionId, runId, accessToken, onConversationUpdate, onUsageChanged],
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        void submit()
      }
    },
    [submit],
  )

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      void submit()
    },
    [submit],
  )

  const suggestions =
    latestSuggestions.length > 0
      ? latestSuggestions
      : ([...conversation].reverse().find((t) => t.role === 'assistant')?.suggestions ?? [])

  return (
    <Card className="flex flex-col overflow-hidden">
      <div className="px-4 pt-4 pb-3 border-b">
        <h3 className="font-semibold text-sm">Follow-up Questions</h3>
        <p className="text-xs text-muted-foreground mt-0.5">Grounded to the same research sources.</p>
      </div>

      <div className="max-h-[520px] overflow-y-auto">
        <div className="px-4 py-4 flex flex-col gap-4">
          {conversation.length === 0 && !streaming && (
            <p className="text-sm text-muted-foreground text-center py-4">No questions yet. Ask something below.</p>
          )}

          {conversation.map((turn, index) =>
            turn.role === 'user' ? (
              <div key={`${turn.role}-${index}`} className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm px-3 py-2 text-sm bg-primary text-primary-foreground">
                  {turn.content}
                </div>
              </div>
            ) : (
              <div key={`${turn.role}-${index}`} className="flex gap-2 items-start">
                <div className={cn(assistantAvatarClassName, 'size-7')}>AI</div>
                <div className={cn('max-w-[80%]', assistantBubbleClassName)}>
                  <ChatMarkdown content={turn.content} />
                  <CitationBadges citations={turn.citations} />
                </div>
              </div>
            ),
          )}

          {streaming && (
            <div className="flex gap-2 items-start">
              <div className={cn(assistantAvatarClassName, 'size-7')}>AI</div>
              <div className={cn('max-w-[80%]', assistantBubbleClassName)}>
                <ChatMarkdown content={streamingText || 'Thinking...'} />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {suggestions.length > 0 && (
        <div className="px-3 pt-2 flex flex-wrap gap-2">
          {suggestions.map((text) => (
            <Button
              key={text}
              type="button"
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() => void submit(text)}
              disabled={streaming}
            >
              {text}
            </Button>
          ))}
        </div>
      )}

      {error && <p className="text-destructive text-xs px-4 pb-1">{error}</p>}

      <form className="flex items-end gap-2 px-3 pb-3 pt-2 border-t" onSubmit={handleSubmit}>
        <Textarea
          ref={textareaRef}
          value={question}
          onChange={handleQuestionChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a follow-up question... (Enter to send, Shift+Enter for newline)"
          disabled={streaming}
          className="resize-none min-h-10 max-h-40"
          rows={1}
        />
        <Button type="submit" size="icon" disabled={streaming || !question.trim()} aria-label="Send">
          <SendHorizontal size={16} />
        </Button>
      </form>
    </Card>
  )
}
