import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Paperclip, SendHorizontal, Square, X } from 'lucide-react'
import { assistantAvatarClassName, assistantBubbleClassName, ChatMarkdown } from '@/components/chat/ChatMarkdown'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { RagChatMessage, SessionAttachment } from '@/types'
import type { ChatTransport } from './transports'
import { getStopEditState, replaceLastEditableUserMessage } from './chatThreadState'
import { ToolMenuButton } from './ToolMenuButton'
import { defaultToolConfig } from './toolConfig'
import type { ToolConfig } from './toolConfig'

const ACCEPTED_MIME = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'text/markdown',
].join(',')

type PendingFileUpload = {
  id: string
  file: File
  status: 'uploading' | 'failed'
  error?: string
}

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
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
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

function AttachmentShelf({
  attachments,
  onDelete,
  deletingAttachmentId,
}: {
  attachments: SessionAttachment[]
  onDelete?: (attachmentId: string) => void
  deletingAttachmentId?: string | null
}) {
  if (attachments.length === 0) return null
  return (
    <div className="mx-6 mb-3 flex flex-wrap gap-2 max-md:mx-4">
      {attachments.map((a) => (
        <div
          key={a.attachment_id}
          className="flex items-center gap-1.5 rounded-md border border-border/60 bg-muted/40 px-2 py-1"
        >
          <Paperclip size={11} className="shrink-0 text-muted-foreground" />
          <span className="max-w-[120px] truncate text-[11px] text-foreground">{a.filename}</span>
          <Badge
            variant="outline"
            className={cn('text-[10px] px-1 py-0 h-4', {
              'border-green-500/40 text-green-600': a.state === 'ready',
              'border-destructive/40 text-destructive': a.state === 'failed',
              'border-muted-foreground/30 text-muted-foreground': a.state === 'processing' || a.state === 'uploaded',
            })}
          >
            {a.state}
          </Badge>
          {a.state === 'ready' && onDelete && (
            <button
              type="button"
              aria-label={`Remove ${a.filename}`}
              className="rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
              disabled={deletingAttachmentId === a.attachment_id}
              onClick={() => onDelete(a.attachment_id)}
            >
              <X size={10} />
            </button>
          )}
        </div>
      ))}
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
}: Props) {
  const [messages, setMessages] = useState<RagChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [chatting, setChatting] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [webUsedLastReply, setWebUsedLastReply] = useState(false)
  const [latestSuggestions, setLatestSuggestions] = useState<string[]>([])
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [toolConfig, setToolConfig] = useState<ToolConfig>(() => defaultToolConfig())
  const [pendingUploads, setPendingUploads] = useState<PendingFileUpload[]>([])
  const [sessionAttachments, setSessionAttachments] = useState<SessionAttachment[]>([])
  const [deletingAttachmentId, setDeletingAttachmentId] = useState<string | null>(null)
  const uploadingAttachments = pendingUploads.some((upload) => upload.status === 'uploading')

  const messagesRequestRef = useRef(0)
  const loadedSessionRef = useRef<string | null>(null)
  const currentTransportKeyRef = useRef(transport.key)
  const chatAbortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  // Tracks whether the last user+assistant exchange in `messages` has been persisted server-side.
  // Set to true in onDone; reset to false when a new send begins.
  const lastExchangePersistedRef = useRef(false)

  useEffect(() => {
    currentTransportKeyRef.current = transport.key
  }, [transport.key])

  useEffect(() => {
    messagesRequestRef.current += 1
    chatAbortRef.current?.abort()
    chatAbortRef.current = null
    loadedSessionRef.current = null
    lastExchangePersistedRef.current = false
    setSessionId(null)
    setMessages([])
    setInput('')
    setStreamingText('')
    setError(null)
    setWebUsedLastReply(false)
    setLatestSuggestions([])
    setEditingMessageId(null)
    setToolConfig(defaultToolConfig())
    setPendingUploads([])
    setSessionAttachments([])
    setDeletingAttachmentId(null)
  }, [transport.key])

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
        setError(null)
        lastExchangePersistedRef.current = true

        if (transport.loadSessionAttachments) {
          try {
            const attachments = await transport.loadSessionAttachments(res.session_id, accessToken)
            if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
              setSessionAttachments(attachments)
            }
          } catch {
            // attachment list failure is non-fatal
          }
        }
      } catch (err) {
        if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
          setError(err instanceof Error ? err.message : 'Failed to load chat session.')
        }
      }
    },
    [accessToken, transport],
  )

  const refreshSessionAttachments = useCallback(
    async (targetSessionId: string) => {
      if (!transport.loadSessionAttachments) return
      try {
        const attachments = await transport.loadSessionAttachments(targetSessionId, accessToken)
        setSessionAttachments(attachments)
      } catch {
        // attachment list failure is non-fatal
      }
    },
    [accessToken, transport],
  )

  const uploadPendingFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0 || !transport.uploadAttachments || !transport.ensureSession) return

      const uploadIds: string[] = files.map(() => crypto.randomUUID())
      setPendingUploads((prev) => [
        ...prev,
        ...files.map((file, index) => ({
          id: uploadIds[index],
          file,
          status: 'uploading' as const,
        })),
      ])

      try {
        let targetSessionId = sessionId
        if (!targetSessionId) {
          targetSessionId = await transport.ensureSession(accessToken, files[0]?.name)
          loadedSessionRef.current = targetSessionId
          setSessionId(targetSessionId)
          onSessionActivated(targetSessionId)
        }
        const uploaded = await transport.uploadAttachments(targetSessionId, files, accessToken)
        setSessionAttachments((prev) => {
          const byId = new Map(prev.map((attachment) => [attachment.attachment_id, attachment]))
          for (const attachment of uploaded) {
            byId.set(attachment.attachment_id, attachment)
          }
          return Array.from(byId.values())
        })
        setPendingUploads((prev) => prev.filter((upload) => !uploadIds.includes(upload.id)))
        onSessionsChanged()
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Upload failed.'
        setPendingUploads((prev) =>
          prev.map((upload) =>
            uploadIds.includes(upload.id)
              ? { ...upload, status: 'failed', error: message }
              : upload,
          ),
        )
        setError(message)
      }
    },
    [accessToken, onSessionActivated, onSessionsChanged, sessionId, transport],
  )

  const handleDeleteAttachment = useCallback(
    async (attachmentId: string) => {
      if (!sessionId || !transport.deleteAttachment) return
      setDeletingAttachmentId(attachmentId)
      try {
        await transport.deleteAttachment(sessionId, attachmentId, accessToken)
        setSessionAttachments((prev) =>
          prev.filter((attachment) => attachment.attachment_id !== attachmentId),
        )
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to remove attachment.')
      } finally {
        setDeletingAttachmentId(null)
      }
    },
    [accessToken, sessionId, transport],
  )

  useEffect(() => {
    if (!activeSessionId) {
      if (activeSessionId === null && loadedSessionRef.current !== null) {
        loadedSessionRef.current = null
        lastExchangePersistedRef.current = false
        setSessionId(null)
        setMessages([])
        setSessionAttachments([])
      }
      return
    }
    void openSession(activeSessionId)
  }, [activeSessionId, openSession])

  const send = async (
    overrideText?: string,
    opts?: {
      skipOptimisticAppend?: boolean
      previouslyPersisted?: boolean
      restoreDraftOnFailure?: { input: string; editingMessageId: string }
    },
  ) => {
    const text = overrideText ?? input
    if (!text.trim() || chatting || uploadingAttachments) return
    const question = text.trim()
    const requestId = ++messagesRequestRef.current
    lastExchangePersistedRef.current = false

    const readyAttachmentNames = sessionAttachments
      .filter((attachment) => attachment.state === 'ready')
      .map((attachment) => attachment.filename)

    if (!opts?.skipOptimisticAppend) {
      const optimisticUserMessage: RagChatMessage = {
        message_id: `tmp-user-${requestId}`,
        session_id: sessionId ?? 'pending',
        agent_id: null,
        owner_id: '',
        role: 'user',
        content: readyAttachmentNames.length > 0
          ? `${question}\n\n📎 ${readyAttachmentNames.join(', ')}`
          : question,
        citations: [],
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, optimisticUserMessage])
    }

    setInput('')
    setStreamingText('')
    setStreamingStatus(null)
    setChatting(true)
    setError(null)
    setLatestSuggestions([])

    chatAbortRef.current?.abort()
    const controller = new AbortController()
    chatAbortRef.current = controller

    if (opts?.previouslyPersisted && sessionId !== null) {
      try {
        await transport.deleteLastExchange(sessionId, accessToken)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete previous exchange.')
        if (opts.restoreDraftOnFailure) {
          setInput(opts.restoreDraftOnFailure.input)
          setEditingMessageId(opts.restoreDraftOnFailure.editingMessageId)
        }
        setChatting(false)
        if (chatAbortRef.current === controller) chatAbortRef.current = null
        return
      }
    }

    let streamedSessionId = sessionId
    let accumulated = ''
    let finalCitations: RagChatMessage['citations'] = []
    let pendingSuggestions: string[] = []
    let streamFailed = false

    try {
      await transport.streamMessage(question, sessionId, accessToken, {
        signal: controller.signal,
        onSession: (nextSessionId) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          streamedSessionId = nextSessionId
          loadedSessionRef.current = nextSessionId
          setSessionId(nextSessionId)
          if (!sessionId) onSessionActivated(nextSessionId)
        },
        onStatus: (message) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          if (controller.signal.aborted) return
          setStreamingStatus(message)
        },
        onChunk: (textChunk) => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          if (controller.signal.aborted) return
          accumulated += textChunk
          setStreamingStatus(null)
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
        onWebUsed: () => {
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          setWebUsedLastReply(true)
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
          setStreamingStatus(null)
          lastExchangePersistedRef.current = true
          onSessionsChanged()
          if (finalSessionId !== 'pending') {
            void refreshSessionAttachments(finalSessionId)
          }
        },
        onError: (streamError) => {
          streamFailed = true
          if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
          setError(streamError)
          setStreamingText('')
          setStreamingStatus(null)
        },
      }, toolConfig)
    } catch (err) {
      if (controller.signal.aborted) return
      if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
        setError(err instanceof Error ? err.message : 'Chat failed.')
        setStreamingText('')
        setStreamingStatus(null)
      }
    } finally {
      if (chatAbortRef.current === controller) chatAbortRef.current = null
      if (requestId === messagesRequestRef.current && currentTransportKeyRef.current === transport.key) {
        setChatting(false)
        if (streamFailed) onSessionsChanged()
      }
    }
  }

  const submitEdit = async () => {
    const text = input.trim()
    if (!text || !editingMessageId) return
    const previouslyPersisted = lastExchangePersistedRef.current && sessionId !== null
    setEditingMessageId(null)
    setMessages((prev) => {
      return replaceLastEditableUserMessage(prev, text)
    })
    setInput('')
    await send(text, {
      skipOptimisticAppend: true,
      previouslyPersisted,
      restoreDraftOnFailure: { input: text, editingMessageId },
    })
  }

  const suggestions = useMemo(
    () =>
      latestSuggestions.length > 0
        ? latestSuggestions
        : ([...messages].reverse().find((m) => m.role === 'assistant')?.suggestions ?? []),
    [latestSuggestions, messages],
  )

  const lastUserMessageId = useMemo(
    () => [...messages].reverse().find((m) => m.role === 'user')?.message_id ?? null,
    [messages],
  )

  return (
    <div className="flex h-dvh flex-col max-md:h-full">
      <div className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-6 max-md:px-4">
        <div className="min-w-0">
          <p className="font-medium text-sm">{title}</p>
          {subtitle && <p className="text-xs text-muted-foreground truncate max-w-md">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-3">
          {resourceLabel && <span className="shrink-0 text-xs text-muted-foreground">{resourceLabel}</span>}
          {webUsedLastReply && <Badge variant="outline">Web used</Badge>}
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1 px-6 py-6 max-md:px-4">
        <div className="space-y-4">
          {messages.length === 0 && <p className="py-8 text-center text-sm text-muted-foreground">{emptyState}</p>}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.message_id} className="flex flex-col items-end gap-1">
                <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground max-md:max-w-[86%]">{m.content}</div>
                {!chatting && lastUserMessageId === m.message_id && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs text-muted-foreground"
                    onClick={() => {
                      setInput(m.content)
                      setEditingMessageId(m.message_id)
                    }}
                  >
                    Edit
                  </Button>
                )}
              </div>
            ) : (
              <div key={m.message_id} className="flex flex-col gap-2">
                <div className="flex gap-2 items-start">
                  <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                  <div className={cn('max-w-[75%] max-md:max-w-[86%]', assistantBubbleClassName)}>
                    <ChatMarkdown content={m.content} />
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
              <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
              <div className={cn('max-w-[75%] max-md:max-w-[86%]', assistantBubbleClassName)}>
                <ChatMarkdown content={streamingText || streamingStatus || 'Thinking...'} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {sessionAttachments.length > 0 && (
        <AttachmentShelf
          attachments={sessionAttachments}
          onDelete={transport.deleteAttachment ? handleDeleteAttachment : undefined}
          deletingAttachmentId={deletingAttachmentId}
        />
      )}

      {error && <p role="alert" className="mx-6 mb-2 shrink-0 rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-xs text-destructive max-md:mx-4">{error}</p>}

      <div className="shrink-0 border-t bg-background px-6 py-4 max-md:px-4">
        {editingMessageId && (
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Editing message</span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs"
              onClick={() => {
                setEditingMessageId(null)
                setInput('')
              }}
            >
              Cancel
            </Button>
          </div>
        )}

        {pendingUploads.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {pendingUploads.map((upload) => (
              <div
                key={upload.id}
                className="flex items-center gap-1 rounded-md border border-border/60 bg-muted/50 px-2 py-1 text-[11px]"
              >
                <Paperclip size={10} className="shrink-0 text-muted-foreground" />
                <span className="max-w-[120px] truncate text-foreground">{upload.file.name}</span>
                <span className="text-muted-foreground">({formatBytes(upload.file.size)})</span>
                <Badge
                  variant="outline"
                  className={cn('text-[10px] px-1 py-0 h-4', {
                    'border-muted-foreground/30 text-muted-foreground': upload.status === 'uploading',
                    'border-destructive/40 text-destructive': upload.status === 'failed',
                  })}
                >
                  {upload.status === 'uploading' ? 'uploading' : 'failed'}
                </Badge>
                <button
                  type="button"
                  aria-label={`Remove ${upload.file.name}`}
                  className="ml-0.5 rounded text-muted-foreground hover:text-foreground"
                  disabled={upload.status === 'uploading'}
                  onClick={() => setPendingUploads((prev) => prev.filter((item) => item.id !== upload.id))}
                >
                  <X size={10} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2 items-end">
          {transport.supportsFileUpload && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={ACCEPTED_MIME}
                className="hidden"
                onChange={(e) => {
                  const selected = Array.from(e.target.files ?? [])
                  if (selected.length > 0) {
                    void uploadPendingFiles(selected)
                  }
                  e.target.value = ''
                }}
              />
              <Button
                type="button"
                size="icon"
                variant="ghost"
                className="shrink-0"
                aria-label="Attach files"
                disabled={chatting || uploadingAttachments}
                onClick={() => fileInputRef.current?.click()}
              >
                <Paperclip size={15} />
              </Button>
            </>
          )}
          <ToolMenuButton
            toolConfig={toolConfig}
            onToggle={(id, enabled) =>
              setToolConfig((prev) => ({ ...prev, [id]: enabled }))
            }
            disabled={chatting}
          />
          <Textarea
            className="resize-none min-h-10 max-h-32 text-sm"
            placeholder="Ask a question..."
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void (editingMessageId ? submitEdit() : send())
              }
            }}
            disabled={chatting || uploadingAttachments}
          />
          {chatting ? (
            <Button
              size="icon"
              variant="secondary"
              onClick={() => {
                chatAbortRef.current?.abort()
                setStreamingText('')
                setChatting(false)
                setError(null)
                const stopEditState = getStopEditState(messages)
                if (stopEditState) {
                  setInput(stopEditState.draft)
                  setEditingMessageId(stopEditState.editingMessageId)
                }
              }}
              aria-label="Stop generating"
            >
              <Square size={15} />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={() => void (editingMessageId ? submitEdit() : send())}
              disabled={!input.trim() || uploadingAttachments}
              aria-label={editingMessageId ? 'Update and resend' : 'Send message'}
            >
              <SendHorizontal size={15} />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
