import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, Paperclip, SendHorizontal, Square, X } from 'lucide-react'
import { assistantAvatarClassName, assistantBubbleClassName, ChatMarkdown } from '@/components/chat/ChatMarkdown'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { RagChatMessage, SessionAttachment } from '@/types'
import type { ChatTransport } from './transports'
import {
  consumeStream,
  getStreamStartBlocker,
  peekStream,
  sessionStreamKey,
  startChatStream,
  stopChatStream,
  useChatStream,
} from './chatStreamStore'
import { removeLastExchange } from './chatThreadState'
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
              'border-success/40 text-success': a.state === 'ready',
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
  const [loadingSession, setLoadingSession] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [pendingStreamKey, setPendingStreamKey] = useState<string | null>(null)
  const [input, setInput] = useState('')
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
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Stream lifecycle lives in chatStreamStore, keyed by session, so streams
  // survive session switches and multiple sessions can stream in parallel.
  const currentStreamKey = sessionId
    ? sessionStreamKey(transport.key, sessionId)
    : pendingStreamKey
  const stream = useChatStream(currentStreamKey)
  const chatting = stream?.status === 'streaming'

  useEffect(() => {
    currentTransportKeyRef.current = transport.key
  }, [transport.key])

  useEffect(() => {
    messagesRequestRef.current += 1
    loadedSessionRef.current = null
    setSessionId(null)
    setPendingStreamKey(null)
    setMessages([])
    setLoadingSession(false)
    setInput('')
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
      // A stream that finished in the background is already persisted server-side:
      // drop the store entry and refetch so the messages include that exchange.
      const streamKey = sessionStreamKey(transport.key, nextSessionId)
      const finished = peekStream(streamKey)
      const hadFinishedStream = finished !== undefined && finished.status !== 'streaming'
      if (hadFinishedStream) consumeStream(streamKey)
      if (!hadFinishedStream && loadedSessionRef.current === nextSessionId) return
      const requestId = ++messagesRequestRef.current
      setLoadingSession(true)
      try {
        const res = await transport.loadSessionMessages(nextSessionId, accessToken)
        if (requestId !== messagesRequestRef.current || currentTransportKeyRef.current !== transport.key) return
        loadedSessionRef.current = res.session_id
        setSessionId(res.session_id)
        setMessages(res.messages)
        setLatestSuggestions([])
        setWebUsedLastReply(false)
        if (finished?.status === 'error') {
          // the stream failed while this session was in the background: surface
          // the error and give the user their question back
          setError(finished.error ?? 'Chat failed.')
          setInput(finished.question)
        } else {
          setError(null)
        }

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
      } finally {
        if (requestId === messagesRequestRef.current) setLoadingSession(false)
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
        setSessionId(null)
        setPendingStreamKey(null)
        setMessages([])
        setSessionAttachments([])
      }
      messagesRequestRef.current += 1
      setLoadingSession(false)
      return
    }
    void openSession(activeSessionId)
  }, [activeSessionId, openSession])

  // A stream started without a session adopts the server-assigned id once it arrives.
  useEffect(() => {
    if (!stream?.resolvedSessionId || sessionId === stream.resolvedSessionId) return
    loadedSessionRef.current = stream.resolvedSessionId
    setSessionId(stream.resolvedSessionId)
    onSessionActivated(stream.resolvedSessionId)
  }, [stream, sessionId, onSessionActivated])

  // When the stream for the session being viewed finishes, fold its result into the thread.
  useEffect(() => {
    if (!stream || stream.status === 'streaming') return
    const finished = stream
    if (finished.status === 'done') {
      const finalSessionId = finished.resolvedSessionId ?? sessionId ?? 'pending'
      setMessages((prev) => {
        // ponytail: naive content dedupe covers the rare race where a refetch
        // already delivered this exchange before the store entry was consumed
        const last = prev.at(-1)
        if (last?.role === 'assistant' && last.content === finished.streamingText.trim()) return prev
        const userMessage: RagChatMessage = {
          message_id: `tmp-user-${crypto.randomUUID()}`,
          session_id: finalSessionId,
          agent_id: null,
          owner_id: '',
          role: 'user',
          content: finished.displayQuestion,
          citations: [],
          created_at: new Date().toISOString(),
        }
        const assistantMessage: RagChatMessage = {
          message_id: `tmp-assistant-${crypto.randomUUID()}`,
          session_id: finalSessionId,
          agent_id: null,
          owner_id: '',
          role: 'assistant',
          content: finished.streamingText.trim(),
          citations: finished.citations,
          suggestions: finished.suggestions,
          created_at: new Date().toISOString(),
        }
        return [...prev, userMessage, assistantMessage]
      })
      setLatestSuggestions(finished.suggestions)
      if (finished.webUsed) setWebUsedLastReply(true)
      if (finished.resolvedSessionId) void refreshSessionAttachments(finished.resolvedSessionId)
    } else {
      setError(finished.error ?? 'Chat failed.')
    }
    consumeStream(finished.key)
  }, [stream, sessionId, refreshSessionAttachments])

  const send = (overrideText?: string) => {
    const text = overrideText ?? input
    if (!text.trim() || chatting || uploadingAttachments) return
    const question = text.trim()

    const readyAttachmentNames = sessionAttachments
      .filter((attachment) => attachment.state === 'ready')
      .map((attachment) => attachment.filename)
    const displayQuestion = readyAttachmentNames.length > 0
      ? `${question}\n\n📎 ${readyAttachmentNames.join(', ')}`
      : question

    setInput('')
    setError(null)
    setLatestSuggestions([])

    try {
      const key = startChatStream({
        transport,
        sessionId,
        question,
        displayQuestion,
        accessToken,
        toolConfig,
        onSessionsChanged,
      })
      if (!sessionId) setPendingStreamKey(key)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat failed.')
      setInput(question)
    }
  }

  const submitEdit = async () => {
    const text = input.trim()
    if (!text || !editingMessageId || sessionId === null) return
    const blocker = getStreamStartBlocker(sessionStreamKey(transport.key, sessionId))
    if (blocker) {
      setError(blocker)
      return
    }
    try {
      await transport.deleteLastExchange(sessionId, accessToken)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete previous exchange.')
      return
    }
    setEditingMessageId(null)
    setMessages((prev) => removeLastExchange(prev))
    send(text)
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
          {loadingSession ? (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground animate-fade-in">
              <Loader2 size={14} className="animate-spin" />
              Loading discussion...
            </div>
          ) : (
            <>
              {messages.length === 0 && (
                <p className="py-8 text-center text-sm text-muted-foreground animate-fade-in">{emptyState}</p>
              )}
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
              {stream && (
                <>
                  <div className="flex flex-col items-end gap-1">
                    <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground max-md:max-w-[86%]">{stream.displayQuestion}</div>
                  </div>
                  <div className="flex gap-2 items-start">
                    <div className={cn('mt-0.5', assistantAvatarClassName)}>AI</div>
                    <div className={cn('max-w-[75%] max-md:max-w-[86%]', assistantBubbleClassName)}>
                      {stream.streamingText || stream.streamingStatus ? (
                        <ChatMarkdown content={stream.streamingText || stream.streamingStatus || ''} />
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                          <Loader2 size={14} className="animate-spin" />
                          Thinking...
                        </span>
                      )}
                    </div>
                  </div>
                </>
              )}
            </>
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
                const question = stream?.question ?? ''
                if (currentStreamKey) stopChatStream(currentStreamKey)
                setError(null)
                if (question) setInput(question)
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
