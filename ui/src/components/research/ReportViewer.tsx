import { useCallback, useState } from 'react'
import { Download, ThumbsDown, ThumbsUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

type Props = {
  report: string
  streamingReport?: string
  query: string
  isStreaming: boolean
  error: string | null
  feedbackEnabled?: boolean
  feedbackUnavailableReason?: string | null
  feedbackSubmittedAt?: string | null
  feedbackHelpful?: boolean | null
  feedbackPending?: boolean
  feedbackError?: string | null
  onFeedbackSubmit?: (helpful: boolean, comment: string | null) => void
}

function toSafeFileStem(value: string): string {
  return (
    value
      .toLowerCase()
      .trim()
      .replace(/[^\w\s.-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 80) || 'research-report'
  )
}

export function ReportViewer({
  report,
  streamingReport = '',
  query,
  isStreaming,
  error,
  feedbackEnabled = false,
  feedbackUnavailableReason = null,
  feedbackSubmittedAt = null,
  feedbackHelpful = null,
  feedbackPending = false,
  feedbackError = null,
  onFeedbackSubmit,
}: Props) {
  const visibleReport = report || (isStreaming ? streamingReport : '')

  const download = useCallback(() => {
    if (!visibleReport.trim()) return
    const blob = new Blob([visibleReport], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${toSafeFileStem(query)}-${new Date().toISOString().slice(0, 10)}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [query, visibleReport])

  if (!visibleReport && !error && !isStreaming) return null

  return (
    <div className="space-y-3">
      {visibleReport && (
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-muted-foreground truncate max-w-[70%]">{query}</h2>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={download}>
              <Download size={12} />
              Download
            </Button>
          </div>
        </div>
      )}
      {error && (
        <p role="alert" className="rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      )}
      {!error && !visibleReport && isStreaming && (
        <p className="rounded-md border bg-muted/35 px-3 py-2 text-sm text-muted-foreground">
          Drafting final report...
        </p>
      )}
      {visibleReport && (
        <div className="space-y-4">
          {(feedbackEnabled || feedbackUnavailableReason) && (
            <RunFeedbackPanel
              key={`${query}:${feedbackSubmittedAt ?? 'pending'}`}
              feedbackEnabled={feedbackEnabled}
              feedbackUnavailableReason={feedbackUnavailableReason}
              feedbackSubmittedAt={feedbackSubmittedAt}
              feedbackHelpful={feedbackHelpful}
              feedbackPending={feedbackPending}
              feedbackError={feedbackError}
              onFeedbackSubmit={onFeedbackSubmit}
            />
          )}
          <div className="overflow-x-auto">
            <article className="prose prose-sm dark:prose-invert max-w-none prose-table:my-2 prose-th:border prose-th:border-border prose-th:px-2 prose-th:py-1 prose-th:text-left prose-td:border prose-td:border-border prose-td:px-2 prose-td:py-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{visibleReport}</ReactMarkdown>
            </article>
          </div>
        </div>
      )}
    </div>
  )
}

type RunFeedbackPanelProps = {
  feedbackEnabled: boolean
  feedbackUnavailableReason: string | null
  feedbackSubmittedAt: string | null
  feedbackHelpful: boolean | null
  feedbackPending: boolean
  feedbackError: string | null
  onFeedbackSubmit?: (helpful: boolean, comment: string | null) => void
}

function RunFeedbackPanel({
  feedbackEnabled,
  feedbackUnavailableReason,
  feedbackSubmittedAt,
  feedbackHelpful,
  feedbackPending,
  feedbackError,
  onFeedbackSubmit,
}: RunFeedbackPanelProps) {
  const [showDownFeedback, setShowDownFeedback] = useState(false)
  const [comment, setComment] = useState('')

  return (
    <div className="rounded-md border border-border/70 bg-muted/20 px-3 py-3">
      {feedbackSubmittedAt ? (
        <p className="text-sm text-muted-foreground">
          Thanks for the feedback{feedbackHelpful === false ? ' — we’ll review the issue.' : '.'}
        </p>
      ) : !feedbackEnabled || !onFeedbackSubmit ? (
        <p className="text-sm text-muted-foreground">
          {feedbackUnavailableReason ?? 'Feedback is not available for this run.'}
        </p>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm text-muted-foreground">Was this research result helpful?</span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={feedbackPending}
              onClick={() => onFeedbackSubmit(true, null)}
            >
              <ThumbsUp size={14} />
              Yes
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={feedbackPending}
              onClick={() => setShowDownFeedback((prev) => !prev)}
            >
              <ThumbsDown size={14} />
              No
            </Button>
          </div>
          {showDownFeedback && (
            <div className="space-y-2">
              <Textarea
                value={comment}
                onChange={(event) => setComment(event.target.value)}
                placeholder="Optional note about what was missing or incorrect"
                className="min-h-[88px]"
                disabled={feedbackPending}
              />
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  disabled={feedbackPending}
                  onClick={() => onFeedbackSubmit(false, comment || null)}
                >
                  Submit feedback
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={feedbackPending}
                  onClick={() => {
                    setShowDownFeedback(false)
                    setComment('')
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
          {feedbackError && <p className="text-xs text-destructive">{feedbackError}</p>}
        </div>
      )}
    </div>
  )
}
