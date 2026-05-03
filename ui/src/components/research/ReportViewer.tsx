import { useCallback } from 'react'
import { Download } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button } from '@/components/ui/button'

type Props = {
  report: string
  streamingReport?: string
  query: string
  isStreaming: boolean
  error: string | null
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

export function ReportViewer({ report, streamingReport = '', query, isStreaming, error }: Props) {
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
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={download}>
            <Download size={12} />
            Download
          </Button>
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
        <div className="overflow-x-auto">
          <article className="prose prose-sm dark:prose-invert max-w-none prose-table:my-2 prose-th:border prose-th:border-border prose-th:px-2 prose-th:py-1 prose-th:text-left prose-td:border prose-td:border-border prose-td:px-2 prose-td:py-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{visibleReport}</ReactMarkdown>
          </article>
        </div>
      )}
    </div>
  )
}
