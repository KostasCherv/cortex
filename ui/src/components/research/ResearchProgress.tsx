import { CheckCircle2, CircleDot, Loader2, XCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { ResearchStreamEvent } from '@/types'

const NODE_ALIASES: Record<string, string> = {
  search: 'search_node',
  search_and_memory: 'search_and_memory_node',
  retrieve: 'retrieve_node',
  memory_context: 'memory_context_node',
  rerank: 'rerank_node',
  summarize: 'summarize_node',
  report: 'report_node',
  abort: '__error__',
}

const NODE_LABELS: Record<string, string> = {
  search_and_memory_node: 'Searching & Context',
  search_node: 'Searching',
  retrieve_node: 'Retrieving Sources',
  memory_context_node: 'Building Context',
  rerank_node: 'Reranking Sources',
  summarize_node: 'Summarizing Sources',
  report_node: 'Drafting Final Report',
  __end__: 'Completed',
  __error__: 'Failed',
}

const PIPELINE_PHASES = [
  'search_and_memory_node',
  'rerank_node',
  'summarize_node',
  'report_node',
]

// Customize this list to display only selected phases in the UI.
const VISIBLE_PHASES = PIPELINE_PHASES

type Props = {
  events?: ResearchStreamEvent[]
  latestNode?: string | null
  status?: 'idle' | 'running' | 'completed' | 'failed'
  isStreaming: boolean
}

export function ResearchProgress({ events = [], latestNode = null, status = 'idle', isStreaming }: Props) {
  if (events.length === 0 && !isStreaming) return null

  const normalizedLatestNode = latestNode ? NODE_ALIASES[latestNode] ?? latestNode : null
  const normalizedNodes = [
    ...events.map((e) => NODE_ALIASES[e.node] ?? e.node),
    ...(normalizedLatestNode ? [normalizedLatestNode] : []),
  ]
  const seenNodes = new Set(normalizedNodes)
  const latestVisibleNode = [...normalizedNodes].reverse().find((node) => VISIBLE_PHASES.includes(node)) ?? null
  const furthestKnownPhaseIndex = normalizedNodes.reduce((max, node) => {
    const idx = PIPELINE_PHASES.indexOf(node)
    return idx > max ? idx : max
  }, -1)
  const hasError = status === 'failed' || seenNodes.has('__error__')
  const ended = status === 'completed' || seenNodes.has('__end__')
  const hasCurrent = isStreaming && !ended && !hasError

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2" aria-live="polite">
          {VISIBLE_PHASES.map((node) => {
            const phaseIndex = PIPELINE_PHASES.indexOf(node)
            const complete = ended || (phaseIndex !== -1 && phaseIndex < furthestKnownPhaseIndex)
            const current =
              hasCurrent &&
              (latestVisibleNode === node ||
                (!latestVisibleNode &&
                  furthestKnownPhaseIndex === -1 &&
                  node === VISIBLE_PHASES[0]))
            return (
              <li key={node} className="flex items-center gap-2 text-sm">
                {complete ? (
                  <CheckCircle2 size={16} className="text-green-500 shrink-0" />
                ) : current ? (
                  <Loader2 size={16} className="animate-spin text-primary shrink-0" />
                ) : (
                  <CircleDot
                    size={16}
                    className={cn(
                      'shrink-0',
                      seenNodes.has(node) ? 'text-foreground' : 'text-muted-foreground',
                    )}
                  />
                )}
                <span
                  className={cn(
                    complete || seenNodes.has(node) ? 'text-foreground' : 'text-muted-foreground',
                  )}
                >
                  {NODE_LABELS[node] ?? node}
                </span>
              </li>
            )
          })}
          {hasError && (
            <li className="flex items-center gap-2 text-sm">
              <XCircle size={16} className="text-destructive shrink-0" />
              <span className="text-destructive">Failed</span>
            </li>
          )}
          {ended && (
            <li className="flex items-center gap-2 text-sm">
              <CheckCircle2 size={16} className="text-green-500 shrink-0" />
              <span>Completed</span>
            </li>
          )}
        </ul>
      </CardContent>
    </Card>
  )
}
