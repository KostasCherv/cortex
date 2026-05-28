import { useCallback, useEffect, useMemo, useState } from 'react'
import { Download, Loader2 } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { generateSoftwareDevPlan, getSoftwareDevPlan, listSoftwareDevPlans } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import type { SavedSoftwareDevPlan, SavedSoftwareDevPlanSummary } from '@/types'

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

function toPlanSummary(plan: SavedSoftwareDevPlan): SavedSoftwareDevPlanSummary {
  return {
    plan_id: plan.plan_id,
    title: plan.plan.title,
    summary: plan.plan.summary,
    prompt_preview: plan.prompt_preview,
    created_at: plan.created_at,
    updated_at: plan.updated_at,
  }
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function SelectedPlanView({ result }: { result: SavedSoftwareDevPlan }) {
  const plan = result.plan
  const highlightedFiles = Array.from(new Set(plan.file_map.map((item) => item.path)))

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">{plan.title}</CardTitle>
          <CardDescription>{plan.summary}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Recommended approach</h2>
            <p className="text-sm leading-6">{plan.recommended_approach}</p>
          </section>
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Markdown preview</h2>
            <article className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.markdown}</ReactMarkdown>
            </article>
          </section>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Planner brief</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <p className="font-medium">Outcome</p>
              <p className="text-muted-foreground">{result.planning_brief.desired_outcome}</p>
            </div>
            <div>
              <p className="font-medium">Repo fit</p>
              <p className="text-muted-foreground">{plan.repo_fit}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Relevant files</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {plan.file_map.map((item) => (
                <li key={`${item.path}:${item.reason}`}>
                  <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{item.path}</code>
                  <p className="mt-1 text-muted-foreground">{item.reason}</p>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Implementation phases</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              {plan.phases.map((phase) => (
                <section key={phase.id} className="rounded-md border p-3">
                  <p className="font-medium">{phase.title}</p>
                  <p className="mt-1 text-muted-foreground">{phase.objective}</p>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                    {phase.files.map((path) => (
                      <li key={`${phase.id}:${path}`}>
                        <code>{path}</code>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Planning options</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              {result.planning_options.approaches.map((approach) => (
                <section key={approach.name} className="rounded-md border p-3">
                  <p className="font-medium">{approach.name}</p>
                  <p className="mt-1 text-muted-foreground">{approach.summary}</p>
                  {approach.tradeoffs.length > 0 && (
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                      {approach.tradeoffs.map((tradeoff) => (
                        <li key={`${approach.name}:${tradeoff}`}>{tradeoff}</li>
                      ))}
                    </ul>
                  )}
                </section>
              ))}
            </div>
          </CardContent>
        </Card>

        {highlightedFiles.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Quick file map</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 text-xs">
                {highlightedFiles.map((path) => (
                  <code key={path} className="rounded bg-muted px-2 py-1">
                    {path}
                  </code>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export function SoftwarePlannerPage({ authSession }: { authSession: Session | null }) {
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState<SavedSoftwareDevPlan | null>(null)
  const [history, setHistory] = useState<SavedSoftwareDevPlanSummary[]>([])
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)

  const signedIn = Boolean(authSession?.access_token)
  const canSubmit = signedIn && prompt.trim().length > 0 && !loading
  const selectedSummary = useMemo(
    () => history.find((item) => item.plan_id === selectedPlanId) ?? null,
    [history, selectedPlanId],
  )

  const loadHistory = useCallback(async () => {
    if (!authSession?.access_token) {
      setHistory([])
      setHistoryError(null)
      return
    }

    setHistoryLoading(true)
    try {
      const response = await listSoftwareDevPlans(authSession.access_token)
      setHistory(response.plans)
      setHistoryError(null)
    } catch (historyLoadError) {
      setHistoryError(historyLoadError instanceof Error ? historyLoadError.message : 'Failed to load saved plans.')
    } finally {
      setHistoryLoading(false)
    }
  }, [authSession?.access_token])

  useEffect(() => {
    if (!signedIn) {
      setHistory([])
      setResult(null)
      setSelectedPlanId(null)
      setHistoryError(null)
      return
    }
    void loadHistory()
  }, [loadHistory, signedIn])

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

      setDetailLoading(true)
      setError(null)
      try {
        const savedPlan = await getSoftwareDevPlan(planId, authSession.access_token)
        setResult(savedPlan)
      } catch (detailError) {
        setError(detailError instanceof Error ? detailError.message : 'Failed to load saved plan.')
      } finally {
        setDetailLoading(false)
      }
    },
    [authSession?.access_token, result?.plan_id],
  )

  const handleSubmit = useCallback(async () => {
    const normalizedPrompt = prompt.trim()
    if (!normalizedPrompt) {
      setError('Describe the feature or implementation goal first.')
      return
    }
    if (!authSession?.access_token) {
      setError('Sign in to generate a software implementation plan.')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await generateSoftwareDevPlan(normalizedPrompt, authSession.access_token)
      setResult(response)
      setSelectedPlanId(response.plan_id)
      setHistory((prev) => {
        const nextSummary = toPlanSummary(response)
        return [nextSummary, ...prev.filter((item) => item.plan_id !== response.plan_id)]
      })
      setHistoryError(null)
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : 'Failed to generate plan.')
    } finally {
      setLoading(false)
    }
  }, [authSession?.access_token, prompt])

  return (
    <main className="h-full overflow-y-auto bg-background px-4 py-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Software implementation planner</CardTitle>
            <CardDescription>
              Turn a feature request or architecture change into a repo-grounded implementation plan.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="software-planner-prompt" className="text-sm font-medium">
                What should the planner design?
              </label>
              <Textarea
                id="software-planner-prompt"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Describe the feature, goal, constraints, and any files or areas of the repo that matter."
                className="min-h-[160px]"
                disabled={loading}
              />
            </div>
            {!signedIn && (
              <p className="text-sm text-muted-foreground">Sign in to generate a software implementation plan.</p>
            )}
            {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" onClick={() => void handleSubmit()} disabled={!canSubmit}>
                {loading ? <Loader2 className="animate-spin" size={16} /> : null}
                Generate plan
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

        <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Saved plans</CardTitle>
              <CardDescription>Reopen recent implementation plans for this account.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {!signedIn ? (
                <p className="text-sm text-muted-foreground">Sign in to view saved plans.</p>
              ) : historyLoading ? (
                <p className="text-sm text-muted-foreground">Loading saved plans...</p>
              ) : historyError ? (
                <p role="alert" className="text-sm text-destructive">{historyError}</p>
              ) : history.length === 0 ? (
                <p className="text-sm text-muted-foreground">No saved plans yet.</p>
              ) : (
                <div className="space-y-2">
                  {history.map((item) => {
                    const isSelected = item.plan_id === selectedPlanId
                    return (
                      <button
                        key={item.plan_id}
                        type="button"
                        onClick={() => void handleSelectPlan(item.plan_id)}
                        className={`w-full rounded-md border p-3 text-left transition-colors ${
                          isSelected ? 'border-primary bg-muted' : 'hover:bg-muted/60'
                        }`}
                      >
                        <p className="font-medium text-sm">{item.title}</p>
                        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{item.prompt_preview}</p>
                        <p className="mt-2 text-xs text-muted-foreground">{formatTimestamp(item.created_at)}</p>
                      </button>
                    )
                  })}
                </div>
              )}
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
            <SelectedPlanView result={result} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Select a saved plan</CardTitle>
                <CardDescription>
                  {selectedSummary
                    ? `Loading ${selectedSummary.title}...`
                    : 'Generate a new plan or choose one from history to view its details.'}
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </div>
      </div>
    </main>
  )
}
