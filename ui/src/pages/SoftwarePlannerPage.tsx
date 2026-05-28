import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Loader2 } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { generateSoftwareDevPlan, getSoftwareDevPlan } from '@/api/client'
import { SavedPlanDetailView } from '@/components/planner/SavedPlanDetailView'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import type { SavedSoftwareDevPlan } from '@/types'

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

export function SoftwarePlannerPage({
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
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState<SavedSoftwareDevPlan | null>(null)
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
        const savedPlan = await getSoftwareDevPlan(planId, authSession.access_token)
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
      setError('Sign in to generate a software implementation plan.')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await generateSoftwareDevPlan(normalizedPrompt, authSession.access_token)
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
      </div>
    </main>
  )
}
