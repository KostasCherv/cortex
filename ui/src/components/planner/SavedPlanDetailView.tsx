import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { SavedPRD } from '@/types'

export function SavedPlanDetailView({ result }: { result: SavedPRD }) {
  const plan = result.plan
  const requestPreview = result.prompt_preview || result.prompt

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">{plan?.title}</CardTitle>
          <CardDescription>{plan?.executive_summary}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {requestPreview && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Original request</h2>
              <p className="text-sm leading-6 text-muted-foreground">{requestPreview}</p>
            </section>
          )}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">PRD document</h2>
            <article className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.markdown}</ReactMarkdown>
            </article>
          </section>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Problem & outcome</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <p className="font-medium">Problem statement</p>
              <p className="text-muted-foreground">{plan?.problem_statement}</p>
            </div>
            <div>
              <p className="font-medium">Desired outcome</p>
              <p className="text-muted-foreground">{result.planning_brief?.desired_outcome}</p>
            </div>
          </CardContent>
        </Card>

        {(plan?.goals ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Goals</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1 text-sm list-disc pl-4">
                {(plan?.goals ?? []).map((goal, i) => (
                  <li key={i} className="text-muted-foreground">{goal}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {(plan?.target_users ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Target users</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1 text-sm list-disc pl-4">
                {(plan?.target_users ?? []).map((persona, i) => (
                  <li key={i} className="text-muted-foreground">{persona}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {(plan?.requirements ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Requirements (MoSCoW)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                {(['Must Have', 'Should Have', 'Could Have', "Won't Have"] as const).map((priority) => {
                  const group = (plan?.requirements ?? []).filter((r) => r.priority === priority)
                  if (!group.length) return null
                  return (
                    <div key={priority}>
                      <p className="font-medium text-xs uppercase tracking-wide text-muted-foreground mb-1">{priority}</p>
                      {group.map((req) => (
                        <div key={req.id} className="rounded border p-2 mb-1">
                          <p><span className="font-mono text-xs text-muted-foreground">{req.id}</span> {req.description}</p>
                        </div>
                      ))}
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {(plan?.success_metrics ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Success metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1 text-sm list-disc pl-4">
                {(plan?.success_metrics ?? []).map((metric, i) => (
                  <li key={i} className="text-muted-foreground">{metric}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {(plan?.milestones ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Milestones</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                {(plan?.milestones ?? []).map((milestone) => (
                  <section key={milestone.id} className="rounded-md border p-3">
                    <p className="font-medium">{milestone.id}: {milestone.title}</p>
                    <p className="mt-1 text-muted-foreground">{milestone.description}</p>
                    {milestone.deliverables.length > 0 && (
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                        {milestone.deliverables.map((d, i) => (
                          <li key={i}>{d}</li>
                        ))}
                      </ul>
                    )}
                  </section>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {(plan?.risks ?? []).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Risks</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1 text-sm list-disc pl-4">
                {(plan?.risks ?? []).map((risk, i) => (
                  <li key={i} className="text-muted-foreground">{risk}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
