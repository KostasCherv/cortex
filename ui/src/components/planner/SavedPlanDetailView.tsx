import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { SavedSoftwareDevPlan } from '@/types'

export function SavedPlanDetailView({ result }: { result: SavedSoftwareDevPlan }) {
  const plan = result.plan
  const fileMap = plan?.file_map ?? []
  const phases = plan?.phases ?? []
  const approaches = result.planning_options?.approaches ?? []
  const highlightedFiles = Array.from(new Set(fileMap.map((item) => item.path)))
  const requestPreview = result.prompt_preview || result.prompt

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">{plan?.title}</CardTitle>
          <CardDescription>{plan?.summary}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {requestPreview && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Saved request</h2>
              <p className="text-sm leading-6 text-muted-foreground">{requestPreview}</p>
            </section>
          )}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Recommended approach</h2>
            <p className="text-sm leading-6">{plan?.recommended_approach}</p>
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
              <p className="text-muted-foreground">{result.planning_brief?.desired_outcome}</p>
            </div>
            <div>
              <p className="font-medium">Repo fit</p>
              <p className="text-muted-foreground">{plan?.repo_fit}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Relevant files</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {fileMap.map((item) => (
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
              {phases.map((phase) => (
                <section key={phase.id} className="rounded-md border p-3">
                  <p className="font-medium">{phase.title}</p>
                  <p className="mt-1 text-muted-foreground">{phase.objective}</p>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                    {(phase.files ?? []).map((path) => (
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
              {approaches.map((approach) => (
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
