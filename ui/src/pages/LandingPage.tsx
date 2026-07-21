import { ArrowRight, Bot, MessageSquare, Moon, Network, ShieldCheck, Sun, Telescope, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { useTheme } from '@/hooks/useTheme'

type Props = {
  onSignIn: () => void
}

const STATS = [
  {
    icon: Zap,
    value: '1.2s median / 2.0s p95',
    label: 'Agent-loop turn latency with a real LLM call',
  },
  {
    icon: ShieldCheck,
    value: '13,197 requests, 0% errors',
    label: 'Sustained at up to 200 req/s in production load testing',
  },
  {
    icon: Network,
    value: '60 of 70 admitted',
    label: 'Concurrent requests allowed per IP under rate limiting; the rest returned HTTP 429',
  },
]

const FEATURES = [
  {
    icon: Telescope,
    title: 'Multi-step Research',
    description:
      'A LangGraph workflow searches, retrieves, reranks, and reports — with explicit paths for success, empty results, and failure.',
  },
  {
    icon: MessageSquare,
    title: 'Source-grounded Chat',
    description: 'Follow-up conversation stays grounded in the sources and context from your research run.',
  },
  {
    icon: Network,
    title: 'GraphRAG Retrieval',
    description: 'Neo4j graph context and reranking enrich retrieval so answers cite real evidence.',
  },
  {
    icon: Bot,
    title: 'RAG Agents & Resources',
    description: 'Build custom agents grounded in your own uploaded documents and resources.',
  },
]

export function LandingPage({ onSignIn }: Props) {
  const { theme, toggle } = useTheme()

  return (
    <div className="flex h-dvh flex-col overflow-y-auto bg-background text-foreground">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4 sm:px-6">
        <span className="font-semibold tracking-tight">Cortex</span>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="size-8 text-muted-foreground hover:text-foreground"
            onClick={toggle}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </Button>
          <Button size="sm" onClick={onSignIn}>
            Sign in with Google
          </Button>
        </div>
      </header>

      <main className="flex-1">
        <section className="mx-auto max-w-3xl px-4 py-16 text-center sm:px-6 sm:py-24">
          <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">
            A research platform for multi-step web research
          </h1>
          <p className="mt-4 text-base text-muted-foreground sm:text-lg">
            Cortex searches and retrieves sources, enriches them with GraphRAG context, and keeps every answer
            grounded in evidence — with source-grounded chat for the follow-up questions that matter.
          </p>
          <div className="mt-8 flex justify-center">
            <Button size="lg" onClick={onSignIn}>
              Sign in with Google
              <ArrowRight size={16} />
            </Button>
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-4 pb-16 sm:px-6">
          <div className="grid gap-4 sm:grid-cols-3">
            {STATS.map((stat) => (
              <Card key={stat.label}>
                <CardHeader>
                  <stat.icon className="size-5 text-primary" />
                  <CardTitle className="text-xl">{stat.value}</CardTitle>
                  <CardDescription>{stat.label}</CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-4 pb-16 sm:px-6">
          <h2 className="text-center text-2xl font-semibold tracking-tight sm:text-3xl">How Cortex works</h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((feature) => (
              <Card key={feature.title}>
                <CardHeader>
                  <feature.icon className="size-5 text-primary" />
                  <CardTitle className="text-base">{feature.title}</CardTitle>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-3xl px-4 pb-16 sm:px-6">
          <h2 className="text-center text-2xl font-semibold tracking-tight sm:text-3xl">Simple pricing</h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Free</CardTitle>
                <CardDescription>Daily limits on research and questions — enough to try it out.</CardDescription>
              </CardHeader>
              <CardFooter>
                <Button variant="outline" className="w-full" onClick={onSignIn}>
                  Get started
                </Button>
              </CardFooter>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <CardTitle>Pro</CardTitle>
                <CardDescription>Higher daily limits on research and questions for regular use.</CardDescription>
              </CardHeader>
              <CardFooter>
                <Button className="w-full" onClick={onSignIn}>
                  Upgrade
                </Button>
              </CardFooter>
            </Card>
          </div>
        </section>

        <section className="border-t border-border px-4 py-12 text-center sm:px-6">
          <h2 className="text-xl font-semibold tracking-tight sm:text-2xl">Ready to start researching?</h2>
          <div className="mt-6 flex justify-center">
            <Button size="lg" onClick={onSignIn}>
              Sign in with Google
            </Button>
          </div>
        </section>
      </main>

      <footer className="shrink-0 border-t border-border px-4 py-6 text-center text-xs text-muted-foreground sm:px-6">
        Cortex — research, grounded.
      </footer>
    </div>
  )
}
