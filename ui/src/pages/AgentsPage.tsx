import { useCallback, useEffect, useMemo, useState } from 'react'
import { Plus } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import {
  createCheckoutSession,
  createPortalSession,
  createRagAgent,
  deleteRagAgent,
  getBillingUsage,
  listRagAgents,
  listRagResources,
} from '@/api/client'
import { AgentCard } from '@/components/agents/AgentCard'
import { NewAgentSheet } from '@/components/agents/NewAgentSheet'
import { Button } from '@/components/ui/button'
import type { BillingUsageSummary, RagAgent, RagResource } from '@/types'

export function AgentsPage({ authSession }: { authSession: Session | null }) {
  const [agents, setAgents] = useState<RagAgent[]>([])
  const [resources, setResources] = useState<RagResource[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sheetOpen, setSheetOpen] = useState(false)
  const [billingUsage, setBillingUsage] = useState<BillingUsageSummary | null>(null)

  const readyResources = useMemo(() => resources.filter((r) => r.state === 'ready'), [resources])

  const load = useCallback(async () => {
    if (!authSession?.access_token) return
    setLoading(true)
    try {
      const [agentData, resourceData] = await Promise.all([
        listRagAgents(authSession.access_token),
        listRagResources(authSession.access_token),
      ])
      setAgents(agentData.agents)
      setResources(resourceData.resources)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load agents.')
    } finally {
      setLoading(false)
    }
  }, [authSession?.access_token])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    if (!authSession?.access_token) return
    void getBillingUsage(authSession.access_token)
      .then(setBillingUsage)
      .catch(() => {})
  }, [authSession?.access_token])

  const handleCreate = async (payload: {
    name: string
    description: string
    system_instructions: string
    linked_resource_ids: string[]
  }) => {
    if (!authSession?.access_token) return
    try {
      await createRagAgent(payload, authSession.access_token)
      await load()
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create agent.')
    }
  }

  const handleDelete = async (agentId: string) => {
    if (!authSession?.access_token) return
    try {
      await deleteRagAgent(agentId, authSession.access_token)
      setAgents((prev) => prev.filter((a) => a.agent_id !== agentId))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete agent.')
    }
  }

  return (
    <main className="max-w-screen-lg mx-auto px-4 py-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Agents</h1>
        {authSession && (
          <Button size="sm" onClick={() => setSheetOpen(true)}>
            <Plus size={14} />
            New agent
          </Button>
        )}
      </div>
      {authSession && billingUsage && (
        <div className="rounded-md border bg-card p-3 text-xs text-muted-foreground">
          <div>
            Plan: <span className="font-medium">{billingUsage.plan}</span> · Questions:{' '}
            {billingUsage.usage.total_questions_count}/{billingUsage.limits.total_questions_daily}
          </div>
          {billingUsage.plan === 'free' && (
            <button
              type="button"
              className="mt-2 text-primary underline"
              onClick={() => {
                void createCheckoutSession(authSession.access_token).then((res) => {
                  window.location.href = res.url
                })
              }}
            >
              Upgrade to Pro
            </button>
          )}
          {billingUsage.plan === 'pro' && (
            <button
              type="button"
              className="mt-2 text-primary underline"
              onClick={() => {
                void createPortalSession(authSession.access_token)
                  .then((res) => {
                    window.location.href = res.url
                  })
                  .catch((portalError) => {
                    setError(
                      portalError instanceof Error
                        ? portalError.message
                        : 'Failed to open billing portal.',
                    )
                  })
              }}
            >
              Manage Subscription
            </button>
          )}
        </div>
      )}
      {error && <p className="text-destructive text-sm">{error}</p>}
      {!authSession ? (
        <p className="text-muted-foreground text-sm">Sign in to create and manage agents.</p>
      ) : loading ? (
        <p className="text-muted-foreground text-sm">Loading...</p>
      ) : agents.length === 0 ? (
        <p className="text-muted-foreground text-sm text-center py-12">
          No agents yet. Create one to get started.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((a) => (
            <AgentCard key={a.agent_id} agent={a} onChat={() => undefined} onDelete={handleDelete} />
          ))}
        </div>
      )}
      <NewAgentSheet
        open={sheetOpen}
        onOpenChange={setSheetOpen}
        readyResources={readyResources}
        onCreate={handleCreate}
      />
    </main>
  )
}
