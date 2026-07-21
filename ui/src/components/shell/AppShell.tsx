import { useCallback, useEffect, useMemo, useState } from 'react'
import type { AuthChangeEvent, Session } from '@supabase/supabase-js'
import { checkHealth, createRagAgent, generateRagAgentDraft, listRagAgents, listRagResources, updateRagAgent } from '@/api/client'
import { AgentChat } from '@/components/agents/AgentChat'
import { GenericChat } from '@/components/chat/GenericChat'
import { NewAgentSheet } from '@/components/agents/NewAgentSheet'
import { AgentRail } from '@/components/shell/AgentRail'
import { Button } from '@/components/ui/button'
import { supabase } from '@/lib/supabase'
import { LandingPage } from '@/pages/LandingPage'
import { MemoryPage } from '@/pages/MemoryPage'
import { ResearchPage } from '@/pages/ResearchPage'
import { ResourcesPage } from '@/pages/ResourcesPage'
import type { HealthResponse, RagAgent, RagResource } from '@/types'

type HealthState = 'loading' | 'online' | 'offline'

export type ActiveView =
  | { type: 'chat' }
  | { type: 'research' }
  | { type: 'rag-agent'; agent: RagAgent }
  | { type: 'memory' }
  | { type: 'resources' }

export function AppShell() {
  const [health, setHealth] = useState<HealthState>('loading')
  const [authSession, setAuthSession] = useState<Session | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [activeView, setActiveView] = useState<ActiveView>({ type: 'chat' })
  const [ragAgents, setRagAgents] = useState<RagAgent[]>([])
  const [resources, setResources] = useState<RagResource[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [sessionRefreshToken, setSessionRefreshToken] = useState(0)
  const [newAgentSheetOpen, setNewAgentSheetOpen] = useState(false)
  const [editingAgent, setEditingAgent] = useState<RagAgent | null>(null)

  const accessToken = authSession?.access_token ?? null
  const readyResources = useMemo(() => resources.filter((r) => r.state === 'ready'), [resources])
  const hasPendingResources = useMemo(
    () => resources.some((r) => r.state === 'uploaded' || r.state === 'processing'),
    [resources],
  )

  /** Apply a server list without dropping locally-known pending uploads still missing from a stale response. */
  const applyResourcesList = useCallback((incoming: RagResource[]) => {
    setResources((prev) => {
      const byId = new Map(incoming.map((r) => [r.resource_id, r]))
      for (const resource of prev) {
        if (
          !byId.has(resource.resource_id) &&
          (resource.state === 'uploaded' || resource.state === 'processing')
        ) {
          byId.set(resource.resource_id, resource)
        }
      }
      return Array.from(byId.values()).sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
    })
  }, [])

  const loadResources = useCallback(async () => {
    if (!accessToken) {
      setResources([])
      return []
    }
    const { resources: data } = await listRagResources(accessToken)
    applyResourcesList(data)
    return data
  }, [accessToken, applyResourcesList])

  const upsertResource = useCallback((resource: RagResource) => {
    setResources((prev) => {
      const index = prev.findIndex((r) => r.resource_id === resource.resource_id)
      if (index >= 0) {
        const next = [...prev]
        next[index] = resource
        return next
      }
      return [resource, ...prev]
    })
  }, [])

  useEffect(() => {
    void checkHealth()
      .then((r: HealthResponse) => setHealth(r.status === 'ok' ? 'online' : 'offline'))
      .catch(() => setHealth('offline'))
  }, [])

  useEffect(() => {
    void supabase.auth.getSession().then(({ data }) => {
      setAuthSession(data.session)
      setAuthChecked(true)
      if (!data.session) {
        setRagAgents([])
        setResources([])
      }
    })
    const { data } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session) => {
      setAuthSession(session)
      if (!session) {
        setRagAgents([])
        setResources([])
      }
    })
    return () => data.subscription.unsubscribe()
  }, [])

  useEffect(() => {
    if (!accessToken) return
    void listRagAgents(accessToken)
      .then(({ agents }) => setRagAgents(agents))
      .catch(() => setRagAgents([]))
    void listRagResources(accessToken)
      .then(({ resources: data }) => applyResourcesList(data))
      .catch(() => setResources([]))
  }, [accessToken, applyResourcesList])

  useEffect(() => {
    if (!accessToken || !hasPendingResources) return

    const refresh = () => {
      void listRagResources(accessToken)
        .then(({ resources: data }) => applyResourcesList(data))
        .catch(() => {})
    }

    refresh()
    const intervalId = window.setInterval(refresh, 2000)
    return () => window.clearInterval(intervalId)
  }, [accessToken, hasPendingResources, applyResourcesList])

  const signInWithGoogle = useCallback(async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: `${window.location.origin}/` },
    })
  }, [])

  const signOut = useCallback(async () => {
    await supabase.auth.signOut()
  }, [])

  const handleViewChange = useCallback((view: ActiveView) => {
    setActiveView(view)
    setActiveSessionId(null)
  }, [])

  const handleSessionActivated = useCallback((id: string | null) => {
    setActiveSessionId(id)
    if (id) setSessionRefreshToken((n) => n + 1)
  }, [])

  const handleSessionsChanged = useCallback(() => {
    setSessionRefreshToken((n) => n + 1)
  }, [])

  const handleAgentDeleted = useCallback(
    (agentId: string) => {
      setRagAgents((prev) => prev.filter((a) => a.agent_id !== agentId))
      setActiveView((prev) =>
        prev.type === 'rag-agent' && prev.agent.agent_id === agentId
          ? { type: 'research' }
          : prev,
      )
    },
    [],
  )

  const handleCreateAgent = useCallback(
    async (payload: {
      name: string
      description: string
      system_instructions: string
      linked_resource_ids: string[]
    }) => {
      if (!accessToken) return
      const { agent } = await createRagAgent(payload, accessToken)
      setRagAgents((prev) => [...prev, agent])
    },
    [accessToken],
  )

  const handleGenerateAgentDraft = useCallback(
    async (prompt: string) => {
      if (!accessToken) throw new Error('You must be signed in to create an agent.')
      const { draft } = await generateRagAgentDraft(prompt, accessToken)
      return draft
    },
    [accessToken],
  )

  const handleUpdateAgent = useCallback(
    async (
      agentId: string,
      payload: {
        name: string
        description: string
        system_instructions: string
        linked_resource_ids: string[]
      },
    ) => {
      if (!accessToken) throw new Error('You must be signed in to update an agent.')
      const { agent } = await updateRagAgent(agentId, payload, accessToken)
      setRagAgents((prev) => prev.map((a) => (a.agent_id === agent.agent_id ? agent : a)))
      setActiveView((prev) =>
        prev.type === 'rag-agent' && prev.agent.agent_id === agent.agent_id
          ? { type: 'rag-agent', agent }
          : prev,
      )
    },
    [accessToken],
  )

  const handleAgentSheetOpenChange = useCallback((open: boolean) => {
    setNewAgentSheetOpen(open)
    if (!open) setEditingAgent(null)
  }, [])

  if (!authChecked) {
    return <div className="h-dvh bg-background" />
  }

  if (!authSession) {
    return (
      <div className="animate-fade-in">
        <LandingPage onSignIn={() => void signInWithGoogle()} />
      </div>
    )
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-background max-md:flex-col animate-fade-in">
      <AgentRail
        health={health}
        authSession={authSession}
        activeView={activeView}
        ragAgents={ragAgents}
        activeSessionId={activeSessionId}
        sessionRefreshToken={sessionRefreshToken}
        onViewChange={handleViewChange}
        onSessionSelect={setActiveSessionId}
        onSignIn={() => void signInWithGoogle()}
        onSignOut={() => void signOut()}
        onEditAgent={(agent) => {
          setEditingAgent(agent)
          setNewAgentSheetOpen(true)
        }}
        onAgentDeleted={handleAgentDeleted}
        onNewAgent={() => {
          setEditingAgent(null)
          setNewAgentSheetOpen(true)
        }}
        onNewResearch={() => {
          setActiveView({ type: 'research' })
          setActiveSessionId(null)
        }}
        onNewChat={() => {
          setActiveView({ type: 'chat' })
          setActiveSessionId(null)
        }}
      />
      <div className="flex-1 min-w-0 overflow-hidden max-md:min-h-0">
        {activeView.type === 'chat' &&
          (authSession ? (
            <GenericChat
              accessToken={authSession.access_token}
              activeSessionId={activeSessionId}
              onSessionActivated={handleSessionActivated}
              onSessionsChanged={handleSessionsChanged}
            />
          ) : (
            <main className="flex h-full items-center justify-center px-6">
              <div className="flex flex-col items-center gap-4 text-center">
                <p className="text-sm text-muted-foreground">Sign in to start a workspace chat.</p>
                <Button size="sm" onClick={() => void signInWithGoogle()}>
                  Sign in with Google
                </Button>
              </div>
            </main>
          ))}
        {activeView.type === 'research' && (
          <ResearchPage
            authSession={authSession}
            activeSessionId={activeSessionId}
            onSessionActivated={handleSessionActivated}
            onSessionsChanged={handleSessionsChanged}
          />
        )}
        {activeView.type === 'rag-agent' && authSession && (
          <AgentChat
            agent={activeView.agent}
            accessToken={authSession.access_token}
            activeSessionId={activeSessionId}
            onSessionActivated={handleSessionActivated}
            onSessionsChanged={handleSessionsChanged}
          />
        )}
        {activeView.type === 'memory' && <MemoryPage authSession={authSession} />}
        {activeView.type === 'resources' && (
          <ResourcesPage
            authSession={authSession}
            resources={resources}
            onResourcesChange={loadResources}
            onResourceUploaded={upsertResource}
          />
        )}
      </div>
      <NewAgentSheet
        open={newAgentSheetOpen}
        onOpenChange={handleAgentSheetOpenChange}
        agent={editingAgent}
        readyResources={readyResources}
        onGenerateDraft={handleGenerateAgentDraft}
        onCreate={handleCreateAgent}
        onUpdate={handleUpdateAgent}
      />
    </div>
  )
}
