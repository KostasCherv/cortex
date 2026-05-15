import { useCallback, useEffect, useRef, useState } from 'react'
import { Bot, FolderOpen, Loader2, LogOut, MessageSquare, Moon, MoreHorizontal, Pencil, Plus, Sun, Telescope, Trash2 } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import {
  createCheckoutSession,
  createPortalSession,
  deleteRagAgent,
  deleteRagAgentChatSession,
  deleteRagWorkspaceChatSession,
  deleteSession,
  getBillingUsage,
  listRagAgentChatSessions,
  listRagWorkspaceChatSessions,
  listSessions,
  updateRagAgentChatSessionTitle,
  updateRagWorkspaceChatSessionTitle,
  updateSessionTitle,
} from '@/api/client'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'
import type { BillingUsageSummary, RagAgent, RagChatSessionSummary, SessionSummary } from '@/types'
import type { ActiveView } from './AppShell'

type Props = {
  health: 'loading' | 'online' | 'offline'
  authSession: Session | null
  activeView: ActiveView
  ragAgents: RagAgent[]
  activeSessionId: string | null
  sessionRefreshToken: number
  onViewChange: (view: ActiveView) => void
  onSessionSelect: (id: string | null) => void
  onSignIn: () => void
  onSignOut: () => void
  onEditAgent: (agent: RagAgent) => void
  onAgentDeleted: (agentId: string) => void
  onNewAgent: () => void
  onNewResearch: () => void
  onNewChat: () => void
}

type ResearchSession = SessionSummary
type AgentSession = RagChatSessionSummary

function formatDateLabel(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function daysUntilLabel(iso: string): string {
  const target = new Date(iso)
  if (Number.isNaN(target.getTime())) return ''
  const now = new Date()
  const msPerDay = 24 * 60 * 60 * 1000
  const days = Math.max(0, Math.ceil((target.getTime() - now.getTime()) / msPerDay))
  return `Expires in ${days} day${days === 1 ? '' : 's'}`
}

function ResearchSessionList({
  accessToken,
  activeSessionId,
  refreshToken,
  onSelect,
}: {
  accessToken: string
  activeSessionId: string | null
  refreshToken: number
  onSelect: (id: string | null) => void
}) {
  const [sessions, setSessions] = useState<ResearchSession[] | null>(null)
  const fetchedTokenRef = useRef(-1)
  const [contextMenu, setContextMenu] = useState<{
    id: string
    title: string
    x: number
    y: number
  } | null>(null)
  const [renameTarget, setRenameTarget] = useState<{ id: string; title: string } | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renamePending, setRenamePending] = useState(false)
  const [renameError, setRenameError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null)
  const [deletePending, setDeletePending] = useState(false)

  const fetchSessions = useCallback(() => {
    void listSessions(accessToken)
      .then(({ sessions: data }) => setSessions(data))
      .catch(() => setSessions([]))
  }, [accessToken])

  useEffect(() => {
    if (fetchedTokenRef.current === refreshToken) return
    fetchedTokenRef.current = refreshToken
    fetchSessions()
  }, [fetchSessions, refreshToken])

  useEffect(() => {
    if (!sessions || sessions.length === 0) return
    const hasRunning = sessions.some((session) => session.latest_run_status === 'running')
    if (!hasRunning) return
    const intervalId = window.setInterval(() => {
      fetchSessions()
    }, 5000)
    return () => {
      window.clearInterval(intervalId)
    }
  }, [fetchSessions, sessions])

  useEffect(() => {
    if (!contextMenu) return
    const closeMenu = () => setContextMenu(null)
    window.addEventListener('click', closeMenu)
    window.addEventListener('scroll', closeMenu, true)
    window.addEventListener('resize', closeMenu)
    return () => {
      window.removeEventListener('click', closeMenu)
      window.removeEventListener('scroll', closeMenu, true)
      window.removeEventListener('resize', closeMenu)
    }
  }, [contextMenu])

  if (sessions === null) {
    return (
      <div className="flex items-center gap-1.5 px-2 py-1.5">
        <Loader2 size={11} className="animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Loading sessions</span>
      </div>
    )
  }

  return (
    <>
      {sessions.slice(0, 10).map((s) => (
        <button
          key={s.session_id}
          type="button"
          onClick={() => onSelect(s.session_id)}
          onContextMenu={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setContextMenu({
              id: s.session_id,
              title: s.title || 'Untitled session',
              x: e.clientX,
              y: e.clientY,
            })
          }}
          className={cn(
            'w-full truncate rounded px-2 py-1 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
            activeSessionId === s.session_id
              ? 'bg-primary/10 text-primary font-medium'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
          )}
          aria-current={activeSessionId === s.session_id ? 'page' : undefined}
          title={s.title || 'Untitled session'}
        >
          <span className="inline-flex min-w-0 items-center gap-1.5">
            {s.latest_run_status === 'running' && (
              <span className="size-1.5 rounded-full bg-primary animate-pulse" aria-label="Running" />
            )}
            {s.latest_run_status === 'failed' && (
              <span className="size-1.5 rounded-full bg-destructive" aria-label="Failed" />
            )}
            <span className="truncate">{s.title || 'Untitled session'}</span>
          </span>
        </button>
      ))}
      {contextMenu && (
        <div
          className="fixed z-50 min-w-32 rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground"
            onClick={() => {
              setRenameTarget({ id: contextMenu.id, title: contextMenu.title })
              setRenameValue(contextMenu.title)
              setRenameError(null)
              setContextMenu(null)
            }}
          >
            Edit
          </button>
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm text-destructive hover:bg-accent"
            onClick={() => {
              setDeleteTarget({ id: contextMenu.id, title: contextMenu.title })
              setContextMenu(null)
            }}
          >
            Delete
          </button>
        </div>
      )}
      <Dialog
        open={!!renameTarget}
        onOpenChange={(open: boolean) => {
          if (!open && !renamePending) {
            setRenameTarget(null)
            setRenameError(null)
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Discussion Title</DialogTitle>
            <DialogDescription>Choose a clearer name for this research discussion.</DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            maxLength={120}
            autoFocus
            onKeyDown={(e) => {
              if (e.key !== 'Enter' || renamePending || !renameTarget || !renameValue.trim()) return
              setRenamePending(true)
              setRenameError(null)
              void updateSessionTitle(renameTarget.id, renameValue.trim(), accessToken)
                .then(() => {
                  const title = renameValue.trim()
                  setSessions((prev) =>
                    prev?.map((session) =>
                      session.session_id === renameTarget.id ? { ...session, title } : session,
                    ) ?? prev,
                  )
                  setRenameTarget(null)
                })
                .catch((error) => {
                  setRenameError(error instanceof Error ? error.message : 'Failed to rename discussion.')
                })
                .finally(() => setRenamePending(false))
            }}
          />
          {renameError && <p className="text-sm text-destructive">{renameError}</p>}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setRenameTarget(null)
                setRenameError(null)
              }}
              disabled={renamePending}
            >
              Cancel
            </Button>
            <Button
              disabled={renamePending || !renameTarget || !renameValue.trim()}
              onClick={() => {
                if (!renameTarget || !renameValue.trim()) return
                setRenamePending(true)
                setRenameError(null)
                void updateSessionTitle(renameTarget.id, renameValue.trim(), accessToken)
                  .then(() => {
                    const title = renameValue.trim()
                    setSessions((prev) =>
                      prev?.map((session) =>
                        session.session_id === renameTarget.id ? { ...session, title } : session,
                      ) ?? prev,
                    )
                    setRenameTarget(null)
                  })
                  .catch((error) => {
                    setRenameError(error instanceof Error ? error.message : 'Failed to rename discussion.')
                  })
                  .finally(() => setRenamePending(false))
              }}
            >
              {renamePending ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!deleteTarget}
        onOpenChange={(open: boolean) => {
          if (!open && !deletePending) setDeleteTarget(null)
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Discussion</DialogTitle>
            <DialogDescription>
              This will permanently delete "{deleteTarget?.title ?? 'this discussion'}".
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)} disabled={deletePending}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={deletePending || !deleteTarget}
              onClick={() => {
                if (!deleteTarget) return
                setDeletePending(true)
                void deleteSession(deleteTarget.id, accessToken)
                  .then(() => {
                    setSessions((prev) => prev?.filter((session) => session.session_id !== deleteTarget.id) ?? prev)
                    if (activeSessionId === deleteTarget.id) onSelect(null)
                    setDeleteTarget(null)
                  })
                  .finally(() => setDeletePending(false))
              }}
            >
              {deletePending ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function AgentSessionList({
  agentId,
  accessToken,
  activeSessionId,
  refreshToken,
  onSelect,
}: {
  agentId: string
  accessToken: string
  activeSessionId: string | null
  refreshToken: number
  onSelect: (id: string | null) => void
}) {
  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [pending, setPending] = useState(true)
  const fetchedTokenRef = useRef(-1)
  const [contextMenu, setContextMenu] = useState<{
    id: string
    title: string
    x: number
    y: number
  } | null>(null)
  const [renameTarget, setRenameTarget] = useState<{ id: string; title: string } | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renamePending, setRenamePending] = useState(false)
  const [renameError, setRenameError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null)
  const [deletePending, setDeletePending] = useState(false)

  useEffect(() => {
    if (fetchedTokenRef.current === refreshToken) return
    fetchedTokenRef.current = refreshToken
    void listRagAgentChatSessions(agentId, accessToken)
      .then(({ sessions: data }) => {
        setSessions(data)
        setPending(false)
      })
      .catch(() => {
        setSessions([])
        setPending(false)
      })
  }, [agentId, accessToken, refreshToken])

  useEffect(() => {
    if (!contextMenu) return
    const closeMenu = () => setContextMenu(null)
    window.addEventListener('click', closeMenu)
    window.addEventListener('scroll', closeMenu, true)
    window.addEventListener('resize', closeMenu)
    return () => {
      window.removeEventListener('click', closeMenu)
      window.removeEventListener('scroll', closeMenu, true)
      window.removeEventListener('resize', closeMenu)
    }
  }, [contextMenu])

  if (pending) {
    return (
      <div className="flex items-center gap-1.5 px-2 py-1.5">
        <Loader2 size={11} className="animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Loading chats</span>
      </div>
    )
  }

  return (
    <>
      {sessions.slice(0, 10).map((s) => (
        <button
          key={s.session_id}
          type="button"
          onClick={() => onSelect(s.session_id)}
          onContextMenu={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setContextMenu({
              id: s.session_id,
              title: s.title || s.last_message_preview || 'New chat',
              x: e.clientX,
              y: e.clientY,
            })
          }}
          className={cn(
            'w-full truncate rounded px-2 py-1 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
            activeSessionId === s.session_id
              ? 'bg-primary/10 text-primary font-medium'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
          )}
          aria-current={activeSessionId === s.session_id ? 'page' : undefined}
          title={s.title || s.last_message_preview || 'New chat'}
        >
          {s.title || s.last_message_preview || 'New chat'}
        </button>
      ))}
      {contextMenu && (
        <div
          className="fixed z-50 min-w-32 rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground"
            onClick={() => {
              setRenameTarget({ id: contextMenu.id, title: contextMenu.title })
              setRenameValue(contextMenu.title)
              setRenameError(null)
              setContextMenu(null)
            }}
          >
            Edit
          </button>
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm text-destructive hover:bg-accent"
            onClick={() => {
              setDeleteTarget({ id: contextMenu.id, title: contextMenu.title })
              setContextMenu(null)
            }}
          >
            Delete
          </button>
        </div>
      )}
      <Dialog
        open={!!renameTarget}
        onOpenChange={(open: boolean) => {
          if (!open && !renamePending) {
            setRenameTarget(null)
            setRenameError(null)
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Discussion Title</DialogTitle>
            <DialogDescription>Choose a clearer name for this agent discussion.</DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            maxLength={120}
            autoFocus
            onKeyDown={(e) => {
              if (e.key !== 'Enter' || renamePending || !renameTarget || !renameValue.trim()) return
              setRenamePending(true)
              setRenameError(null)
              void updateRagAgentChatSessionTitle(agentId, renameTarget.id, renameValue.trim(), accessToken)
                .then(() => {
                  const title = renameValue.trim()
                  setSessions((prev) =>
                    prev.map((session) => (session.session_id === renameTarget.id ? { ...session, title } : session)),
                  )
                  setRenameTarget(null)
                })
                .catch((error) => {
                  setRenameError(error instanceof Error ? error.message : 'Failed to rename discussion.')
                })
                .finally(() => setRenamePending(false))
            }}
          />
          {renameError && <p className="text-sm text-destructive">{renameError}</p>}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setRenameTarget(null)
                setRenameError(null)
              }}
              disabled={renamePending}
            >
              Cancel
            </Button>
            <Button
              disabled={renamePending || !renameTarget || !renameValue.trim()}
              onClick={() => {
                if (!renameTarget || !renameValue.trim()) return
                setRenamePending(true)
                setRenameError(null)
                void updateRagAgentChatSessionTitle(agentId, renameTarget.id, renameValue.trim(), accessToken)
                  .then(() => {
                    const title = renameValue.trim()
                    setSessions((prev) =>
                      prev.map((session) => (session.session_id === renameTarget.id ? { ...session, title } : session)),
                    )
                    setRenameTarget(null)
                  })
                  .catch((error) => {
                    setRenameError(error instanceof Error ? error.message : 'Failed to rename discussion.')
                  })
                  .finally(() => setRenamePending(false))
              }}
            >
              {renamePending ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!deleteTarget}
        onOpenChange={(open: boolean) => {
          if (!open && !deletePending) setDeleteTarget(null)
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Discussion</DialogTitle>
            <DialogDescription>
              This will permanently delete "{deleteTarget?.title ?? 'this discussion'}".
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)} disabled={deletePending}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={deletePending || !deleteTarget}
              onClick={() => {
                if (!deleteTarget) return
                setDeletePending(true)
                void deleteRagAgentChatSession(agentId, deleteTarget.id, accessToken)
                  .then(() => {
                    setSessions((prev) => prev.filter((session) => session.session_id !== deleteTarget.id))
                    if (activeSessionId === deleteTarget.id) onSelect(null)
                    setDeleteTarget(null)
                  })
                  .finally(() => setDeletePending(false))
              }}
            >
              {deletePending ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function WorkspaceChatSessionList({
  accessToken,
  activeSessionId,
  refreshToken,
  onSelect,
}: {
  accessToken: string
  activeSessionId: string | null
  refreshToken: number
  onSelect: (id: string | null) => void
}) {
  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [pending, setPending] = useState(true)
  const fetchedTokenRef = useRef(-1)
  const [contextMenu, setContextMenu] = useState<{ id: string; title: string; x: number; y: number } | null>(null)

  useEffect(() => {
    if (fetchedTokenRef.current === refreshToken) return
    fetchedTokenRef.current = refreshToken
    void listRagWorkspaceChatSessions(accessToken)
      .then(({ sessions: data }) => {
        setSessions(data)
        setPending(false)
      })
      .catch(() => {
        setSessions([])
        setPending(false)
      })
  }, [accessToken, refreshToken])

  if (pending) {
    return (
      <div className="flex items-center gap-1.5 px-2 py-1.5">
        <Loader2 size={11} className="animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Loading chats</span>
      </div>
    )
  }

  return (
    <>
      {sessions.slice(0, 10).map((s) => (
        <button
          key={s.session_id}
          type="button"
          onClick={() => onSelect(s.session_id)}
          onContextMenu={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setContextMenu({
              id: s.session_id,
              title: s.title || s.last_message_preview || 'New chat',
              x: e.clientX,
              y: e.clientY,
            })
          }}
          className={cn(
            'w-full truncate rounded px-2 py-1 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
            activeSessionId === s.session_id
              ? 'bg-primary/10 text-primary font-medium'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
          )}
        >
          {s.title || s.last_message_preview || 'New chat'}
        </button>
      ))}
      {contextMenu && (
        <div className="fixed z-50 min-w-32 rounded-md border bg-popover p-1 text-popover-foreground shadow-md" style={{ left: contextMenu.x, top: contextMenu.y }}>
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground"
            onClick={() => {
              const nextTitle = window.prompt('Rename chat', contextMenu.title)
              if (!nextTitle?.trim()) return
              void updateRagWorkspaceChatSessionTitle(contextMenu.id, nextTitle.trim(), accessToken).then(() => {
                setSessions((prev) => prev.map((session) => (session.session_id === contextMenu.id ? { ...session, title: nextTitle.trim() } : session)))
              })
              setContextMenu(null)
            }}
          >
            Edit
          </button>
          <button
            type="button"
            className="w-full rounded-sm px-2 py-1.5 text-left text-sm text-destructive hover:bg-accent"
            onClick={() => {
              const targetId = contextMenu.id
              void deleteRagWorkspaceChatSession(targetId, accessToken).then(() => {
                setSessions((prev) => prev.filter((session) => session.session_id !== targetId))
                if (activeSessionId === targetId) onSelect(null)
              })
              setContextMenu(null)
            }}
          >
            Delete
          </button>
        </div>
      )}
    </>
  )
}

export function AgentRail({
  health,
  authSession,
  activeView,
  ragAgents,
  activeSessionId,
  sessionRefreshToken,
  onViewChange,
  onSessionSelect,
  onSignIn,
  onSignOut,
  onEditAgent,
  onAgentDeleted,
  onNewAgent,
  onNewResearch,
  onNewChat,
}: Props) {
  const { theme, toggle } = useTheme()
  const accessToken = authSession?.access_token ?? null
  const [billingUsage, setBillingUsage] = useState<BillingUsageSummary | null>(null)

  useEffect(() => {
    if (!accessToken) {
      setBillingUsage(null)
      return
    }
    void getBillingUsage(accessToken)
      .then(setBillingUsage)
      .catch(() => setBillingUsage(null))
  }, [accessToken, sessionRefreshToken])

  const handleDeleteAgent = useCallback(
    async (agent: RagAgent) => {
      if (!accessToken) return
      const confirmed = window.confirm(`Delete agent "${agent.name}"? This cannot be undone.`)
      if (!confirmed) return
      try {
        await deleteRagAgent(agent.agent_id, accessToken)
        onAgentDeleted(agent.agent_id)
      } catch {
        // silently ignore; user sees no change
      }
    },
    [accessToken, onAgentDeleted],
  )

  const isChat = activeView.type === 'chat'
  const isResearch = activeView.type === 'research'
  const isResources = activeView.type === 'resources'

  return (
    <aside className="flex h-dvh w-[280px] min-w-[280px] shrink-0 flex-col border-r border-border bg-secondary max-md:h-[42dvh] max-md:min-h-[260px] max-md:w-full max-md:min-w-0 max-md:border-b max-md:border-r-0">
      {/* Header */}
      <div className="flex h-14 shrink-0 items-center justify-between px-4 max-md:h-12">
        <span className="font-semibold tracking-tight text-foreground">Cortex</span>
        <span
          className={cn(
            'size-1.5 rounded-full',
            health === 'online'
              ? 'bg-green-500'
              : health === 'offline'
                ? 'bg-red-500'
                : 'bg-muted-foreground',
          )}
          title={health === 'online' ? 'Online' : health === 'offline' ? 'Offline' : 'Checking...'}
        />
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto py-2">
        <div className="shrink-0 space-y-0.5 pl-2 pr-4">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onViewChange({ type: 'chat' })}
              className={cn(
                'min-w-0 flex-1 flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
                isChat ? 'bg-background text-foreground font-medium shadow-sm' : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
              )}
            >
              <MessageSquare size={15} className="shrink-0" />
              Chat
            </button>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0 rounded-md border border-border bg-background text-foreground shadow-sm hover:border-primary/30 hover:bg-background hover:text-primary"
              onClick={authSession ? onNewChat : onSignIn}
              aria-label={authSession ? 'New chat' : 'Sign in to create a chat session'}
            >
              <Plus size={14} />
            </Button>
          </div>
          {isChat && authSession && (
            <ScrollArea className="max-h-44 shrink-0">
              <div className="ml-7 mt-0.5 mb-1 space-y-0.5 pr-4">
                <WorkspaceChatSessionList
                  accessToken={authSession.access_token}
                  activeSessionId={activeSessionId}
                  refreshToken={sessionRefreshToken}
                  onSelect={onSessionSelect}
                />
              </div>
            </ScrollArea>
          )}
          {/* Research */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onViewChange({ type: 'research' })}
              className={cn(
                'min-w-0 flex-1 flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
                isResearch
                  ? 'bg-background text-foreground font-medium shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
              )}
            >
              <Telescope size={15} className="shrink-0" />
              Research
            </button>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0 rounded-md border border-border bg-background text-foreground shadow-sm hover:border-primary/30 hover:bg-background hover:text-primary"
              onClick={authSession ? onNewResearch : onSignIn}
              aria-label={authSession ? 'New research' : 'Sign in to create a research session'}
              title={authSession ? 'New research' : 'Sign in to create a research session'}
            >
              <Plus size={14} />
            </Button>
          </div>
          {isResearch && authSession && (
            <ScrollArea className="max-h-44 shrink-0">
              <div className="ml-7 mt-0.5 mb-1 space-y-0.5 pr-4">
                <ResearchSessionList
                  accessToken={authSession.access_token}
                  activeSessionId={activeSessionId}
                  refreshToken={sessionRefreshToken}
                  onSelect={onSessionSelect}
                />
              </div>
            </ScrollArea>
          )}
        </div>

        {/* My Agents section */}
        <div className="flex min-h-0 flex-1 flex-col pt-3">
          <div className="mb-1 flex w-full shrink-0 items-center justify-between gap-2 pl-4 pr-5">
            <span className="min-w-0 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              My Agents
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0 rounded-md border border-border bg-background text-foreground shadow-sm hover:border-primary/30 hover:bg-background hover:text-primary"
              onClick={authSession ? onNewAgent : onSignIn}
              aria-label={authSession ? 'New agent' : 'Sign in to create an agent'}
              title={authSession ? 'New agent' : 'Sign in to create an agent'}
            >
              <Plus size={14} />
            </Button>
          </div>

          <div className="min-h-0 flex-1">
            <div className="space-y-0.5 pl-2 pr-4">
              {!authSession && (
                <p className="px-2 py-1 text-xs text-muted-foreground">Sign in to create agents.</p>
              )}

              {authSession && ragAgents.length === 0 && (
                <p className="px-2 py-1 text-xs text-muted-foreground">No agents yet.</p>
              )}

              {authSession &&
                ragAgents.map((agent) => {
                  const isActive =
                    activeView.type === 'rag-agent' && activeView.agent.agent_id === agent.agent_id

                  return (
                    <div key={agent.agent_id} className="group">
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => onViewChange({ type: 'rag-agent', agent })}
                          className={cn(
                            'flex min-w-0 flex-1 items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
                            isActive
                              ? 'bg-background text-foreground font-medium shadow-sm'
                              : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
                          )}
                        >
                          <Bot size={15} className="shrink-0" />
                          <span className="truncate" title={agent.name}>{agent.name}</span>
                        </button>

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-6 shrink-0 rounded-md opacity-0 text-muted-foreground transition-opacity hover:text-foreground group-hover:opacity-100 group-focus-within:opacity-100"
                              aria-label="Agent options"
                            >
                              <MoreHorizontal size={12} />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-36">
                            <DropdownMenuItem onClick={() => onEditAgent(agent)}>
                              <Pencil size={13} className="mr-2" />
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-destructive focus:text-destructive"
                              onClick={() => void handleDeleteAgent(agent)}
                            >
                              <Trash2 size={13} className="mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>

                      {/* Agent sessions */}
                      {isActive && (
                        <div className="ml-5 mt-0.5 mb-1 space-y-0.5">
                          <AgentSessionList
                            key={agent.agent_id}
                            agentId={agent.agent_id}
                            accessToken={authSession.access_token}
                            activeSessionId={activeSessionId}
                            refreshToken={sessionRefreshToken}
                            onSelect={onSessionSelect}
                          />
                        </div>
                      )}
                    </div>
                  )
                })}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="shrink-0 space-y-0.5 border-t border-border px-2 py-2 max-md:flex max-md:items-center max-md:gap-1 max-md:border-t-0 max-md:pt-0">
        <button
          type="button"
          onClick={() => onViewChange({ type: 'resources' })}
          className={cn(
            'w-full flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-secondary',
            isResources
              ? 'bg-background text-foreground font-medium shadow-sm'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
          )}
        >
          <FolderOpen size={15} className="shrink-0" />
          Resources
        </button>

        <div className="flex items-center gap-1 pt-1 max-md:ml-auto max-md:pt-0">
          <Button
            variant="ghost"
            size="icon"
            className="size-7 text-muted-foreground hover:text-foreground"
            onClick={toggle}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </Button>

          <div className="flex-1" />

          {authSession ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="size-7 rounded-full" aria-label="Account menu">
                  <Avatar className="size-6">
                    <AvatarFallback className="text-[10px]">
                      {authSession.user.email?.[0]?.toUpperCase() ?? '?'}
                    </AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" side="top" className="w-64">
                <DropdownMenuItem className="text-xs text-muted-foreground" disabled>
                  {authSession.user.email}
                </DropdownMenuItem>
                {billingUsage && (
                  <DropdownMenuItem className="text-xs text-muted-foreground" disabled>
                    {billingUsage.plan === 'free'
                      ? `Free · Research ${billingUsage.usage.research_queries_count}/${billingUsage.limits.research_queries_daily} · Questions ${billingUsage.usage.total_questions_count}/${billingUsage.limits.total_questions_daily}`
                      : `Pro · Questions ${billingUsage.usage.total_questions_count}/${billingUsage.limits.total_questions_daily}`}
                  </DropdownMenuItem>
                )}
                {billingUsage?.subscription && (
                  <DropdownMenuItem className="text-xs text-muted-foreground" disabled>
                    {(() => {
                      const subscription = billingUsage.subscription
                      const effectiveEnd = subscription.cancel_at ?? subscription.current_period_end
                      if (subscription.cancel_at_period_end && effectiveEnd) {
                        return `Cancels on ${formatDateLabel(effectiveEnd)} · ${daysUntilLabel(effectiveEnd)}`
                      }
                      if (effectiveEnd && subscription.status === 'active') {
                        return `Renews on ${formatDateLabel(effectiveEnd)} · ${daysUntilLabel(effectiveEnd)}`
                      }
                      return `Status: ${subscription.status}`
                    })()}
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                {billingUsage?.plan === 'free' ? (
                  <DropdownMenuItem
                    onClick={() => {
                      if (!accessToken) return
                      void createCheckoutSession(accessToken).then((res) => {
                        window.location.href = res.url
                      })
                    }}
                  >
                    Upgrade to Pro
                  </DropdownMenuItem>
                ) : (
                  <DropdownMenuItem
                    onClick={() => {
                      if (!accessToken) return
                      void createPortalSession(accessToken).then((res) => {
                        window.location.href = res.url
                      })
                    }}
                  >
                    Manage Subscription
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onSignOut}>
                  <LogOut size={13} className="mr-2" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button size="sm" variant="ghost" className="text-xs h-7 px-2" onClick={onSignIn}>
              Sign in
            </Button>
          )}
        </div>
      </div>
    </aside>
  )
}
