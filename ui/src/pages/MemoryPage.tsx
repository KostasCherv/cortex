import { useCallback, useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { deleteUserMemory, getUserMemory, updateUserMemory } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

type Props = {
  authSession: Session | null
}

export function MemoryPage({ authSession }: Props) {
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [content, setContent] = useState('')
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)
  const [lastRefreshedAt, setLastRefreshedAt] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!authSession?.access_token) return
    setLoading(true)
    try {
      const memory = await getUserMemory(authSession.access_token)
      setContent(memory.content)
      setUpdatedAt(memory.updated_at)
      setLastRefreshedAt(memory.last_refreshed_at)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load memory.')
    } finally {
      setLoading(false)
    }
  }, [authSession?.access_token])

  useEffect(() => {
    void load()
  }, [load])

  const handleSave = async () => {
    if (!authSession?.access_token || !content.trim()) return
    setSaving(true)
    try {
      const memory = await updateUserMemory(content.trim(), authSession.access_token)
      setContent(memory.content)
      setUpdatedAt(memory.updated_at)
      setLastRefreshedAt(memory.last_refreshed_at)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save memory.')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!authSession?.access_token) return
    setDeleting(true)
    try {
      await deleteUserMemory(authSession.access_token)
      setContent('')
      setUpdatedAt(null)
      setLastRefreshedAt(null)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete memory.')
    } finally {
      setDeleting(false)
    }
  }

  if (!authSession) {
    return (
      <main className="mx-auto max-w-screen-lg space-y-4 px-4 py-6">
        <h1 className="text-xl font-semibold">Memory</h1>
        <p className="text-sm text-muted-foreground">Sign in to manage your memory.</p>
      </main>
    )
  }

  return (
    <main className="mx-auto max-w-screen-lg space-y-4 px-4 py-6">
      <div className="space-y-1">
        <h1 className="text-xl font-semibold">Memory</h1>
        <p className="text-sm text-muted-foreground">
          One editable memory the system can reuse across sessions.
        </p>
      </div>

      {error ? (
        <p role="alert" className="rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      ) : null}

      {loading ? (
        <div className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm text-muted-foreground">
          <Loader2 size={14} className="animate-spin" />
          Loading memory
        </div>
      ) : (
        <section className="space-y-3 rounded-lg border p-4">
          <div className="space-y-1">
            <h2 className="font-medium">Your memory</h2>
            <p className="text-sm text-muted-foreground">
              Edit the durable notes you want the system to remember.
            </p>
          </div>
          <label htmlFor="memory-content" className="sr-only">Memory content</label>
          <Textarea
            id="memory-content"
            value={content}
            onChange={(event) => setContent(event.target.value)}
            placeholder="Add the durable preferences or facts you want the system to remember."
            rows={10}
          />
          <div className="text-xs text-muted-foreground">
            {updatedAt ? `Last edited ${new Date(updatedAt).toLocaleString()}` : 'No saved memory yet.'}
            {lastRefreshedAt ? ` Auto-refreshed ${new Date(lastRefreshedAt).toLocaleString()}.` : ''}
          </div>
          <div className="flex gap-2">
            <Button onClick={() => void handleSave()} disabled={saving || !content.trim()}>
              {saving ? 'Saving...' : 'Save'}
            </Button>
            <Button variant="outline" onClick={() => void handleDelete()} disabled={deleting}>
              {deleting ? 'Deleting...' : 'Delete memory'}
            </Button>
          </div>
        </section>
      )}
    </main>
  )
}
