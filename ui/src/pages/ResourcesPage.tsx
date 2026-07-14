import { useCallback, useEffect, useState } from 'react'
import { Loader2, Upload } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { deleteRagResource, uploadRagResource } from '@/api/client'
import { ResourceTable } from '@/components/resources/ResourceTable'
import { UploadFileDialog } from '@/components/resources/UploadFileDialog'
import { Button } from '@/components/ui/button'
import type { RagResource } from '@/types'

type Props = {
  authSession: Session | null
  resources: RagResource[]
  onResourcesChange: () => Promise<RagResource[]>
  onResourceUploaded: (resource: RagResource) => void
}

export function ResourcesPage({
  authSession,
  resources,
  onResourcesChange,
  onResourceUploaded,
}: Props) {
  const [initialLoading, setInitialLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadOpen, setUploadOpen] = useState(false)

  const refresh = useCallback(
    async (options?: { background?: boolean }) => {
      if (!authSession?.access_token) return
      const background = options?.background ?? false
      if (background) {
        setRefreshing(true)
      }
      try {
        await onResourcesChange()
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load resources.')
      } finally {
        if (background) {
          setRefreshing(false)
        }
      }
    },
    [authSession?.access_token, onResourcesChange],
  )

  useEffect(() => {
    if (!authSession?.access_token) {
      setInitialLoading(false)
      return
    }
    void refresh().finally(() => setInitialLoading(false))
  }, [authSession?.access_token, refresh])

  const handleUpload = async (file: File) => {
    if (!authSession?.access_token) return
    try {
      const { resource } = await uploadRagResource(file, authSession.access_token)
      onResourceUploaded(resource)
      await refresh({ background: true })
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload resource.')
      throw err
    }
  }

  const handleDelete = async (id: string) => {
    if (!authSession?.access_token) return
    try {
      await deleteRagResource(id, authSession.access_token)
      await refresh({ background: true })
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete resource.')
    }
  }

  return (
    <main className="mx-auto max-w-screen-lg space-y-4 px-4 py-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Resources</h1>
        <div className="flex items-center gap-2">
          {refreshing && (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Loader2 size={12} className="animate-spin" />
              Refreshing
            </span>
          )}
          {authSession && (
            <Button size="sm" onClick={() => setUploadOpen(true)}>
              <Upload size={14} />
              Upload file
            </Button>
          )}
        </div>
      </div>
      {error && (
        <p role="alert" className="rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      )}
      {!authSession ? (
        <p className="text-muted-foreground text-sm">Sign in to manage your resources.</p>
      ) : initialLoading ? (
        <div className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm text-muted-foreground">
          <Loader2 size={14} className="animate-spin" />
          Loading resources
        </div>
      ) : (
        <ResourceTable resources={resources} onDelete={handleDelete} />
      )}
      <UploadFileDialog open={uploadOpen} onOpenChange={setUploadOpen} onUpload={handleUpload} />
    </main>
  )
}
