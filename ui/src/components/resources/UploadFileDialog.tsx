import { useCallback, useState } from 'react'
import { Loader2, Upload } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onUpload: (file: File) => Promise<void>
}

export function UploadFileDialog({ open, onOpenChange, onUpload }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    setFile(accepted[0] ?? null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: { 'application/pdf': [], 'text/plain': [], 'text/markdown': [] },
  })

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      await onUpload(file)
      setFile(null)
      onOpenChange(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload file</DialogTitle>
        </DialogHeader>
        <div
          {...getRootProps()}
          className={cn(
            'w-full min-w-0 cursor-pointer overflow-hidden rounded-lg border-2 border-dashed p-8 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
            isDragActive
              ? 'border-primary bg-primary/5'
              : 'border-muted-foreground/25 hover:border-primary/50',
          )}
        >
          <input {...getInputProps()} />
          <Upload size={32} className="mx-auto mb-3 text-muted-foreground" />
          {file ? (
            <p className="mx-auto max-w-full truncate px-1 text-sm font-medium" title={file.name}>
              {file.name}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">
              {isDragActive ? 'Drop it here' : 'Drag & drop or click to select'}
            </p>
          )}
          <p className="text-xs text-muted-foreground mt-1">PDF, TXT, Markdown</p>
        </div>
        {error && (
          <p role="alert" className="rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={uploading}>
            Cancel
          </Button>
          <Button onClick={() => void handleUpload()} disabled={!file || uploading}>
            {uploading && <Loader2 size={14} className="animate-spin" />}
            {uploading ? 'Uploading' : 'Upload'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
