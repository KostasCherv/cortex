import { useState } from 'react'
import { Loader2, SendHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

type Props = {
  onSubmit: (query: string) => Promise<void>
  disabled: boolean
  isStreaming: boolean
  showSubmit?: boolean
}

export function QueryComposer({ onSubmit, disabled, isStreaming, showSubmit = true }: Props) {
  const [query, setQuery] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || disabled) return
    await onSubmit(query)
    setQuery('')
  }

  return (
    <form onSubmit={(e) => void handleSubmit(e)} className="space-y-3">
      <Textarea
        placeholder="e.g. Compare Model Context Protocol server frameworks in Python vs TypeScript."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        rows={4}
        disabled={disabled}
        required
        className="resize-none text-sm"
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault()
            void handleSubmit(e as unknown as React.FormEvent)
          }
        }}
      />
      {showSubmit && (
        <div className="flex justify-end">
          <Button type="submit" disabled={disabled || !query.trim()} size="sm">
            {isStreaming ? (
              <>
                <Loader2 size={13} className="animate-spin" />
                Running...
              </>
            ) : (
              <>
                <SendHorizontal size={13} />
                {disabled ? 'Sign in to run' : 'Run research'}
              </>
            )}
          </Button>
        </div>
      )}
    </form>
  )
}
