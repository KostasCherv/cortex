import { useState } from 'react'
import { Loader2, SendHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'

type Props = {
  onSubmit: (query: string) => Promise<void>
  disabled: boolean
  isStreaming: boolean
}

export function QueryForm({ onSubmit, disabled, isStreaming }: Props) {
  const [query, setQuery] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || disabled) return
    await onSubmit(query)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">New Research</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4">
          <Textarea
            placeholder="e.g. Compare Model Context Protocol server frameworks in Python vs TypeScript."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={4}
            disabled={disabled}
            required
            className="resize-none"
          />
          <div className="flex justify-end">
            <Button type="submit" disabled={disabled || !query.trim()} size="sm">
              {isStreaming ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <SendHorizontal size={14} />
                  {disabled ? 'Sign in to run' : 'Run Research'}
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
