import { Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function CompletedRunLoader() {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Loading final report</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
          <Loader2 size={16} className="animate-spin text-primary shrink-0" />
          <span>Research is complete. Preparing the final report...</span>
        </div>
      </CardContent>
    </Card>
  )
}
