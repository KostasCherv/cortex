import { cn } from '@/lib/utils'

type Props = {
  health: 'loading' | 'online' | 'offline'
}

export function HealthDot({ health }: Props) {
  return (
    <span
      className={cn(
        'size-1.5 rounded-full',
        health === 'online' ? 'bg-success' : health === 'offline' ? 'bg-destructive' : 'bg-muted-foreground',
      )}
      title={health === 'online' ? 'Online' : health === 'offline' ? 'Offline' : 'Checking...'}
    />
  )
}
