import { Plus } from 'lucide-react'
import { useState, type MouseEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import { TOOL_DEFINITIONS, defaultToolConfig } from './toolConfig'
import type { ToolConfig } from './toolConfig'

type Props = {
  toolConfig: ToolConfig
  onToggle: (id: string, enabled: boolean) => void
  disabled?: boolean
}

export function ToolMenuButton({ toolConfig, onToggle, disabled }: Props) {
  const [open, setOpen] = useState(false)
  const defaults = defaultToolConfig()
  const hasNonDefault = TOOL_DEFINITIONS.some((t) => toolConfig[t.id] !== defaults[t.id])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className={cn('relative shrink-0 h-9 w-9', disabled && 'opacity-50 pointer-events-none')}
          aria-label="Toggle tools"
          disabled={disabled}
        >
          <Plus size={16} />
          {hasNonDefault && (
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        className="w-56 p-1"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <div className="flex flex-col gap-0.5">
          {TOOL_DEFINITIONS.map((tool) => {
            const Icon = tool.icon
            const enabled = toolConfig[tool.id] ?? tool.defaultEnabled
            return (
              <button
                key={tool.id}
                type="button"
                className="flex items-center justify-between gap-3 rounded-md px-2 py-2 text-sm hover:bg-muted transition-colors"
                onClick={() => onToggle(tool.id, !enabled)}
              >
                <span className="flex items-center gap-2 text-foreground">
                  <Icon size={15} className="shrink-0 text-muted-foreground" />
                  {tool.label}
                </span>
                <Switch
                  checked={enabled}
                  onCheckedChange={(v: boolean) => onToggle(tool.id, v)}
                  onClick={(e: MouseEvent) => e.stopPropagation()}
                  aria-label={`Toggle ${tool.label}`}
                />
              </button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
