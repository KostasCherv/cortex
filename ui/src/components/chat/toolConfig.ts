import { Globe, Puzzle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export type ToolDefinition = {
  id: string
  label: string
  icon: LucideIcon
  defaultEnabled: boolean
}

export type ToolConfig = Record<string, boolean>

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  { id: 'web_search', label: 'Web search', icon: Globe, defaultEnabled: true },
  { id: 'composio', label: 'Connected apps', icon: Puzzle, defaultEnabled: false },
]

export function defaultToolConfig(): ToolConfig {
  return Object.fromEntries(TOOL_DEFINITIONS.map((t) => [t.id, t.defaultEnabled]))
}
