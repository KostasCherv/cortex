import { BookOpen, FileText, Globe, Library, Puzzle } from 'lucide-react'
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
  { id: 'wikipedia', label: 'Wikipedia', icon: BookOpen, defaultEnabled: true },
  { id: 'arxiv', label: 'arXiv', icon: FileText, defaultEnabled: false },
  { id: 'open_library', label: 'Open Library', icon: Library, defaultEnabled: false },
  { id: 'composio', label: 'Connected apps', icon: Puzzle, defaultEnabled: false },
]

export function defaultToolConfig(): ToolConfig {
  return Object.fromEntries(TOOL_DEFINITIONS.map((t) => [t.id, t.defaultEnabled]))
}
