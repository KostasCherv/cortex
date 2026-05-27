import { useEffect, useId, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Textarea } from '@/components/ui/textarea'
import type { RagAgent, RagAgentDraft, RagResource } from '@/types'

type NewAgentPayload = {
  name: string
  description: string
  system_instructions: string
  linked_resource_ids: string[]
}

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  agent?: RagAgent | null
  readyResources: RagResource[]
  onGenerateDraft: (prompt: string) => Promise<RagAgentDraft>
  onCreate: (payload: NewAgentPayload) => Promise<void>
  onUpdate?: (agentId: string, payload: NewAgentPayload) => Promise<void>
}

export function NewAgentSheet({
  open,
  onOpenChange,
  agent,
  readyResources,
  onGenerateDraft,
  onCreate,
  onUpdate,
}: Props) {
  const [planningPrompt, setPlanningPrompt] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [instructions, setInstructions] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stage, setStage] = useState<'brief' | 'review'>('brief')
  const [generating, setGenerating] = useState(false)
  const planningPromptId = useId()
  const nameId = useId()
  const descriptionId = useId()
  const instructionsId = useId()
  const isEditing = Boolean(agent)

  useEffect(() => {
    if (!open) return
    setPlanningPrompt('')
    setName(agent?.name ?? '')
    setDescription(agent?.description ?? '')
    setInstructions(agent?.system_instructions ?? '')
    setSelectedIds(agent?.linked_resource_ids ?? [])
    setStage(agent ? 'review' : 'brief')
    setGenerating(false)
    setError(null)
  }, [agent, open])

  const toggle = (id: string) =>
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))

  const handleSave = async () => {
    if (!name.trim()) return
    setSaving(true)
    setError(null)
    const payload = {
      name: name.trim(),
      description: description.trim(),
      system_instructions: instructions.trim(),
      linked_resource_ids: selectedIds,
    }
    try {
      if (agent) {
        if (!onUpdate) throw new Error('Missing update handler.')
        await onUpdate(agent.agent_id, payload)
      } else {
        await onCreate(payload)
      }
      onOpenChange(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${isEditing ? 'update' : 'create'} agent.`)
    } finally {
      setSaving(false)
    }
  }

  const handleGenerateDraft = async () => {
    if (!planningPrompt.trim()) return
    setGenerating(true)
    setError(null)
    try {
      const draft = await onGenerateDraft(planningPrompt.trim())
      setName(draft.name)
      setDescription(draft.description)
      setInstructions(draft.system_instructions)
      setStage('review')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate agent draft.')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="flex flex-col gap-0 p-0">
        <SheetHeader className="px-4 py-4 border-b">
          <SheetTitle>{isEditing ? 'Edit Agent' : stage === 'brief' ? 'Plan New Agent' : 'Review Agent Draft'}</SheetTitle>
          <SheetDescription>
            {isEditing
              ? 'Update the saved agent definition and linked resources.'
              : stage === 'brief'
                ? 'Describe the agent you want, then review the generated draft before creating it.'
                : 'Approve or edit the generated draft before the agent is created.'}
          </SheetDescription>
        </SheetHeader>
        <ScrollArea className="flex-1">
          <div className="px-4 py-4 space-y-4">
            {!isEditing && stage === 'brief' ? (
              <div className="space-y-1.5">
                <Label htmlFor={planningPromptId}>Planning brief</Label>
                <Textarea
                  id={planningPromptId}
                  placeholder="Describe the agent you want, the tasks it should handle, and any constraints."
                  rows={8}
                  value={planningPrompt}
                  onChange={(e) => setPlanningPrompt(e.target.value)}
                />
              </div>
            ) : (
              <>
                <div className="space-y-1.5">
                  <Label htmlFor={nameId}>Name</Label>
                  <Input
                    id={nameId}
                    placeholder="e.g. Research Assistant"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor={descriptionId}>Description</Label>
                  <Input
                    id={descriptionId}
                    placeholder="Brief description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor={instructionsId}>System instructions</Label>
                  <Textarea
                    id={instructionsId}
                    placeholder="How should this agent behave?"
                    rows={6}
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Resources</Label>
                  {readyResources.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No ready resources. Upload some first.</p>
                  ) : (
                    readyResources.map((r) => (
                      <div key={r.resource_id} className="flex items-center gap-2">
                        <Checkbox
                          id={r.resource_id}
                          checked={selectedIds.includes(r.resource_id)}
                          onCheckedChange={() => toggle(r.resource_id)}
                        />
                        <Label htmlFor={r.resource_id} className="font-normal cursor-pointer text-sm">
                          {r.filename}
                        </Label>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
            {error && (
              <p role="alert" className="rounded-md border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </p>
            )}
          </div>
        </ScrollArea>
        <SheetFooter className="px-4 py-4 border-t">
          {!isEditing && stage === 'review' && (
            <Button variant="outline" onClick={() => setStage('brief')} disabled={saving || generating}>
              Back to brief
            </Button>
          )}
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          {!isEditing && stage === 'brief' ? (
            <Button onClick={() => void handleGenerateDraft()} disabled={!planningPrompt.trim() || generating}>
              {generating && <Loader2 size={14} className="animate-spin" />}
              {generating ? 'Generating' : 'Generate draft'}
            </Button>
          ) : (
            <Button onClick={() => void handleSave()} disabled={!name.trim() || saving}>
              {saving && <Loader2 size={14} className="animate-spin" />}
              {saving ? (isEditing ? 'Saving' : 'Creating') : isEditing ? 'Save changes' : 'Create agent'}
            </Button>
          )}
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
