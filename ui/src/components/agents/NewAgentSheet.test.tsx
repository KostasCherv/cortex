import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { NewAgentSheet } from './NewAgentSheet'

describe('NewAgentSheet', () => {
  it('requires draft generation before agent creation in create mode', async () => {
    const onGenerateDraft = vi.fn().mockResolvedValue({
      name: 'Policy Analyst',
      description: 'Compares policy documents.',
      system_instructions: 'Answer from the linked policy resources first.',
    })
    const onCreate = vi.fn().mockResolvedValue(undefined)

    render(
      <NewAgentSheet
        open
        onOpenChange={() => {}}
        readyResources={[]}
        onGenerateDraft={onGenerateDraft}
        onCreate={onCreate}
      />,
    )

    expect(screen.getByLabelText(/planning brief/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /generate draft/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /create agent/i })).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/planning brief/i), {
      target: { value: 'Create an agent that compares policy documents and explains the differences.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /generate draft/i }))

    await waitFor(() =>
      expect(onGenerateDraft).toHaveBeenCalledWith(
        'Create an agent that compares policy documents and explains the differences.',
      ),
    )

    expect(screen.getByDisplayValue('Policy Analyst')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Compares policy documents.')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Answer from the linked policy resources first.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /create agent/i }))

    await waitFor(() =>
      expect(onCreate).toHaveBeenCalledWith({
        name: 'Policy Analyst',
        description: 'Compares policy documents.',
        system_instructions: 'Answer from the linked policy resources first.',
        linked_resource_ids: [],
      }),
    )
  })
})
