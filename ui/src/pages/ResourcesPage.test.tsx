import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ResourcesPage } from './ResourcesPage'

describe('ResourcesPage', () => {
  it('shows the signed-out state without loading resources', () => {
    const onResourcesChange = vi.fn()

    render(
      <ResourcesPage
        authSession={null}
        resources={[]}
        onResourcesChange={onResourcesChange}
        onResourceUploaded={vi.fn()}
      />,
    )

    expect(screen.getByText(/sign in to manage your resources/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /upload file/i })).not.toBeInTheDocument()
    expect(onResourcesChange).not.toHaveBeenCalled()
  })
})
