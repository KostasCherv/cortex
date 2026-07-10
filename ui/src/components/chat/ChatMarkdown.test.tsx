import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ChatMarkdown } from './ChatMarkdown'

describe('ChatMarkdown', () => {
  it('renders markdown through the streaming-aware renderer', () => {
    render(<ChatMarkdown content={'- First item\n- Second item'} />)

    const list = screen.getByRole('list')
    expect(list).toHaveAttribute('data-streamdown', 'unordered-list')
    expect(screen.getByText('First item')).toBeInTheDocument()
    expect(screen.getByText('Second item')).toBeInTheDocument()
  })
})
