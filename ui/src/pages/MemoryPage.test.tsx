import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryPage } from './MemoryPage'

const {
  deleteUserMemoryMock,
  getUserMemoryMock,
  updateUserMemoryMock,
} = vi.hoisted(() => ({
  deleteUserMemoryMock: vi.fn(),
  getUserMemoryMock: vi.fn(),
  updateUserMemoryMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  deleteUserMemory: deleteUserMemoryMock,
  getUserMemory: getUserMemoryMock,
  updateUserMemory: updateUserMemoryMock,
}))

describe('MemoryPage', () => {
  beforeEach(() => {
    getUserMemoryMock.mockReset()
    updateUserMemoryMock.mockReset()
    deleteUserMemoryMock.mockReset()

    getUserMemoryMock.mockResolvedValue({
      content: 'Prefers concise answers.',
      updated_at: '2026-06-04T10:00:00Z',
      last_refreshed_at: '2026-06-04T10:05:00Z',
    })
    updateUserMemoryMock.mockResolvedValue({
      content: 'Works in fintech.',
      updated_at: '2026-06-04T11:00:00Z',
      last_refreshed_at: '2026-06-04T10:05:00Z',
    })
    deleteUserMemoryMock.mockResolvedValue({ deleted: true })
  })

  it('shows sign-in state when the user is not authenticated', () => {
    render(<MemoryPage authSession={null} />)

    expect(screen.getByText(/sign in to manage your memory/i)).toBeInTheDocument()
  })

  it('loads, saves, and deletes the single memory document', async () => {
    render(<MemoryPage authSession={{ access_token: 'token' } as never} />)

    await waitFor(() => expect(getUserMemoryMock).toHaveBeenCalledWith('token'))
    expect(screen.getByDisplayValue('Prefers concise answers.')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/memory content/i), {
      target: { value: 'Works in fintech.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^save$/i }))

    await waitFor(() => expect(updateUserMemoryMock).toHaveBeenCalledWith('Works in fintech.', 'token'))
    expect(screen.getByDisplayValue('Works in fintech.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /delete memory/i }))

    await waitFor(() => expect(deleteUserMemoryMock).toHaveBeenCalledWith('token'))
    await waitFor(() => expect(screen.getByDisplayValue('')).toBeInTheDocument())
  })
})
