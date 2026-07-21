import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import { LandingPage } from './LandingPage'

function renderLandingPage(onSignIn = vi.fn()) {
  return {
    onSignIn,
    ...render(
      <ThemeProvider>
        <LandingPage onSignIn={onSignIn} />
      </ThemeProvider>,
    ),
  }
}

describe('LandingPage', () => {
  it('renders the hero heading', () => {
    renderLandingPage()

    expect(
      screen.getByRole('heading', { name: /research platform/i, level: 1 }),
    ).toBeInTheDocument()
  })

  it('renders a Sign in with Google button and calls onSignIn when clicked', () => {
    const { onSignIn } = renderLandingPage()

    const buttons = screen.getAllByRole('button', { name: /sign in with google/i })
    expect(buttons.length).toBeGreaterThan(0)

    fireEvent.click(buttons[0])

    expect(onSignIn).toHaveBeenCalledTimes(1)
  })
})
