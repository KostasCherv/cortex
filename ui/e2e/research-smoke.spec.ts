import { expect, test } from '@playwright/test'

const authSession = {
  access_token: 'e2e-access-token',
  refresh_token: 'e2e-refresh-token',
  expires_in: 3600,
  expires_at: Math.floor(Date.now() / 1000) + 3600,
  token_type: 'bearer',
  user: {
    id: '00000000-0000-4000-8000-000000000001',
    aud: 'authenticated',
    role: 'authenticated',
    email: 'engineer@example.com',
    app_metadata: { provider: 'google', providers: ['google'] },
    user_metadata: { full_name: 'E2E Engineer' },
    identities: [],
    created_at: '2026-01-01T00:00:00.000Z',
  },
}

test('signed-in user creates research and receives a streamed final report', async ({ page }) => {
  await page.addInitScript((session) => {
    window.localStorage.setItem('sb-e2e-auth-token', JSON.stringify(session))
  }, authSession)

  await page.goto('/')

  // This exercises the post-OAuth authenticated path without external credentials.
  await expect(page.getByLabel('Account menu', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Research', exact: true }).click()

  const query = 'How does retrieval quality affect grounded AI answers?'
  await page.getByPlaceholder(/Compare Model Context Protocol/).fill(query)
  const createSessionRequest = page.waitForRequest((request) =>
    request.method() === 'POST' && request.url().endsWith('/sessions'),
  )
  const startResearchRequest = page.waitForRequest((request) =>
    request.method() === 'POST' && request.url().endsWith('/sessions/session-e2e-001/research'),
  )
  await page.getByRole('button', { name: 'Run research' }).click()

  const [createRequest, researchRequest] = await Promise.all([
    createSessionRequest,
    startResearchRequest,
  ])
  expect(createRequest.headers().authorization).toBe('Bearer e2e-access-token')
  expect(createRequest.postDataJSON()).toEqual({ query })
  expect(researchRequest.postDataJSON()).toEqual({ query })

  await expect(page.getByText('Searching & Context')).toBeVisible()
  await expect(page.getByText('Drafting Final Report', { exact: true })).toBeVisible()
  await expect(page.getByText('Evidence is being synthesized')).toBeVisible()

  await expect(page.getByRole('heading', { name: 'Grounded AI systems' })).toBeVisible()
  await expect(page.getByText('Retrieval quality is a release criterion')).toBeVisible()
  await expect(page.getByText('Was this research result helpful?')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download' })).toBeEnabled()
})
