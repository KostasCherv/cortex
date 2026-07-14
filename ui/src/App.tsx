import { Analytics } from '@vercel/analytics/react'
import { AppShell } from './components/shell/AppShell'

function App() {
  return (
    <>
      <AppShell />
      <Analytics />
    </>
  )
}

export default App
