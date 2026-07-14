import { createServer } from 'node:http'

const host = '127.0.0.1'
const port = 8010
const sessionId = 'session-e2e-001'
const runId = 'run-e2e-001'
const query = 'How does retrieval quality affect grounded AI answers?'
const completedReport = [
  '# Grounded AI systems',
  '',
  'Retrieval quality is a release criterion for grounded AI systems.',
  '',
  '## Evidence',
  '',
  '- Relevant context improves factual precision.',
  '- Evaluation catches regressions before release.',
].join('\n')

const corsHeaders = {
  'Access-Control-Allow-Origin': 'http://127.0.0.1:4173',
  'Access-Control-Allow-Headers': 'authorization, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
}

const json = (response, status, body) => {
  response.writeHead(status, { ...corsHeaders, 'Content-Type': 'application/json' })
  response.end(JSON.stringify(body))
}

const sessionDetail = () => ({
  session_id: sessionId,
  title: query,
  runs: [{
    run_id: runId,
    query,
    source_urls: ['https://example.com/evidence'],
    report: completedReport,
    partial_report: completedReport,
    status: 'completed',
    latest_node: '__end__',
    latest_event_at: '2026-07-12T10:00:04.000Z',
    feedback_submitted_at: null,
    feedback_helpful: null,
    created_at: '2026-07-12T10:00:00.000Z',
  }],
  conversation: [],
  created_at: '2026-07-12T10:00:00.000Z',
})

const server = createServer((request, response) => {
  const url = new URL(request.url ?? '/', `http://${host}:${port}`)

  if (request.method === 'OPTIONS') {
    response.writeHead(204, corsHeaders)
    response.end()
    return
  }
  if (url.pathname === '/health') return json(response, 200, { status: 'ok', version: 'e2e' })
  if (url.pathname === '/api/rag/agents') return json(response, 200, { agents: [] })
  if (url.pathname === '/api/rag/resources') return json(response, 200, { resources: [] })
  if (url.pathname === '/api/rag/chat/sessions') return json(response, 200, { sessions: [] })
  if (url.pathname === '/api/billing/usage') {
    return json(response, 200, {
      plan: 'free', date: '2026-07-12',
      limits: { research_queries_daily: 10, total_questions_daily: 30 },
      usage: { research_queries_count: 0, total_questions_count: 0 },
      resets_at: '2026-07-13T00:00:00.000Z', subscription: null,
    })
  }
  if (url.pathname === '/sessions' && request.method === 'GET') return json(response, 200, { sessions: [] })
  if (url.pathname === '/sessions' && request.method === 'POST') {
    return json(response, 200, { session_id: sessionId, title: query, created_at: '2026-07-12T10:00:00.000Z' })
  }
  if (url.pathname === `/sessions/${sessionId}/research` && request.method === 'POST') {
    return json(response, 202, { run_id: runId, status: 'running' })
  }
  if (url.pathname === `/sessions/${sessionId}` && request.method === 'GET') return json(response, 200, sessionDetail())
  if (url.pathname === `/sessions/${sessionId}/runs/${runId}/stream`) {
    response.writeHead(200, { ...corsHeaders, 'Cache-Control': 'no-cache', Connection: 'keep-alive', 'Content-Type': 'text/event-stream' })
    const events = [
      { delay: 80, data: { type: 'progress', node: 'search_and_memory_node', status: 'running', updated_at: '2026-07-12T10:00:01.000Z' } },
      { delay: 350, data: { type: 'progress', node: 'report_node', status: 'running', updated_at: '2026-07-12T10:00:02.000Z' } },
      { delay: 700, data: { type: 'report_chunk', text: '# Grounded AI systems\n\nEvidence is being synthesized' } },
      { delay: 1250, data: { type: 'done' } },
    ]
    for (const event of events) setTimeout(() => response.write(`data: ${JSON.stringify(event.data)}\n\n`), event.delay)
    setTimeout(() => response.end(), 1300)
    return
  }
  return json(response, 404, { detail: `No E2E fixture for ${request.method} ${url.pathname}` })
})

server.listen(port, host, () => process.stdout.write(`E2E API listening on http://${host}:${port}\n`))
const shutdown = () => server.close(() => process.exit(0))
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
