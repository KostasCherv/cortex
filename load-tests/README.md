# Local k6 Benchmarks

This folder is the local-first benchmark harness for Cortex.

Use it to:

- exercise the current API under repeatable load
- capture raw k6 summary output
- render a markdown report with latency, error rate, and bottleneck hints

Do not use local-only numbers as public production-capacity claims. Local runs are for bottleneck discovery and sizing the next production-like test.

## Prerequisites

Install [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) and start the backend locally.

```bash
docker compose up -d
uv run uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload
```

Optional backend flags that make benchmark analysis easier:

```bash
RAG_PERF_HEADERS=true
RAG_SUGGESTIONS_DEFERRED=true
```

## Shared environment

All scripts respect these variables:

- `BENCHMARK_BASE_URL` default: `http://127.0.0.1:8000`
- `BENCHMARK_RATE` requests per second for the constant-arrival-rate executor
- `BENCHMARK_DURATION` test duration such as `1m`
- `BENCHMARK_PREALLOCATED_VUS` initial VUs reserved by k6
- `BENCHMARK_MAX_VUS` upper VU limit for the scenario
- `BENCHMARK_P95_MS` latency threshold used by k6
- `BENCHMARK_MAX_ERROR_RATE` allowed `http_req_failed` threshold

## Scenarios

### 1. Health

No auth required. Good for measuring raw API responsiveness.

```bash
mkdir -p reports/benchmarks

k6 run \
  --summary-export reports/benchmarks/health-summary.json \
  load-tests/health.js

uv run python scripts/render_k6_report.py \
  --summary-json reports/benchmarks/health-summary.json \
  --scenario health \
  --environment local-dev \
  --target "20 req/s for 1 minute" \
  --notes "Local liveness baseline." \
  --output reports/benchmarks/health-report.md
```

### 2. Agent chat

Requires a real user token and an existing agent id.

```bash
export BENCHMARK_BEARER_TOKEN=...
export BENCHMARK_AGENT_ID=...
export BENCHMARK_CHAT_MESSAGE="Give me the top 3 grounded takeaways from my materials."

k6 run \
  --summary-export reports/benchmarks/agent-chat-summary.json \
  load-tests/agent_chat.js

uv run python scripts/render_k6_report.py \
  --summary-json reports/benchmarks/agent-chat-summary.json \
  --scenario agent-chat \
  --environment local-dev \
  --target "2 req/s for 1 minute" \
  --notes "Local authenticated chat load." \
  --output reports/benchmarks/agent-chat-report.md
```

Optional:

- `BENCHMARK_CHAT_SESSION_ID` to reuse an existing chat session

### 3. Research workflow

Requires a real user token and an existing session id. This endpoint queues research work and is best for measuring request admission behavior, error rate, and dispatch overhead locally.

```bash
export BENCHMARK_BEARER_TOKEN=...
export BENCHMARK_SESSION_ID=...
export BENCHMARK_RESEARCH_QUERY="Summarize the market outlook for this topic."

k6 run \
  --summary-export reports/benchmarks/research-summary.json \
  load-tests/research.js

uv run python scripts/render_k6_report.py \
  --summary-json reports/benchmarks/research-summary.json \
  --scenario research \
  --environment local-dev \
  --target "1 req/s for 1 minute" \
  --notes "Local research queue pressure." \
  --output reports/benchmarks/research-report.md
```

### 4. Internal agent-loop benchmark

This path is for local benchmarking when session persistence is unavailable or intentionally bypassed. It exercises the real `_run_agent_loop` path over HTTP behind the internal bearer secret.

Start the backend with an explicit internal secret:

```bash
INTERNAL_DISPATCH_SECRET=bench-secret \
INNGEST_DEV=1 \
INNGEST_SIGNING_KEY=local-benchmark-signing-key \
LLM_PROVIDER=openai \
uv run uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
```

Then run the benchmark:

```bash
export BENCHMARK_INTERNAL_SECRET=bench-secret
export BENCHMARK_BIND_TOOLS=false

k6 run \
  --summary-export reports/benchmarks/internal-agent-loop-summary.json \
  load-tests/internal_agent_loop.js

uv run python scripts/render_k6_report.py \
  --summary-json reports/benchmarks/internal-agent-loop-summary.json \
  --scenario internal-agent-loop \
  --environment local-dev \
  --target "1 req/s for 30 seconds" \
  --notes "Local HTTP benchmark for the internal agent loop." \
  --output reports/benchmarks/internal-agent-loop-report.md
```

Useful toggles:

- `BENCHMARK_BIND_TOOLS=true` to include Composio tool binding overhead
- `BENCHMARK_ALLOW_WEB_SEARCH=true` if you explicitly want web-enabled behavior

## Reading the report

The generated markdown includes:

- request count and rate
- avg, median, p95, p99, and max latency
- error rate
- a short bottleneck hint based on waiting time, blocked time, and failures

Cross-check the report with:

- FastAPI logs
- `docker stats`
- LangSmith and LangFuse traces
- Supabase, Neo4j, and Redis availability
- external provider limits such as OpenAI and Tavily
