# Observability

Cortex traces research execution and generation behavior, exposes health signals for the runtime platform, and provides an opt-in metrics stack for benchmark analysis.

## LangSmith

LangSmith captures traces across LangGraph nodes and external dependencies. Production configuration:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=cortex
LANGSMITH_REDACTION_MODE=redacted_default
LANGSMITH_SAMPLING_RATE=1.0
```

`LANGSMITH_API_KEY` is stored as a production secret. Redaction mode can replace input and output values while preserving their shape, allowing operators to debug control flow without treating the literal `"[REDACTED]"` marker as application data.

Repository agents have a standard workflow for investigating a LangSmith run ID, correlating child spans with graph nodes, and reporting the failing source location. See [Agent tooling](agent-tooling.md).

## LangFuse

LangFuse provides generation-level telemetry, user scoring, and evaluation datasets. Its integration and data model are documented in [LANGFUSE.md](../LANGFUSE.md).

## Runtime health

The backend separates process liveness from dependency readiness:

- `GET /health` proves that the API process can answer.
- `GET /ready` checks the configured LLM plus Supabase, Neo4j, and optional Redis.
- Critical dependency failures return HTTP 503.
- Optional dependency failures report a degraded status without taking the service out of rotation.

See [Production deployment](deployment.md#health-and-readiness) for probe behavior.

## Production monitoring

Cloud Run is the metrics platform for the deployed backend. The service does not export Prometheus metrics in production; the platform already records the request-level signals an operator needs, without running a metrics stack alongside the application.

### Cloud Run built-in metrics

Available under **Cloud Run → cortex → Metrics** in the Google Cloud console, or in Metrics Explorer under the `run.googleapis.com` namespace:

| Metric | What it tells you |
|---|---|
| Request count (by response code class) | Traffic volume and 4xx/5xx rates |
| Request latency (p50/p95/p99) | End-to-end latency including SSE streams |
| Instance count (active/idle) | Scaling behavior against the `maxScale: 3` cap |
| Startup latency | Cold-start cost; the startup probe allows up to five minutes |
| Container CPU/memory utilization | Headroom against the 1 CPU / 1 Gi limits |

Probe outcomes appear in Cloud Logging (`resource.type="cloud_run_revision"`); a failing readiness probe logs the `/ready` 503 body, which names the failing dependency.

### Production alerting baseline

Alert policies and the uptime check are provisioned per project by the idempotent setup script. The notification email must be verified before relying on delivery:

```bash
GCP_PROJECT=<project-id> \
ALERT_EMAIL=<operator-email> \
./scripts/setup_alerting.sh --dry-run

GCP_PROJECT=<project-id> \
ALERT_EMAIL=<operator-email> \
./scripts/setup_alerting.sh
```

The script creates or updates:

1. **Backend uptime** — checks `GET /health` each minute from the USA, Europe, and Asia-Pacific. Two failing regions for one minute opens an incident.
2. **5xx burst** — `request_count` filtered to `response_code_class = "5xx"`, aggregated across revisions. Three errors within five minutes opens an incident.
3. **Instance crash / probe failure** — log-match alert for container termination and startup, liveness, or readiness-probe failures, with a 15-minute notification rate limit.

Policies auto-close after 30 healthy minutes. The email channel is named `cortex-ops-email`; test it in **Cloud Monitoring → Alerting → Notification channels** after setup.

### How degradation is detected

Which signal fires for which failure class:

- **Critical dependency outage** (Supabase, Neo4j, LLM config) — `/ready` returns 503, Cloud Run marks the instance unready, and the probe-failure log alert fires. New revisions fail their startup probe and never receive traffic.
- **Elevated request errors** — the 5xx-ratio alert fires; drill into Cloud Logging for the failing route, and Sentry (when `SENTRY_DSN` is set) captures the exception with stack trace, error-capture only.
- **Latency regression** — investigate Cloud Run latency metrics manually; long-lived SSE routes make a single global p95 alarm noisy.
- **LLM-provider failure** — surfaces as elevated 5xx plus errored runs in LangSmith/LangFuse traces; the provider's own status page confirms.
- **Ingestion backlog** — the `outbox-dispatcher` Inngest function runs every two minutes. Configure an Inngest email alert for terminal failures and no successful run for six minutes. A zero-dispatch run is successful, so idle traffic does not alert. Final outbox delivery failures are also sent to Sentry.
- **Service fully down** — the `/health` uptime check fails within a minute even with zero user traffic.

### Error tracking

Setting `SENTRY_DSN` locally, or `sentry_dsn` inside the production `PROVIDER_CONFIG_JSON` secret, enables Sentry for unhandled and selected handled-exception capture (error-capture only — tracing stays with LangSmith/LangFuse, PII and stack-trace locals are not sent). Production uses the Cloud Run revision as the Sentry release. Configure one Sentry issue alert for a new or regressed issue in `production`, delivered by email with a 30-minute per-issue cooldown. See [Production configuration](env-vars-production.md).

Handled-exception metadata is allowlisted to operational identifiers such as run, session, job, and outbox event IDs. Prompts, chat messages, documents, tokens, and authorization values are never attached.

### Inngest alert activation

In the Inngest production environment, create email alerts for:

- terminal failure in `rag-ingestion`, `research-run`, `user-memory-refresh`, or `outbox-dispatcher`;
- no successful `outbox-dispatcher` run for six minutes.

Test the rule in a non-production Inngest environment before enabling production delivery.

### Cost guardrails

The baseline uses built-in Cloud Run metrics, one standard uptime check, one metric policy, one log-match policy, the Sentry Developer plan, and Inngest's existing alerting. At current low traffic it should remain within free allowances except for additional Secret Manager versions and future GCP metric-alert evaluation charges. Do not replace the uptime check with a synthetic monitor or deploy the benchmark Prometheus/Grafana stack to production.

## Benchmark dashboards

For local benchmark investigation, start the opt-in observability stack:

```bash
docker compose -f docker-compose.observability.yml up -d
```

It provides:

- Prometheus at `http://localhost:9090`
- Grafana at `http://localhost:3000`

This stack is benchmark-only and is never deployed to production — Cloud Run's built-in metrics already cover the production signals (see [Production monitoring](#production-monitoring)). The k6 scenarios and reporting workflow are documented in the [load-test guide](../load-tests/README.md).
