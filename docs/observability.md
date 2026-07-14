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

### Recommended alerting policy

The following is the documented operational policy for this deployment. Alert policies and uptime checks are created per project in Cloud Monitoring and are **not** provisioned by the deploy scripts — verify they exist in your project before relying on them.

Create in **Cloud Monitoring → Alerting → Create policy** (or `gcloud alpha monitoring policies create`):

1. **5xx ratio** — `request_count` filtered to `response_code_class = "5xx"` above 1% of total requests over a 5-minute window. Catches elevated errors regardless of cause.
2. **p95 latency** — `request_latencies` p95 above 5 s over 10 minutes for non-SSE routes. Long-lived `/research` SSE streams inflate raw latency percentiles, so scope the alert or set the threshold with streaming in mind.
3. **Instance crash / probe failure** — log-based alert on `resource.type="cloud_run_revision"` with `severity>=ERROR` matching container exit or readiness-probe failure messages. This is the signal that `/ready` is returning 503.
4. **Uptime check** — **Cloud Monitoring → Uptime checks** against `GET /health` on the service URL, one-minute cadence, with an attached alert on failure. This detects the service being fully down even when no traffic is arriving to generate error metrics.

Route notification channels (email, PagerDuty, Slack) per project in **Alerting → Notification channels**.

### How degradation is detected

Which signal fires for which failure class:

- **Critical dependency outage** (Supabase, Neo4j, LLM config) — `/ready` returns 503, Cloud Run marks the instance unready, and the probe-failure log alert fires. New revisions fail their startup probe and never receive traffic.
- **Elevated request errors** — the 5xx-ratio alert fires; drill into Cloud Logging for the failing route, and Sentry (when `SENTRY_DSN` is set) captures the exception with stack trace, error-capture only.
- **Latency regression** — the p95 alert fires; LangSmith traces show which graph node or external call slowed down.
- **LLM-provider failure** — surfaces as elevated 5xx plus errored runs in LangSmith/LangFuse traces; the provider's own status page confirms.
- **Ingestion backlog** — the `outbox-dispatcher` Inngest function runs every two minutes; a stall shows as failed or missing runs in the Inngest dashboard and as `document_outbox` rows stuck in `pending` in Supabase. There is no automatic alert for this today; check the Inngest dashboard when uploads stop completing.
- **Service fully down** — the `/health` uptime check fails within a minute even with zero user traffic.

### Error tracking

Setting `SENTRY_DSN` enables Sentry for unhandled-exception capture (error-capture only — tracing stays with LangSmith/LangFuse, PII and stack-trace locals are not sent). See [Production configuration](env-vars-production.md).

## Benchmark dashboards

For local benchmark investigation, start the opt-in observability stack:

```bash
docker compose -f docker-compose.observability.yml up -d
```

It provides:

- Prometheus at `http://localhost:9090`
- Grafana at `http://localhost:3000`

This stack is benchmark-only and is never deployed to production — Cloud Run's built-in metrics already cover the production signals (see [Production monitoring](#production-monitoring)). The k6 scenarios and reporting workflow are documented in the [load-test guide](../load-tests/README.md).

