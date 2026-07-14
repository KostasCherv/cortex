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

## Benchmark dashboards

For local benchmark investigation, start the opt-in observability stack:

```bash
docker compose -f docker-compose.observability.yml up -d
```

It provides:

- Prometheus at `http://localhost:9090`
- Grafana at `http://localhost:3000`

This stack is intended for benchmark analysis rather than the default development environment. The k6 scenarios and reporting workflow are documented in the [load-test guide](../load-tests/README.md).

