# Testing and quality

Cortex applies deterministic code, UI, browser, AI-contract, supply-chain, and performance checks at different points in the delivery lifecycle.

## Backend checks

```bash
uv run pytest -v
uv run pytest --cov=src --cov-fail-under=73 --cov-report=term-missing
uv run ruff check src
uv run mypy src
```

CI blocks lint, type, test, and coverage failures. Backend coverage must remain at or above the measured 73% baseline.

## Frontend checks

```bash
cd ui
npm run lint
npm run test:coverage
npx playwright install chromium   # first run only
npm run test:e2e
```

The UI coverage floors are 19% for statements, 19% for branches, 12% for functions, and 20% for lines. These are measured baselines, not final targets; raise them as the suite expands.

The Playwright smoke journey uses local fixtures, so it requires no Google, Supabase, or model credentials. It restores an authenticated session, creates a research session, consumes paced SSE progress/report events, and verifies the completed report. CI retains the HTML report, trace, screenshot, and video when the journey fails.

## AI regression gate

Every pull request runs a credential-free contract gate over 20 versioned cases in `src/evals/ai_regression_set.json`. The cases cover:

- Validated chat-router decisions
- RAG and tool citation provenance
- Progressive finance-tool selection and call planning

CI requires 100% overall and within every category, then uploads a commit-keyed JSON score artifact.

```bash
uv run python -m src.evals.regression_gate
```

Add a uniquely named deterministic case whenever one of these boundaries gains behavior or a production failure needs a permanent regression test. Use the model-backed workflows in [Model evaluation](model-evaluation.md) for semantic generation quality.

## Supply-chain security

CI builds the production image on every pull request and push to `main`, then:

- Scans repository dependencies and configuration with Trivy
- Scans the container image with Trivy
- Blocks HIGH or CRITICAL vulnerabilities with available fixes
- Blocks serious configuration and secret findings
- Produces a CycloneDX container SBOM
- Retains the SBOM artifact for 30 days

Unfixed vulnerabilities remain visible without blocking delivery so the gate stays actionable. Dependabot surfaces dependency updates when fixes become available.

## Load testing

The local-first k6 harness lives in [`load-tests/`](../load-tests/README.md). A typical run:

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
  --output reports/benchmarks/health-report.md
```

Additional scenarios:

- `load-tests/agent_chat.js` for authenticated agent-chat pressure
- `load-tests/research.js` for research queue-admission pressure

Local results are diagnostic measurements, not production-capacity claims. The consolidated production and agent-loop evidence is in the [load-test report](../reports/benchmarks/LOAD_TEST_REPORT.md).

