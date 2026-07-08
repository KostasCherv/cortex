# Cortex Prod-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take Cortex from "impressive side project" to promotable, deployable product: remove dead/demo code, add CI + LICENSE, slim dependencies, flatten over-layered modules, split the API god-file, and harden for production.

**Architecture:** No feature work. Six independent phases: (1) repo hygiene, (2) CI, (3) dependency diet, (4) billing flatten, (5) API router split, (6) production hardening + README polish. Phases 1–2 must go first (everything later relies on CI as the safety net); 3–6 are independent of each other. One branch + PR per phase.

**Tech Stack:** Python 3.12 / uv / FastAPI / pytest / ruff; React 19 / Vite / vitest / eslint; GitHub Actions; Docker (Cloud Run).

**Decisions already made by the owner (do not re-litigate):**
- License: **MIT**, copyright holder `Konstantinos Chervatidis`.
- Eval/fine-tune deps become **optional extras**.
- `yfinance` is imported by **nothing** in `src/` or `scripts/` — delete it outright.
- **Deviation from the original "extras" decision:** `composio-langchain`, `arxiv-mcp-server`, `mcp`, `langchain-mcp-adapters` stay **core** — they are imported at module level in `src/api/endpoints.py:35,76` and `src/api/rag_chat_helpers.py:44` (prod boot path). Making them optional would require lazy-import refactors for no size win. Flagged to owner; keep core unless told otherwise.

**Baseline (verify before starting):**
- `uv run python -m pytest tests/ -q` → all tests pass (owner fixed suite isolation in commit `f09d70f`). If any fail at baseline, record them; the bar for every task is "no new failures."
- `cd ui && npx vitest run` → all pass.
- Pre-existing, out-of-scope-until-Phase-2: 9 ruff errors, 2 tsc errors, 1 eslint error, 38 mypy errors (mypy stays non-blocking).

---

## Phase 1 — Repo hygiene

Branch: `chore/repo-hygiene`

### Task 1: Remove tracked dev-session artifacts

**Files:**
- Delete (git rm): see exact list in Step 1
- Modify: `.gitignore`

- [ ] **Step 1: Remove the artifacts**

```bash
git rm -r --quiet \
  test_results_artifact.txt \
  .deepeval/.deepeval_telemetry.txt \
  scripts/PIPELINE_FINDINGS.md \
  .archon \
  .claude/skills/archon \
  docs/superpowers/plans/2026-05-17-archon-develop-workflow.md \
  docs/superpowers/plans/2026-05-23-prompt-optimization-production.md \
  docs/superpowers/plans/2026-06-05-chat-tool-toggles.md \
  docs/superpowers/plans/2026-06-11-agent-chat-session-uploads.md \
  docs/superpowers/plans/2026-06-11-agent-chat-uploads-ui.md \
  docs/superpowers/plans/2026-06-11-session-uploads-review.md \
  docs/superpowers/plans/2026-06-12-agui-planner-chat.md \
  docs/superpowers/plans/2026-06-25-local-benchmark-harness.md \
  docs/superpowers/specs/2026-05-17-archon-develop-workflow-design.md \
  docs/superpowers/specs/2026-06-03-composio-agent-integration-design.md \
  docs/superpowers/specs/2026-06-05-chat-tool-toggles-design.md \
  docs/superpowers/specs/2026-06-11-agent-chat-session-uploads-design.md
```

Note: `docs/superpowers/plans/2026-07-08-prod-readiness.md` (this plan) stays until the final task. `.cursor/` and `.claude/skills/graphify` stay — they are the owner's active tooling. `docs/env-vars-production.md` stays — it is real docs.

- [ ] **Step 2: Append ignore rules to `.gitignore`**

```gitignore
# dev-session artifacts
.deepeval/
.archon/
test_results_artifact.txt
*.egg-info/
```

- [ ] **Step 3: Verify nothing else references the deleted files**

Run: `grep -rn "PIPELINE_FINDINGS\|test_results_artifact\|archon" src tests ui/src scripts --include='*.py' --include='*.ts' --include='*.tsx' --include='*.sh' | grep -v __pycache__`
Expected: no output (references in `.claude/skills/archon` itself are gone with it).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: remove dev-session artifacts from repo"
```

### Task 2: Delete dead `src/knowledge_graph/` package

Verified dead: no file in `src/` imports `src.knowledge_graph`; only `tests/test_knowledge_graph.py` does. The live GraphRAG path is `src/tools/neo4j_graph_store.py` (do NOT touch it).

**Files:**
- Delete: `src/knowledge_graph/` (entire package), `tests/test_knowledge_graph.py`

- [ ] **Step 1: Re-verify it is still dead** (code may have changed since the audit)

Run: `grep -rln "knowledge_graph" src --include='*.py' | grep -v __pycache__ | grep -v "^src/knowledge_graph"`
Expected: no output. If there IS output, STOP — do not delete; report to owner.

- [ ] **Step 2: Delete**

```bash
git rm -r --quiet src/knowledge_graph tests/test_knowledge_graph.py
```

- [ ] **Step 3: Full test suite**

Run: `uv run python -m pytest tests/ -q`
Expected: same pass count as baseline minus the deleted test file's tests; zero new failures.

- [ ] **Step 4: Commit**

```bash
git commit -am "chore: remove unused knowledge_graph package"
```

### Task 3: Remove empty dirs and stale build metadata

- [ ] **Step 1: Remove**

```bash
rmdir ralph tasks 2>/dev/null; rm -rf cortex.egg-info
```

(These are untracked; no git rm needed. `*.egg-info/` is ignored via Task 1 Step 2.)

- [ ] **Step 2: Commit if .gitignore changed, otherwise nothing to commit**

### Task 4: Add MIT LICENSE

**Files:**
- Create: `LICENSE`
- Modify: `README.md` (append section), `pyproject.toml` (license field)

- [ ] **Step 1: Write `LICENSE`** (full standard MIT text)

```text
MIT License

Copyright (c) 2026 Konstantinos Chervatidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Add to `pyproject.toml` under `[project]`**

```toml
license = { file = "LICENSE" }
```

- [ ] **Step 3: Append to `README.md`**

```markdown
## License

MIT — see [LICENSE](LICENSE).
```

- [ ] **Step 4: Commit, push, open PR for Phase 1**

```bash
git commit -am "chore: add MIT license"
git push -u origin chore/repo-hygiene
gh pr create --title "chore: repo hygiene (artifacts, dead code, LICENSE)" --body "Removes dev-session artifacts, dead knowledge_graph package, adds MIT LICENSE. No behavior change. Full suite green."
```

---

## Phase 2 — CI

Branch: `ci/github-actions` (branch from main after Phase 1 merges)

### Task 5: Fix the 9 pre-existing ruff errors

All are auto-fixable unused imports in `src/api/rag_chat_helpers.py`, `tests/finetune/test_generate_router_dataset.py`, `tests/test_dspy_optimizer.py`.

- [ ] **Step 1: Auto-fix**

Run: `uv run --extra dev ruff check --fix src tests`
Expected: `Found 9 errors (9 fixed, 0 remaining).` (count may drift slightly; end state must be `All checks passed!`)

- [ ] **Step 2: Full test suite** (unused-import removal can theoretically break `patch()` targets)

Run: `uv run python -m pytest tests/ -q`
Expected: zero new failures vs baseline.

- [ ] **Step 3: Commit**

```bash
git commit -am "chore: fix ruff unused-import errors"
```

### Task 6: Fix the 2 tsc errors and 1 eslint error in the UI

**Files:**
- Modify: `ui/src/components/chat/ChatThreadContainer.tsx:296`
- Modify: `ui/src/components/chat/transports.ts:161`

- [ ] **Step 1: tsc fix.** `crypto.randomUUID()` returns the UUID template-literal type; `upload.id` is `string`. Widen at the source. In `ChatThreadContainer.tsx` line 296 change:

```typescript
      const uploadIds = files.map(() => crypto.randomUUID())
```

to:

```typescript
      const uploadIds: string[] = files.map(() => crypto.randomUUID())
```

- [ ] **Step 2: eslint fix.** In `transports.ts` line 161, `_files` is an unused trailing parameter — TypeScript allows implementations with fewer params, so delete it:

```typescript
  streamMessage: async (message, sessionId, accessToken, callbacks, tools) => {
```

- [ ] **Step 3: Verify all three UI gates**

```bash
cd ui && npx tsc --noEmit -p tsconfig.app.json && npx eslint src --max-warnings=0 && npx vitest run
```

Expected: tsc silent, eslint clean, all vitest tests pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "fix: resolve pre-existing tsc and eslint errors in UI"
```

### Task 7: Add GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

All backend `Settings` fields have defaults (verified), so no secrets are needed — tests run fully mocked.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - name: Install
        run: uv sync --extra dev
      - name: Lint
        run: uv run ruff check src tests
      - name: Typecheck (non-blocking)
        run: uv run mypy src || true
      - name: Test
        run: uv run python -m pytest tests/ -q

  ui:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ui
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: ui/package-lock.json
      - name: Install
        run: npm ci
      - name: Lint
        run: npx eslint src --max-warnings=0
      - name: Typecheck
        run: npx tsc --noEmit -p tsconfig.app.json
      - name: Test
        run: npx vitest run

  docker:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    needs: [backend, ui]
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t cortex:${{ github.sha }} .
```

- [ ] **Step 2: Push branch, open PR, watch the run**

```bash
git add .github && git commit -m "ci: add GitHub Actions workflow (backend + ui + docker build)"
git push -u origin ci/github-actions
gh pr create --title "ci: GitHub Actions for backend, UI, and Docker build" --body "ruff + pytest, eslint + tsc + vitest, docker build on main. mypy non-blocking (38 pre-existing errors)."
gh pr checks --watch
```

Expected: backend + ui jobs green (docker job skips on PR). If a job fails on something that passes locally, fix forward on the branch — do not merge red.

---

## Phase 3 — Dependency diet

Branch: `chore/dependency-extras`

### Task 8: Delete yfinance, move eval/finetune deps to extras

Verified import sites: `dspy`/`pandas` only in `src/prompts/dspy_optimizer.py`, `src/evals/model_comparison.py`, `scripts/`; `datasets`/`huggingface-hub` only in `scripts/finetune/`; `deepeval` only in tests/scripts; `yfinance` nowhere. `dspy_optimizer` is only imported lazily inside functions in `scripts/optimize_prompts.py:67` and `scripts/prompt_optimization_pipeline.py:454` — no prod import path.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `[project]` dependencies** — remove these six lines (five move to extras, `yfinance` is deleted outright):

```toml
    "dspy[optuna]>=2.5.0",
    "deepeval>=3.0.0",
    "pandas>=2.2.0",
    "yfinance>=0.2.40",
    "datasets>=2.20",
    "huggingface-hub>=0.23",
```

(that's six strings; `yfinance` is deleted outright, the other five move)

- [ ] **Step 2: Add extras under `[project.optional-dependencies]`** (alongside the existing `dev`):

```toml
evals = [
    "dspy[optuna]>=2.5.0",
    "deepeval>=3.0.0",
    "pandas>=2.2.0",
]
finetune = [
    "datasets>=2.20",
    "huggingface-hub>=0.23",
]
```

- [ ] **Step 3: Re-lock and verify prod boot without extras**

```bash
uv sync                      # prod + no extras
uv run python -c "from src.api.endpoints import app; print(len(app.routes), 'routes OK')"
```

Expected: prints route count, no ImportError.

- [ ] **Step 4: Verify tests still run** (dev extra now needs evals for `tests/test_dspy_optimizer.py`; if it fails with ImportError, add `pytest.importorskip("dspy")` at the top of that test file OR run tests with `--extra evals`. Prefer updating CI: change the CI install line to `uv sync --extra dev --extra evals`.)

```bash
uv sync --extra dev --extra evals
uv run python -m pytest tests/ -q
```

Expected: zero new failures.

- [ ] **Step 5: Update `.github/workflows/ci.yml` install step** to `uv sync --extra dev --extra evals`, update README stack list (remove yfinance mention, note extras), commit, PR.

```bash
git commit -am "chore: move eval/finetune deps to extras, drop unused yfinance"
git push -u origin chore/dependency-extras && gh pr create --fill
```

---

## Phase 4 — Flatten billing package

Branch: `refactor/flatten-billing`

### Task 9: Collapse 4-layer billing into one module

Current: `src/billing/{domain/{models,policy,errors},application/{ports,service},infrastructure/{stripe_gateway,supabase_repositories},interfaces/http}` — 3 Protocols with exactly one implementation each. Target: single `src/billing.py`, identical public names so callers only change import paths.

**Files:**
- Create: `src/billing.py`
- Delete: `src/billing/` (whole package)
- Modify: `src/api/endpoints.py` (imports only), `tests/test_billing_service.py` (imports only)

- [ ] **Step 1: Inventory current consumers** (must match this list; if new consumers appeared, include them)

Run: `grep -rn "from src.billing" src tests --include='*.py' | grep -v __pycache__ | grep -v "^src/billing"`
Expected consumers: `src/api/endpoints.py` (3 imports: `BillingService`, `UsageIncrement`; `BillingSyncError`, `QuotaExceededError`; `build_billing_service`, `usage_summary_to_response`) and `tests/test_billing_service.py`.

- [ ] **Step 2: Create `src/billing.py` by concatenating, in this order, the bodies of:**
  1. `src/billing/domain/errors.py` (exception classes)
  2. `src/billing/domain/models.py` (dataclasses)
  3. `src/billing/domain/policy.py`
  4. `src/billing/application/ports.py` (the 3 Protocols — keep them; they type the service and are cheap)
  5. `src/billing/application/service.py` (`BillingService`, `UsageIncrement`)
  6. `src/billing/infrastructure/stripe_gateway.py` (`StripeHttpGateway`, `NoopStripeGateway`)
  7. `src/billing/infrastructure/supabase_repositories.py`
  8. `src/billing/interfaces/http.py` (`build_billing_service`, `usage_summary_to_response`)

  Mechanics: strip each file's `from src.billing.*` intra-package imports; merge the remaining stdlib/third-party imports into one deduplicated header; keep every public class/function name and signature byte-identical; keep module docstring `"""Billing: plans, quotas, Stripe subscriptions (flattened from src/billing/ package)."""`.

- [ ] **Step 3: Delete the package and rewrite consumer imports**

```bash
git rm -r --quiet src/billing
```

In `src/api/endpoints.py` replace the three import lines:

```python
from src.billing.application import BillingService, UsageIncrement
from src.billing.domain import BillingSyncError, QuotaExceededError
from src.billing.interfaces.http import build_billing_service, usage_summary_to_response
```

with:

```python
from src.billing import (
    BillingService,
    BillingSyncError,
    QuotaExceededError,
    UsageIncrement,
    build_billing_service,
    usage_summary_to_response,
)
```

In `tests/test_billing_service.py` (and `tests/test_api.py` if it imports `src.billing.*` submodules) rewrite every `from src.billing.<anything> import X` to `from src.billing import X`. Also update any `patch("src.billing.<layer>...")` mock targets to `patch("src.billing...")`.

- [ ] **Step 4: Verify**

```bash
uv run --extra dev ruff check src/billing.py
uv run python -m pytest tests/test_billing_service.py tests/test_api.py -q
uv run python -m pytest tests/ -q
```

Expected: ruff clean, billing tests pass unchanged, full suite zero new failures.

- [ ] **Step 5: Commit + PR**

```bash
git commit -am "refactor: flatten billing package into single module"
git push -u origin refactor/flatten-billing && gh pr create --fill
```

---

## Phase 5 — Split `src/api/endpoints.py`

Branch: `refactor/api-routers`

### Task 10: Extract routers by resource

Current: ~3,300 lines, 51 routes. Tags map cleanly: Meta 1, Sessions 9, RAG 32, Memory 3, Billing 4, Internal 2. RAG is further split by URL prefix.

**Files:**
- Create: `src/api/deps.py`, `src/api/routers/__init__.py`, `src/api/routers/sessions.py`, `src/api/routers/rag_resources.py`, `src/api/routers/rag_agents.py`, `src/api/routers/rag_chat.py`, `src/api/routers/memory.py`, `src/api/routers/billing.py`, `src/api/routers/internal.py`
- Modify: `src/api/endpoints.py` (shrinks to app factory: middleware, lifespan/startup, exception handlers, `/health`, `include_router` calls)

Rules for the move (pure code motion, zero logic change):
- Each router file starts with `router = APIRouter()`; decorators change `@app.<method>` → `@router.<method>`, keeping path, tags, and signature identical.
- Shared helpers move to `src/api/deps.py`: `get_authenticated_user` usage stays imported from `src.auth`; move `_get_billing_service`, `_raise_rag_validation_error`, `_parse_rag_chat_request`, request/response Pydantic models used by more than one router. Models used by exactly one router move into that router's file.
- Routing table (by path prefix):
  - `sessions.py`: everything under `/sessions` (9 routes)
  - `rag_resources.py`: `/api/rag/resources*` routes
  - `rag_agents.py`: `/api/rag/agents*` routes (including agent chat sessions)
  - `rag_chat.py`: `/api/rag/chat*` routes (workspace chat + stream + attachments)
  - `memory.py`: `/api/memory*` (3 routes)
  - `billing.py`: `/api/billing*` (4 routes)
  - `internal.py`: `/internal/*` (2 routes: dispatch-outbox, benchmark/agent-loop)
  - stays in `endpoints.py`: `/health` (Meta), Inngest serve wiring, startup/shutdown, exception handlers, CORS.
- **Route-count invariant is the acceptance test** (Step 1 records it, Step 3 asserts it).

- [ ] **Step 1: Record the invariant before touching anything**

```bash
uv run python -c "from src.api.endpoints import app; rs=sorted((r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r,'methods')); print(len(rs)); [print(p, m) for p,m in rs]" > /tmp/routes_before.txt
cat /tmp/routes_before.txt | head -3
```

- [ ] **Step 2: Do the extraction, one router file at a time, in this order (smallest → largest): `internal`, `billing`, `memory`, `sessions`, `rag_resources`, `rag_agents`, `rag_chat`.** After EACH file: `uv run python -c "from src.api.endpoints import app"` must import cleanly, then run that router's test file (`tests/test_api.py` covers most; `tests/test_billing_service.py`, `tests/test_benchmark_api.py` for billing/internal). Commit after each router:

```bash
git commit -am "refactor: extract <name> router"
```

- [ ] **Step 3: Assert the invariant**

```bash
uv run python -c "from src.api.endpoints import app; rs=sorted((r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r,'methods')); print(len(rs)); [print(p, m) for p,m in rs]" > /tmp/routes_after.txt
diff /tmp/routes_before.txt /tmp/routes_after.txt && echo "ROUTES IDENTICAL"
```

Expected: `ROUTES IDENTICAL`.

- [ ] **Step 4: Fix test mock targets.** `tests/test_api.py` patches `src.api.endpoints.<function>` heavily. Every moved function changes its patch target to its new module (e.g. `patch("src.api.routers.rag_chat.retrieve_context_for_query", ...)`). Run the suite and fix each `AttributeError: <module> does not have the attribute` mechanically:

```bash
uv run python -m pytest tests/ -q
```

Expected: zero new failures vs baseline.

- [ ] **Step 5: Final commit + PR**

```bash
git commit -am "refactor: split endpoints.py into resource routers"
git push -u origin refactor/api-routers && gh pr create --fill
```

---

## Phase 6 — Production hardening

Branch: `feat/prod-hardening`

### Task 11: Replace deprecated `@app.on_event` with lifespan

**Files:**
- Modify: `src/api/endpoints.py` (startup handler at ~line 183, shutdown at ~line 244; lines shift after Phase 5)

- [ ] **Step 1: Convert.** The existing startup function body (named `validate_session_store_configuration`) and shutdown body move into a lifespan context manager:

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _startup()      # rename existing startup handler body to _startup()
    yield
    await _shutdown()     # rename existing shutdown handler body to _shutdown()


app = FastAPI(..., lifespan=_lifespan)  # keep all existing FastAPI(...) kwargs
```

Delete both `@app.on_event(...)` decorators. Do NOT touch the unrelated `on_event` callback parameter used by the tool loop (same name, different thing — it appears in `_run_tool_loop` and stream handlers).

- [ ] **Step 2: Verify** — deprecation warnings gone:

Run: `uv run python -m pytest tests/test_api.py -q 2>&1 | grep -c "on_event is deprecated"`
Expected: `0`. Full suite green.

- [ ] **Step 3: Commit** `git commit -am "refactor: migrate to FastAPI lifespan handlers"`

### Task 12: Rate limiting on public endpoints

**Files:**
- Modify: `pyproject.toml` (add `"slowapi>=0.1.9"`), `src/api/endpoints.py`, `src/config.py`

- [ ] **Step 1: Add dependency** `slowapi>=0.1.9` to `[project]` dependencies, `uv sync --extra dev --extra evals`.

- [ ] **Step 2: Add config field** to `src/config.py` `Settings`:

```python
    rate_limit_default: str = Field(
        default="60/minute", description="Default per-IP rate limit for API endpoints"
    )
```

- [ ] **Step 3: Wire limiter in `src/api/endpoints.py`** (app assembly section):

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

- [ ] **Step 4: Exempt streaming + internal routes if the default is too tight** — only if tests/manual checks show SSE reconnects tripping it; use `@limiter.exempt` on the stream endpoints. Default: leave all limited.

- [ ] **Step 5: Test.** Add to `tests/test_api.py`:

```python
def test_rate_limit_returns_429_after_burst():
    from src.api.endpoints import app, limiter

    limiter.reset()
    with TestClient(app) as client:
        responses = [client.get("/health") for _ in range(70)]
    assert responses[0].status_code == 200
    assert any(r.status_code == 429 for r in responses)
```

Run: `uv run python -m pytest tests/test_api.py::test_rate_limit_returns_429_after_burst -v`
Expected: PASS. **Then run the FULL suite** — if the global limit 429s other tests (they hammer TestClient), set `settings.rate_limit_default = "1000/minute"` via env in `tests/conftest.py` (`os.environ.setdefault("RATE_LIMIT_DEFAULT", "1000/minute")` before settings import) and have the burst test override it explicitly.

- [ ] **Step 6: Commit** `git commit -am "feat: add per-IP rate limiting via slowapi"`

### Task 13: Sentry error tracking

**Files:**
- Modify: `pyproject.toml` (add `"sentry-sdk[fastapi]>=2.0"`), `src/api/endpoints.py`, `src/config.py`, `.env.example`, `docs/env-vars-production.md`

- [ ] **Step 1: Add dep + config field**

```python
    sentry_dsn: str = Field(default="", description="Sentry DSN; empty disables error tracking")
```

- [ ] **Step 2: Init at the very top of app assembly in `src/api/endpoints.py`** (before `app = FastAPI(...)`):

```python
import sentry_sdk

if settings.sentry_dsn:
    sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.0, send_default_pii=False)
```

`traces_sample_rate=0.0` on purpose — LangFuse/LangSmith own tracing; Sentry only gets exceptions.

- [ ] **Step 3: Document** — add `SENTRY_DSN=` to `.env.example` and a row to `docs/env-vars-production.md`.

- [ ] **Step 4: Verify no-DSN path (the default) boots clean**

Run: `uv run python -c "from src.api.endpoints import app; print('boot OK')"`
Expected: `boot OK`. Full suite green.

- [ ] **Step 5: Commit** `git commit -am "feat: optional Sentry error tracking"`

### Task 14: Dockerfile hardening

**Files:**
- Modify: `Dockerfile`

Current issues: installs from `pyproject.toml` only (not the lockfile — non-reproducible), runs as root, Python 3.11 while dev env is 3.12, no healthcheck.

- [ ] **Step 1: Replace `Dockerfile` with:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Locked, prod-only install (layer cached)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN useradd --create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8080"]
```

Note: `--no-install-project` + running from source needs the venv on PATH (handled above). If `uv sync` complains the project itself is required, drop `--no-install-project` and add `COPY src/ src/` before the sync instead.

- [ ] **Step 2: Build and smoke it**

```bash
docker build -t cortex:local .
docker run -d --rm -p 8080:8080 --name cortex-smoke cortex:local
sleep 5 && curl -sf http://localhost:8080/health && echo OK
docker exec cortex-smoke whoami   # expected: appuser
docker stop cortex-smoke
```

Expected: `/health` returns 200 (`OK` printed), user is `appuser`.

- [ ] **Step 3: Commit + PR for Phase 6**

```bash
git commit -am "chore: harden Dockerfile (lockfile install, non-root, healthcheck, py3.12)"
git push -u origin feat/prod-hardening && gh pr create --fill
```

---

## Phase 7 — Promotion polish

Branch: `docs/readme-polish`

### Task 15: README quickstart + measurement story

**Files:**
- Modify: `README.md`
- Delete (final step): `docs/superpowers/plans/2026-07-08-prod-readiness.md` (this plan; `docs/superpowers/` dir goes with it if empty)

- [ ] **Step 1: Add a "Quickstart" section near the top** (after "What it does"), content verified against `docker-compose.yml` (adjust service names to what the compose file actually defines — read it first):

```markdown
## Quickstart (local)

```bash
cp .env.example .env        # fill in SUPABASE_URL, SUPABASE_SECRET_KEY, and one LLM provider key
docker compose up -d        # API on :8080
cd ui && npm ci && npm run dev   # UI on :5173
```

Requires: Docker, Node 22+. Optional: Redis/Neo4j/Langfuse via `docker compose -f docker-compose.observability.yml up -d`.
```

- [ ] **Step 2: Add a "How we measure it" section** referencing the existing k6 + Grafana stack (`load-tests/`, `monitoring/`, `LANGFUSE.md`): what's measured (latency percentiles per endpoint, agent-loop benchmark), how to run it (`scripts/run_backends.sh`, k6 commands from `load-tests/README.md`), and one dashboard screenshot committed to `docs/images/`.

- [ ] **Step 3: Add an "Architecture decisions" section** — three short paragraphs: why the transactional outbox (exactly-once-ish ingestion without distributed transactions), why a fine-tuned Qwen2.5-3B router (latency + cost vs frontier-model routing, scored via `scripts/score_router.py`), why explicit success/empty/failure routing in the LangGraph research flow.

- [ ] **Step 4: Sweep README for staleness** — remove/adjust: `yfinance` in the stack list (deleted in Phase 3), anything referencing itinerary/PRD planner (feature removed), stack items now behind extras get an "(optional extra)" note.

- [ ] **Step 5: Delete this plan file, commit, PR**

```bash
git rm docs/superpowers/plans/2026-07-08-prod-readiness.md
git commit -am "docs: README quickstart, measurement story, architecture decisions"
git push -u origin docs/readme-polish && gh pr create --fill
```

---

## Definition of done (whole plan)

- [ ] `main` has: no dev artifacts, no dead packages, MIT LICENSE, green CI badge-worthy pipeline.
- [ ] `uv sync` (no extras) boots the API; extras cover evals/finetune tooling.
- [ ] `src/billing.py` single module; `src/api/endpoints.py` under ~600 lines with routers extracted; route set byte-identical to pre-refactor.
- [ ] No `on_event` deprecation warnings; rate limiting active; Sentry opt-in via env; Docker runs non-root from the lockfile.
- [ ] README sells the outbox, the fine-tuned router, and the observability story, and a stranger can run it in under 5 minutes.
- [ ] After each phase: full `pytest` + `vitest` + `tsc` + `eslint` green (or matching recorded baseline), `graphify update .` run once per merged phase.
