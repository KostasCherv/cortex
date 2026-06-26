# Local Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-first benchmark harness for Cortex that can exercise health, agent chat, and research endpoints with k6 and turn raw run output into a shareable markdown report with bottleneck hints.

**Architecture:** Keep load generation in a new `load-tests/` folder with thin k6 scripts and shared config helpers. Put report parsing and markdown rendering in a small Python module under `src/benchmarking/` so we can cover the non-trivial logic with pytest and reuse it from a CLI script.

**Tech Stack:** k6, Python 3.11+, pytest, FastAPI endpoint contracts already in the repo

---

### Task 1: Add tested report parsing and rendering core

**Files:**
- Create: `src/benchmarking/__init__.py`
- Create: `src/benchmarking/k6_report.py`
- Create: `tests/test_k6_report.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.benchmarking.k6_report import build_report, render_markdown


def test_build_report_extracts_core_http_metrics():
    summary = {
        "root_group": {"name": "", "checks": []},
        "metrics": {
            "http_reqs": {"type": "counter", "values": {"count": 1200, "rate": 20}},
            "http_req_failed": {"type": "rate", "values": {"rate": 0.005}},
            "http_req_duration": {
                "type": "trend",
                "values": {"avg": 820.4, "med": 700.0, "p(95)": 1450.0, "p(99)": 2100.0, "max": 3000.0},
            },
            "http_req_waiting": {"type": "trend", "values": {"avg": 790.0, "p(95)": 1400.0}},
            "http_req_blocked": {"type": "trend", "values": {"avg": 18.0, "p(95)": 55.0}},
            "iterations": {"type": "counter", "values": {"count": 1200, "rate": 20}},
            "vus": {"type": "gauge", "values": {"value": 12, "min": 0, "max": 12}},
        },
        "state": {"testRunDurationMs": 60000},
    }

    report = build_report(summary, scenario_name="agent-chat")

    assert report.scenario_name == "agent-chat"
    assert report.total_requests == 1200
    assert report.request_rate == 20
    assert report.error_rate == 0.005
    assert report.p95_ms == 1450.0
    assert "Application processing time dominates" in report.bottleneck_hints


def test_render_markdown_includes_claim_ready_sections():
    summary = {
        "root_group": {"name": "", "checks": []},
        "metrics": {
            "http_reqs": {"type": "counter", "values": {"count": 300, "rate": 5}},
            "http_req_failed": {"type": "rate", "values": {"rate": 0.0}},
            "http_req_duration": {
                "type": "trend",
                "values": {"avg": 400.0, "med": 350.0, "p(95)": 650.0, "p(99)": 900.0, "max": 1100.0},
            },
            "http_req_waiting": {"type": "trend", "values": {"avg": 300.0, "p(95)": 500.0}},
            "http_req_blocked": {"type": "trend", "values": {"avg": 60.0, "p(95)": 120.0}},
            "vus": {"type": "gauge", "values": {"value": 6, "min": 0, "max": 6}},
        },
        "state": {"testRunDurationMs": 60000},
    }

    report = build_report(summary, scenario_name="health")
    markdown = render_markdown(
        report,
        environment_label="local-dev",
        target_description="5 req/s for 1 minute",
        notes="Local-only bottleneck discovery run.",
    )

    assert "# k6 Benchmark Report" in markdown
    assert "Environment" in markdown
    assert "local-dev" in markdown
    assert "5 req/s for 1 minute" in markdown
    assert "Local-only bottleneck discovery run." in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_k6_report.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `src.benchmarking.k6_report`

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass


@dataclass
class K6Report:
    scenario_name: str
    total_requests: int
    request_rate: float | None
    error_rate: float | None
    p95_ms: float | None
    bottleneck_hints: list[str]


def build_report(summary: dict, *, scenario_name: str) -> K6Report:
    ...


def render_markdown(
    report: K6Report,
    *,
    environment_label: str,
    target_description: str,
    notes: str | None = None,
) -> str:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_k6_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmarking/__init__.py src/benchmarking/k6_report.py tests/test_k6_report.py
git commit -m "feat: add k6 report renderer"
```

### Task 2: Add a CLI to turn k6 summary JSON into markdown

**Files:**
- Create: `scripts/render_k6_report.py`
- Test: `tests/test_k6_report.py`

- [ ] **Step 1: Write the failing CLI-focused test**

```python
from pathlib import Path
from subprocess import run


def test_render_k6_report_cli_writes_markdown(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "report.md"
    summary_path.write_text('{"root_group":{"name":"","checks":[]},"metrics":{"http_reqs":{"type":"counter","values":{"count":10,"rate":1}},"http_req_failed":{"type":"rate","values":{"rate":0.0}},"http_req_duration":{"type":"trend","values":{"avg":100.0,"med":90.0,"p(95)":150.0,"p(99)":180.0,"max":200.0}}},"state":{"testRunDurationMs":10000}}')

    result = run(
        [
            "uv",
            "run",
            "python",
            "scripts/render_k6_report.py",
            "--summary-json",
            str(summary_path),
            "--scenario",
            "health",
            "--environment",
            "local-dev",
            "--target",
            "1 req/s",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert "health" in output_path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_k6_report.py -q`
Expected: FAIL because `scripts/render_k6_report.py` does not exist yet

- [ ] **Step 3: Write the minimal implementation**

```python
import argparse
import json
from pathlib import Path

from src.benchmarking.k6_report import build_report, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_k6_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/render_k6_report.py tests/test_k6_report.py
git commit -m "feat: add k6 report cli"
```

### Task 3: Add the local k6 load harness

**Files:**
- Create: `load-tests/lib.js`
- Create: `load-tests/health.js`
- Create: `load-tests/agent_chat.js`
- Create: `load-tests/research.js`

- [ ] **Step 1: Write a thin validation check for script syntax**

```bash
node --check load-tests/lib.js
node --check load-tests/health.js
node --check load-tests/agent_chat.js
node --check load-tests/research.js
```

- [ ] **Step 2: Create minimal scripts with shared env parsing and thresholds**

```javascript
export function requiredEnv(name) {
  const value = __ENV[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}
```

- [ ] **Step 3: Add scenario-specific requests**

```javascript
http.get(`${baseUrl}/health`);
http.post(`${baseUrl}/api/rag/agents/${agentId}/chat`, payload, { headers });
http.post(`${baseUrl}/sessions/${sessionId}/research`, payload, { headers });
```

- [ ] **Step 4: Run syntax checks**

Run:
`node --check load-tests/lib.js`
`node --check load-tests/health.js`
`node --check load-tests/agent_chat.js`
`node --check load-tests/research.js`

Expected: no output, exit 0

- [ ] **Step 5: Commit**

```bash
git add load-tests/lib.js load-tests/health.js load-tests/agent_chat.js load-tests/research.js
git commit -m "feat: add local k6 benchmark scripts"
```

### Task 4: Document local benchmark usage and reporting flow

**Files:**
- Create: `load-tests/README.md`
- Modify: `README.md`

- [ ] **Step 1: Add focused docs for local-only runs**

```markdown
## Local benchmarking

This harness is for local bottleneck discovery first. Do not use local-only numbers as public production-capacity claims.
```

- [ ] **Step 2: Document k6 commands and report rendering**

```bash
k6 run --summary-export reports/benchmarks/health-summary.json load-tests/health.js
uv run python scripts/render_k6_report.py --summary-json reports/benchmarks/health-summary.json ...
```

- [ ] **Step 3: Verify docs reference real files and env vars**

Run: `rg -n "load-tests|render_k6_report|BENCHMARK_" README.md load-tests/README.md`
Expected: matches for each command and variable documented

- [ ] **Step 4: Commit**

```bash
git add README.md load-tests/README.md
git commit -m "docs: add local benchmark workflow"
```
