# Prompt Optimization Productionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DSPy prompt optimization workflow trustworthy, repeatable, and safe enough to serve as the production prompt-improvement pipeline for the research system.

**Architecture:** Keep the existing Jinja2 runtime path as the default serving path while hardening DSPy as an offline optimization and promotion pipeline first. Fix correctness bugs in the current CLI and batch pipeline, add prompt-family-specific datasets and metrics, then add artifact governance and an optional runtime adoption path behind a feature flag.

**Tech Stack:** Python 3.11, DSPy, DeepEval, pytest, Jinja2 prompt registry, existing research graph/API runtime.

---

## File Map

| File | Action | Why |
|---|---|---|
| `scripts/optimize_prompts.py` | Modify | Fix module coverage and metric-aware compare behavior |
| `scripts/prompt_optimization_pipeline.py` | Modify | Add real template filtering, prompt-family datasets, guardrails, and artifact reporting |
| `src/prompts/dspy_optimizer.py` | Modify | Make comparison and example-building APIs metric-aware and reusable |
| `src/evals/golden_set.json` | Leave as compatibility fixture | Preserve current minimal cross-prompt set while migrating |
| `src/evals/prompt_optimization/` | Create | Store prompt-family-specific optimization datasets |
| `tests/test_dspy_optimizer.py` | Modify | Cover metric selection and compare behavior |
| `tests/test_prompt_optimization_pipeline.py` | Create | Cover pipeline filtering, dataset mapping, checkpoints, and report generation |
| `README.md` | Modify | Document the production prompt optimization workflow and rollout constraints |
| `docs/superpowers/plans/2026-05-23-prompt-optimization-production.md` | Create | Track the implementation plan |
| `src/prompts/registry.py` | Optional modify | Only if the team chooses runtime loading of optimized artifacts |
| `src/graph/nodes.py` | Optional modify | Only if summarize/report runtime adoption is enabled behind a flag |
| `src/api/endpoints.py` | Optional modify | Only if router/repair runtime adoption is enabled behind a flag |

---

## Recommended Rollout Decision

Use a two-stage productionization strategy:

1. **Stage 1: Offline production line only**
   Harden optimization, metrics, datasets, reports, and promotion rules. Continue serving the existing `.j2` prompts in production.

2. **Stage 2: Selective runtime adoption behind a feature flag**
   After offline scores and integration tests are trustworthy, allow one prompt family at a time to load an optimized artifact in non-default environments.

This avoids coupling current pipeline bugs to live request handling.

---

### Task 1: Fix the correctness bugs in the current optimizer interfaces

The current implementation has two issues that make results misleading even before metric quality is addressed:

- `scripts/optimize_prompts.py` passes `routing_accuracy_metric` into `optimize()` for `web_search_decision`, but `compare()` still uses `self.metric`, which defaults to `default_overlap_metric`.
- `scripts/prompt_optimization_pipeline.py --template summarize` still processes every template, then filters the results afterward, which wastes budget and can overwrite unrelated optimized artifacts.

**Files:**
- Modify: `scripts/optimize_prompts.py`
- Modify: `src/prompts/dspy_optimizer.py`
- Modify: `scripts/prompt_optimization_pipeline.py`
- Modify: `tests/test_dspy_optimizer.py`
- Create: `tests/test_prompt_optimization_pipeline.py`

- [ ] **Step 1: Make `compare()` metric-aware**

Add an optional `metric: Callable | None = None` parameter to `DspyPromptOptimizer.compare()` and use `active_metric = metric or self.metric` consistently when scoring original and optimized predictions.

Expected code shape:

```python
def compare(
    self,
    original_module: dspy.Module,
    optimized_path: str | Path,
    golden_set: list[dict],
    module_type: str = "summarize",
    *,
    metric: Callable | None = None,
) -> list[dict[str, Any]]:
    active_metric = metric or self.metric
    ...
    results.append(
        {
            "query": case["query"],
            "original_score": active_metric(example, original_pred),
            "optimized_score": active_metric(example, optimized_pred),
            "expected": case.get("expected_answer", ""),
        }
    )
```

- [ ] **Step 2: Pass module-specific metrics into CLI comparison**

In `scripts/optimize_prompts.py`, reuse `MODULE_METRICS.get(args.module)` for both `optimize()` and `compare()`.

Expected code shape:

```python
metric = MODULE_METRICS.get(args.module)
result = optimizer.optimize(..., metric=metric, ...)
...
comparisons = optimizer.compare(module, saved, golden_set, args.module, metric=metric)
```

- [ ] **Step 3: Make `--template` a real pipeline filter**

Thread a `template_name: str | None = None` argument into `run_pipeline()`. Filter `template_paths` before optimization rather than filtering `results` after the fact.

Expected code shape:

```python
def run_pipeline(..., template_name: str | None = None) -> list[OptimizationRun]:
    template_paths = sorted(PROMPTS_DIR.glob("*.j2"))
    if template_name:
        template_paths = [p for p in template_paths if p.stem == template_name]
```

- [ ] **Step 4: Add tests for both bugs**

Add tests that verify:

- `compare(metric=...)` uses the supplied metric instead of `self.metric`
- `--template summarize` only evaluates or optimizes `summarize.j2`
- no unrelated `*_optimized.json` files are written during a filtered run

Run:

```bash
uv run pytest tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/optimize_prompts.py scripts/prompt_optimization_pipeline.py src/prompts/dspy_optimizer.py tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py
git commit -m "fix: make prompt optimization metrics and template filtering correct"
```

---

### Task 2: Replace generic example mapping with prompt-family datasets

The current golden set only contains `query`, `retrieved_contents`, and `expected_answer`, but several prompts require additional fields such as `history_block`, `report_block`, `web_results_json`, `rag_is_insufficient`, and router labels. Production optimization cannot rely on placeholder values for these inputs.

**Files:**
- Create: `src/evals/prompt_optimization/README.md`
- Create: `src/evals/prompt_optimization/summarize.json`
- Create: `src/evals/prompt_optimization/report.json`
- Create: `src/evals/prompt_optimization/rag_chat_system.json`
- Create: `src/evals/prompt_optimization/followup_answer.json`
- Create: `src/evals/prompt_optimization/web_search_decision.json`
- Create: `src/evals/prompt_optimization/web_search_repair.json`
- Modify: `scripts/prompt_optimization_pipeline.py`
- Modify: `src/prompts/dspy_optimizer.py`
- Create: `tests/test_prompt_optimization_pipeline.py`

- [ ] **Step 1: Define per-template dataset schema**

Document each file’s required fields in `src/evals/prompt_optimization/README.md`.

Minimum schema:

```json
{
  "case_id": "router-current-info-001",
  "inputs": {
    "history_block": "...",
    "rag_context": "...",
    "rag_is_insufficient": "true",
    "message_urls": "None",
    "history_urls": "None",
    "message": "What changed in EU AI Act enforcement this month?"
  },
  "expected": {
    "action": "web_search",
    "reason": "needs_fresh_information",
    "query": "EU AI Act enforcement latest updates",
    "url": ""
  },
  "tags": ["freshness", "router"]
}
```

- [ ] **Step 2: Create starter datasets with at least 8-12 cases per prompt family**

Cover the highest-risk prompt behaviors first:

- `summarize`: source fidelity, multiple sources, weak source text, URL preservation
- `report`: synthesis quality, structured markdown, memory context usage
- `rag_chat_system`: grounded answers vs unsupported claims
- `followup_answer`: reuse prior report context and history correctly
- `web_search_decision`: direct answer vs RAG vs search vs fetch vs clarify
- `web_search_repair`: repair false “I can’t access URLs” behavior using retrieved content

- [ ] **Step 3: Replace `build_examples()` with dataset-driven adapters**

Refactor the pipeline so it loads prompt-family data from `src/evals/prompt_optimization/<template>.json` and converts `inputs`/`expected` directly into `dspy.Example` objects.

Expected code shape:

```python
def load_template_dataset(template_name: str) -> list[dict]:
    path = EVALS_DIR / "prompt_optimization" / f"{template_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))

def build_examples(spec: PromptSpec, cases: list[dict]) -> list[Any]:
    import dspy
    examples = []
    input_keys = [name for name, _, _ in spec.input_fields]
    for case in cases:
        row = {**case["inputs"], **case["expected"]}
        row["expected_output"] = json.dumps(case["expected"], sort_keys=True)
        examples.append(dspy.Example(**row).with_inputs(*input_keys))
    return examples
```

- [ ] **Step 4: Keep legacy golden-set support only for summarize/report smoke tests**

Do not delete `src/evals/golden_set.json` yet. Keep a small compatibility path for existing tests, but stop using it as the primary production optimization dataset.

- [ ] **Step 5: Add dataset validation tests**

Add tests that fail fast when:

- a template dataset is missing required input keys
- expected router outputs omit `action` or include invalid actions
- JSON datasets are malformed

Run:

```bash
uv run pytest tests/test_prompt_optimization_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evals/prompt_optimization scripts/prompt_optimization_pipeline.py src/prompts/dspy_optimizer.py tests/test_prompt_optimization_pipeline.py
git commit -m "feat: add prompt-family optimization datasets"
```

---

### Task 3: Make metrics production-meaningful instead of structurally convenient

`word_overlap` is too weak for markdown, JSON, and router outputs, and current router scoring rewards structural validity without checking decision correctness. Production optimization needs metrics aligned to each prompt’s job.

**Files:**
- Modify: `scripts/prompt_optimization_pipeline.py`
- Modify: `src/prompts/dspy_optimizer.py`
- Modify: `tests/test_dspy_optimizer.py`
- Modify: `tests/test_prompt_optimization_pipeline.py`

- [ ] **Step 1: Split metrics by prompt family**

Implement metric selection roughly like:

```python
PROMPT_METRICS = {
    "summarize": summarize_quality_metric,
    "report": report_quality_metric,
    "rag_chat_system": grounded_answer_metric,
    "followup_answer": grounded_answer_metric,
    "web_search_decision": router_decision_metric,
    "web_search_repair": repair_quality_metric,
}
```

- [ ] **Step 2: Add a real router metric**

Router scoring should compare predicted `action`, `reason`, `query`, and `url` against labeled expectations, not just check whether fields are non-empty.

Suggested scoring:

- `0.5` exact `action` match
- `0.2` query/url presence rules correct
- `0.2` expected query/url semantic overlap
- `0.1` reason non-empty and not obviously malformed

- [ ] **Step 3: Add text-prompt metrics that tolerate format differences**

Use prompt-family-aware evaluation:

- `summarize`: JSON parse success, source URL preservation, key-fact coverage
- `report`: section presence, answer coverage, no unsupported claims
- `rag_chat_system` / `followup_answer` / `web_search_repair`: DeepEval faithfulness and relevancy, with fallback heuristics only when DeepEval fails

- [ ] **Step 4: Make fallback metrics loud, not silent**

Record whether scoring came from:

- primary metric
- DeepEval fallback
- heuristic fallback
- scoring exception

Add this provenance to `pipeline_report.json`.

- [ ] **Step 5: Add tests for metric behavior**

Test at least:

- router metric penalizes wrong `action`
- report metric does not collapse to near-zero for valid markdown
- summarize metric requires source URL preservation

Run:

```bash
uv run pytest tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/prompt_optimization_pipeline.py src/prompts/dspy_optimizer.py tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py
git commit -m "feat: add production metrics for prompt optimization"
```

---

### Task 4: Add pipeline guardrails, artifact governance, and resumability

Once metrics are trustworthy, the next production blocker is operational safety. Optimization runs need bounded cost, resumability, and reviewable outputs.

**Files:**
- Modify: `scripts/prompt_optimization_pipeline.py`
- Modify: `scripts/optimize_prompts.py`
- Create: `optimized_prompts/manifests/`
- Create: `tests/test_prompt_optimization_pipeline.py`
- Modify: `README.md`

- [ ] **Step 1: Add explicit budget and concurrency controls**

Add CLI flags such as:

```bash
--max-cases 10
--max-templates 2
--max-cost-usd 5
--threads 2
--stop-on-regression
```

At minimum, enforce case and template count caps before long runs begin.

- [ ] **Step 2: Add checkpoint/resume support**

Persist one JSON checkpoint per template run under `optimized_prompts/manifests/`.

Minimum fields:

```json
{
  "template": "summarize",
  "metric": "summarize_quality_v1",
  "status": "completed",
  "started_at": "2026-05-23T10:00:00Z",
  "completed_at": "2026-05-23T10:07:12Z",
  "score_before": 0.42,
  "score_after": 0.58,
  "optimized_path": "optimized_prompts/summarize_optimized.json"
}
```

- [ ] **Step 3: Produce promotion-friendly artifacts**

For every optimized template, save:

- optimized DSPy JSON
- metrics summary JSON
- per-case comparison JSON
- human-readable diff/summary markdown

If DSPy does not expose a clean prompt diff, save a structured “before vs after behavior” artifact instead of pretending there is a line diff.

- [ ] **Step 4: Add regression gates**

Fail the pipeline when:

- score delta is negative beyond a small tolerance
- required artifact files are missing
- fallback metrics were used for more than an allowed threshold

- [ ] **Step 5: Add tests for checkpoint and artifact generation**

Run:

```bash
uv run pytest tests/test_prompt_optimization_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimize_prompts.py scripts/prompt_optimization_pipeline.py optimized_prompts/manifests README.md tests/test_prompt_optimization_pipeline.py
git commit -m "feat: add prompt optimization guardrails and artifacts"
```

---

### Task 5: Define how optimized artifacts are promoted into production

At this point the team needs an explicit contract for “optimized” becoming “approved for use.” Without this, the optimizer remains a sidecar experiment.

**Files:**
- Modify: `README.md`
- Create: `docs/env-vars-production.md` or a prompt-specific production doc section
- Optional create: `src/prompts/optimized_loader.py`
- Optional modify: `src/prompts/registry.py`
- Optional modify: `src/graph/nodes.py`
- Optional modify: `src/api/endpoints.py`
- Create: `tests/test_prompt_runtime_adoption.py`

- [ ] **Step 1: Choose the promotion model**

Pick one of these and document it in `README.md`:

1. **Recommended now:** offline optimization + human approval + manual `.j2` prompt update
2. **Later:** runtime loading of DSPy optimized artifacts behind a feature flag

The recommended near-term model is option 1 because current production code paths in `src/graph/nodes.py` and `src/api/endpoints.py` still render Jinja templates directly.

- [ ] **Step 2: If runtime adoption is required, add a feature flag**

Example env flag:

```bash
PROMPT_OPTIMIZATION_RUNTIME=off
PROMPT_OPTIMIZATION_RUNTIME=shadow
PROMPT_OPTIMIZATION_RUNTIME=enabled
```

Suggested semantics:

- `off`: current `.j2` prompts only
- `shadow`: run optimized prompt side-by-side, log metrics, do not serve output
- `enabled`: serve optimized output only for explicitly allowlisted prompt families

- [ ] **Step 3: Start with one allowlisted prompt family**

Do not enable all prompts at once. Start with one of:

- `web_search_decision` if router labeling is complete
- `summarize` if JSON schema and source fidelity metrics are stable

Avoid first-wave runtime adoption for `report` until markdown evaluation is trustworthy.

- [ ] **Step 4: Add production verification tests**

Tests should verify:

- runtime flag off preserves current behavior
- shadow mode does not change served output
- enabled mode only activates for allowlisted prompts

Run:

```bash
uv run pytest tests/test_prompt_runtime_adoption.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/env-vars-production.md src/prompts tests/test_prompt_runtime_adoption.py src/graph/nodes.py src/api/endpoints.py
git commit -m "feat: define prompt optimization promotion path"
```

---

### Task 6: Add end-to-end verification and release criteria

Before declaring the prompt optimization line production-ready, verify the whole workflow from dataset load to artifact generation and, if enabled, runtime gating.

**Files:**
- Modify: `README.md`
- Modify: `tests/test_prompt_optimization_pipeline.py`
- Optional modify: CI workflow files if prompt optimization checks should run in CI

- [ ] **Step 1: Add one end-to-end pipeline smoke test**

The smoke test should stub DSPy optimization and verify:

- the requested template dataset is loaded
- the selected metric is used
- artifacts are written
- the report contains before/after/provenance data

- [ ] **Step 2: Define release criteria in `README.md`**

Required release criteria:

- all pipeline tests pass
- no correctness bugs remain in filter/metric plumbing
- each production-targeted prompt family has labeled datasets
- primary metrics are used for at least 90% of scored cases
- optimization artifacts and manifests are generated for every run
- one human review approves promotion

- [ ] **Step 3: Run the full verification suite**

```bash
uv run pytest tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py tests/test_prompt_runtime_adoption.py -q
```

Expected: PASS.

- [ ] **Step 4: Run a manual smoke pass**

Use:

```bash
uv run python scripts/optimize_prompts.py --module summarize --auto light --compare
uv run python scripts/prompt_optimization_pipeline.py --template summarize --eval-only
uv run python scripts/prompt_optimization_pipeline.py --template web_search_decision --auto light
```

Expected:

- template filter limits work to the named template
- compare output uses the correct metric
- manifests and artifacts are written
- no unrelated optimized files are overwritten during filtered runs

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_dspy_optimizer.py tests/test_prompt_optimization_pipeline.py tests/test_prompt_runtime_adoption.py
git commit -m "test: add release verification for prompt optimization pipeline"
```

---

## Priority Order

Implement in this order:

1. Task 1: correctness bugs
2. Task 2: prompt-family datasets
3. Task 3: production metrics
4. Task 4: guardrails and artifacts
5. Task 5: promotion model and optional runtime flag
6. Task 6: release verification

## Stop/Go Criteria

Stop and realign if any of these are true:

- the team wants direct optimization of `.j2` template text instead of DSPy runtime artifacts
- labeled router data cannot be produced
- DeepEval cost or instability makes it unsuitable as a primary metric
- runtime integration is requested before offline artifact quality is trustworthy

## Definition of Done

The prompt optimization production line is ready when:

- the CLI and batch pipeline report trustworthy, prompt-family-specific scores
- filtered runs only touch the requested template
- optimizer outputs are resumable and reviewable
- promotion criteria are documented and enforced
- either manual `.j2` promotion or feature-flagged runtime adoption is implemented intentionally
- the verification suite passes locally
