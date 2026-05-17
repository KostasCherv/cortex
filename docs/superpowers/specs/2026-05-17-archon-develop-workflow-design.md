# Archon `develop` Workflow — Design Spec

**Date:** 2026-05-17
**Status:** Approved

---

## Overview

Add an Archon-powered AI development workflow to the Cortex repo. The workflow takes a task description from the CLI, plans the work with a smart model, implements it with a fast model, then runs up to three review-refinement cycles before opening a GitHub PR.

Each run is isolated in its own git worktree. The planner presents its plan for human approval before any code is written. A run-summary artifact is produced at the end for audit and debugging.

**Files created:**

| File | Purpose |
|---|---|
| `.archon/config.yaml` | Model defaults, assistant config, worktree settings |
| `.archon/workflows/develop.yaml` | Main workflow definition |
| `.github/pull_request_template.md` | Consistent PR format for all PRs |

---

## Architecture

```
plan (claude:opus)  ← runs in isolated git worktree
  ↓
plan-gate (approval)  ← human reviews .archon/plan.md before any code is written
  ↓
implement (codex loop, max 3)          ← initial implementation + self-correction
  ↓
validate1 (bash: pytest + ruff + mypy)
  ↓
review1 (claude:sonnet, structured JSON)
  ↓ pass=true ──────────────────────────────────────────────────────→ run-summary → create-pr
  ↓ pass=false
refine2 (codex) → validate2 (bash) → review2 (sonnet)
  ↓ pass=true ──────────────────────────────────────────────────────→ run-summary → create-pr
  ↓ pass=false
refine3 (codex) → validate3 (bash) → review3 (sonnet)
  ↓
run-summary (bash)  ← writes .archon/run-summary.md
  ↓
create-pr (claude:sonnet)
```

**Worktree isolation:** `worktree: true` in `.archon/config.yaml` creates a
clean git worktree for each run. The branch is named from the plan title
(kebab-case). No risk of contaminating the working tree mid-run.

**Planner approval gate:** an `approval` node pauses after `plan` so you can
read `.archon/plan.md` and either approve or reject with a reason. If rejected,
`$REJECTION_REASON` is available for a revised plan (not wired in v1 — see
Known constraints).

**Skip-summary node:** a bash node writes `.archon/run-summary.md` before PR
creation, capturing the pass/fail history of all cycles for audit and debugging.

Short-circuit behaviour: each `refineN`/`validateN`/`reviewN` cycle carries
`when: "$reviewN-1.output.pass == false"`. When skipped, Archon marks the node
as completed with null output and downstream nodes proceed normally.
`create-pr` always depends on `review3`; null references in its prompt template
render as empty strings.

---

## Model Matrix

| Node | Provider | Model | Rationale |
|---|---|---|---|
| `plan` | claude | `opus` | Deep codebase exploration, high-effort planning |
| `implement` (loop) | codex | `gpt-5.3-codex` | Fast, cheap, autocomplete-style coding |
| `validate*` | bash | — | Deterministic — no AI tokens |
| `review*` | claude | `sonnet` | Strong reasoning; cheaper than opus for review |
| `refine*` | codex | `gpt-5.3-codex` | Apply structured feedback mechanically |
| `plan-gate` | — | — | Human approval — no AI tokens |
| `run-summary` | bash | — | Writes audit artifact — no AI tokens |
| `create-pr` | claude | `sonnet` | Writes PR description from structured review output |

---

## Node Details

### `plan`
- **Model:** claude:opus
- **Input:** `$ARGUMENTS` — the task description passed on the CLI
- **Output:** writes `.archon/plan.md` with numbered tasks and acceptance criteria
- **Tools:** Read, Write, Bash (for codebase exploration)

### `plan-gate`
- **Type:** approval
- **Message:** "Review `.archon/plan.md`. Approve to start coding, or reply with feedback to abort."
- Zero AI cost — pure human gate
- Requires `interactive: true` at the workflow level

### `implement` (loop)
- **Model:** codex
- **Loop:** `until: DONE`, `max_iterations: 3`, `fresh_context: true`
- **Context:** reads `.archon/plan.md`; `$LOOP_PREV_OUTPUT` carries previous iteration's output for self-correction
- **Inline validation:** runs `uv run pytest -x -q` within each iteration; fixes failures before signalling DONE
- **Tools:** Read, Write, Edit, Bash

### `validate1` / `validate2` / `validate3`
- **Type:** bash
- **Command:** `uv run pytest -x -q && uv run ruff check src && uv run mypy src --ignore-missing-imports`
- Fails fast (`-x`) to surface first failure clearly; blocks reviewer from running on broken code

### `review1` / `review2` / `review3`
- **Model:** claude:sonnet
- **Output format:**
  ```json
  { "pass": boolean, "issues": ["string"], "summary": "string" }
  ```
- `review2` prompt includes `$review1.output.issues` for continuity
- `review3` prompt includes issues from both previous cycles
- **Tools:** Read, Bash

### `refine2` / `refine3`
- **Model:** codex
- **Condition:** `when: "$reviewN.output.pass == false"`
- **Input:** structured issues list from the preceding review node via `$reviewN.output.issues`
- Runs `uv run pytest -x -q` after fixes
- **Tools:** Read, Write, Edit, Bash

### `run-summary`
- **Type:** bash
- **Depends on:** `review3` (always runs, regardless of cycle outcomes)
- Writes `.archon/run-summary.md` with: cycle pass/fail history, issue counts, final status
- Script:
  ```bash
  cat > .archon/run-summary.md << 'EOF'
  # Run summary — $(date -u +"%Y-%m-%dT%H:%M:%SZ")
  | Cycle | Pass | Issues |
  |-------|------|--------|
  | 1     | ...  | ...    |
  | 2     | ...  | ...    |
  | 3     | ...  | ...    |
  EOF
  ```
  The `create-pr` node (claude:sonnet) fills in the actual values from `$reviewN.output.*`.

### `create-pr`
- **Model:** claude:sonnet
- **Depends on:** `run-summary`
- Reads `.archon/run-summary.md` and `.github/pull_request_template.md`
- Uses `gh pr create` to open the PR; attaches run summary and any unresolved issues

---

## `.archon/config.yaml`

```yaml
assistants:
  claude:
    model: sonnet
    settingSources:
      - project
      - user
  codex:
    model: gpt-5.3-codex
    modelReasoningEffort: medium
    webSearchMode: disabled

worktree:
  enabled: true          # each run gets an isolated git worktree
  baseBranch: main       # branch from main
```

---

## `.archon/workflows/develop.yaml` (full)

```yaml
name: develop
description: "Plan (opus) → human gate → Implement (codex) → up to 3 × [Validate → Review (sonnet) → Refine (codex)] → summary → PR"
interactive: true   # required for approval gate to appear in Web UI / CLI

nodes:
  # ── PLAN ──────────────────────────────────────────────────────────────
  - id: plan
    provider: claude
    model: opus
    prompt: |
      Task: $ARGUMENTS

      Explore the codebase thoroughly. Write a numbered implementation plan
      to .archon/plan.md. Include: files to change, acceptance criteria,
      and expected test behaviour.
    allowed_tools: [Read, Write, Bash]

  # ── HUMAN APPROVAL GATE ───────────────────────────────────────────────
  - id: plan-gate
    depends_on: [plan]
    approval:
      message: |
        Review the implementation plan at .archon/plan.md before coding begins.
        Approve to proceed, or reply with your reason to abort.

  # ── INITIAL IMPLEMENTATION ────────────────────────────────────────────
  - id: implement
    depends_on: [plan-gate]
    provider: codex
    loop:
      prompt: |
        Read .archon/plan.md. Implement all numbered tasks.
        Run `uv run pytest -x -q` after changes and fix any failures.
        When all tasks are done and tests pass output: <promise>DONE</promise>

        Previous iteration output (empty on first pass):
        $LOOP_PREV_OUTPUT
      until: DONE
      max_iterations: 3
      fresh_context: true
    allowed_tools: [Read, Write, Edit, Bash]

  # ── CYCLE 1 ───────────────────────────────────────────────────────────
  - id: validate1
    depends_on: [implement]
    bash: "uv run pytest -x -q && uv run ruff check src && uv run mypy src --ignore-missing-imports"

  - id: review1
    depends_on: [validate1]
    provider: claude
    model: sonnet
    prompt: |
      Review all changes against .archon/plan.md.
      Cortex stack: Python 3.11 + FastAPI + LangGraph backend; React 19 + TypeScript frontend.
      Check: correctness, test coverage, type safety, security, code style.
      Be specific and actionable.
    output_format:
      type: object
      properties:
        pass: { type: boolean }
        issues: { type: array, items: { type: string } }
        summary: { type: string }
      required: [pass, issues, summary]
    allowed_tools: [Read, Bash]

  # ── CYCLE 2 (skipped if review1 passed) ──────────────────────────────
  - id: refine2
    depends_on: [review1]
    when: "$review1.output.pass == false"
    provider: codex
    prompt: |
      Fix all issues raised by the reviewer.

      Issues to fix:
      $review1.output.issues

      Run `uv run pytest -x -q` after fixes to confirm they pass.
    allowed_tools: [Read, Write, Edit, Bash]

  - id: validate2
    depends_on: [refine2]
    when: "$review1.output.pass == false"
    bash: "uv run pytest -x -q && uv run ruff check src && uv run mypy src --ignore-missing-imports"

  - id: review2
    depends_on: [validate2]
    when: "$review1.output.pass == false"
    provider: claude
    model: sonnet
    prompt: |
      Re-review all changes against .archon/plan.md.
      Previous issues (cycle 1): $review1.output.issues
      Confirm each is resolved, then check for new issues.
    output_format:
      type: object
      properties:
        pass: { type: boolean }
        issues: { type: array, items: { type: string } }
        summary: { type: string }
      required: [pass, issues, summary]
    allowed_tools: [Read, Bash]

  # ── CYCLE 3 (skipped if review1 or review2 passed) ───────────────────
  - id: refine3
    depends_on: [review2]
    when: "$review2.output.pass == false"
    provider: codex
    prompt: |
      Fix all outstanding reviewer issues.

      Cycle 1 issues: $review1.output.issues
      Cycle 2 issues: $review2.output.issues

      Run `uv run pytest -x -q` after fixes.
    allowed_tools: [Read, Write, Edit, Bash]

  - id: validate3
    depends_on: [refine3]
    when: "$review2.output.pass == false"
    bash: "uv run pytest -x -q && uv run ruff check src && uv run mypy src --ignore-missing-imports"

  - id: review3
    depends_on: [validate3]
    when: "$review2.output.pass == false"
    provider: claude
    model: sonnet
    prompt: |
      Final review against .archon/plan.md.
      All previous issues:
      - Cycle 1: $review1.output.issues
      - Cycle 2: $review2.output.issues
      Confirm everything is resolved. Flag anything still outstanding.
    output_format:
      type: object
      properties:
        pass: { type: boolean }
        issues: { type: array, items: { type: string } }
        summary: { type: string }
      required: [pass, issues, summary]
    allowed_tools: [Read, Bash]

  # ── RUN SUMMARY ──────────────────────────────────────────────────────
  - id: run-summary
    depends_on: [review3]
    provider: claude
    model: sonnet
    prompt: |
      Write a concise run summary to .archon/run-summary.md using this format:

      # Run summary — <ISO timestamp>
      **Task:** $ARGUMENTS

      | Cycle | Pass | Issues |
      |-------|------|--------|
      | 1     | $review1.output.pass | $review1.output.issues |
      | 2     | $review2.output.pass | $review2.output.issues |
      | 3     | $review3.output.pass | $review3.output.issues |

      **Final status:** APPROVED / UNRESOLVED ISSUES
    allowed_tools: [Write]

  # ── CREATE PR ─────────────────────────────────────────────────────────
  - id: create-pr
    depends_on: [run-summary]
    provider: claude
    model: sonnet
    prompt: |
      The branch was created automatically by Archon's worktree feature.

      Open a GitHub PR using:
        gh pr create --title "<title from .archon/plan.md>" --body "<body>"

      Build the PR body from .github/pull_request_template.md, filling in:
      - Summary: what changed and why (from .archon/plan.md)
      - Review cycles table: from .archon/run-summary.md
      - Outstanding issues: any unresolved issues from the final review
      - Test plan: derived from acceptance criteria in .archon/plan.md
    allowed_tools: [Bash]
```

---

## `.github/pull_request_template.md`

```markdown
## Summary
<!-- What changed and why. 2-4 bullet points. -->
-

## Review cycles completed
<!-- Filled in by the Archon develop workflow -->
| Cycle | Result | Summary |
|-------|--------|---------|
| 1     |        |         |
| 2     | —      |         |
| 3     | —      |         |

## Outstanding issues
<!-- Any issues not resolved within the 3 review cycles -->
None

## Test plan
- [ ] `uv run pytest -v` passes
- [ ] `uv run ruff check src` clean
- [ ] `uv run mypy src` clean
- [ ] Manual smoke test: research endpoint responds correctly
- [ ] Manual smoke test: RAG ingestion pipeline processes a test file
```

---

## Usage

```bash
# Run the develop workflow with a task description
archon workflow run develop "Add rate limiting to POST /sessions/{id}/research — max 10 req/min per user"
```

Archon discovers `.archon/workflows/develop.yaml` automatically. The task
description becomes `$ARGUMENTS` inside the workflow.

---

## Known constraints

- **Archon loop is per-node** — multi-node cycles are unrolled explicitly. Three
  cycles means 9 nodes (refine + validate + review × 3). Verbose but transparent.
- **Skipped node outputs** — when a cycle is skipped, `$reviewN.output.*`
  references in downstream prompts render as empty strings, not errors.
- **`create-pr` always runs** — even if all three reviews fail, the PR is created
  with unresolved issues listed. Human reviews the PR before merge.
- **Worktree branch naming** — Archon derives the branch name from the workflow
  name + run ID by default. The `create-pr` node should rename it to a
  meaningful slug (from `.archon/plan.md` title) before opening the PR.
- **Approval rejection not looped** — if `plan-gate` is rejected, the workflow
  aborts. A revised plan requires re-running the workflow. Looping on rejection
  via `$REJECTION_REASON` is a future enhancement.
- **Codex binary** — requires `codexBinaryPath` set in `.archon/config.yaml` or
  `CODEX_BIN_PATH` env var if not on `$PATH`.
