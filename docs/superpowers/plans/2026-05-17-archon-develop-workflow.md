# Archon Develop Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Archon workflow that takes a task description, plans with Claude Opus, implements with Codex, runs up to 3 Sonnet review-refinement cycles, and opens a GitHub PR — all in an isolated git worktree.

**Architecture:** Three config files: `.archon/config.yaml` (model + worktree settings), `.archon/workflows/develop.yaml` (13-node DAG), `.github/pull_request_template.md` (PR body template). Cycles 2 and 3 use `when:` conditions to short-circuit when the preceding review passes. `interactive: true` at the workflow level enables the planner approval gate.

**Tech Stack:** Archon CLI, Claude Code SDK (opus/sonnet), Codex CLI, `gh` CLI for PR creation, `uv run` for Python toolchain validation.

---

## File Map

| File | Action |
|---|---|
| `.archon/config.yaml` | Create |
| `.archon/workflows/develop.yaml` | Create |
| `.github/pull_request_template.md` | Create |
| `.gitignore` | Modify — remove `docs` line so the spec can be committed |

---

### Task 1: Fix `.gitignore` and commit spec doc

The `docs/` directory is currently ignored. Remove that entry so the design spec and this plan can be tracked in git.

**Files:**
- Modify: `.gitignore`
- Commit: `docs/superpowers/specs/2026-05-17-archon-develop-workflow-design.md`
- Commit: `docs/superpowers/plans/2026-05-17-archon-develop-workflow.md` (this file)

- [ ] **Step 1: Remove `docs` from `.gitignore`**

Open `.gitignore` and delete the line that reads exactly `docs`. Do not remove any other entries.

- [ ] **Step 2: Verify docs are now trackable**

```bash
git status docs/
```

Expected: both spec and plan files appear as untracked (not ignored).

- [ ] **Step 3: Stage and commit**

```bash
git add .gitignore docs/superpowers/specs/2026-05-17-archon-develop-workflow-design.md docs/superpowers/plans/2026-05-17-archon-develop-workflow.md
git commit -m "chore: track docs directory and add Archon workflow design spec and plan"
```

---

### Task 2: Create `.archon/config.yaml`

Archon reads this file from the repo root `.archon/` directory. Sets default models for each assistant and enables per-run worktree isolation.

**Files:**
- Create: `.archon/config.yaml`

- [ ] **Step 1: Create the `.archon/` directory and config file**

```yaml
# .archon/config.yaml
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
  enabled: true
  baseBranch: main
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.archon/config.yaml')); print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add .archon/config.yaml
git commit -m "chore: add Archon config with Claude/Codex models and worktree isolation"
```

---

### Task 3: Create `.archon/workflows/develop.yaml` — plan + gate + implement nodes

Build the workflow file incrementally. This task covers the first three nodes: `plan`, `plan-gate`, and `implement`.

**Files:**
- Create: `.archon/workflows/develop.yaml`

- [ ] **Step 1: Create the workflow file with the plan node**

```yaml
# .archon/workflows/develop.yaml
name: develop
description: "Plan (opus) → human gate → Implement (codex) → up to 3 × [Validate → Review (sonnet) → Refine (codex)] → summary → PR"
interactive: true

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
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.archon/workflows/develop.yaml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .archon/workflows/develop.yaml
git commit -m "feat: add Archon develop workflow — plan, approval gate, implement nodes"
```

---

### Task 4: Add Cycle 1 — `validate1` + `review1`

Append the first validation bash node and the first structured reviewer node to the workflow.

**Files:**
- Modify: `.archon/workflows/develop.yaml`

- [ ] **Step 1: Append Cycle 1 nodes after the `implement` node**

Add the following at the end of the `nodes:` list (after the `implement` node block):

```yaml
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
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.archon/workflows/develop.yaml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .archon/workflows/develop.yaml
git commit -m "feat: add Cycle 1 validate and review nodes to develop workflow"
```

---

### Task 5: Add Cycle 2 — `refine2` + `validate2` + `review2`

Append the second cycle. All three nodes carry `when: "$review1.output.pass == false"` so they are skipped if the first review passed.

**Files:**
- Modify: `.archon/workflows/develop.yaml`

- [ ] **Step 1: Append Cycle 2 nodes at the end of the `nodes:` list**

```yaml
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
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.archon/workflows/develop.yaml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .archon/workflows/develop.yaml
git commit -m "feat: add Cycle 2 refine/validate/review nodes (skipped when review1 passes)"
```

---

### Task 6: Add Cycle 3 — `refine3` + `validate3` + `review3`

Append the third and final cycle. Guards on `$review2.output.pass == false`.

**Files:**
- Modify: `.archon/workflows/develop.yaml`

- [ ] **Step 1: Append Cycle 3 nodes at the end of the `nodes:` list**

```yaml
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
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.archon/workflows/develop.yaml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .archon/workflows/develop.yaml
git commit -m "feat: add Cycle 3 refine/validate/review nodes (final review cycle)"
```

---

### Task 7: Add `run-summary` + `create-pr` — complete the workflow

Append the final two nodes: the audit summary writer and the PR creation node.

**Files:**
- Modify: `.archon/workflows/develop.yaml`

- [ ] **Step 1: Append `run-summary` and `create-pr` nodes at the end of the `nodes:` list**

```yaml
  # ── RUN SUMMARY ───────────────────────────────────────────────────────
  - id: run-summary
    depends_on: [review3]
    provider: claude
    model: sonnet
    prompt: |
      Write a concise run summary to .archon/run-summary.md using this exact format:

      # Run summary — <current ISO timestamp>
      **Task:** $ARGUMENTS

      | Cycle | Pass | Issues |
      |-------|------|--------|
      | 1     | $review1.output.pass | $review1.output.issues |
      | 2     | $review2.output.pass | $review2.output.issues |
      | 3     | $review3.output.pass | $review3.output.issues |

      **Final status:** APPROVED if any cycle passed, otherwise UNRESOLVED ISSUES

      Replace null values with `—` for skipped cycles.
    allowed_tools: [Write]

  # ── CREATE PR ─────────────────────────────────────────────────────────
  - id: create-pr
    depends_on: [run-summary]
    provider: claude
    model: sonnet
    prompt: |
      The branch was created automatically by Archon's worktree feature.

      1. Read .archon/plan.md to derive the PR title (first non-blank line, title-case).
      2. Read .archon/run-summary.md for the review cycle table.
      3. Read .github/pull_request_template.md for the PR body structure.
      4. Fill in all template sections:
         - Summary: what changed and why (from .archon/plan.md)
         - Review cycles table: copy from .archon/run-summary.md
         - Outstanding issues: any unresolved issues from the final completed review
           (use $review1.output.issues / $review2.output.issues / $review3.output.issues
            from the last cycle that ran)
         - Test plan: derived from acceptance criteria in .archon/plan.md
      5. Run: gh pr create --title "<derived title>" --body "<filled template body>"
    allowed_tools: [Read, Bash]
```

- [ ] **Step 2: Validate the complete YAML file**

```bash
python3 -c "import yaml; data = yaml.safe_load(open('.archon/workflows/develop.yaml')); print(f'OK — {len(data[\"nodes\"])} nodes')"
```

Expected: `OK — 13 nodes`

- [ ] **Step 3: Verify all `depends_on` references point to real node IDs**

```bash
python3 - << 'EOF'
import yaml
data = yaml.safe_load(open('.archon/workflows/develop.yaml'))
ids = {n['id'] for n in data['nodes']}
errors = []
for node in data['nodes']:
    for dep in node.get('depends_on', []):
        if dep not in ids:
            errors.append(f"Node '{node['id']}' depends on unknown '{dep}'")
if errors:
    print('\n'.join(errors))
else:
    print(f'OK — all depends_on references valid ({len(ids)} nodes)')
EOF
```

Expected: `OK — all depends_on references valid (13 nodes)`

- [ ] **Step 4: Commit**

```bash
git add .archon/workflows/develop.yaml
git commit -m "feat: complete Archon develop workflow with run-summary and create-pr nodes"
```

---

### Task 8: Create `.github/pull_request_template.md`

GitHub automatically uses this file as the default body for all new PRs opened in the repo via the web UI or `gh pr create` without an explicit `--body`.

**Files:**
- Create: `.github/pull_request_template.md`

- [ ] **Step 1: Create the `.github/` directory and template file**

```markdown
## Summary
<!-- What changed and why. 2-4 bullet points. -->
-

## Review cycles completed
<!-- Filled in by the Archon develop workflow (see .archon/run-summary.md) -->
| Cycle | Result | Summary |
|-------|--------|---------|
| 1     |        |         |
| 2     | —      |         |
| 3     | —      |         |

## Outstanding issues
<!-- Any issues not resolved within the 3 review cycles. "None" if all cycles passed. -->
None

## Test plan
- [ ] `uv run pytest -v` passes
- [ ] `uv run ruff check src` clean
- [ ] `uv run mypy src` clean
- [ ] Manual smoke test: `POST /sessions/{id}/research` returns a streaming response
- [ ] Manual smoke test: RAG ingestion pipeline processes a test file end-to-end
```

- [ ] **Step 2: Verify the file renders as valid Markdown**

```bash
python3 -c "
content = open('.github/pull_request_template.md').read()
assert '## Summary' in content
assert '## Review cycles completed' in content
assert '## Test plan' in content
print('OK — all required sections present')
"
```

Expected: `OK — all required sections present`

- [ ] **Step 3: Commit**

```bash
git add .github/pull_request_template.md
git commit -m "chore: add GitHub PR template with review cycle table and test plan checklist"
```

---

### Task 9: Smoke-test workflow discovery

Verify Archon can discover and parse the completed workflow before declaring done.

**Files:** none (read-only verification)

- [ ] **Step 1: Check Archon is installed**

```bash
archon --version
```

If not installed: `npm install -g @archonhq/cli` (or `bun add -g @archonhq/cli`).

- [ ] **Step 2: List workflows and confirm `develop` appears**

```bash
archon workflow list
```

Expected output includes a line like:
```
develop   Plan (opus) → human gate → Implement (codex) → ...
```

If Archon is not yet installed, run the Python structural check instead:

```bash
python3 - << 'EOF'
import yaml
data = yaml.safe_load(open('.archon/workflows/develop.yaml'))
node_ids = [n['id'] for n in data['nodes']]
expected = [
    'plan', 'plan-gate', 'implement',
    'validate1', 'review1',
    'refine2', 'validate2', 'review2',
    'refine3', 'validate3', 'review3',
    'run-summary', 'create-pr',
]
assert node_ids == expected, f"Mismatch: {node_ids}"
print(f"OK — {len(node_ids)} nodes in correct order")
EOF
```

Expected: `OK — 13 nodes in correct order`

- [ ] **Step 3: Verify `.archon/config.yaml` is also valid**

```bash
python3 -c "
import yaml
cfg = yaml.safe_load(open('.archon/config.yaml'))
assert cfg['assistants']['claude']['model'] == 'sonnet'
assert cfg['assistants']['codex']['model'] == 'gpt-5.3-codex'
assert cfg['worktree']['enabled'] == True
print('OK — config valid')
"
```

Expected: `OK — config valid`

- [ ] **Step 4: Final commit if any files were adjusted during smoke-test**

```bash
git status
# Only commit if there are changes
git add -p   # review interactively
git commit -m "chore: verify Archon workflow smoke-test passes"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| `.archon/config.yaml` with Claude + Codex + worktree | Task 2 |
| `plan` node — claude:opus, writes `.archon/plan.md` | Task 3 |
| `plan-gate` approval node | Task 3 |
| `implement` loop — codex, max 3, `$LOOP_PREV_OUTPUT` | Task 3 |
| `validate1/2/3` bash nodes | Tasks 4, 5, 6 |
| `review1/2/3` — sonnet, `output_format` with pass/issues/summary | Tasks 4, 5, 6 |
| `when:` short-circuit on each cycle | Tasks 5, 6 |
| `refine2/3` — codex, `$reviewN.output.issues` injected | Tasks 5, 6 |
| `run-summary` node — writes `.archon/run-summary.md` | Task 7 |
| `create-pr` — reads template, fills review table | Task 7 |
| `.github/pull_request_template.md` with all sections | Task 8 |
| `interactive: true` at workflow level | Task 3 |
| Fix `.gitignore` so `docs/` is tracked | Task 1 |

All spec requirements covered. No placeholders.

**Type/reference consistency:**
- Node IDs used in `depends_on` and `when:` (`review1`, `review2`) match the `id:` fields exactly
- `$review1.output.pass`, `$review1.output.issues`, `$review1.output.summary` match the `output_format` schema defined in `review1`
- Same pattern holds for `review2` and `review3`
- `$LOOP_PREV_OUTPUT` and `$ARGUMENTS` are Archon built-in variables — no definition needed
