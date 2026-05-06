# LangFuse Integration

## Environment

Set these environment variables to enable LangFuse:

- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY=pk-lf-...`
- `LANGFUSE_SECRET_KEY=sk-lf-...`
- `LANGFUSE_HOST=https://cloud.langfuse.com`
- `LANGFUSE_RELEASE=<optional app version>`

## Ownership

- LangSmith continues to trace workflow and node execution for the LangGraph pipeline.
- LangFuse records individual non-streaming LLM generations wrapped by `_invoke_llm()`.
- LangFuse also stores user feedback scores and the synced golden-query dataset.

## User Feedback

The API exposes:

- `POST /sessions/{session_id}/runs/{run_id}/feedback`

Request body:

```json
{
  "helpful": true,
  "comment": "Optional note"
}
```

Rules:

- Only one feedback submission is allowed per run in this iteration.
- The run must have LangFuse trace metadata persisted on it.
- Comments are optional, trimmed, and limited to 500 characters.

The research UI shows a simple thumbs up/down control for the latest visible completed run. A thumbs down reveals an optional comment box.

## Golden Queries

The checked-in source of truth lives at:

- `tests/fixtures/langfuse_golden_queries.json`

Each item contains:

- `id`
- `input`
- `rubric`
- `tags`
- optional `difficulty`
- optional `notes`

This artifact is intentionally rubric-based rather than exact-output based so the benchmark remains stable for generative research/report answers.

## Dataset Sync

Use the CLI to sync the checked-in artifact into LangFuse:

```bash
uv run python -m src.main langfuse-sync-dataset
```

Optional flags:

- `--dataset-name research-agent/golden-queries`
- `--source tests/fixtures/langfuse_golden_queries.json`

The sync is explicit and idempotent. It creates the dataset if missing and upserts items by stable item id.

## Production Promotion Workflow

1. Review weak traces or generations in LangFuse.
2. Curate the useful failures into `tests/fixtures/langfuse_golden_queries.json`.
3. Re-run dataset sync.
4. Use the LangFuse dataset and experiments for regression tracking.

This keeps git as the benchmark source of truth while still using LangFuse UI as the operational discovery surface.
