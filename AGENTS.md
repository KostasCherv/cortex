## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- For deploy/ops questions, check README.md's "Production deployment" section before re-deriving from scripts/.
- When code changes affect setup, deploy, or architecture, update README.md to match.
- For UI design-system questions (theming, component conventions, shadcn primitives), check ui/DESIGN.md before re-deriving from source.
- When adding/changing UI components, variants, or design tokens under ui/src/, update ui/DESIGN.md to match.

## Debugging a LangSmith run

Credentials come from `.env` (`LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`) via `src.config.settings` — never ask the user for keys, never print them.

Use whenever given a `smith.langchain.com` URL, a bare run/trace UUID, or asked why a run failed or is slow:

1. Extract the run ID: the UUID trailing the `/r/` segment of the URL, or the bare UUID itself.
2. Fetch the run with the project's existing settings and the `langsmith` SDK (already a dependency) — don't hand-roll HTTP calls or new env loading:
   ```bash
   uv run python -c "
   from langsmith import Client
   from src.config import settings
   client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
   run = client.read_run('<RUN_ID>')
   print(run.name, run.run_type, run.status, run.error)
   print('inputs:', run.inputs)
   print('outputs:', run.outputs)
   "
   ```
3. If the root run succeeded but something downstream looks wrong, pull child runs:
   ```bash
   uv run python -c "
   from langsmith import Client
   from src.config import settings
   client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
   for r in client.list_runs(trace_id='<RUN_ID>'):
       print(r.id, r.name, r.run_type, r.status, r.error, r.latency)
   "
   ```
4. Correlate run/span names with source: `src/graph/nodes.py`, `src/graph/edges.py`, `src/graph/graph.py`, `src/observability/langsmith.py`.
5. Report: root cause, the failing node/span, the file:line in source, and a suggested fix. Inputs/outputs may be redacted per `LANGSMITH_REDACTION_MODE` — reason about shape, not the literal `"[REDACTED]"` string. If `LANGSMITH_API_KEY` is empty or `read_run` 404s, say so — don't guess at run content.
