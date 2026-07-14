# Agent tooling

The authoritative rules for AI coding agents are [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md), which are kept in sync. This guide explains the repository-specific tools referenced by those rules.

## Documentation ownership

When behavior changes, update the most specific guide:

- Product positioning and headline capabilities: `README.md`
- Local setup: `docs/getting-started.md`
- Architecture and system flows: `docs/architecture.md`
- Production releases: `docs/deployment.md`
- UI components, variants, and design tokens: `ui/DESIGN.md`

This keeps the root README useful as a project landing page while ensuring operational instructions stay close to their audience.

## Graphify knowledge graph

The generated graph in `graphify-out/` maps code, documentation, symbols, communities, and cross-file relationships. Prefer a scoped query over reading the complete graph report.

### Install

```bash
uv tool install graphifyy
ollama pull gemma4:31b-cloud
```

For Graphify's OpenAI-compatible extraction API:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_API_KEY=ollama
export GRAPHIFY_OLLAMA_MODEL=gemma4:31b-cloud
```

Do not reuse an embedding URL that omits `/v1`.

### Commands

| Task | Command |
|---|---|
| Ask about the codebase | `graphify query "How does billing work?"` |
| Explain one concept | `graphify explain "ResearchState"` |
| Find a relationship | `graphify path "endpoints" "outbox"` |
| Refresh code structure | `graphify update .` |
| Full semantic rebuild | `./scripts/graphify-rebuild.sh` |
| Incremental Markdown rebuild | `./scripts/graphify-rebuild.sh --incremental` |
| Use the local 8B model | `./scripts/graphify-rebuild.sh --local` |
| Regenerate report and HTML | `graphify cluster-only .` |

Generated outputs include `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, and `graphify-out/graph.html`.

### Git hooks

```bash
graphify hook install
./scripts/install-graphify-post-commit.sh
```

The first hook refreshes AST data after code commits. The repository hook starts an incremental semantic extraction after Markdown commits. Follow its log with:

```bash
tail -f ~/.cache/graphify-rebuild.log
```

Skip the semantic hook once with `GRAPHIFY_SKIP_HOOK=1 git commit`. Reinstall the repository hook after reinstalling Graphify hooks.

## Production deployment

When asked to ship to production, agents follow `.cursor/skills/deploy-prod/SKILL.md` and the [deployment runbook](deployment.md). That workflow covers the Cloud Run backend, Vercel UI, smoke verification, and rollback behavior.

## LangSmith diagnosis

When given a LangSmith run URL or UUID, agents use the configured `langsmith` SDK and credentials from `src.config.settings`; they never request, print, or hand-load the API key. The investigation should identify the root cause, failing node or span, relevant source location, and a suggested fix.

