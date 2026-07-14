# Getting started

This guide covers the complete Cortex development setup. For the short path, start with the [root README](../README.md#quickstart).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm
- Docker with Docker Compose
- Credentials for at least one supported LLM provider
- Supabase credentials for authenticated sessions, uploads, and session-scoped RAG

## 1. Install dependencies

```bash
uv sync
cp .env.example .env
```

`arxiv-mcp-server` is installed with the backend dependencies. The API launches it over `stdio` when needed, so it does not require a separate service. Backend startup validates that it can load the arXiv tools and fails early if the package is unavailable.

Optional dependency groups:

```bash
uv sync --extra evals       # DeepEval, DSPy, and evaluation data tooling
uv sync --extra finetune    # router dataset and Hugging Face tooling
```

## 2. Start local infrastructure

```bash
docker compose up -d
```

This starts:

- Redis on `localhost:6379`
- Neo4j Bolt on `localhost:7687`
- Neo4j Browser at `http://localhost:7474`
- Local Neo4j credentials: `neo4j` / `devpassword`

The backend creates the required Neo4j indexes and constraints on first connection.

## 3. Configure the environment

Choose an LLM provider in `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Supported alternatives:

- OpenRouter: `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`
- Ollama: `OLLAMA_BASE_URL` and `OLLAMA_MODEL`

Local GraphRAG defaults:

```dotenv
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=devpassword
NEO4J_DATABASE=neo4j

REDIS_URL=redis://localhost:6379/0
```

The embedding dimensions must match the Neo4j vector index. Production normally uses OpenAI `text-embedding-3-small` with 1,536 dimensions; do not point local development at that database while using the 768-dimensional local model.

For market data, configure:

```dotenv
ASSET_PRICE_PROVIDER=alphavantage_mcp
ALPHA_VANTAGE_API_KEY=...
```

`ALPHA_VANTAGE_MCP_TOOL_REFRESH_SECONDS` controls the in-memory MCP tool-catalog refresh interval. `ALPHA_VANTAGE_MCP_URL` can override the complete remote MCP URL.

See [.env.example](../.env.example) for the complete development configuration and [Production configuration](env-vars-production.md) for deployed environments.

## 4. Start the backend

```bash
uv run uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8010 --reload
```

Useful endpoints:

- API: `http://localhost:8010`
- Process liveness: `http://localhost:8010/health`
- Dependency readiness: `http://localhost:8010/ready`

## 5. Start the frontend

```bash
cd ui
npm install
npm run dev
```

The UI runs at `http://localhost:5173`.

## 6. Start background jobs

Run the Inngest development server in another terminal:

```bash
npx --ignore-scripts=false inngest-cli@latest dev \
  -u http://127.0.0.1:8010/api/inngest \
  --no-discovery
```

It invokes the `outbox-dispatcher` cron every two minutes. No separate dispatcher process is needed locally.

To flush pending outbox events immediately:

```bash
uv run python scripts/dispatch_outbox.py --limit 100
```

## Verify the setup

Run the core checks:

```bash
uv run pytest -v
uv run ruff check src
uv run mypy src
```

See [Testing and quality](testing-and-quality.md) for coverage, UI, browser, security, and load-test commands.

## Local and production GraphRAG

| | Local development | Production |
|---|---|---|
| Instance | Docker (`bolt://localhost:7687`) | Neo4j Aura (`neo4j+s://...`) |
| Database | `neo4j` | Aura database name |
| Embedding model | `nomic-embed-text` | `text-embedding-3-small` |
| Dimensions | 768 | 1,536 |

On first connection Cortex creates:

- Vector index `chunk_embedding_index` on `Chunk.embedding`
- B-tree index on `Chunk.run_id`
- B-tree index on `Document.resource_id`
- B-tree index on `Entity.normalized_name`

