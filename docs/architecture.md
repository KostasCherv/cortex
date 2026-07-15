# Architecture

Cortex is a stateful research and retrieval platform built around four cooperating paths: research execution, routed chat, asynchronous resource ingestion, and session-scoped persistence.

## System context

```mermaid
flowchart TB
    user["Researcher"]

    subgraph cortex["Cortex application"]
        direction TB
        ui["React workspace"] --> api["FastAPI API and SSE streaming"]
        api --> research["LangGraph research workflow"]
        api --> chat["ReAct-lite chat routing"]
        api --> ingestion["Transactional ingestion outbox"]
    end

    user --> ui
    research --> ai["LLM, search, and retrieval providers"]
    chat --> ai
    research --> data["Supabase and Neo4j"]
    chat --> data
    ingestion --> jobs["Inngest workers"]
    jobs --> data

    api -. telemetry .-> observe["LangSmith and LangFuse"]
    jobs -. telemetry .-> observe
```

## Research execution

```mermaid
flowchart LR
    query["User query"] --> api["Research endpoint"]
    api --> search["Search"]
    search -->|"continue"| retrieve["Retrieve and parse"]
    search -->|"abort"| abort["Abort"]
    retrieve -->|"ok"| memory["GraphRAG context"]
    retrieve -->|"empty"| empty["Empty result"]
    memory --> rerank["Rerank"]
    rerank --> summarize["Summarize"]
    summarize --> report["Generate report"]
    report --> stream["SSE stream"]
    stream --> done["End"]
    abort --> done
    empty --> done
```

Search, retrieval, memory context, reranking, summarization, and report generation are separate graph nodes. The graph represents empty and aborted execution explicitly instead of collapsing every non-happy path into a generic exception.

## Chat routing

```mermaid
flowchart LR
    message["User message"] --> context["Available local context"]
    context --> router["Schema-validated ReAct-lite router"]
    router -->|"answer_direct"| direct["Direct answer"]
    router -->|"answer_from_rag"| rag["Grounded RAG answer"]
    router -->|"web_search"| web["Web search"]
    router -->|"fetch_url"| fetch["Fetch URL"]
    router -->|"ask_clarifying"| clarify["Clarifying question"]
```

Routing policy:

- Greetings and other social turns normally resolve directly.
- Weak or empty RAG context does not automatically trigger web search.
- Web search is selected for external, fresh, or otherwise web-dependent information.
- A URL in the message or history is available context, not an automatic fetch instruction.
- Direct URL fetching happens only when inspecting the resource is necessary.
- Agent chat, workspace chat, and streaming/non-streaming endpoints use the same policy.

The workspace-wide document collection is deny-by-default: Cortex retrieves it only when the router explicitly selects `answer_from_rag`. A custom agent's linked resources and session attachments are explicitly scoped resources, so they remain available on every turn. A routing failure after one structured-output repair returns `router_error` and does not retrieve documents.

## Reliable ingestion

```mermaid
flowchart LR
    upload["Upload resource"] --> api["Upload endpoint"]
    api --> tx["Atomic database RPC"]
    tx --> resource["Resource row"]
    tx --> job["Queued ingestion job"]
    tx --> event["Pending outbox event"]
    dispatcher["Outbox dispatcher"] --> claim["Claim event"]
    claim --> publish["Publish to Inngest"]
    publish --> worker["Idempotent ingestion worker"]
    worker --> ingest["Ingest signed storage URL"]
    ingest --> artifacts["GraphRAG artifacts"]
    claim -->|"send error"| retry["Backoff and retry"]
```

Resource creation, job creation, and the intent to publish are committed together. The dispatcher claims outbox rows before publishing to Inngest, and the worker claims a queued job before processing it. Retries therefore preserve the event intent without allowing duplicate workers to process the same terminal job.

## Persistence and isolation

- Supabase provides Postgres persistence, authentication, and object storage.
- Research sessions and retrieved context are scoped to the authenticated user.
- Neo4j stores document chunks, vector embeddings, and graph relationships used for retrieval.
- Redis accelerates auth, search, and session hot paths but degrades gracefully when unavailable.

## Key engineering decisions

### Transactional outbox instead of direct dispatch

A direct `database write → queue API call` creates a dual-write failure window: the process can crash after saving the job but before publishing its event. Cortex records the resource, job, and outbox intent in one Supabase transaction. A separate dispatcher publishes and retries independently, avoiding a distributed transaction between Postgres and Inngest.

### Model-directed routing instead of fixed heuristics

Whether a turn needs RAG, the web, a URL fetch, clarification, or no tool at all depends on intent and conversation context. A validated model decision keeps that policy consistent across chat surfaces. The optional fine-tuned router can offload this frequent classification step to a smaller model; see [Router fine-tuning](router-fine-tuning.md).

### Explicit graph outcomes instead of a linear pipeline

An empty search result is not the same as a crashed provider call. Explicit graph edges preserve the distinction so the API, UI, and telemetry can report accurate outcomes and recovery behavior.

## Related documentation

- [Getting started](getting-started.md)
- [Production deployment](deployment.md)
- [Observability](observability.md)
- [Model evaluation](model-evaluation.md)
- [UI design system](../ui/DESIGN.md)
