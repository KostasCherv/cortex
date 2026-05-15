"""LangGraph state definition for the research agent."""

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    """Shared mutable state passed between all graph nodes.

    Fields are optional (total=False) so nodes only need to populate
    the fields they produce.
    """

    # Input
    query: str                      # The original user research query

    # Search node output
    search_results: list[dict]      # Raw Tavily results [{url, title, content}, ...]

    # Retrieve node output
    retrieved_contents: list[dict]  # [{url, title, raw_text}, ...]

    # Rerank node output
    reranked_contents: list[dict]   # [{url, title, raw_text, score?}, ...]
    rerank_metadata: dict           # {fallback, reason?, model, input_count, output_count}

    # Summarize node output
    summaries: list[dict]           # [{url, title, summary}, ...]

    # Final report node output
    report: str                     # Final markdown report
    report_metadata: dict           # {title, sources, generated_at}

    # Session context (populated by API layer when running inside a session)
    session_id: str | None          # Parent session ID
    run_id: str | None              # This run's unique ID
    user_id: str | None             # Authenticated user ID when available
    active_source_urls: list[str]   # URLs retrieved in this run
    conversation_history: list[dict] # [{role, content, run_id}] prior turns
    langfuse_trace_id: str | None
    langfuse_observation_id: str | None

    # Control flow
    error: str | None               # Set on unrecoverable errors
    use_vector_store: bool          # Deprecated compatibility flag (graph-first mode ignores it)

    memory_context: str | None      # Memory context for the LLM
    graph_context: str | None       # Graph-aware context block for prompting
    graph_chunks: list[dict]        # Structured graph chunk hits for citations/debug
    graph_entities: list[str]       # Expanded entity set used during retrieval
