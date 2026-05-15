"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(
        default="openai", description="LLM provider: 'ollama' or 'openai'"
    )
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini", description="OpenRouter model slug"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: 'ollama' or 'openai'",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the embedding provider when required (for example Ollama).",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Expected embedding vector size for the configured Pinecone index.",
    )

    # Search
    tavily_api_key: str = Field(default="", description="Tavily search API key")
    max_search_results: int = Field(default=5, description="Max Tavily search results")
    web_search_provider: str = Field(
        default="tavily",
        description="Active web search provider for agent chat tool calls.",
    )

    # Vector store (Pinecone)
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index_name: str = Field(
        default="research-agent", description="Pinecone index name"
    )

    # Graph store (Neo4j)
    neo4j_uri: str = Field(
        default="",
        description="Neo4j Bolt URI (e.g. neo4j://localhost:7687).",
    )
    neo4j_username: str = Field(
        default="",
        description="Neo4j username.",
    )
    neo4j_password: str = Field(
        default="",
        description="Neo4j password.",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name.",
    )
    graph_rag_top_k: int = Field(
        default=8,
        description="Top chunks to keep after graph-aware score fusion.",
    )
    graph_rag_max_hops: int = Field(
        default=2,
        description="Max RELATES_TO hop expansion during graph retrieval (1-2).",
    )

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Comma-separated in env: CORS_ORIGINS=https://app.example.com,https://staging.example.com",
    )
    enforce_session_auth: bool = Field(
        default=True,
        description="Require authentication for session endpoints.",
    )

    # Supabase
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_service_role_key: str = Field(
        default="",
        description="Supabase service role key used by backend for PostgREST.",
    )
    supabase_jwks_url: str = Field(
        default="",
        description="Supabase Auth JWKS URL for JWT verification.",
    )
    supabase_jwt_audience: str = Field(
        default="authenticated",
        description="Expected JWT audience for Supabase access tokens.",
    )
    supabase_jwt_secret: str = Field(
        default="",
        description="Supabase JWT secret (used only for HS256 token verification fallback).",
    )

    # Observability (LangSmith)
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing for workflow and node spans.",
    )
    langsmith_project: str = Field(
        default="research-agent",
        description="LangSmith project name for traced runs.",
    )
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint.",
    )
    langsmith_redaction_mode: str = Field(
        default="redacted_default",
        description="Trace payload policy: full_payloads|redacted_default|metadata_only",
    )
    langsmith_sampling_rate: float = Field(
        default=1.0,
        description="Fraction of runs to trace (0.0 to 1.0).",
    )

    # Observability (LangFuse)
    langfuse_enabled: bool = Field(
        default=True,
        description="Enable LangFuse generation tracing and scoring.",
    )
    langfuse_public_key: str = Field(default="", description="LangFuse public key")
    langfuse_secret_key: str = Field(default="", description="LangFuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="LangFuse base URL.",
    )
    langfuse_release: str = Field(
        default="",
        description="Optional application release/version attached to LangFuse traces.",
    )
    langfuse_env: str = Field(
        default="",
        description="Optional LangFuse environment label (for example dev, staging, prod).",
    )

    # RAG Agent
    rag_max_file_size_mb: int = Field(
        default=25,
        description="Maximum upload size per RAG resource file in megabytes.",
    )
    rag_max_resources_per_workspace: int = Field(
        default=500,
        description="Maximum number of RAG resources allowed per workspace.",
    )
    rag_max_resources_per_agent: int = Field(
        default=25,
        description="Maximum number of linked resources allowed per RAG agent.",
    )
    rag_storage_bucket: str = Field(
        default="rag-resources",
        description="Supabase Storage bucket used for raw RAG resource files.",
    )
    rag_signed_url_ttl_seconds: int = Field(
        default=600,
        description="TTL in seconds for Supabase signed download URLs passed to sidecar.",
    )

    # Billing / Stripe
    stripe_secret_key: str = Field(default="", description="Stripe secret key.")
    stripe_webhook_secret: str = Field(default="", description="Stripe webhook signing secret.")
    stripe_pro_price_id: str = Field(default="", description="Stripe price id for the pro plan.")
    stripe_success_url: str = Field(
        default="http://localhost:5173/billing/success",
        description="Redirect URL after successful checkout.",
    )
    stripe_cancel_url: str = Field(
        default="http://localhost:5173/billing/cancel",
        description="Redirect URL after canceled checkout.",
    )
    stripe_portal_return_url: str = Field(
        default="http://localhost:5173",
        description="Return URL from Stripe customer portal.",
    )

    # Redis cache
    redis_url: str = Field(
        default="",
        description="Redis connection URL (e.g. redis://localhost:6379/0). Leave empty to disable caching.",
    )
    redis_cache_ttl_db_list_seconds: int = Field(
        default=30,
        description="TTL in seconds for cached DB-heavy list/read-all responses.",
    )
    redis_cache_ttl_auth_seconds: int = Field(
        default=300,
        description="TTL in seconds for cached auth userinfo lookups.",
    )
    redis_cache_ttl_search_seconds: int = Field(
        default=1800,
        description="TTL in seconds for cached web search responses.",
    )


settings = Settings()
