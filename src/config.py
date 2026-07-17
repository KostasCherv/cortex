"""Application settings loaded from environment variables."""

import json

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_bundled_secret(
    raw: str,
    config_name: str,
    required: tuple[str, ...],
) -> dict[str, str]:
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_name} must be valid JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object.")

    values = {name: config.get(name) for name in required}
    if any(not isinstance(value, str) or not value for value in values.values()):
        fields = ", ".join(required)
        raise ValueError(f"{config_name} must include non-empty {fields} values.")
    return values


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(default="openai", description="LLM provider: 'ollama' or 'openai'")
    provider_config_json: str = Field(
        default="",
        description="JSON object containing provider, cache, and Langfuse settings for production.",
    )
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", description="OpenRouter model slug")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", description="LM Studio base URL"
    )
    lmstudio_model: str = Field(default="model", description="LM Studio model name")

    # Router LLM (always-on action classification; defaults to the main LLM)
    router_llm_provider: str = Field(
        default="",
        description="Router LLM provider override: 'ollama'|'openai'|'openrouter'|'lmstudio'. Empty falls back to llm_provider.",
    )
    router_openai_model: str = Field(
        default="",
        description="OpenAI model override for the router. Empty falls back to openai_model.",
    )
    router_openrouter_model: str = Field(
        default="",
        description="OpenRouter model slug override for the router. Empty falls back to openrouter_model.",
    )
    router_ollama_model: str = Field(
        default="",
        description="Ollama model tag override for the router. Empty falls back to ollama_model.",
    )
    router_ollama_base_url: str = Field(
        default="",
        description="Ollama base URL override for the router. Empty falls back to ollama_base_url.",
    )
    router_lmstudio_model: str = Field(
        default="",
        description="LM Studio model override for the router. Empty falls back to lmstudio_model.",
    )
    router_temperature: float = Field(
        default=0.0, description="Sampling temperature for the router LLM."
    )
    router_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Maximum duration of each router LLM request.",
    )
    router_prompt_path: str = Field(
        default="",
        description=(
            "Path to an optimized router prompt artifact (JSON with a 'system_prompt' key), "
            "e.g. optimized_prompts/router_action_optimized.json. Empty uses the built-in prompt."
        ),
    )
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
        description="Expected embedding vector size for Neo4j vector indexes.",
    )

    # Search
    tavily_api_key: str = Field(default="", description="Tavily search API key")
    max_search_results: int = Field(default=5, description="Max Tavily search results")
    web_search_provider: str = Field(
        default="tavily",
        description="Active web search provider for agent chat tool calls.",
    )
    arxiv_mcp_storage_path: str = Field(
        default="~/.arxiv-mcp-server/papers",
        description="Local cache directory for arxiv-mcp-server paper downloads.",
    )
    arxiv_mcp_command: str = Field(
        default="",
        description="Optional override path to the arxiv-mcp-server executable.",
    )
    # Composio
    composio_api_key: str = Field(
        default="",
        description="Composio API key for the service account.",
    )
    composio_user_id: str = Field(
        default="default",
        description="Composio user ID whose connected accounts are used for tool calls.",
    )
    composio_enabled: bool = Field(
        default=True,
        description="Enable Composio tool integration.",
    )
    composio_apps: list[str] | str = Field(
        default=[],
        description="Comma-separated allowlist of Composio app names to expose. Empty = all connected apps.",
    )
    composio_tool_refresh_seconds: int = Field(
        default=3600,
        description="Seconds to cache Composio Tool Router sessions per user before refresh.",
    )
    composio_max_agent_turns: int = Field(
        default=5,
        description="Maximum agent loop turns before returning the last response.",
    )
    rag_chat_max_history_messages: int = Field(
        default=20,
        description="Maximum prior chat messages included in the agent prompt.",
    )
    rag_chat_conditional_tools: bool = Field(
        default=True,
        description="When true, still binds tools for all messages if apps are connected (linked docs never disable Composio).",
    )
    rag_suggestions_deferred: bool = Field(
        default=False,
        description="Generate follow-up suggestions in the background instead of blocking the response.",
    )
    rag_chat_title_llm_background: bool = Field(
        default=True,
        description="Use a fast fallback session title immediately; optional LLM title upgrade runs in background.",
    )
    rag_perf_headers: bool = Field(
        default=False,
        description="Include X-Rag-Perf JSON timing header on RAG chat responses.",
    )

    @field_validator("composio_apps", mode="before")
    @classmethod
    def parse_composio_apps(cls, v: object) -> object:
        if isinstance(v, str):
            return [app.strip() for app in v.split(",") if app.strip()]
        return v

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
    graph_rag_min_cosine_score: float = Field(
        default=0.15,
        description="Minimum chunk cosine similarity before graph score fusion.",
    )
    cohere_api_key: str = Field(default="", description="Cohere API key for reranking.")
    rerank_top_k: int = Field(
        default=5,
        description="Maximum RAG chunks to keep after cross-encoder reranking.",
    )
    rerank_relevance_threshold: float = Field(
        default=0.1,
        description="Minimum reranker relevance score required for a citation.",
    )
    rerank_timeout_seconds: float = Field(
        default=10.0,
        description="HTTP timeout in seconds for Cohere reranking requests.",
    )

    # API
    api_host: str = Field(default="0.0.0.0")  # nosec B104 — intentional; containerised service binds all interfaces
    api_port: int = Field(default=8010)
    app_log_level: str = Field(
        default="INFO",
        description="Application logger level for Cortex loggers.",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Comma-separated in env: CORS_ORIGINS=https://app.example.com,https://staging.example.com",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    enforce_session_auth: bool = Field(
        default=True,
        description="Require authentication for session endpoints.",
    )
    rate_limit_default: str = Field(
        default="60/minute", description="Default per-IP rate limit for API endpoints"
    )
    internal_dispatch_secret: str = Field(
        default="",
        description="Secret token required to call POST /internal/dispatch-outbox. Set a strong random value in prod.",
    )
    sentry_dsn: str = Field(
        default="", description="Sentry DSN for error tracking; empty disables it"
    )
    readiness_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        description="Maximum duration of each live dependency readiness check.",
    )
    readiness_require_supabase: bool = Field(
        default=False,
        description="Return unready when Supabase is missing or unreachable.",
    )
    readiness_require_neo4j: bool = Field(
        default=False,
        description="Return unready when Neo4j is missing or unreachable.",
    )

    # Supabase
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SUPABASE_SECRET_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ),
        description=(
            "Supabase secret API key (sb_secret_...) for backend PostgREST/Storage. "
            "Legacy service_role JWT is still accepted via SUPABASE_SERVICE_ROLE_KEY."
        ),
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
        default="cortex",
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
    billing_config_json: str = Field(
        default="",
        description=(
            "JSON object containing Stripe secret_key, webhook_secret, and pro_price_id. "
            "Used in production to keep billing credentials in one secret version."
        ),
    )
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

    @model_validator(mode="after")
    def apply_provider_config(self) -> "Settings":
        """Load bundled provider/cache credentials while preserving local split-variable support."""
        if not self.provider_config_json.strip():
            return self
        values = _parse_bundled_secret(
            self.provider_config_json,
            "PROVIDER_CONFIG_JSON",
            ("openai_api_key", "tavily_api_key"),
        )
        for name, value in values.items():
            setattr(self, name, value)
        redis_url = json.loads(self.provider_config_json).get("redis_url")
        if redis_url is not None:
            if not isinstance(redis_url, str) or not redis_url:
                raise ValueError("PROVIDER_CONFIG_JSON redis_url must be a non-empty string.")
            self.redis_url = redis_url
        langfuse = json.loads(self.provider_config_json)
        langfuse_fields = {
            "langfuse_public_key": "langfuse_public_key",
            "langfuse_secret_key": "langfuse_secret_key",
            "langfuse_base_url": "langfuse_host",
        }
        for source, target in langfuse_fields.items():
            value = langfuse.get(source)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"PROVIDER_CONFIG_JSON {source} must be a non-empty string.")
                setattr(self, target, value)
        return self

    @model_validator(mode="after")
    def apply_billing_config(self) -> "Settings":
        """Load bundled Stripe credentials while preserving local split-variable support."""
        if not self.billing_config_json.strip():
            return self
        values = _parse_bundled_secret(
            self.billing_config_json,
            "BILLING_CONFIG_JSON",
            ("stripe_secret_key", "stripe_webhook_secret", "stripe_pro_price_id"),
        )
        for name, value in values.items():
            setattr(self, name, value)
        return self

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
