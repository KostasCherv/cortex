"""LLM factory — returns the configured chat model (Ollama, OpenAI, or OpenRouter)."""

import os

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.errors import ConfigurationError
from src.observability.context import build_trace_metadata, build_trace_tags


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return a chat model based on the configured LLM_PROVIDER.

    Args:
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ConfigurationError: If the provider is unknown or required keys are missing.
    """
    provider = settings.llm_provider.lower()
    common_kwargs = {
        "tags": build_trace_tags(["llm"]),
        "metadata": build_trace_metadata({"provider": provider}),
    }

    if getattr(settings, "langsmith_tracing", False) is True:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = str(
            getattr(settings, "langsmith_project", "cortex")
        )
        langsmith_api_key = getattr(settings, "langsmith_api_key", "")
        if langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = str(langsmith_api_key)
        langsmith_endpoint = getattr(settings, "langsmith_endpoint", "")
        if langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = str(langsmith_endpoint)

    return _build_chat_model(
        provider,
        model_name="",
        base_url="",
        temperature=temperature,
        common_kwargs=common_kwargs,
    )


def _build_chat_model(
    provider: str,
    *,
    model_name: str,
    base_url: str,
    temperature: float,
    common_kwargs: dict,
) -> BaseChatModel:
    """Construct a chat model for the given provider.

    model_name/base_url override the provider's default settings field when
    non-empty (used by get_router_llm's fallback resolution); get_llm always
    passes empty strings so it keeps reading directly from settings.
    """
    if provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            timeout=60,
            **common_kwargs,
        )

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or settings.openrouter_model,
            temperature=temperature,
            api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
            base_url="https://openrouter.ai/api/v1",
            timeout=60,
            **common_kwargs,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or settings.ollama_model,
            temperature=temperature,
            base_url=base_url or settings.ollama_base_url,
            **common_kwargs,
        )

    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or settings.lmstudio_model,
            temperature=temperature,
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",
            timeout=60,
            **common_kwargs,
        )

    raise ConfigurationError(
        f"Unknown LLM_PROVIDER '{provider}'. Choose 'openai', 'ollama', 'openrouter', or 'lmstudio'."
    )


def get_router_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a chat model for the fast action-routing decision.

    Resolves router-specific settings (router_llm_provider, router_*_model,
    router_ollama_base_url) and falls back to the main llm_provider config
    for any field left unset, so the router can be introduced without
    requiring new environment variables.
    """
    provider = (settings.router_llm_provider or settings.llm_provider).lower()
    resolved_temperature = (
        temperature if temperature is not None else settings.router_temperature
    )
    common_kwargs = {
        "tags": build_trace_tags(["llm", "router"]),
        "metadata": build_trace_metadata({"provider": provider, "role": "router"}),
    }

    model_name = {
        "openai": settings.router_openai_model,
        "openrouter": settings.router_openrouter_model,
        "ollama": settings.router_ollama_model,
        "lmstudio": settings.router_lmstudio_model,
    }.get(provider, "")

    return _build_chat_model(
        provider,
        model_name=model_name,
        base_url=settings.router_ollama_base_url,
        temperature=resolved_temperature,
        common_kwargs=common_kwargs,
    )
