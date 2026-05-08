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
            getattr(settings, "langsmith_project", "research-agent")
        )
        langsmith_api_key = getattr(settings, "langsmith_api_key", "")
        if langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = str(langsmith_api_key)
        langsmith_endpoint = getattr(settings, "langsmith_endpoint", "")
        if langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = str(langsmith_endpoint)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            **common_kwargs,
        )

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            temperature=temperature,
            api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
            base_url="https://openrouter.ai/api/v1",
            **common_kwargs,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            **common_kwargs,
        )

    raise ConfigurationError(
        f"Unknown LLM_PROVIDER '{provider}'. Choose 'openai', 'ollama', or 'openrouter'."
    )
