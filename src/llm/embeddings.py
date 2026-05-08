"""Embedding provider abstraction for vector-store operations."""

from __future__ import annotations

from src.config import settings
from src.errors import ConfigurationError, VectorStoreError


class EmbeddingClient:
    """Embed text batches using the configured provider."""

    def __init__(self, embed_model: object | None = None) -> None:
        self._embed_model = embed_model

    def _get_embed_model(self):
        if self._embed_model is not None:
            return self._embed_model

        provider = settings.embedding_provider.lower()

        if provider == "openai":
            if not settings.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

            from llama_index.embeddings.openai import OpenAIEmbedding

            self._embed_model = OpenAIEmbedding(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
            return self._embed_model

        if provider == "ollama":
            base_url = settings.embedding_base_url.rstrip("/")
            if not base_url:
                raise ConfigurationError(
                    "EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=ollama."
                )

            from llama_index.embeddings.ollama import OllamaEmbedding

            self._embed_model = OllamaEmbedding(
                model_name=settings.embedding_model,
                base_url=base_url,
            )
            return self._embed_model

        raise ConfigurationError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Choose 'openai' or 'ollama'."
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for *texts* using the configured embedding provider."""
        try:
            model = self._get_embed_model()
            return model.get_text_embedding_batch(texts)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Embedding request failed: {exc}") from exc
