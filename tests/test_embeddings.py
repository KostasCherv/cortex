"""Tests for embedding provider selection and requests."""

from unittest.mock import MagicMock, patch

import pytest

from src.errors import ConfigurationError, VectorStoreError
from src.llm.embeddings import EmbeddingClient


def test_embed_texts_openai_returns_vectors():
    with patch("src.llm.embeddings.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.openai_api_key = "sk-test"

        embed_model = MagicMock()
        embed_model.get_text_embedding_batch.return_value = [[0.1, 0.2]]

        client = EmbeddingClient(embed_model=embed_model)
        vectors = client.embed_texts(["hello"])

        assert vectors == [[0.1, 0.2]]
        embed_model.get_text_embedding_batch.assert_called_once_with(["hello"])


def test_embed_texts_openai_requires_api_key():
    with patch("src.llm.embeddings.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.openai_api_key = ""

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            EmbeddingClient().embed_texts(["hello"])


def test_embed_texts_ollama_requires_base_url():
    with patch("src.llm.embeddings.settings") as mock_settings:
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "nomic-embed-text"
        mock_settings.embedding_base_url = ""

        with pytest.raises(ConfigurationError, match="EMBEDDING_BASE_URL"):
            EmbeddingClient().embed_texts(["hello"])


def test_embed_texts_wraps_model_errors():
    embed_model = MagicMock()
    embed_model.get_text_embedding_batch.side_effect = RuntimeError("connection refused")

    with pytest.raises(VectorStoreError, match="connection refused"):
        EmbeddingClient(embed_model=embed_model).embed_texts(["hello"])


def test_embed_texts_unknown_provider_raises_configuration_error():
    with patch("src.llm.embeddings.settings") as mock_settings:
        mock_settings.embedding_provider = "wat"

        with pytest.raises(ConfigurationError, match="Unknown EMBEDDING_PROVIDER"):
            EmbeddingClient().embed_texts(["hello"])
