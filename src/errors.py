"""Custom exceptions for the research agent."""


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""


class SearchError(ResearchAgentError):
    """Raised when the Tavily search fails after all retries."""


class FetchError(ResearchAgentError):
    """Raised when URL content fetching fails."""


class LLMError(ResearchAgentError):
    """Raised when an LLM call fails."""


class VectorStoreError(ResearchAgentError):
    """Raised when a vector store operation fails."""


class ConfigurationError(ResearchAgentError):
    """Raised when required configuration is missing or invalid."""


class CacheError(ResearchAgentError):
    """Raised when a Redis cache operation fails."""
