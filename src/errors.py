"""Custom exceptions for Cortex."""


class CortexError(Exception):
    """Base exception for all Cortex errors."""


class SearchError(CortexError):
    """Raised when the Tavily search fails after all retries."""


class FetchError(CortexError):
    """Raised when URL content fetching fails."""


class LLMError(CortexError):
    """Raised when an LLM call fails."""


class StructuredOutputError(CortexError):
    """Raised when structured LLM output parsing or validation fails."""


class StructuredOutputParseError(StructuredOutputError):
    """Raised when structured LLM output cannot be decoded as JSON."""


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured LLM output does not match the expected schema."""


class VectorStoreError(CortexError):
    """Raised when a vector store operation fails."""


class ConfigurationError(CortexError):
    """Raised when required configuration is missing or invalid."""


class CacheError(CortexError):
    """Raised when a Redis cache operation fails."""
