import os

import pytest

# Must be set before any src imports so inngest.Inngest initialises in dev mode.
os.environ.setdefault("INNGEST_DEV", "1")


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset provider singletons between tests to prevent state leaks."""
    yield
    from src.db import provider
    provider._session_store = None
    provider._storage_adapter = None
