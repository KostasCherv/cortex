"""Unit tests for bounded dependency readiness checks."""

from unittest.mock import AsyncMock, patch

import pytest

from src import readiness


@pytest.mark.asyncio
async def test_checks_disabled_optional_dependencies_without_network_calls():
    with (
        patch.object(readiness.settings, "llm_provider", "ollama"),
        patch.object(readiness.settings, "ollama_base_url", "http://localhost:11434"),
        patch.object(readiness.settings, "ollama_model", "llama3.2"),
        patch.object(readiness.settings, "supabase_url", ""),
        patch.object(readiness.settings, "supabase_secret_key", ""),
        patch.object(readiness.settings, "neo4j_uri", ""),
        patch.object(readiness.settings, "neo4j_username", ""),
        patch.object(readiness.settings, "neo4j_password", ""),
        patch.object(readiness.settings, "readiness_require_supabase", False),
        patch.object(readiness.settings, "readiness_require_neo4j", False),
        patch("src.readiness.get_cache", return_value=None),
    ):
        checks = await readiness.run_readiness_checks()

    assert checks["llm_provider"].status == "ok"
    assert checks["supabase"].status == "disabled"
    assert checks["neo4j"].status == "disabled"
    assert checks["redis"].status == "disabled"


@pytest.mark.asyncio
async def test_required_missing_dependencies_are_critical():
    with (
        patch.object(readiness.settings, "llm_provider", "openai"),
        patch.object(readiness.settings, "openai_api_key", ""),
        patch.object(readiness.settings, "supabase_url", ""),
        patch.object(readiness.settings, "supabase_secret_key", ""),
        patch.object(readiness.settings, "neo4j_uri", ""),
        patch.object(readiness.settings, "neo4j_username", ""),
        patch.object(readiness.settings, "neo4j_password", ""),
        patch.object(readiness.settings, "readiness_require_supabase", True),
        patch.object(readiness.settings, "readiness_require_neo4j", True),
        patch("src.readiness.get_cache", return_value=None),
    ):
        checks = await readiness.run_readiness_checks()

    assert checks["llm_provider"].status == "misconfigured"
    assert checks["supabase"].status == "missing"
    assert checks["supabase"].critical is True
    assert checks["neo4j"].status == "missing"
    assert checks["neo4j"].critical is True


@pytest.mark.asyncio
async def test_unreachable_redis_is_optional_and_degraded():
    cache = AsyncMock()
    cache.ping.return_value = False
    with (
        patch.object(readiness.settings, "llm_provider", "ollama"),
        patch.object(readiness.settings, "ollama_base_url", "http://localhost:11434"),
        patch.object(readiness.settings, "ollama_model", "llama3.2"),
        patch.object(readiness.settings, "supabase_url", ""),
        patch.object(readiness.settings, "supabase_secret_key", ""),
        patch.object(readiness.settings, "neo4j_uri", ""),
        patch.object(readiness.settings, "neo4j_username", ""),
        patch.object(readiness.settings, "neo4j_password", ""),
        patch.object(readiness.settings, "readiness_require_supabase", False),
        patch.object(readiness.settings, "readiness_require_neo4j", False),
        patch("src.readiness.get_cache", return_value=cache),
    ):
        checks = await readiness.run_readiness_checks()

    assert checks["redis"].status == "unavailable"
    assert checks["redis"].critical is False
