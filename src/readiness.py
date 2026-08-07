"""Bounded dependency checks for the API readiness probe."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from neo4j import GraphDatabase

from src.cache.client import get_cache
from src.config import settings
from src.supabase_keys import supabase_api_headers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadinessCheck:
    status: str
    critical: bool


def _configured(value: str) -> bool:
    return bool(value.strip())


def _llm_configuration_check() -> ReadinessCheck:
    provider = settings.llm_provider.strip().lower()
    configured = {
        "openai": _configured(settings.openai_api_key),
        "openrouter": _configured(settings.openrouter_api_key),
        "ollama": _configured(settings.ollama_base_url) and _configured(settings.ollama_model),
        "lmstudio": _configured(settings.lmstudio_base_url)
        and _configured(settings.lmstudio_model),
    }.get(provider, False)
    return ReadinessCheck(status="ok" if configured else "misconfigured", critical=True)


async def _supabase_check() -> ReadinessCheck:
    configured = _configured(settings.supabase_url) and _configured(settings.supabase_secret_key)
    if not configured:
        status = "missing" if settings.readiness_require_supabase else "disabled"
        return ReadinessCheck(status=status, critical=settings.readiness_require_supabase)

    headers = supabase_api_headers(settings.supabase_secret_key)
    try:
        timeout = httpx.Timeout(settings.readiness_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{settings.supabase_url.rstrip('/')}/rest/v1/",
                headers=headers,
            )
        response.raise_for_status()
        return ReadinessCheck(status="ok", critical=settings.readiness_require_supabase)
    except Exception:
        return ReadinessCheck(status="unavailable", critical=settings.readiness_require_supabase)


_neo4j_readiness_driver = None


def _get_neo4j_readiness_driver():
    # Reused across checks (the driver owns a pooled connection) instead of
    # opening a fresh Bolt handshake every 10s — that churn was tripping
    # Aura's connection limits and causing intermittent "defunct connection"
    # failures even while the instance itself was healthy.
    global _neo4j_readiness_driver
    if _neo4j_readiness_driver is None:
        _neo4j_readiness_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            notifications_min_severity="OFF",
        )
    return _neo4j_readiness_driver


def _verify_neo4j_connectivity() -> None:
    # Run a real query (not just verify_connectivity) so scheduled /ready
    # pings count as activity for AuraDB Free's 72h auto-pause timer.
    _get_neo4j_readiness_driver().execute_query(
        "RETURN 1",
        database_=settings.neo4j_database or None,
    )


async def _neo4j_check() -> ReadinessCheck:
    configured = all(
        _configured(value)
        for value in (settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    )
    if not configured:
        status = "missing" if settings.readiness_require_neo4j else "disabled"
        return ReadinessCheck(status=status, critical=settings.readiness_require_neo4j)

    try:
        await asyncio.wait_for(
            asyncio.to_thread(_verify_neo4j_connectivity),
            timeout=settings.readiness_timeout_seconds,
        )
        return ReadinessCheck(status="ok", critical=settings.readiness_require_neo4j)
    except Exception:
        logger.warning("Neo4j readiness check failed", exc_info=True)
        return ReadinessCheck(status="unavailable", critical=settings.readiness_require_neo4j)


async def _redis_check() -> ReadinessCheck:
    cache = get_cache()
    if cache is None:
        return ReadinessCheck(status="disabled", critical=False)
    try:
        reachable = await asyncio.wait_for(cache.ping(), timeout=settings.readiness_timeout_seconds)
    except Exception:
        reachable = False
    return ReadinessCheck(status="ok" if reachable else "unavailable", critical=False)


async def run_readiness_checks() -> dict[str, ReadinessCheck]:
    """Check required configuration and live dependencies concurrently."""
    checks: dict[str, ReadinessCheck] = {"llm_provider": _llm_configuration_check()}
    probes: dict[str, Callable[[], Awaitable[ReadinessCheck]]] = {
        "supabase": _supabase_check,
        "neo4j": _neo4j_check,
        "redis": _redis_check,
    }
    results = await asyncio.gather(*(probe() for probe in probes.values()))
    checks.update(zip(probes, results, strict=True))
    return checks
