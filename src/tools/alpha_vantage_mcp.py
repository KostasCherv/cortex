"""Alpha Vantage MCP-backed asset pricing tool with optional caching."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from src.cache.client import get_cache
from src.config import settings
from src.errors import AssetPriceError, McpClientError
from src.tools.alpha_vantage_mcp_client import (
    alpha_vantage_mcp_url,
    get_alpha_vantage_mcp_client,
)

_FIAT_CODES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
_ALPHA_VANTAGE_DOCS_URL = "https://www.alphavantage.co/documentation/"


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    raw_symbols = [symbols] if isinstance(symbols, str) else list(symbols)
    normalized = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    if not normalized:
        raise AssetPriceError("At least one asset symbol is required.")
    return normalized


def _alpha_vantage_mcp_url() -> str:
    return alpha_vantage_mcp_url()


def validate_alpha_vantage_mcp_configuration() -> None:
    """Validate Alpha Vantage MCP configuration without consuming market-data quota."""
    _alpha_vantage_mcp_url()


def _cache_key_for_symbols(symbols: Sequence[str], currency: str | None) -> str:
    normalized_currency = (currency or "").strip().upper()
    normalized_symbols = ",".join(_normalize_symbols(symbols))
    return f"{normalized_symbols}|{normalized_currency}"


async def _call_alpha_vantage_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = await get_alpha_vantage_mcp_client().call_tool(tool_name, arguments)
    if not isinstance(payload, dict):
        raise AssetPriceError(f"Alpha Vantage MCP returned an unexpected payload for {tool_name}.")
    return payload


def _build_request_candidates(
    symbol: str,
    currency: str | None,
) -> list[tuple[str, dict[str, Any], str, str]]:
    requested = symbol.strip().upper()
    target_currency = (currency or "").strip().upper()

    if requested.endswith("=F") or requested.startswith("^"):
        raise AssetPriceError(
            f"Alpha Vantage MCP v1 does not support generic quote routing for '{requested}'."
        )

    pair_candidate = requested.replace("=X", "").replace("-", "").replace("/", "")
    candidates: list[tuple[str, dict[str, Any], str, str]] = []

    if len(pair_candidate) == 6 and pair_candidate.isalpha():
        base, quote = pair_candidate[:3], pair_candidate[3:]
        if quote in _FIAT_CODES:
            asset_type = "forex" if base in _FIAT_CODES else "crypto"
            candidates.append(
                (
                    "CURRENCY_EXCHANGE_RATE",
                    {
                        "from_currency": base,
                        "to_currency": target_currency or quote,
                        "datatype": "json",
                        "return_full_data": True,
                    },
                    asset_type,
                    f"{base}{target_currency or quote}",
                )
            )

    candidates.append(
        (
            "GLOBAL_QUOTE",
            {
                "symbol": requested,
                "datatype": "json",
                "return_full_data": True,
            },
            "equity",
            requested,
        )
    )

    if not target_currency and len(requested) <= 10 and requested.isalpha():
        target_currency = "USD"

    if target_currency and requested.isalpha():
        asset_type = "forex" if requested in _FIAT_CODES else "crypto"
        exchange_candidate = (
            "CURRENCY_EXCHANGE_RATE",
            {
                "from_currency": requested,
                "to_currency": target_currency,
                "datatype": "json",
                "return_full_data": True,
            },
            asset_type,
            f"{requested}{target_currency}",
        )
        if exchange_candidate not in candidates:
            candidates.append(exchange_candidate)

    return candidates


def _normalize_global_quote(
    *,
    requested_symbol: str,
    payload: dict[str, Any],
    asset_type: str,
) -> dict[str, Any]:
    quote = payload.get("Global Quote") or payload.get("global quote") or payload
    if not isinstance(quote, dict):
        raise AssetPriceError(f"Alpha Vantage quote payload for {requested_symbol} was malformed.")

    price = quote.get("05. price") or quote.get("price")
    previous_close = quote.get("08. previous close")
    latest_day = quote.get("07. latest trading day") or quote.get("latest trading day") or ""
    if price is None:
        raise AssetPriceError(f"Alpha Vantage quote payload for {requested_symbol} was missing price.")

    return {
        "symbol": requested_symbol,
        "title": f"Alpha Vantage MCP · {requested_symbol}",
        "url": _ALPHA_VANTAGE_DOCS_URL,
        "content": (
            f"Source: Alpha Vantage MCP (GLOBAL_QUOTE)\n"
            f"{requested_symbol} price {float(price)} USD as of {latest_day}"
        ).strip(),
        "asset_type": asset_type,
        "name": requested_symbol,
        "price": float(price),
        "currency": "USD",
        "as_of": str(latest_day),
        "source": "alphavantage_mcp",
        "raw": {
            "resolved_symbol": quote.get("01. symbol") or requested_symbol,
            "previous_close": previous_close or "",
            "change_percent": quote.get("10. change percent") or "",
        },
    }


def _normalize_exchange_rate(
    *,
    requested_symbol: str,
    payload: dict[str, Any],
    asset_type: str,
) -> dict[str, Any]:
    quote = (
        payload.get("Realtime Currency Exchange Rate")
        or payload.get("realtime currency exchange rate")
        or payload
    )
    if not isinstance(quote, dict):
        raise AssetPriceError(
            f"Alpha Vantage exchange-rate payload for {requested_symbol} was malformed."
        )

    from_code = str(quote.get("1. From_Currency Code") or quote.get("from_currency") or "").upper()
    to_code = str(quote.get("3. To_Currency Code") or quote.get("to_currency") or "").upper()
    rate = quote.get("5. Exchange Rate") or quote.get("exchange_rate")
    as_of = quote.get("6. Last Refreshed") or quote.get("last_refreshed") or ""
    if rate is None:
        raise AssetPriceError(
            f"Alpha Vantage exchange-rate payload for {requested_symbol} was missing rate."
        )

    return {
        "symbol": requested_symbol,
        "title": f"Alpha Vantage MCP · {from_code}/{to_code}" if from_code and to_code else "Alpha Vantage MCP",
        "url": _ALPHA_VANTAGE_DOCS_URL,
        "content": (
            f"Source: Alpha Vantage MCP (CURRENCY_EXCHANGE_RATE)\n"
            f"{requested_symbol} price {float(rate)} {to_code} as of {as_of}"
        ).strip(),
        "asset_type": asset_type,
        "name": f"{from_code}/{to_code}" if from_code and to_code else requested_symbol,
        "price": float(rate),
        "currency": to_code,
        "as_of": str(as_of),
        "source": "alphavantage_mcp",
        "raw": {
            "resolved_symbol": f"{from_code}{to_code}" if from_code and to_code else requested_symbol,
            "from_currency_name": quote.get("2. From_Currency Name") or "",
            "to_currency_name": quote.get("4. To_Currency Name") or "",
            "time_zone": quote.get("7. Time Zone") or "",
        },
    }


async def _get_alpha_vantage_prices_uncached(
    symbols: str | Sequence[str],
    currency: str | None = None,
) -> list[dict[str, Any]]:
    normalized_symbols = _normalize_symbols(symbols)
    quotes: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        last_exc: AssetPriceError | None = None
        for tool_name, arguments, asset_type, requested_symbol in _build_request_candidates(
            symbol, currency
        ):
            try:
                payload = await _call_alpha_vantage_tool(tool_name, arguments)
                if tool_name == "GLOBAL_QUOTE":
                    quotes.append(
                        _normalize_global_quote(
                            requested_symbol=requested_symbol,
                            payload=payload,
                            asset_type=asset_type,
                        )
                    )
                else:
                    quotes.append(
                        _normalize_exchange_rate(
                            requested_symbol=requested_symbol,
                            payload=payload,
                            asset_type=asset_type,
                        )
                    )
                break
            except (AssetPriceError, McpClientError) as exc:
                last_exc = AssetPriceError(str(exc))
        else:
            raise AssetPriceError(
                f"Alpha Vantage MCP could not resolve a quote for '{symbol}'."
            ) from last_exc
    return quotes


async def get_alpha_vantage_prices_cached(
    symbols: str | Sequence[str],
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Cached async wrapper around Alpha Vantage MCP price retrieval."""
    normalized_symbols = _normalize_symbols(symbols)
    cache = get_cache()
    if cache is None:
        return await _get_alpha_vantage_prices_uncached(normalized_symbols, currency)

    cache_key = cache.hash_key(
        "asset_prices:alphavantage",
        _cache_key_for_symbols(normalized_symbols, currency),
    )
    cached = await cache.get(cache_key)
    if isinstance(cached, list):
        return cached

    results = await _get_alpha_vantage_prices_uncached(normalized_symbols, currency)
    await cache.set(cache_key, results, settings.redis_cache_ttl_asset_price_seconds)
    return results


def get_alpha_vantage_prices(
    symbols: str | Sequence[str],
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for Alpha Vantage MCP price retrieval.

    This intentionally bypasses the async Redis cache because the sync provider path
    is executed inside worker threads from the chat router, while the shared Redis
    async client is initialized on the main application loop.
    """
    return asyncio.run(_get_alpha_vantage_prices_uncached(symbols, currency))
