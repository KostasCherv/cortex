"""Tests for src/tools/alpha_vantage_mcp.py."""

from unittest.mock import AsyncMock, patch

import pytest

from src.errors import AssetPriceError
from src.tools.alpha_vantage_mcp import (
    get_alpha_vantage_prices,
    get_alpha_vantage_prices_cached,
    validate_alpha_vantage_mcp_configuration,
)


def test_get_alpha_vantage_prices_normalizes_equity_quote():
    payload = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "190.12",
            "07. latest trading day": "2026-05-27",
            "08. previous close": "189.33",
            "10. change percent": "0.42%",
        }
    }

    with patch(
        "src.tools.alpha_vantage_mcp._call_alpha_vantage_tool",
        new=AsyncMock(return_value=payload),
    ):
        results = get_alpha_vantage_prices("AAPL")

    assert results == [
        {
            "symbol": "AAPL",
            "title": "Alpha Vantage MCP · AAPL",
            "url": "https://www.alphavantage.co/documentation/",
            "content": "Source: Alpha Vantage MCP (GLOBAL_QUOTE)\nAAPL price 190.12 USD as of 2026-05-27",
            "asset_type": "equity",
            "name": "AAPL",
            "price": 190.12,
            "currency": "USD",
            "as_of": "2026-05-27",
            "source": "alphavantage_mcp",
            "raw": {
                "resolved_symbol": "AAPL",
                "previous_close": "189.33",
                "change_percent": "0.42%",
            },
        }
    ]


def test_get_alpha_vantage_prices_sync_wrapper_bypasses_cached_path():
    payload = [
        {
            "symbol": "AAPL",
            "asset_type": "equity",
            "name": "AAPL",
            "price": 190.12,
            "currency": "USD",
            "as_of": "2026-05-27",
            "source": "alphavantage_mcp",
            "raw": {"resolved_symbol": "AAPL", "previous_close": "", "change_percent": ""},
        }
    ]

    with (
        patch(
            "src.tools.alpha_vantage_mcp._get_alpha_vantage_prices_uncached",
            new=AsyncMock(return_value=payload),
        ) as mock_uncached,
        patch("src.tools.alpha_vantage_mcp.get_alpha_vantage_prices_cached") as mock_cached,
    ):
        results = get_alpha_vantage_prices("AAPL")

    assert results == payload
    mock_uncached.assert_awaited_once_with("AAPL", None)
    mock_cached.assert_not_called()


def test_get_alpha_vantage_prices_normalizes_crypto_pair():
    payload = {
        "Realtime Currency Exchange Rate": {
            "1. From_Currency Code": "BTC",
            "2. From_Currency Name": "Bitcoin",
            "3. To_Currency Code": "USD",
            "4. To_Currency Name": "United States Dollar",
            "5. Exchange Rate": "68000.10",
            "6. Last Refreshed": "2026-05-27 16:00:00",
            "7. Time Zone": "UTC",
        }
    }

    with patch(
        "src.tools.alpha_vantage_mcp._call_alpha_vantage_tool",
        new=AsyncMock(return_value=payload),
    ):
        results = get_alpha_vantage_prices("BTCUSD")

    assert results[0]["symbol"] == "BTCUSD"
    assert results[0]["title"] == "Alpha Vantage MCP · BTC/USD"
    assert results[0]["url"] == "https://www.alphavantage.co/documentation/"
    assert "Source: Alpha Vantage MCP" in results[0]["content"]
    assert results[0]["asset_type"] == "crypto"
    assert results[0]["price"] == 68000.10
    assert results[0]["currency"] == "USD"
    assert results[0]["raw"]["resolved_symbol"] == "BTCUSD"


def test_get_alpha_vantage_prices_falls_back_from_global_quote_to_exchange_rate():
    payload = {
        "Realtime Currency Exchange Rate": {
            "1. From_Currency Code": "BTC",
            "2. From_Currency Name": "Bitcoin",
            "3. To_Currency Code": "USD",
            "4. To_Currency Name": "United States Dollar",
            "5. Exchange Rate": "68000.10",
            "6. Last Refreshed": "2026-05-27 16:00:00",
            "7. Time Zone": "UTC",
        }
    }

    call_mock = AsyncMock(
        side_effect=[
            AssetPriceError("quote not found"),
            payload,
        ]
    )

    with patch("src.tools.alpha_vantage_mcp._call_alpha_vantage_tool", new=call_mock):
        results = get_alpha_vantage_prices("BTC")

    assert results[0]["symbol"] == "BTCUSD"
    assert results[0]["asset_type"] == "crypto"
    assert call_mock.await_args_list[0].args == (
        "GLOBAL_QUOTE",
        {"symbol": "BTC", "datatype": "json", "return_full_data": True},
    )
    assert call_mock.await_args_list[1].args == (
        "CURRENCY_EXCHANGE_RATE",
        {
            "from_currency": "BTC",
            "to_currency": "USD",
            "datatype": "json",
            "return_full_data": True,
        },
    )


def test_validate_alpha_vantage_mcp_configuration_requires_url_or_key():
    with patch("src.tools.alpha_vantage_mcp.settings.alpha_vantage_mcp_url", ""), patch(
        "src.tools.alpha_vantage_mcp.settings.alpha_vantage_api_key", ""
    ):
        with pytest.raises(AssetPriceError, match="not configured"):
            validate_alpha_vantage_mcp_configuration()


@pytest.mark.asyncio
async def test_get_alpha_vantage_prices_cached_uses_cache_key():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "asset_prices:alphavantage:key"
    mock_cache.get.return_value = [{"symbol": "AAPL", "price": 190.12}]

    with patch("src.tools.alpha_vantage_mcp.get_cache", return_value=mock_cache):
        results = await get_alpha_vantage_prices_cached("AAPL")

    assert results == [{"symbol": "AAPL", "price": 190.12}]
