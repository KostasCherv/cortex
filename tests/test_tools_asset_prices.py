"""Tests for src/tools/asset_prices.py."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.errors import AssetPriceError
from src.tools.asset_prices import get_asset_prices_cached


def _single_history(price: float = 123.45) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime(2026, 1, 2, tzinfo=UTC)])
    return pd.DataFrame({"Close": [price]}, index=index)


def _multi_history() -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime(2026, 1, 2, tzinfo=UTC)])
    columns = pd.MultiIndex.from_product(
        [["AAPL", "BTC-USD"], ["Close"]],
        names=["Ticker", "Field"],
    )
    return pd.DataFrame([[123.45, 67890.12]], index=index, columns=columns)


def test_get_asset_prices_single_symbol_returns_normalized_quote():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _single_history()
    mock_ticker.fast_info = {"currency": "USD"}
    mock_ticker.info = {"shortName": "Apple Inc.", "quoteType": "EQUITY", "exchange": "NMS"}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        from src.tools.asset_prices import get_asset_prices

        results = get_asset_prices("AAPL")

    assert results == [
        {
            "symbol": "AAPL",
            "asset_type": "equity",
            "name": "Apple Inc.",
            "price": 123.45,
            "currency": "USD",
            "as_of": "2026-01-02T00:00:00+00:00",
            "source": "yfinance",
            "raw": {"quote_type": "EQUITY", "exchange": "NMS"},
        }
    ]


def test_get_asset_prices_multiple_symbols_returns_requested_order():
    def _ticker_factory(symbol: str):
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"currency": "USD"}
        if symbol == "AAPL":
            mock_ticker.info = {"shortName": "Apple Inc.", "quoteType": "EQUITY"}
        else:
            mock_ticker.info = {"shortName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}
        return mock_ticker

    with (
        patch("yfinance.download", return_value=_multi_history()) as mock_download,
        patch("yfinance.Ticker", side_effect=_ticker_factory),
    ):
        from src.tools.asset_prices import get_asset_prices

        results = get_asset_prices(["BTC-USD", "AAPL"])

    assert [quote["symbol"] for quote in results] == ["BTC-USD", "AAPL"]
    assert results[0]["asset_type"] == "crypto"
    assert results[0]["price"] == 67890.12
    assert results[1]["asset_type"] == "equity"
    assert results[1]["price"] == 123.45
    mock_download.assert_called_once()


def test_get_asset_prices_raises_when_history_is_empty():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker.fast_info = {}
    mock_ticker.info = {}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        from src.tools.asset_prices import get_asset_prices

        with pytest.raises(AssetPriceError, match="No recent price history returned"):
            get_asset_prices("AAPL")


def test_get_asset_prices_retries_on_failure():
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = [
        RuntimeError("network error"),
        RuntimeError("network error"),
        _single_history(),
    ]
    mock_ticker.fast_info = {"currency": "USD"}
    mock_ticker.info = {"shortName": "Apple Inc.", "quoteType": "EQUITY"}

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch("src.tools.search.time.sleep"),
    ):
        from src.tools.asset_prices import get_asset_prices

        results = get_asset_prices("AAPL")

    assert results[0]["symbol"] == "AAPL"
    assert mock_ticker.history.call_count == 3


def test_get_asset_prices_resolves_common_crypto_symbol_without_dash():
    invalid_ticker = MagicMock()
    invalid_ticker.history.return_value = pd.DataFrame()

    valid_ticker = MagicMock()
    valid_ticker.history.return_value = _single_history(67890.12)
    valid_ticker.fast_info = {"currency": "USD"}
    valid_ticker.info = {"shortName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}

    def _ticker_factory(symbol: str):
        return {"BTCUSD": invalid_ticker, "BTC-USD": valid_ticker}[symbol]

    with patch("yfinance.Ticker", side_effect=_ticker_factory):
        from src.tools.asset_prices import get_asset_prices

        results = get_asset_prices("BTCUSD")

    assert results[0]["symbol"] == "BTCUSD"
    assert results[0]["asset_type"] == "crypto"
    assert results[0]["price"] == 67890.12
    assert results[0]["raw"]["resolved_symbol"] == "BTC-USD"


@pytest.mark.asyncio
async def test_get_asset_prices_cached_returns_cached_results():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "asset_prices:key"
    mock_cache.get.return_value = [{"symbol": "AAPL", "price": 123.45}]

    with (
        patch("src.tools.asset_prices.get_cache", return_value=mock_cache),
        patch("src.tools.asset_prices._get_asset_prices_uncached") as mock_uncached,
    ):
        results = await get_asset_prices_cached("AAPL")

    assert results == [{"symbol": "AAPL", "price": 123.45}]
    mock_uncached.assert_not_called()


@pytest.mark.asyncio
async def test_get_asset_prices_cached_stores_results_on_miss():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "asset_prices:key"
    mock_cache.get.return_value = None
    uncached = [{"symbol": "AAPL", "price": 123.45}]

    with (
        patch("src.tools.asset_prices.get_cache", return_value=mock_cache),
        patch("src.tools.asset_prices.settings") as mock_settings,
        patch("src.tools.asset_prices.asyncio.to_thread", new=AsyncMock(return_value=uncached)),
    ):
        mock_settings.redis_cache_ttl_asset_price_seconds = 300
        results = await get_asset_prices_cached(["AAPL"], currency=None)

    assert results == uncached
    mock_cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_asset_prices_cached_normalizes_symbols_for_cache_key():
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None

    with (
        patch("src.tools.asset_prices.get_cache", return_value=mock_cache),
        patch("src.tools.asset_prices.settings") as mock_settings,
        patch("src.tools.asset_prices.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_settings.redis_cache_ttl_asset_price_seconds = 300
        await get_asset_prices_cached([" aapl "], currency=None)
        await get_asset_prices_cached("AAPL", currency=None)

    left = mock_cache.hash_key.call_args_list[0].args[1]
    right = mock_cache.hash_key.call_args_list[1].args[1]
    assert left == right
