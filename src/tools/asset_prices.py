"""Yahoo Finance asset pricing tool with retry and optional caching."""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.cache.client import get_cache
from src.config import settings
from src.errors import AssetPriceError

logger = logging.getLogger(__name__)

_SINGLE_LOOKBACK_PERIOD = "5d"
_COMMON_QUOTE_CURRENCIES = ("USD", "USDT", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF")
_FX_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}


def with_asset_price_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Retry transient provider failures while preserving domain-level price errors."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except AssetPriceError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                            attempt, max_attempts, fn.__name__, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts, fn.__name__, exc,
                        )

            raise AssetPriceError(
                f"'{fn.__name__}' failed after {max_attempts} attempts"
            ) from last_exc

        return wrapper

    return decorator


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    raw_symbols = [symbols] if isinstance(symbols, str) else list(symbols)
    normalized = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    if not normalized:
        raise AssetPriceError("At least one Yahoo Finance symbol is required.")
    return normalized


def _candidate_symbols(symbol: str) -> list[str]:
    candidates = [symbol]
    if any(marker in symbol for marker in ("-", "=", "^")):
        return candidates

    if symbol.isalpha() and len(symbol) == 6:
        base, quote = symbol[:3], symbol[3:]
        if base in _FX_CURRENCY_CODES and quote in _FX_CURRENCY_CODES:
            candidates.append(f"{symbol}=X")

    for quote in _COMMON_QUOTE_CURRENCIES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if re.fullmatch(r"[A-Z0-9]{2,10}", base):
                candidates.append(f"{base}-{quote}")
            break

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _cache_key_for_symbols(symbols: Sequence[str], currency: str | None) -> str:
    normalized_currency = (currency or "").strip().upper()
    normalized_symbols = ",".join(_normalize_symbols(symbols))
    return f"{normalized_symbols}|{normalized_currency}"


def _coerce_iso_timestamp(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    return str(value)


def _last_close_from_history(history: pd.DataFrame, symbol: str) -> tuple[float, str]:
    if history.empty or "Close" not in history:
        raise AssetPriceError(f"No recent price history returned for {symbol}.")

    close_series = history["Close"].dropna()
    if close_series.empty:
        raise AssetPriceError(f"No recent closing price returned for {symbol}.")

    latest_index = close_series.index[-1]
    latest_price = close_series.iloc[-1]
    try:
        price = float(latest_price)
    except (TypeError, ValueError) as exc:
        raise AssetPriceError(f"Latest price for {symbol} is not numeric.") from exc
    return price, _coerce_iso_timestamp(latest_index)


def _try_history_candidates(
    *,
    history_loader,
    symbol: str,
) -> tuple[str, pd.DataFrame]:
    last_error: AssetPriceError | None = None
    for candidate in _candidate_symbols(symbol):
        history = history_loader(candidate)
        try:
            _last_close_from_history(history, candidate)
            return candidate, history
        except AssetPriceError as exc:
            last_error = exc
    raise AssetPriceError(f"No recent price history returned for {symbol}.") from last_error


def _safe_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_metadata_for_symbol(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    fast_info = _safe_mapping(getattr(ticker, "fast_info", None))

    info: dict[str, Any] = {}
    try:
        info = _safe_mapping(ticker.info)
    except Exception as exc:
        logger.debug("Ticker info unavailable for %s: %s", symbol, exc)

    return {
        "name": info.get("shortName") or info.get("longName") or info.get("displayName") or "",
        "currency": info.get("currency") or fast_info.get("currency") or "",
        "quote_type": info.get("quoteType") or info.get("typeDisp") or "",
        "exchange": info.get("exchange") or fast_info.get("exchange") or "",
    }


def _infer_asset_type(symbol: str, metadata: dict[str, Any]) -> str:
    quote_type = str(metadata.get("quote_type") or "").strip().lower()
    if quote_type in {
        "equity",
        "crypto",
        "cryptocurrency",
        "currency",
        "forex",
        "future",
        "futures",
        "index",
        "etf",
    }:
        if quote_type == "cryptocurrency":
            return "crypto"
        if quote_type == "currency":
            return "forex"
        if quote_type == "futures":
            return "future"
        return quote_type

    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=X"):
        return "forex"
    if symbol.endswith("=F"):
        return "future"
    if symbol.startswith("^"):
        return "index"
    return "equity"


def _normalize_quote(
    symbol: str,
    *,
    price: float,
    as_of: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_type": _infer_asset_type(symbol, metadata),
        "name": metadata.get("name") or symbol,
        "price": price,
        "currency": metadata.get("currency") or "",
        "as_of": as_of,
        "source": "yfinance",
        "raw": {
            "quote_type": metadata.get("quote_type") or "",
            "exchange": metadata.get("exchange") or "",
        },
    }


def _extract_batch_history(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if history.empty:
        raise AssetPriceError(f"No recent price history returned for {symbol}.")

    if isinstance(history.columns, pd.MultiIndex):
        if symbol not in history.columns.get_level_values(0):
            raise AssetPriceError(f"No recent price history returned for {symbol}.")
        return history[symbol]

    return history


@with_asset_price_retry(max_attempts=3, base_delay=1.0)
def _get_asset_prices_uncached(
    symbols: str | Sequence[str],
    currency: str | None = None,
) -> list[dict[str, Any]]:
    import yfinance as yf

    del currency
    normalized_symbols = _normalize_symbols(symbols)
    if len(normalized_symbols) == 1:
        symbol = normalized_symbols[0]
        resolved_symbol, history = _try_history_candidates(
            history_loader=lambda candidate: yf.Ticker(candidate).history(
                period=_SINGLE_LOOKBACK_PERIOD,
                auto_adjust=False,
            ),
            symbol=symbol,
        )
        price, as_of = _last_close_from_history(history, resolved_symbol)
        metadata = _safe_metadata_for_symbol(resolved_symbol)
        quote = _normalize_quote(resolved_symbol, price=price, as_of=as_of, metadata=metadata)
        quote["symbol"] = symbol
        if resolved_symbol != symbol:
            quote["raw"]["resolved_symbol"] = resolved_symbol
        return [quote]

    history = yf.download(
        normalized_symbols,
        period=_SINGLE_LOOKBACK_PERIOD,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )

    quotes: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        symbol_history = _extract_batch_history(history, symbol)
        price, as_of = _last_close_from_history(symbol_history, symbol)
        metadata = _safe_metadata_for_symbol(symbol)
        quotes.append(_normalize_quote(symbol, price=price, as_of=as_of, metadata=metadata))
    return quotes


async def get_asset_prices_cached(
    symbols: str | Sequence[str],
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Cached async wrapper around direct Yahoo Finance asset price retrieval."""
    normalized_symbols = _normalize_symbols(symbols)
    cache = get_cache()
    if cache is None:
        return await asyncio.to_thread(_get_asset_prices_uncached, normalized_symbols, currency)

    cache_key = cache.hash_key(
        "asset_prices:symbols",
        _cache_key_for_symbols(normalized_symbols, currency),
    )
    cached = await cache.get(cache_key)
    if isinstance(cached, list):
        return cached

    results = await asyncio.to_thread(_get_asset_prices_uncached, normalized_symbols, currency)
    await cache.set(cache_key, results, settings.redis_cache_ttl_asset_price_seconds)
    return results


get_asset_prices = _get_asset_prices_uncached
