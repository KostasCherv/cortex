"""Pluggable asset-pricing tool interface and provider registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from src.config import settings
from src.errors import AssetPriceError, ConfigurationError
from src.tools.alpha_vantage_mcp import (
    get_alpha_vantage_prices,
    validate_alpha_vantage_mcp_configuration,
)
from src.tools.asset_prices import get_asset_prices

logger = logging.getLogger(__name__)


class AssetPriceTool(Protocol):
    """Provider-agnostic asset price contract."""

    provider_name: str

    def quote(self, symbols: str | list[str], currency: str | None = None) -> list[dict]:
        """Return normalized market quotes for one or more symbols."""


@dataclass(slots=True)
class YFinanceAssetPriceTool:
    """Yahoo Finance-backed implementation of ``AssetPriceTool``."""

    provider_name: str = "yfinance"

    def quote(self, symbols: str | list[str], currency: str | None = None) -> list[dict]:
        return get_asset_prices(symbols=symbols, currency=currency)


@dataclass(slots=True)
class AlphaVantageMcpAssetPriceTool:
    """Alpha Vantage MCP-backed implementation of ``AssetPriceTool``."""

    provider_name: str = "alphavantage_mcp"

    def quote(self, symbols: str | list[str], currency: str | None = None) -> list[dict]:
        try:
            return get_alpha_vantage_prices(symbols=symbols, currency=currency)
        except AssetPriceError as exc:
            logger.warning(
                "[asset_price] Alpha Vantage MCP failed for symbols=%s currency=%s; "
                "falling back to yfinance: %s",
                symbols,
                currency,
                exc,
            )
            fallback_quotes = get_asset_prices(symbols=symbols, currency=currency)
            for quote in fallback_quotes:
                raw = quote.get("raw")
                if not isinstance(raw, dict):
                    raw = {}
                    quote["raw"] = raw
                raw["fallback_provider"] = "yfinance"
                raw["primary_provider_error"] = str(exc)
            return fallback_quotes


def get_asset_price_tool(provider: str | None = None) -> AssetPriceTool:
    """Return the configured asset-pricing tool adapter."""
    active = (provider or settings.asset_price_provider).strip().lower()
    if active == "alphavantage_mcp":
        return AlphaVantageMcpAssetPriceTool()
    if active == "yfinance":
        return YFinanceAssetPriceTool()
    raise ConfigurationError(
        "Unknown ASSET_PRICE_PROVIDER "
        f"'{active}'. Supported providers: alphavantage_mcp, yfinance."
    )


def validate_asset_price_provider_health() -> None:
    """Fail fast when the configured asset-pricing provider is unavailable."""
    tool = get_asset_price_tool()
    if isinstance(tool, AlphaVantageMcpAssetPriceTool):
        try:
            validate_alpha_vantage_mcp_configuration()
        except AssetPriceError as exc:
            raise ConfigurationError(
                "Configured asset price provider is unavailable. "
                "The server requires a working asset price tool at startup."
            ) from exc
        return
    try:
        tool.quote(["AAPL"])
    except AssetPriceError as exc:
        raise ConfigurationError(
            "Configured asset price provider is unavailable. "
            "The server requires a working asset price tool at startup."
        ) from exc
