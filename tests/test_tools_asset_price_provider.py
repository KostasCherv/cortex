from unittest.mock import MagicMock, patch

import pytest

from src.errors import AssetPriceError, ConfigurationError
from src.tools.asset_price_provider import (
    AlphaVantageMcpAssetPriceTool,
    YFinanceAssetPriceTool,
    get_asset_price_tool,
    validate_asset_price_provider_health,
)


def test_get_asset_price_tool_returns_alpha_vantage_adapter_by_default_setting():
    with patch("src.tools.asset_price_provider.settings.asset_price_provider", "alphavantage_mcp"):
        tool = get_asset_price_tool()
    assert isinstance(tool, AlphaVantageMcpAssetPriceTool)


def test_get_asset_price_tool_returns_yfinance_adapter():
    with patch("src.tools.asset_price_provider.settings.asset_price_provider", "yfinance"):
        tool = get_asset_price_tool()
    assert isinstance(tool, YFinanceAssetPriceTool)


def test_get_asset_price_tool_raises_for_unknown_provider():
    with patch("src.tools.asset_price_provider.settings.asset_price_provider", "unknown"):
        with pytest.raises(ConfigurationError, match="Unknown ASSET_PRICE_PROVIDER"):
            get_asset_price_tool()


def test_validate_asset_price_provider_health_runs_yfinance_probe():
    mock_tool = MagicMock(spec=YFinanceAssetPriceTool)
    mock_tool.provider_name = "yfinance"
    with patch("src.tools.asset_price_provider.get_asset_price_tool", return_value=mock_tool):
        validate_asset_price_provider_health()
    mock_tool.quote.assert_called_once_with(["AAPL"])


def test_validate_asset_price_provider_health_checks_alpha_vantage_config_only():
    with (
        patch(
            "src.tools.asset_price_provider.get_asset_price_tool",
            return_value=AlphaVantageMcpAssetPriceTool(),
        ),
        patch("src.tools.asset_price_provider.validate_alpha_vantage_mcp_configuration") as validate_cfg,
    ):
        validate_asset_price_provider_health()
    validate_cfg.assert_called_once_with()


def test_alpha_vantage_provider_falls_back_to_yfinance_on_failure():
    tool = AlphaVantageMcpAssetPriceTool()
    fallback_quotes = [
        {
            "symbol": "BTCUSD",
            "asset_type": "crypto",
            "name": "Bitcoin USD",
            "price": 74908.01,
            "currency": "USD",
            "as_of": "2026-05-27T19:34:20",
            "source": "yfinance",
            "raw": {},
        }
    ]

    with (
        patch(
            "src.tools.asset_price_provider.get_alpha_vantage_prices",
            side_effect=AssetPriceError("rate limited"),
        ),
        patch("src.tools.asset_price_provider.get_asset_prices", return_value=fallback_quotes),
    ):
        results = tool.quote(["BTCUSD"])

    assert results[0]["source"] == "yfinance"
    assert results[0]["raw"]["fallback_provider"] == "yfinance"
    assert results[0]["raw"]["primary_provider_error"] == "rate limited"
