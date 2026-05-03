from unittest.mock import MagicMock, patch

import pytest

from src.errors import ConfigurationError
from src.tools.web_search import (
    TavilyWebSearchTool,
    get_web_search_tool,
    validate_web_search_provider_health,
)


def test_get_web_search_tool_returns_tavily_adapter():
    with patch("src.tools.web_search.settings.web_search_provider", "tavily"):
        tool = get_web_search_tool()
    assert isinstance(tool, TavilyWebSearchTool)


def test_get_web_search_tool_raises_for_unknown_provider():
    with patch("src.tools.web_search.settings.web_search_provider", "unknown"):
        with pytest.raises(ConfigurationError, match="Unknown WEB_SEARCH_PROVIDER"):
            get_web_search_tool()


def test_validate_web_search_provider_health_runs_probe():
    mock_tool = MagicMock()
    with patch("src.tools.web_search.get_web_search_tool", return_value=mock_tool):
        validate_web_search_provider_health()
    mock_tool.search.assert_called_once_with("health check", max_results=1)
