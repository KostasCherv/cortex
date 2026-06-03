"""Tests for Composio config fields."""
from unittest.mock import patch

from src.config import Settings


def test_composio_defaults():
    with patch.dict("os.environ", {}, clear=True):
        s = Settings(_env_file=None)
    assert s.composio_api_key == ""
    assert s.composio_enabled is True
    assert s.composio_apps == []
    assert s.composio_tool_refresh_seconds == 3600
    assert s.composio_max_agent_turns == 5


def test_composio_enabled_false_from_env():
    with patch.dict("os.environ", {"COMPOSIO_ENABLED": "false"}, clear=True):
        s = Settings(_env_file=None)
        assert s.composio_enabled is False


def test_composio_apps_parsed_from_env():
    with patch.dict("os.environ", {"COMPOSIO_APPS": "github,gmail,slack"}, clear=True):
        s = Settings(_env_file=None)
        assert s.composio_apps == ["github", "gmail", "slack"]
