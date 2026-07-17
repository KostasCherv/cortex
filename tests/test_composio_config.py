"""Tests for Composio config fields."""
import json
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


def test_billing_config_json_overrides_split_stripe_variables():
    bundled = {
        "stripe_secret_key": "sk_test_bundled",
        "stripe_webhook_secret": "whsec_bundled",
        "stripe_pro_price_id": "price_bundled",
    }
    with patch.dict(
        "os.environ",
        {
            "BILLING_CONFIG_JSON": json.dumps(bundled),
            "STRIPE_SECRET_KEY": "sk_test_split",
        },
        clear=True,
    ):
        s = Settings(_env_file=None)
    assert s.stripe_secret_key == "sk_test_bundled"
    assert s.stripe_webhook_secret == "whsec_bundled"
    assert s.stripe_pro_price_id == "price_bundled"


def test_provider_config_json_overrides_split_provider_variables():
    bundled = {
        "openai_api_key": "sk-bundled",
        "tavily_api_key": "tvly-bundled",
        "redis_url": "rediss://bundled",
        "langfuse_public_key": "pk-bundled",
        "langfuse_secret_key": "sk-bundled",
        "langfuse_base_url": "https://langfuse.example.test",
    }
    with patch.dict(
        "os.environ",
        {
            "PROVIDER_CONFIG_JSON": json.dumps(bundled),
            "OPENAI_API_KEY": "sk-split",
        },
        clear=True,
    ):
        s = Settings(_env_file=None)
    assert s.openai_api_key == "sk-bundled"
    assert s.tavily_api_key == "tvly-bundled"
    assert s.redis_url == "rediss://bundled"
    assert s.langfuse_public_key == "pk-bundled"
    assert s.langfuse_secret_key == "sk-bundled"
    assert s.langfuse_host == "https://langfuse.example.test"
