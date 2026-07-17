"""Tests for bundled Inngest production credentials."""

import json
import os

from src.inngest_client import _apply_bundled_inngest_config


def test_bundled_inngest_config_sets_sdk_environment(monkeypatch):
    monkeypatch.setenv(
        "INNGEST_CONFIG_JSON",
        json.dumps(
            {
                "inngest_event_key": "event-bundled",
                "inngest_signing_key": "signing-bundled",
            }
        ),
    )
    monkeypatch.delenv("INNGEST_EVENT_KEY", raising=False)
    monkeypatch.delenv("INNGEST_SIGNING_KEY", raising=False)

    _apply_bundled_inngest_config()

    assert os.environ["INNGEST_EVENT_KEY"] == "event-bundled"
    assert os.environ["INNGEST_SIGNING_KEY"] == "signing-bundled"
