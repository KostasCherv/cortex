# tests/finetune/test_label_inputs.py
from unittest.mock import MagicMock, patch

import httpx

from scripts.finetune.label_inputs import label_input

_VALID_JSON = (
    '{"action":"web_search","reason":"needs live data",'
    '"query":"Apple stock news","symbols":[],"currency":""}'
)
_INVALID_TEXT = "I cannot determine the action from this message."
_ASSET_NO_SYMBOLS = (
    '{"action":"asset_price","reason":"stock price","query":"","symbols":[],"currency":""}'
)


def _mock_ollama(content: str) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    return resp


def test_returns_training_record_on_valid_response():
    with patch("scripts.finetune.label_inputs.httpx.post", return_value=_mock_ollama(_VALID_JSON)):
        result = label_input(message="Latest Apple stock news?")

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"


def test_returns_none_on_unparseable_response():
    with patch("scripts.finetune.label_inputs.httpx.post", return_value=_mock_ollama(_INVALID_TEXT)):
        result = label_input(message="test")

    assert result is None


def test_returns_none_on_schema_violation():
    with patch("scripts.finetune.label_inputs.httpx.post", return_value=_mock_ollama(_ASSET_NO_SYMBOLS)):
        result = label_input(message="Price of AAPL?")

    assert result is None


def test_returns_none_on_http_error():
    with patch("scripts.finetune.label_inputs.httpx.post", side_effect=httpx.HTTPError("timeout")):
        result = label_input(message="test")

    assert result is None


def test_assistant_content_is_valid_json():
    import json
    with patch("scripts.finetune.label_inputs.httpx.post", return_value=_mock_ollama(_VALID_JSON)):
        result = label_input(message="Latest Apple stock news?")

    parsed = json.loads(result["messages"][2]["content"])
    assert parsed["action"] == "web_search"
