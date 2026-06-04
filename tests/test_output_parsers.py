"""Tests for shared structured LLM output parsing."""

import pytest
from pydantic import TypeAdapter

from src.errors import StructuredOutputParseError, StructuredOutputValidationError
from src.llm.output_parsers import (
    ChatActionDecisionPayload,
    FinanceToolSelectionPayload,
    format_validation_error_details,
    parse_entity_relation_extraction_json,
    parse_chat_action_json,
    parse_finance_tool_call_plan_json,
    parse_finance_tool_selection_json,
    parse_model_json,
    parse_research_summaries_json,
    parse_type_json,
)


def test_parse_chat_action_json_accepts_markdown_fenced_payload():
    parsed = parse_chat_action_json(
        '```json\n{"action":"answer_direct","reason":"small_talk","query":"","symbols":[],"currency":""}\n```'
    )

    assert parsed.action == "answer_direct"
    assert parsed.reason == "small_talk"


def test_parse_research_summaries_json_unwraps_summaries_key():
    parsed = parse_research_summaries_json(
        (
            '{"summaries":['
            '{"url":"https://a.com","title":"A","summary":"Summary A"}'
            "]} "
        ).strip(),
    )

    assert len(parsed) == 1
    assert parsed[0].url == "https://a.com"


def test_parse_model_json_raises_validation_error_for_missing_field():
    with pytest.raises(StructuredOutputValidationError):
        parse_model_json(
            '{"action":"answer_direct","query":"","symbols":[],"currency":""}',
            model=ChatActionDecisionPayload,
        )


def test_parse_type_json_raises_parse_error_for_non_json():
    with pytest.raises(StructuredOutputParseError):
        parse_type_json(
            "hello there",
            adapter=TypeAdapter(ChatActionDecisionPayload),
        )


def test_parse_chat_action_json_requires_query_for_web_search():
    with pytest.raises(StructuredOutputValidationError):
        parse_chat_action_json(
            '{"action":"web_search","reason":"need fresh info","query":"","symbols":[],"currency":""}'
        )


def test_parse_chat_action_json_rejects_fetch_url_action():
    with pytest.raises(Exception):  # Pydantic literal_error since fetch_url is no longer valid
        parse_chat_action_json(
            '{"action":"fetch_url","reason":"need page content","query":"","symbols":[],"currency":""}'
        )


def test_parse_chat_action_json_requires_symbols_for_asset_price():
    with pytest.raises(StructuredOutputValidationError):
        parse_chat_action_json(
            '{"action":"asset_price","reason":"needs_quote","query":"","symbols":[],"currency":""}'
        )


def test_parse_chat_action_json_requires_query_for_search_finance_tools():
    with pytest.raises(StructuredOutputValidationError):
        parse_chat_action_json(
            '{"action":"search_finance_tools","reason":"needs_finance_tool","query":"","symbols":[],"currency":""}'
        )


def test_parse_finance_tool_selection_json_parses_choice():
    parsed = parse_finance_tool_selection_json(
        '{"tool_name":"CRYPTO_INTRADAY","reason":"best_match_for_24h_change"}'
    )

    assert isinstance(parsed, FinanceToolSelectionPayload)
    assert parsed.tool_name == "CRYPTO_INTRADAY"


def test_parse_finance_tool_call_plan_json_requires_clarifying_question_when_not_callable():
    with pytest.raises(StructuredOutputValidationError):
        parse_finance_tool_call_plan_json(
            '{"should_call":false,"reason":"missing_symbol","arguments":{},"clarifying_question":""}'
        )


def test_parse_research_summaries_json_rejects_blank_summary():
    with pytest.raises(StructuredOutputValidationError):
        parse_research_summaries_json(
            '[{"url":"https://a.com","title":"A","summary":"   "}]'
        )


def test_parse_entity_relation_extraction_json_parses_nested_envelope():
    parsed = parse_entity_relation_extraction_json(
        """
        {
          "entities": [{"name": "OpenAI", "entity_type": " Org ", "confidence": 0.9}],
          "relations": [{"source": "OpenAI", "target": "GPT-4", "type": " BUILDS ", "confidence": 0.4}]
        }
        """
    )

    assert parsed.entities[0].name == "OpenAI"
    assert parsed.entities[0].entity_type == "Org"
    assert parsed.relations[0].type == "BUILDS"


def test_parse_entity_relation_extraction_json_rejects_confidence_above_one():
    with pytest.raises(StructuredOutputValidationError):
        parse_entity_relation_extraction_json(
            """
            {
              "entities": [{"name": "OpenAI", "entity_type": "Org", "confidence": 1.5}],
              "relations": []
            }
            """
        )


def test_format_validation_error_details_returns_compact_sanitized_lines():
    try:
        parse_chat_action_json(
            '{"action":"web_search","reason":"need fresh info","query":"","symbols":[],"currency":""}'
        )
    except StructuredOutputValidationError as exc:
        details = format_validation_error_details(exc)
    else:
        raise AssertionError("expected validation error")

    assert "value_error" in details
