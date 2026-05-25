"""Tests for shared structured LLM output parsing."""

import pytest
from pydantic import TypeAdapter

from src.errors import StructuredOutputParseError, StructuredOutputValidationError
from src.llm.output_parsers import (
    ChatActionDecisionPayload,
    parse_chat_action_json,
    parse_model_json,
    parse_research_summaries_json,
    parse_type_json,
)


def test_parse_chat_action_json_accepts_markdown_fenced_payload():
    parsed = parse_chat_action_json(
        '```json\n{"action":"answer_direct","reason":"small_talk","query":"","url":""}\n```'
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
            '{"action":"answer_direct","query":"","url":""}',
            model=ChatActionDecisionPayload,
        )


def test_parse_type_json_raises_parse_error_for_non_json():
    with pytest.raises(StructuredOutputParseError):
        parse_type_json(
            "hello there",
            adapter=TypeAdapter(ChatActionDecisionPayload),
        )
