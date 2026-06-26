# tests/finetune/test_router_prompt.py
from scripts.finetune.router_prompt import (
    ROUTER_SYSTEM_PROMPT,
    format_user_turn,
    build_training_record,
)


def test_system_prompt_contains_all_actions():
    for action in (
        "answer_direct", "answer_from_rag", "web_search",
        "asset_price", "search_finance_tools", "ask_clarifying",
    ):
        assert action in ROUTER_SYSTEM_PROMPT


def test_format_user_turn_no_context():
    result = format_user_turn(message="What is a P/E ratio?")
    assert "User message: What is a P/E ratio?" in result
    assert "Available RAG context: no" in result
    assert "Conversation history: none" in result


def test_format_user_turn_with_rag():
    result = format_user_turn(message="What does it say?", rag_context="earnings report")
    assert "yes — earnings report" in result


def test_format_user_turn_with_history():
    history = [{"role": "user", "content": "Tell me about Tesla"}]
    result = format_user_turn(message="And the P/E?", history=history)
    assert "Tell me about Tesla" in result


def test_build_training_record_structure():
    record = build_training_record(
        message="Price of AAPL?",
        assistant_json='{"action":"asset_price","reason":"stock price request","query":"","symbols":["AAPL"],"currency":""}',
    )
    assert "messages" in record
    assert len(record["messages"]) == 3
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert '"action"' in record["messages"][2]["content"]
