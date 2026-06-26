from types import SimpleNamespace

from src.benchmarking.rag_chat import coerce_agent_loop_benchmark_result


def test_coerce_agent_loop_benchmark_result_accepts_agent_loop_like_object():
    result = SimpleNamespace(answer="hello world", web_used=True, citations=[{"id": "c1"}])

    summary = coerce_agent_loop_benchmark_result(result)

    assert summary.answer == "hello world"
    assert summary.web_used is True
    assert summary.citation_count == 1
    assert summary.answer_chars == 11


def test_coerce_agent_loop_benchmark_result_accepts_plain_string():
    summary = coerce_agent_loop_benchmark_result("plain answer")

    assert summary.answer == "plain answer"
    assert summary.web_used is False
    assert summary.citation_count == 0
    assert summary.answer_chars == len("plain answer")
