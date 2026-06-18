# tests/finetune/test_score_router.py
from scripts.score_router import compute_accuracy, format_results


def test_compute_accuracy_perfect():
    scored = [
        {"action_label": "answer_direct", "predicted": "answer_direct"},
        {"action_label": "web_search", "predicted": "web_search"},
    ]
    result = compute_accuracy(scored)
    assert result["answer_direct"] == (1, 1)
    assert result["web_search"] == (1, 1)


def test_compute_accuracy_partial():
    scored = [
        {"action_label": "answer_direct", "predicted": "answer_direct"},
        {"action_label": "answer_direct", "predicted": "web_search"},
    ]
    result = compute_accuracy(scored)
    assert result["answer_direct"] == (1, 2)


def test_compute_accuracy_handles_none_prediction():
    scored = [
        {"action_label": "web_search", "predicted": None},
        {"action_label": "web_search", "predicted": "web_search"},
    ]
    result = compute_accuracy(scored)
    assert result["web_search"] == (1, 2)


def test_format_results_contains_overall():
    accuracy = {"answer_direct": (8, 10), "web_search": (9, 10)}
    output = format_results("gpt-4o-mini", accuracy)
    assert "OVERALL" in output
    assert "gpt-4o-mini" in output


def test_format_results_shows_percentage():
    accuracy = {"answer_direct": (8, 10)}
    output = format_results("test-model", accuracy)
    assert "80%" in output
