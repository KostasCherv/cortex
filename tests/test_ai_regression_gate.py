import json

from src.evals.regression_gate import DEFAULT_DATASET, evaluate, main


def test_committed_ai_regression_set_is_green():
    report = evaluate(DEFAULT_DATASET)

    assert report["case_count"] == 20
    assert report["passed_count"] == 20
    assert report["score"] == 1.0
    assert report["category_scores"] == {
        "citation": 1.0,
        "router": 1.0,
        "tool_selection": 1.0,
    }


def test_cli_writes_machine_readable_report(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["regression_gate", "--output", str(output)])

    assert main() == 0
    report = json.loads(output.read_text())
    assert report["score"] == 1.0
    assert report["commit_sha"]
