from pathlib import Path
from subprocess import run

from src.benchmarking.k6_report import build_report, render_markdown


def test_build_report_extracts_core_http_metrics():
    summary = {
        "root_group": {"name": "", "checks": []},
        "metrics": {
            "http_reqs": {"type": "counter", "values": {"count": 1200, "rate": 20}},
            "http_req_failed": {"type": "rate", "values": {"rate": 0.005}},
            "http_req_duration": {
                "type": "trend",
                "values": {
                    "avg": 820.4,
                    "med": 700.0,
                    "p(95)": 1450.0,
                    "p(99)": 2100.0,
                    "max": 3000.0,
                },
            },
            "http_req_waiting": {"type": "trend", "values": {"avg": 790.0, "p(95)": 1400.0}},
            "http_req_blocked": {"type": "trend", "values": {"avg": 18.0, "p(95)": 55.0}},
            "iterations": {"type": "counter", "values": {"count": 1200, "rate": 20}},
            "vus": {"type": "gauge", "values": {"value": 12, "min": 0, "max": 12}},
        },
        "state": {"testRunDurationMs": 60000},
    }

    report = build_report(summary, scenario_name="agent-chat")

    assert report.scenario_name == "agent-chat"
    assert report.total_requests == 1200
    assert report.request_rate == 20
    assert report.error_rate == 0.005
    assert report.p95_ms == 1450.0
    assert "Application processing time dominates request latency." in report.bottleneck_hints


def test_render_markdown_includes_claim_ready_sections():
    summary = {
        "root_group": {"name": "", "checks": []},
        "metrics": {
            "http_reqs": {"type": "counter", "values": {"count": 300, "rate": 5}},
            "http_req_failed": {"type": "rate", "values": {"rate": 0.0}},
            "http_req_duration": {
                "type": "trend",
                "values": {
                    "avg": 400.0,
                    "med": 350.0,
                    "p(95)": 650.0,
                    "p(99)": 900.0,
                    "max": 1100.0,
                },
            },
            "http_req_waiting": {"type": "trend", "values": {"avg": 300.0, "p(95)": 500.0}},
            "http_req_blocked": {"type": "trend", "values": {"avg": 60.0, "p(95)": 120.0}},
            "vus": {"type": "gauge", "values": {"value": 6, "min": 0, "max": 6}},
        },
        "state": {"testRunDurationMs": 60000},
    }

    report = build_report(summary, scenario_name="health")
    markdown = render_markdown(
        report,
        environment_label="local-dev",
        target_description="5 req/s for 1 minute",
        notes="Local-only bottleneck discovery run.",
    )

    assert "# k6 Benchmark Report" in markdown
    assert "Environment" in markdown
    assert "local-dev" in markdown
    assert "5 req/s for 1 minute" in markdown
    assert "Local-only bottleneck discovery run." in markdown


def test_build_report_supports_real_k6_summary_export_shape():
    summary = {
        "root_group": {"name": "", "checks": {}},
        "metrics": {
            "http_reqs": {"count": 600, "rate": 20.0007},
            "http_req_failed": {"value": 0},
            "http_req_duration": {
                "avg": 2.19,
                "med": 2.25,
                "p(95)": 3.13,
                "p(99)": 4.05,
                "max": 5.74,
            },
            "http_req_waiting": {"avg": 2.01, "p(95)": 2.92},
            "http_req_blocked": {"avg": 0.02, "p(95)": 0.03},
            "vus": {"min": 0, "max": 1, "value": 0},
        },
    }

    report = build_report(summary, scenario_name="health")

    assert report.total_requests == 600
    assert report.request_rate == 20.0007
    assert report.error_rate == 0
    assert report.p95_ms == 3.13
    assert report.active_vus == 1


def test_render_k6_report_cli_writes_markdown(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "report.md"
    summary_path.write_text(
        '{"root_group":{"name":"","checks":[]},'
        '"metrics":{"http_reqs":{"type":"counter","values":{"count":10,"rate":1}},'
        '"http_req_failed":{"type":"rate","values":{"rate":0.0}},'
        '"http_req_duration":{"type":"trend","values":{"avg":100.0,"med":90.0,"p(95)":150.0,"p(99)":180.0,"max":200.0}}},'
        '"state":{"testRunDurationMs":10000}}'
    )

    result = run(
        [
            "uv",
            "run",
            "python",
            "scripts/render_k6_report.py",
            "--summary-json",
            str(summary_path),
            "--scenario",
            "health",
            "--environment",
            "local-dev",
            "--target",
            "1 req/s",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd="/Users/kostas/Dev/research_agent/.worktrees/codex-local-benchmark",
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert "health" in output_path.read_text()
