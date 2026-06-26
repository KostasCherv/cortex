import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_k6_lib_includes_default_benchmark_tags():
    source = (ROOT / "load-tests" / "lib.js").read_text()

    assert "BENCHMARK_TEST_ID" in source
    assert re.search(r"testid\s*:", source)
    assert re.search(r"scenario\s*:", source)
    assert re.search(r"environment\s*:", source)
    assert "local-dev" in source


def test_observability_compose_exposes_grafana_and_prometheus():
    compose_path = ROOT / "docker-compose.observability.yml"
    assert compose_path.exists()

    compose = yaml.safe_load(compose_path.read_text())
    services = compose["services"]

    assert {"prometheus", "grafana"} <= set(services)
    assert "9090:9090" in services["prometheus"]["ports"]
    assert "3000:3000" in services["grafana"]["ports"]
    assert any("remote-write-receiver" in str(item) for item in services["prometheus"].get("command", []))

    volumes = compose["volumes"]
    assert {"prometheus_data", "grafana_data"} <= set(volumes)


def test_grafana_provisioning_references_repo_dashboard():
    datasource_path = ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    dashboards_path = ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
    dashboard_json_path = ROOT / "monitoring" / "grafana" / "dashboards" / "k6-benchmark.json"

    assert datasource_path.exists()
    assert dashboards_path.exists()
    assert dashboard_json_path.exists()

    datasource = yaml.safe_load(datasource_path.read_text())
    dashboards = yaml.safe_load(dashboards_path.read_text())

    prometheus_ds = datasource["datasources"][0]
    assert prometheus_ds["type"] == "prometheus"
    assert prometheus_ds["url"] == "http://prometheus:9090"

    provider = dashboards["providers"][0]
    assert "Benchmark" in provider["name"]
    assert provider["options"]["path"].endswith("/dashboards")


def test_k6_dashboard_supports_run_and_scenario_filtering():
    dashboard_path = ROOT / "monitoring" / "grafana" / "dashboards" / "k6-benchmark.json"
    dashboard = json.loads(dashboard_path.read_text())

    templating = dashboard["templating"]["list"]
    variable_names = {item["name"] for item in templating}
    assert {"testid", "scenario"} <= variable_names

    panel_titles = {
        panel["title"]
        for panel in dashboard["panels"]
        if isinstance(panel, dict) and panel.get("title")
    }
    assert {
        "Request Rate",
        "Total Requests",
        "Error Rate",
        "Dropped Iterations",
        "Virtual Users",
        "Avg Latency",
        "Median Latency",
        "P95 Latency",
        "P99 Latency",
        "Max Latency",
    } <= panel_titles
