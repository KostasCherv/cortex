"""Helpers for turning k6 summary exports into shareable reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class K6Report:
    scenario_name: str
    duration_ms: float | None
    total_requests: int
    request_rate: float | None
    error_rate: float | None
    avg_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    active_vus: float | None
    waiting_avg_ms: float | None
    blocked_avg_ms: float | None
    bottleneck_hints: list[str]


def _metric_values(summary: dict, metric_name: str) -> dict:
    metric = (summary.get("metrics") or {}).get(metric_name) or {}
    values = metric.get("values")
    if isinstance(values, dict):
        return values
    if isinstance(metric, dict):
        return metric
    return {}


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _bottleneck_hints(*, error_rate: float | None, avg_ms: float | None, waiting_avg_ms: float | None, blocked_avg_ms: float | None) -> list[str]:
    hints: list[str] = []
    if error_rate is not None and error_rate > 0.01:
        hints.append("Error rate exceeds 1%, so reliability is the first limit before throughput.")
    if waiting_avg_ms is not None and avg_ms is not None and waiting_avg_ms >= avg_ms * 0.8:
        hints.append("Application processing time dominates request latency.")
    if blocked_avg_ms is not None and avg_ms is not None and blocked_avg_ms >= max(avg_ms * 0.15, 20.0):
        hints.append("Connection blocking is noticeable; inspect local socket pressure, DNS, or client-side concurrency.")
    if not hints:
        hints.append("No obvious single bottleneck surfaced from summary metrics alone; inspect traces and host metrics alongside this run.")
    return hints


def build_report(summary: dict, *, scenario_name: str) -> K6Report:
    duration = _as_float(((summary.get("state") or {}).get("testRunDurationMs")))
    http_reqs = _metric_values(summary, "http_reqs")
    http_failed = _metric_values(summary, "http_req_failed")
    http_duration = _metric_values(summary, "http_req_duration")
    http_waiting = _metric_values(summary, "http_req_waiting")
    http_blocked = _metric_values(summary, "http_req_blocked")
    vus = _metric_values(summary, "vus")

    avg_ms = _as_float(http_duration.get("avg"))
    error_rate = _as_float(http_failed.get("rate"))
    if error_rate is None:
        error_rate = _as_float(http_failed.get("value"))
    waiting_avg_ms = _as_float(http_waiting.get("avg"))
    blocked_avg_ms = _as_float(http_blocked.get("avg"))

    return K6Report(
        scenario_name=scenario_name,
        duration_ms=duration,
        total_requests=_as_int(http_reqs.get("count")),
        request_rate=_as_float(http_reqs.get("rate")),
        error_rate=error_rate,
        avg_ms=avg_ms,
        median_ms=_as_float(http_duration.get("med")),
        p95_ms=_as_float(http_duration.get("p(95)")),
        p99_ms=_as_float(http_duration.get("p(99)")),
        max_ms=_as_float(http_duration.get("max")),
        active_vus=_as_float(vus.get("max") if "max" in vus else vus.get("value")),
        waiting_avg_ms=waiting_avg_ms,
        blocked_avg_ms=blocked_avg_ms,
        bottleneck_hints=_bottleneck_hints(
            error_rate=error_rate,
            avg_ms=avg_ms,
            waiting_avg_ms=waiting_avg_ms,
            blocked_avg_ms=blocked_avg_ms,
        ),
    )


def _fmt(value: float | None, *, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def render_markdown(
    report: K6Report,
    *,
    environment_label: str,
    target_description: str,
    notes: str | None = None,
) -> str:
    lines = [
        "# k6 Benchmark Report",
        "",
        f"- Scenario: `{report.scenario_name}`",
        f"- Environment: `{environment_label}`",
        f"- Target: {target_description}",
        f"- Duration: {_fmt(report.duration_ms, decimals=0, suffix=' ms')}",
        "",
        "## Results",
        "",
        f"- Total requests: `{report.total_requests}`",
        f"- Request rate: `{_fmt(report.request_rate, suffix=' req/s')}`",
        f"- Error rate: `{_fmt((report.error_rate or 0.0) * 100, decimals=2, suffix='%')}`",
        f"- Avg latency: `{_fmt(report.avg_ms, suffix=' ms')}`",
        f"- Median latency: `{_fmt(report.median_ms, suffix=' ms')}`",
        f"- P95 latency: `{_fmt(report.p95_ms, suffix=' ms')}`",
        f"- P99 latency: `{_fmt(report.p99_ms, suffix=' ms')}`",
        f"- Max latency: `{_fmt(report.max_ms, suffix=' ms')}`",
        f"- Peak VUs: `{_fmt(report.active_vus, decimals=0)}`",
        "",
        "## Bottleneck Hints",
        "",
    ]
    lines.extend(f"- {hint}" for hint in report.bottleneck_hints)
    if notes:
        lines.extend(["", "## Notes", "", notes])
    return "\n".join(lines) + "\n"
