#!/usr/bin/env python3
"""Render a markdown report from a k6 summary export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmarking.k6_report import build_report, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown report from k6 summary JSON.")
    parser.add_argument("--summary-json", required=True, help="Path to a k6 --summary-export JSON file.")
    parser.add_argument("--scenario", required=True, help="Scenario name to print in the report.")
    parser.add_argument("--environment", required=True, help="Environment label, for example local-dev.")
    parser.add_argument("--target", required=True, help="Human-readable load target, for example 5 req/s for 1 minute.")
    parser.add_argument("--notes", default="", help="Optional notes to include in the report.")
    parser.add_argument("--output", required=True, help="Where to write the markdown report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_path = Path(args.output)

    summary = json.loads(summary_path.read_text())
    report = build_report(summary, scenario_name=args.scenario)
    markdown = render_markdown(
        report,
        environment_label=args.environment,
        target_description=args.target,
        notes=args.notes or None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    print(output_path)


if __name__ == "__main__":
    main()
