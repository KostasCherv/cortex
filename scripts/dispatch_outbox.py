#!/usr/bin/env python3
"""Dispatch pending outbox events to Inngest (cron or dev loop)."""

from __future__ import annotations

import argparse
import asyncio

from src.outbox import dispatch_outbox_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch pending outbox events to Inngest.")
    parser.add_argument("--limit", type=int, default=50, help="Max events per run.")
    args = parser.parse_args()
    sent = asyncio.run(dispatch_outbox_events(limit=args.limit))
    print(f"Dispatched {sent} outbox event(s)")


if __name__ == "__main__":
    main()
