#!/usr/bin/env python3
"""Fail-fast smoke checks for a newly deployed Cortex API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes


def request(base_url: str, path: str, timeout: float, token: str | None = None) -> Response:
    headers = {"Accept": "text/event-stream" if path.endswith("/stream") else "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(urljoin(base_url.rstrip("/") + "/", path.lstrip("/")), headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return Response(
                response.status, response.headers.get_content_type(), response.read(64 * 1024)
            )
    except HTTPError as exc:
        return Response(exc.code, exc.headers.get_content_type(), exc.read(64 * 1024))
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(
            f"request failed: {exc.reason if isinstance(exc, URLError) else exc}"
        ) from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_object(response: Response, endpoint: str) -> dict:
    try:
        value = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{endpoint} did not return valid JSON") from exc
    require(isinstance(value, dict), f"{endpoint} did not return a JSON object")
    return value


def run(base_url: str, timeout: float, token: str | None) -> None:
    health = request(base_url, "/health", timeout)
    require(health.status == 200, f"/health returned HTTP {health.status}, expected 200")
    require(json_object(health, "/health").get("status") == "ok", "/health status is not 'ok'")
    print("PASS /health: live")

    ready = request(base_url, "/ready", timeout)
    require(ready.status == 200, f"/ready returned HTTP {ready.status}, expected 200")
    ready_status = str(json_object(ready, "/ready").get("status", "")).lower()
    require(
        ready_status in {"ok", "ready", "degraded"},
        f"/ready reported unexpected status {ready_status!r}",
    )
    print(f"PASS /ready: {ready_status}")

    protected = request(base_url, "/sessions", timeout)
    require(
        protected.status == 401,
        f"unauthenticated /sessions returned HTTP {protected.status}, expected 401",
    )
    print("PASS /sessions: unauthenticated access rejected")

    stream = request(base_url, "/health/stream", timeout)
    require(stream.status == 200, f"/health/stream returned HTTP {stream.status}, expected 200")
    require(
        stream.content_type == "text/event-stream",
        f"/health/stream content type is {stream.content_type!r}",
    )
    require(
        b"event: ready" in stream.body and b'data: {"status":"ok"}' in stream.body,
        "/health/stream did not emit the ready event",
    )
    print("PASS /health/stream: SSE event received")

    if token:
        authenticated = request(base_url, "/sessions", timeout, token)
        require(
            authenticated.status == 200,
            f"authenticated /sessions returned HTTP {authenticated.status}, expected 200",
        )
        print("PASS /sessions: authenticated request accepted")
    else:
        print("SKIP authenticated /sessions: SMOKE_TEST_TOKEN is not set")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="deployed API origin, for example https://service.run.app")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="per-request timeout in seconds (default: 10)"
    )
    args = parser.parse_args()
    try:
        run(args.base_url, args.timeout, os.environ.get("SMOKE_TEST_TOKEN"))
    except RuntimeError as exc:
        print(f"FAIL post-deployment smoke test: {exc}", file=sys.stderr)
        return 1
    print("PASS all required post-deployment smoke checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
