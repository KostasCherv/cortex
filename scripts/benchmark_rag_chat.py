#!/usr/bin/env python3
"""Benchmark RAG agent chat latency against the running API or in-process TestClient.

Usage:
  # Live server (requires BENCHMARK_BEARER_TOKEN and BENCHMARK_AGENT_ID in .env or env):
  uv run python scripts/benchmark_rag_chat.py --live

  # In-process (auth override, real integrations):
  uv run python scripts/benchmark_rag_chat.py --inprocess

  # Enable perf headers + deferred suggestions for the run:
  RAG_PERF_HEADERS=true RAG_SUGGESTIONS_DEFERRED=true uv run python scripts/benchmark_rag_chat.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from src.benchmarking.rag_chat import coerce_agent_loop_benchmark_result

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env before importing settings
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.strip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

os.environ.setdefault("INNGEST_DEV", "1")
os.environ.setdefault("INNGEST_SIGNING_KEY", "local-benchmark-signing-key")


def _parse_perf_header(headers: dict) -> dict | None:
    raw = headers.get("x-rag-perf") or headers.get("X-Rag-Perf")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _print_result(label: str, wall_s: float, status: int, perf: dict | None, extra: dict | None = None) -> None:
    print(f"\n=== {label} ===")
    print(f"  status={status}  wall_s={wall_s:.3f}")
    if perf:
        for key in (
            "total_ms",
            "prepare_ms",
            "session_ms",
            "agent_loop_ms",
            "suggestions_ms",
            "persist_ms",
            "tools_bound",
            "tool_skip_reason",
        ):
            if key in perf:
                print(f"  {key}={perf[key]}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}={v}")


def benchmark_live(*, base_url: str, token: str, agent_id: str, message: str, session_id: str | None) -> None:
    import httpx

    url = f"{base_url.rstrip('/')}/api/rag/agents/{agent_id}/chat"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": message, "session_id": session_id}

    t0 = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json=payload, headers=headers)
    wall_s = time.perf_counter() - t0
    perf = _parse_perf_header(dict(response.headers))
    extra = {}
    if response.status_code == 200:
        body = response.json()
        reply = body.get("reply") or {}
        extra["suggestions_count"] = len(reply.get("suggestions") or [])
        extra["reply_chars"] = len(reply.get("content") or "")
    _print_result("live POST /chat", wall_s, response.status_code, perf, extra)
    if response.status_code != 200:
        print(response.text[:500])


def benchmark_inprocess(*, agent_id: str, message: str, session_id: str | None) -> None:
    from fastapi.testclient import TestClient

    from src.api.endpoints import app
    from src.auth import AuthenticatedUser, get_authenticated_user

    def _auth() -> AuthenticatedUser:
        return AuthenticatedUser(user_id=os.environ.get("BENCHMARK_USER_ID", "test-user"), email="bench@example.com")

    app.dependency_overrides[get_authenticated_user] = _auth
    client = TestClient(app)
    payload = {"message": message, "session_id": session_id}

    t0 = time.perf_counter()
    response = client.post(f"/api/rag/agents/{agent_id}/chat", json=payload)
    wall_s = time.perf_counter() - t0
    perf = _parse_perf_header(dict(response.headers))
    extra = {}
    if response.status_code == 200:
        body = response.json()
        reply = body.get("reply") or {}
        extra["suggestions_count"] = len(reply.get("suggestions") or [])
        extra["reply_chars"] = len(reply.get("content") or "")
    _print_result("inprocess POST /chat", wall_s, response.status_code, perf, extra)
    if response.status_code != 200:
        print(response.text[:500])


async def benchmark_integration(
    *,
    agent_id: str,
    user_id: str,
    message: str,
    session_id: str | None,
) -> None:
    """Full prepare + agent loop against real Supabase/Neo4j/Ollama (no HTTP auth)."""
    from src.api.rag_chat_helpers import prepare_agent_rag_chat, resolve_suggestions
    from src.api.endpoints import _run_agent_loop
    from src.api.rag_chat_timing import RagChatTimings

    if os.environ.get("COMPOSIO_ENABLED", "true").lower() in ("1", "true", "yes"):
        from src.tools.composio_toolset import initialize_composio_toolset

        await initialize_composio_toolset()

    timings = RagChatTimings()
    wall_start = time.perf_counter()
    prepared = await prepare_agent_rag_chat(
        agent_id=agent_id,
        user_id=user_id,
        normalized_message=message,
        session_id=session_id,
        timings=timings,
    )
    if prepared is None:
        print("prepare failed: agent not found")
        return
    t_loop = time.perf_counter()
    result = await _run_agent_loop(
        messages=prepared.messages,
        metadata={"bench": True},
        bind_tools=prepared.bind_tools,
    )
    answer = coerce_agent_loop_benchmark_result(result)
    timings.agent_loop_ms = (time.perf_counter() - t_loop) * 1000
    suggestions = await resolve_suggestions(
        query=message,
        answer=answer.answer,
        context=prepared.rag_context.context or "",
        timings=timings,
    )
    timings.total_ms = (time.perf_counter() - wall_start) * 1000
    print("\n=== integration prepare+loop ===")
    print(f"  wall_s={timings.total_ms/1000:.3f}")
    print(f"  prepare_ms={timings.prepare_ms:.0f} session_ms={timings.session_ms:.0f}")
    print(f"  agent_loop_ms={timings.agent_loop_ms:.0f} suggestions_ms={timings.suggestions_ms:.0f}")
    print(f"  tools_bound={timings.tools_bound} tool_skip_reason={timings.tool_skip_reason}")
    print(f"  suggestions_count={len(suggestions)} answer_chars={answer.answer_chars}")


async def benchmark_direct_llm(*, bind_tools: bool, message: str) -> None:
    """Time _run_agent_loop only (no Supabase), for A/B on tool binding."""
    from src.api.rag_chat_helpers import build_agent_messages, should_bind_composio_tools
    from src.api.endpoints import _run_agent_loop
    from src.tools.composio_toolset import get_composio_toolset_manager, initialize_composio_toolset

    should_initialize_composio = bind_tools is not False and os.environ.get(
        "COMPOSIO_ENABLED", "true"
    ).lower() in ("1", "true", "yes")
    if should_initialize_composio:
        await initialize_composio_toolset()

    composio_apps = get_composio_toolset_manager().get_connected_app_names()
    use_tools, reason = should_bind_composio_tools(
        message=message,
        resource_ids=["res-bench"],
        composio_apps=composio_apps,
    )
    if bind_tools is not None:
        use_tools = bind_tools
        reason = "forced"

    messages = build_agent_messages(
        system_instructions="Keep answers brief.",
        history=[],
        rag_context="Sample document context for benchmarking.",
        user_memory_context="",
        composio_apps=composio_apps,
        normalized_message=message,
    )
    t0 = time.perf_counter()
    result = await _run_agent_loop(messages=messages, metadata={"bench": True}, bind_tools=use_tools)
    answer = coerce_agent_loop_benchmark_result(result)
    wall_s = time.perf_counter() - t0
    print("\n=== direct _run_agent_loop ===")
    print(f"  wall_s={wall_s:.3f}  bind_tools={use_tools}  reason={reason}")
    print(f"  answer_chars={answer.answer_chars}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG agent chat latency")
    parser.add_argument("--live", action="store_true", help="Hit running server on port 8010")
    parser.add_argument("--inprocess", action="store_true", help="Use FastAPI TestClient in-process")
    parser.add_argument("--base-url", default=os.environ.get("BENCHMARK_BASE_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--agent-id", default=os.environ.get("BENCHMARK_AGENT_ID", ""))
    parser.add_argument("--message", default="Hello, summarize what you know from my documents in two sentences.")
    parser.add_argument("--session-id", default=None, help="Reuse session to skip new-session title path")
    parser.add_argument("--label", default="benchmark", help="Label printed with results")
    parser.add_argument(
        "--direct-llm",
        action="store_true",
        help="Benchmark _run_agent_loop only (no HTTP/Supabase)",
    )
    parser.add_argument(
        "--bind-tools",
        action="store_true",
        default=None,
        help="Force Composio tool binding on/off in --direct-llm mode",
    )
    parser.add_argument(
        "--no-bind-tools",
        action="store_true",
        help="Force no tool binding in --direct-llm mode",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Benchmark prepare+loop with real backends (set BENCHMARK_USER_ID)",
    )
    args = parser.parse_args()

    if args.integration:
        import asyncio

        user_id = os.environ.get("BENCHMARK_USER_ID", "").strip()
        agent_id = args.agent_id.strip() or os.environ.get("BENCHMARK_AGENT_ID", "").strip()
        if not user_id or not agent_id:
            print("ERROR: --integration requires BENCHMARK_USER_ID and --agent-id", file=sys.stderr)
            sys.exit(1)
        asyncio.run(
            benchmark_integration(
                agent_id=agent_id,
                user_id=user_id,
                message=args.message,
                session_id=args.session_id,
            )
        )
        return

    if args.direct_llm:
        import asyncio

        forced = None
        if args.bind_tools:
            forced = True
        if args.no_bind_tools:
            forced = False
        asyncio.run(benchmark_direct_llm(bind_tools=forced, message=args.message))
        return

    if not args.live and not args.inprocess:
        args.inprocess = True

    session_id = args.session_id
    if session_id is None and os.environ.get("BENCHMARK_NEW_SESSION", "").lower() in ("1", "true", "yes"):
        session_id = None
    elif session_id is None:
        session_id = str(uuid.uuid4())

    print(f"label={args.label}")
    print(f"message={args.message!r}")
    print(f"session_id={session_id}")
    print(f"RAG_PERF_HEADERS={os.environ.get('RAG_PERF_HEADERS', '')}")
    print(f"RAG_SUGGESTIONS_DEFERRED={os.environ.get('RAG_SUGGESTIONS_DEFERRED', '')}")
    print(f"COMPOSIO_ENABLED={os.environ.get('COMPOSIO_ENABLED', '')}")

    if args.live:
        token = os.environ.get("BENCHMARK_BEARER_TOKEN", "").strip()
        agent_id = args.agent_id.strip()
        if not token or not agent_id:
            print("ERROR: --live requires BENCHMARK_BEARER_TOKEN and BENCHMARK_AGENT_ID", file=sys.stderr)
            sys.exit(1)
        benchmark_live(
            base_url=args.base_url,
            token=token,
            agent_id=agent_id,
            message=args.message,
            session_id=session_id,
        )
    if args.inprocess:
        agent_id = args.agent_id.strip() or "agent-1"
        benchmark_inprocess(agent_id=agent_id, message=args.message, session_id=session_id)


if __name__ == "__main__":
    main()
