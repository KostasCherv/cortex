from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.deps import AgentLoopResult
from src.api.endpoints import app


client = TestClient(app)


def test_internal_benchmark_endpoint_requires_secret(monkeypatch):
    monkeypatch.setattr("src.api.endpoints.settings.internal_dispatch_secret", "bench-secret")

    response = client.post("/internal/benchmark/agent-loop", json={"message": "hello"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"


def test_internal_benchmark_endpoint_runs_agent_loop(monkeypatch):
    monkeypatch.setattr("src.api.endpoints.settings.internal_dispatch_secret", "bench-secret")

    with patch(
        "src.api.routers.internal._run_agent_loop",
        new=AsyncMock(
            return_value=AgentLoopResult(
                answer="Short answer",
                web_used=False,
                citations=[{"title": "one"}],
            )
        ),
    ) as run_loop:
        response = client.post(
            "/internal/benchmark/agent-loop",
            headers={"Authorization": "Bearer bench-secret"},
            json={
                "message": "Summarize the context",
                "bind_tools": False,
                "rag_context": "Sample benchmark context",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Short answer"
    assert payload["answer_chars"] == len("Short answer")
    assert payload["citation_count"] == 1
    assert payload["bind_tools"] is False
    assert payload["wall_ms"] >= 0
    run_loop.assert_awaited_once()


def test_internal_benchmark_endpoint_reuses_initialized_composio(monkeypatch):
    monkeypatch.setattr("src.api.endpoints.settings.internal_dispatch_secret", "bench-secret")
    monkeypatch.setattr("src.api.endpoints.settings.composio_enabled", True)

    fake_manager = type(
        "FakeManager",
        (),
        {
            "_initialized": True,
            "get_connected_app_names": lambda self: ["alpha_vantage"],
        },
    )()

    with patch("src.api.routers.internal.get_composio_toolset_manager", return_value=fake_manager), patch(
        "src.api.routers.internal.initialize_composio_toolset",
        new=AsyncMock(),
    ) as initialize_composio, patch(
        "src.api.routers.internal._run_agent_loop",
        new=AsyncMock(return_value=AgentLoopResult(answer="ok", web_used=False, citations=[])),
    ):
        response = client.post(
            "/internal/benchmark/agent-loop",
            headers={"Authorization": "Bearer bench-secret"},
            json={
                "message": "Summarize the context",
                "bind_tools": True,
                "rag_context": "Sample benchmark context",
            },
        )

    assert response.status_code == 200
    initialize_composio.assert_not_awaited()
