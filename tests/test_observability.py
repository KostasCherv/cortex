"""Tests for observability helpers and redaction behavior."""

from unittest.mock import MagicMock

from src.observability.context import build_trace_metadata, build_trace_tags
from src.observability.langsmith import end_workflow_run, start_step_span, start_workflow_run
from src.observability.redaction import REDACTED, redact_payload


def test_redaction_default_censors_sensitive_keys():
    payload = {
        "query": "secret query",
        "count": 3,
        "nested": {"prompt": "do not leak", "ok": True},
    }
    out = redact_payload(payload, mode="redacted_default")
    assert out["query"] == REDACTED
    assert out["count"] == 3
    assert out["nested"]["prompt"] == REDACTED
    assert out["nested"]["ok"] is True


def test_redaction_metadata_only_shapes_data():
    payload = {"query": "abc", "items": [1, 2, 3]}
    out = redact_payload(payload, mode="metadata_only")
    assert out == {"type": "dict", "size": 2}


def test_workflow_run_context_populates_metadata_when_disabled(monkeypatch):
    from src.observability import langsmith as ls
    monkeypatch.setattr(ls.settings, "langsmith_tracing", False)

    with start_workflow_run(
        entrypoint="test",
        query="hello",
    ) as ctx:
        assert ctx.workflow_id
        assert ctx.entrypoint == "test"
        assert ctx.tracing_enabled is False

        metadata = build_trace_metadata({"k": "v"})
        tags = build_trace_tags(["extra"])
        assert metadata["workflow_id"] == ctx.workflow_id
        assert metadata["entrypoint"] == "test"
        assert "entrypoint:test" in tags
        assert "extra" in tags

        # Should no-op when tracing is disabled.
        with start_step_span(name="dummy-step", run_type="tool"):
            pass

    # Should no-op when run is disabled.
    end_workflow_run(ctx, status="success", outputs={"ok": True})


def test_langfuse_generation_noops_when_disabled(monkeypatch):
    from src.observability import langfuse as lf

    monkeypatch.setattr(lf.settings, "langfuse_enabled", False)
    monkeypatch.setattr(lf, "get_client", lambda: None)

    with lf.observe_llm_generation(
        step_name="summarize",
        model="gpt-4o-mini",
        prompt="hello",
        metadata={"session_id": "session-1"},
    ) as generation:
        assert generation.trace_id is None
        assert generation.observation_id is None


def test_langfuse_generation_updates_success_metadata(monkeypatch):
    from src.observability import langfuse as lf

    update_mock = MagicMock()
    end_mock = MagicMock()
    generation = MagicMock(
        trace_id="trace-1",
        id="obs-1",
        update=update_mock,
        end=end_mock,
    )
    client = MagicMock()
    client.start_generation.return_value = generation

    monkeypatch.setattr(lf.settings, "langfuse_enabled", True)
    monkeypatch.setattr(lf.settings, "langfuse_release", "test-release")
    monkeypatch.setattr(lf, "get_client", lambda: client)

    with lf.observe_llm_generation(
        step_name="summarize",
        model="gpt-4o-mini",
        prompt="hello",
        metadata={"run_id": "run-1"},
    ) as observed:
        observed.mark_output("world")

    client.start_generation.assert_called_once()
    update_mock.assert_called()
    end_mock.assert_called_once()


def test_langfuse_submit_score_forwards_expected_payload(monkeypatch):
    from src.observability import langfuse as lf

    create_score = MagicMock()
    flush = MagicMock()
    client = MagicMock(create_score=create_score, flush=flush)

    monkeypatch.setattr(lf.settings, "langfuse_enabled", True)
    monkeypatch.setattr(lf, "get_client", lambda: client)

    lf.submit_user_feedback_score(
        trace_id="trace-1",
        observation_id="obs-1",
        helpful=True,
        comment="Useful answer",
    )

    create_score.assert_called_once_with(
        name="user_helpful",
        value=1.0,
        data_type="BOOLEAN",
        trace_id="trace-1",
        observation_id="obs-1",
        comment="Useful answer",
    )
    flush.assert_called_once()


def test_langfuse_start_generation_returns_none_on_typeerror_fallback_failure():
    from src.observability import langfuse as lf

    client = MagicMock()
    client.start_generation.side_effect = [TypeError("bad args"), RuntimeError("still bad")]

    observed = lf._start_generation(  # noqa: SLF001 - intentional direct helper test
        client,
        name="summarize.generation",
        model="gpt-4o-mini",
        prompt="hello",
        metadata={},
        trace_id="trace-1",
    )
    assert observed is None


def test_langfuse_generation_includes_env_and_release_metadata(monkeypatch):
    from src.observability import langfuse as lf

    generation = MagicMock(trace_id="trace-1", id="obs-1")
    client = MagicMock()
    client.start_generation.return_value = generation

    monkeypatch.setattr(lf, "get_client", lambda: client)
    monkeypatch.setattr(lf.settings, "langfuse_enabled", True)
    monkeypatch.setattr(lf.settings, "langfuse_env", "staging")
    monkeypatch.setattr(lf.settings, "langfuse_release", "2026.05.06")

    with lf.observe_llm_generation(
        step_name="summarize",
        model="gpt-4o-mini",
        prompt="hello",
        metadata={"run_id": "run-1"},
    ):
        pass

    called_metadata = client.start_generation.call_args.kwargs["metadata"]
    assert called_metadata["environment"] == "staging"
    assert called_metadata["release"] == "2026.05.06"
