from scripts import post_deploy_smoke as smoke


def _response(
    status: int, content_type: str = "application/json", body: str = "{}"
) -> smoke.Response:
    return smoke.Response(status, content_type, body.encode())


def test_run_checks_required_endpoints(monkeypatch, capsys):
    responses = {
        "/health": _response(200, body='{"status":"ok"}'),
        "/ready": _response(200, body='{"status":"ready"}'),
        "/sessions": _response(401),
        "/health/stream": _response(
            200,
            "text/event-stream",
            'event: ready\ndata: {"status":"ok"}\n\n',
        ),
    }
    monkeypatch.setattr(smoke, "request", lambda _base, path, _timeout, token=None: responses[path])

    smoke.run("https://example.test", 1, None)

    assert "PASS all" not in capsys.readouterr().out


def test_run_checks_optional_authenticated_request(monkeypatch):
    def fake_request(_base, path, _timeout, token=None):
        if path == "/health":
            return _response(200, body='{"status":"ok"}')
        if path == "/ready":
            return _response(200, body='{"status":"degraded"}')
        if path == "/health/stream":
            return _response(200, "text/event-stream", 'event: ready\ndata: {"status":"ok"}\n\n')
        return _response(200 if token else 401)

    monkeypatch.setattr(smoke, "request", fake_request)
    smoke.run("https://example.test", 1, "secret-value")


def test_run_fails_when_readiness_is_unhealthy(monkeypatch):
    def fake_request(_base, path, _timeout, token=None):
        if path == "/health":
            return _response(200, body='{"status":"ok"}')
        return _response(503, body='{"status":"unhealthy"}')

    monkeypatch.setattr(smoke, "request", fake_request)

    try:
        smoke.run("https://example.test", 1, None)
    except RuntimeError as exc:
        assert "/ready returned HTTP 503" in str(exc)
    else:
        raise AssertionError("unhealthy readiness should fail the smoke test")
