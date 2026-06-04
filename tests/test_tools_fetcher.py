"""Tests for src/tools/fetcher.py"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.errors import FetchError
from src.tools.fetcher import _validate_url, fetch_url_content


@pytest.mark.asyncio
async def test_fetch_url_content_returns_text():
    html = "<html><body><p>Hello World</p></body></html>"
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("src.tools.fetcher.httpx.AsyncClient", return_value=mock_client):
        from src.tools.fetcher import fetch_url_content

        result = await fetch_url_content("https://example.com")

    assert "Hello World" in result


@pytest.mark.asyncio
async def test_fetch_url_content_raises_on_http_error():
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 404
    http_error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=http_error)

    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("93.184.216.34")):
        with patch("src.tools.fetcher.httpx.AsyncClient", return_value=mock_client):
            from src.tools.fetcher import fetch_url_content

            with pytest.raises(FetchError, match="404"):
                await fetch_url_content("https://bad.example.com")


def test_clean_html_strips_tags():
    from src.tools.fetcher import clean_html

    html = "<html><head><style>body{}</style></head><body><p>Clean text</p></body></html>"
    result = clean_html(html)
    assert "Clean text" in result
    assert "<" not in result


def test_clean_html_truncates_long_content():
    from src.tools.fetcher import clean_html, _MAX_CHARS

    long_text = "word " * 10_000
    html = f"<body><p>{long_text}</p></body>"
    result = clean_html(html)
    assert len(result) <= _MAX_CHARS


# Helper to make getaddrinfo return a specific IP
def _mock_getaddrinfo(ip: str):
    return [(None, None, None, None, (ip, 0))]


def test_validate_url_raises_for_private_ip():
    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("192.168.1.1")):
        with pytest.raises(FetchError, match="private/reserved"):
            _validate_url("http://internal.example.com/secret")


def test_validate_url_raises_for_loopback():
    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("127.0.0.1")):
        with pytest.raises(FetchError, match="private/reserved"):
            _validate_url("http://localhost/admin")


def test_validate_url_raises_for_link_local():
    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("169.254.169.254")):
        with pytest.raises(FetchError, match="private/reserved"):
            _validate_url("http://169.254.169.254/latest/meta-data/")


def test_validate_url_raises_for_ipv6_loopback():
    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("::1")):
        with pytest.raises(FetchError, match="private/reserved"):
            _validate_url("http://ip6-localhost/")


def test_validate_url_raises_for_non_http_scheme():
    with pytest.raises(FetchError, match="scheme"):
        _validate_url("ftp://example.com/file.txt")


def test_validate_url_raises_for_missing_hostname():
    with pytest.raises(FetchError, match="missing hostname"):
        _validate_url("https:///path")


def test_validate_url_passes_for_public_ip():
    with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("93.184.216.34")):
        _validate_url("https://example.com")  # should not raise


@pytest.mark.asyncio
async def test_fetch_url_content_calls_validate_url():
    with patch("src.tools.fetcher._validate_url") as mock_validate:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "<html><body>hello</body></html>"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.tools.fetcher.httpx.AsyncClient", return_value=mock_client):
            await fetch_url_content("https://example.com")

        mock_validate.assert_called_once_with("https://example.com")
