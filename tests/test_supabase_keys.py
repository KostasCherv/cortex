"""Tests for Supabase API key header helpers."""

from src.supabase_keys import supabase_api_headers


def test_secret_key_uses_apikey_only():
    key = "sb_secret_example_abc12345"
    headers = supabase_api_headers(key)
    assert headers == {"apikey": key}


def test_user_access_token_in_authorization():
    server_key = "sb_secret_example_abc12345"
    user_token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.sig"
    headers = supabase_api_headers(server_key, user_access_token=user_token)
    assert headers["apikey"] == server_key
    assert headers["Authorization"] == f"Bearer {user_token}"
