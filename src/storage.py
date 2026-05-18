"""Supabase Storage adapter for RAG resource file management."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx

from src.config import settings
from src.supabase_keys import supabase_api_headers


@dataclass
class SupabaseObjectRef:
    bucket: str
    key: str


def build_storage_uri(bucket: str, key: str) -> str:
    return f"supabase://{bucket}/{key}"


def parse_storage_uri(storage_uri: str) -> SupabaseObjectRef:
    prefix = "supabase://"
    if not storage_uri.startswith(prefix):
        raise ValueError(f"Invalid Supabase storage URI: {storage_uri}")
    rest = storage_uri[len(prefix):]
    bucket, sep, key = rest.partition("/")
    if not sep or not bucket or not key:
        raise ValueError(f"Invalid Supabase storage URI: {storage_uri}")
    return SupabaseObjectRef(bucket=bucket, key=key)


class SupabaseStorageAdapter:
    """Minimal Storage API wrapper using Supabase secret-key auth."""

    def __init__(self) -> None:
        if not settings.supabase_url or not settings.supabase_secret_key:
            raise RuntimeError(
                "Supabase storage requires SUPABASE_URL and SUPABASE_SECRET_KEY."
            )
        self._base = settings.supabase_url.rstrip("/")
        self._key = settings.supabase_secret_key

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        return supabase_api_headers(self._key, content_type=content_type)

    async def upload_bytes(self, *, key: str, content: bytes, content_type: str) -> str:
        bucket = settings.rag_storage_bucket
        encoded_key = quote(key, safe="/")
        url = f"{self._base}/storage/v1/object/{bucket}/{encoded_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    **self._headers(content_type=content_type),
                    "x-upsert": "true",
                },
                content=content,
            )
        response.raise_for_status()
        return build_storage_uri(bucket, key)

    async def create_signed_download_url(
        self,
        *,
        storage_uri: str,
        expires_in: int | None = None,
    ) -> str:
        ref = parse_storage_uri(storage_uri)
        ttl = expires_in or settings.rag_signed_url_ttl_seconds
        encoded_key = quote(ref.key, safe="/")
        url = f"{self._base}/storage/v1/object/sign/{ref.bucket}/{encoded_key}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                url,
                headers={
                    **self._headers(content_type="application/json"),
                },
                json={"expiresIn": ttl},
            )
        response.raise_for_status()
        payload = response.json()
        signed_path = payload.get("signedURL") or payload.get("signedUrl")
        if not signed_path:
            raise RuntimeError("Supabase storage sign endpoint returned no signed URL.")
        if signed_path.startswith("http"):
            return signed_path
        return f"{self._base}/storage/v1{signed_path}"

    async def delete_object(self, *, storage_uri: str) -> None:
        ref = parse_storage_uri(storage_uri)
        encoded_key = quote(ref.key, safe="/")
        url = f"{self._base}/storage/v1/object/{ref.bucket}/{encoded_key}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.delete(url, headers=self._headers())
        response.raise_for_status()

    async def ensure_bucket_exists(self, *, bucket: str | None = None) -> None:
        target_bucket = bucket or settings.rag_storage_bucket
        list_url = f"{self._base}/storage/v1/bucket"
        async with httpx.AsyncClient(timeout=20.0) as client:
            list_response = await client.get(list_url, headers=self._headers())
            list_response.raise_for_status()
            buckets = list_response.json()
            if any((b.get("id") == target_bucket) for b in buckets):
                return

            create_response = await client.post(
                list_url,
                headers=self._headers(content_type="application/json"),
                json={
                    "id": target_bucket,
                    "name": target_bucket,
                    "public": False,
                },
            )
            create_response.raise_for_status()


async def ensure_rag_storage_ready() -> None:
    """Ensure the configured RAG storage bucket exists."""
    adapter = SupabaseStorageAdapter()
    await adapter.ensure_bucket_exists(bucket=settings.rag_storage_bucket)
