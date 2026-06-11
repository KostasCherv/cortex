"""Supabase-backed persistence for user sessions and runs."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from src.cache.client import get_cache
from src.config import settings
from src.supabase_keys import supabase_api_headers

if TYPE_CHECKING:
    from src.sessions import ConversationTurn, Session, SessionRun

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 20.0


class SupabaseSessionStore:
    """Persist sessions in Supabase PostgREST with strict user scoping."""

    def __init__(self) -> None:
        if not settings.supabase_url or not settings.supabase_secret_key:
            raise RuntimeError(
                "Supabase persistence is not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY."
            )
        self._base_url = f"{settings.supabase_url.rstrip('/')}/rest/v1"
        self._headers = supabase_api_headers(
            settings.supabase_secret_key,
            content_type="application/json",
        )
        self._session_run_extended_fields_supported: bool | None = None

    _SESSION_RUN_BASE_SELECT = (
        "id,query,source_urls,report,status,error_details,"
        "latest_node,latest_event_at,partial_report,created_at"
    )
    _SESSION_RUN_EXTENDED_SELECT = (
        "id,query,source_urls,report,status,error_details,"
        "latest_node,latest_event_at,partial_report,"
        "langfuse_trace_id,langfuse_observation_id,"
        "feedback_submitted_at,feedback_helpful,created_at"
    )
    _SESSION_RUN_OPTIONAL_FIELDS = {
        "langfuse_trace_id",
        "langfuse_observation_id",
        "feedback_submitted_at",
        "feedback_helpful",
    }
    _SESSION_RUN_NON_BASE_FIELDS = {
        "status",
        "error_details",
        "latest_node",
        "latest_event_at",
        "partial_report",
        "langfuse_trace_id",
        "langfuse_observation_id",
        "feedback_submitted_at",
        "feedback_helpful",
    }
    _CACHE_PREFIX_SESSIONS_LIST = "sessions:list"
    _CACHE_PREFIX_RAG_RESOURCES_LIST = "rag:resources:list"
    _CACHE_PREFIX_RAG_AGENTS_LIST = "rag:agents:list"
    _CACHE_PREFIX_RAG_CHAT_SESSIONS_LIST = "rag:chat:sessions:list"
    _CACHE_PREFIX_RAG_CHAT_MESSAGES_LIST = "rag:chat:messages:list"
    _CACHE_PREFIX_PRD_PLANS_LIST = "planner:prd:list"
    _CACHE_PREFIX_ITINERARY_SESSIONS_LIST = "planner:itinerary:list"
    _CACHE_PREFIX_USER_MEMORY = "user:memory"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        headers = dict(self._headers)
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.request(
                method,
                f"{self._base_url}/{path}",
                params=params,
                json=json_body,
                headers=headers,
            )
        response.raise_for_status()
        return response

    @staticmethod
    def _is_missing_session_run_column_error(exc: httpx.HTTPStatusError) -> bool:
        if exc.response.status_code != 400:
            return False
        try:
            payload = exc.response.json()
        except Exception:
            payload = {}
        message = str(payload.get("message", "")).lower()
        code = str(payload.get("code", "")).upper()
        if "does not exist" in message and (
            "session_runs" in message
            or "langfuse_" in message
            or "feedback_" in message
        ):
            return True
        return code.startswith("PGRST")

    def _session_run_select(self) -> str:
        if self._session_run_extended_fields_supported is False:
            return self._SESSION_RUN_BASE_SELECT
        return self._SESSION_RUN_EXTENDED_SELECT

    def _strip_optional_session_run_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in self._SESSION_RUN_OPTIONAL_FIELDS
        }

    def _strip_non_base_session_run_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in self._SESSION_RUN_NON_BASE_FIELDS
        }

    @staticmethod
    def _cache_key(prefix: str, raw: str) -> str:
        cache = get_cache()
        if cache is None:
            return ""
        return cache.hash_key(prefix, raw)

    async def _cache_get_list(self, key: str) -> list[dict[str, Any]] | None:
        cache = get_cache()
        if cache is None or not key:
            return None
        cached = await cache.get(key)
        if isinstance(cached, list):
            return cached
        return None

    async def _cache_set_list(self, key: str, value: list[dict[str, Any]]) -> None:
        cache = get_cache()
        if cache is None or not key:
            return
        await cache.set(key, value, settings.redis_cache_ttl_db_list_seconds)

    async def _cache_delete(self, key: str) -> None:
        cache = get_cache()
        if cache is None or not key:
            return
        await cache.delete(key)

    async def _invalidate_sessions_list_cache(self, user_id: str) -> None:
        key = self._cache_key(self._CACHE_PREFIX_SESSIONS_LIST, user_id)
        await self._cache_delete(key)

    async def _invalidate_rag_resources_list_cache(self, owner_id: str, workspace_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_RAG_RESOURCES_LIST,
            f"{owner_id}:{workspace_id}",
        )
        await self._cache_delete(key)

    async def _invalidate_rag_agents_list_cache(self, owner_id: str, workspace_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_RAG_AGENTS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        await self._cache_delete(key)

    async def _invalidate_rag_chat_sessions_list_cache(self, owner_id: str, agent_id: str | None, chat_scope: str) -> None:
        agent_key = agent_id or "__workspace__"
        key = self._cache_key(
            self._CACHE_PREFIX_RAG_CHAT_SESSIONS_LIST,
            f"{owner_id}:{chat_scope}:{agent_key}",
        )
        await self._cache_delete(key)

    async def _invalidate_rag_chat_messages_list_cache(self, owner_id: str, session_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_RAG_CHAT_MESSAGES_LIST,
            f"{owner_id}:{session_id}",
        )
        await self._cache_delete(key)

    async def _invalidate_prd_plans_list_cache(self, owner_id: str, workspace_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_PRD_PLANS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        await self._cache_delete(key)

    async def _invalidate_itinerary_sessions_list_cache(self, owner_id: str, workspace_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_ITINERARY_SESSIONS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        await self._cache_delete(key)

    async def _invalidate_user_memory_cache(self, owner_id: str, workspace_id: str) -> None:
        key = self._cache_key(
            self._CACHE_PREFIX_USER_MEMORY,
            f"{owner_id}:{workspace_id}",
        )
        await self._cache_delete(key)

    @staticmethod
    def _is_bad_request(exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code == 400

    @staticmethod
    def _session_run_from_row(row: dict[str, Any]) -> "SessionRun":
        from src.sessions import SessionRun

        return SessionRun(
            run_id=row["id"],
            query=row.get("query", ""),
            source_urls=row.get("source_urls") or [],
            report=row.get("report", ""),
            status=row.get("status", "completed"),
            error_details=row.get("error_details"),
            latest_node=row.get("latest_node"),
            latest_event_at=row.get("latest_event_at"),
            partial_report=row.get("partial_report") or "",
            langfuse_trace_id=row.get("langfuse_trace_id"),
            langfuse_observation_id=row.get("langfuse_observation_id"),
            feedback_submitted_at=row.get("feedback_submitted_at"),
            feedback_helpful=row.get("feedback_helpful"),
            created_at=row.get("created_at", ""),
        )

    async def _request_session_runs(
        self,
        *,
        params: dict[str, Any],
    ) -> httpx.Response:
        request_params = dict(params)
        request_params["select"] = self._session_run_select()
        try:
            response = await self._request("GET", "session_runs", params=request_params)
            self._session_run_extended_fields_supported = True
            return response
        except httpx.HTTPStatusError as exc:
            if (
                self._session_run_extended_fields_supported is False
                or not self._is_missing_session_run_column_error(exc)
            ):
                raise
            self._session_run_extended_fields_supported = False
            fallback_params = dict(params)
            fallback_params["select"] = self._SESSION_RUN_BASE_SELECT
            return await self._request("GET", "session_runs", params=fallback_params)

    async def _write_session_run_payload(
        self,
        *,
        method: str,
        params: dict[str, Any] | None,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        request_payload = (
            self._strip_optional_session_run_fields(payload)
            if self._session_run_extended_fields_supported is False
            else payload
        )
        try:
            response = await self._request(
                method,
                "session_runs",
                params=params,
                json_body=request_payload,
                extra_headers=extra_headers,
            )
            self._session_run_extended_fields_supported = True
            return response
        except httpx.HTTPStatusError as exc:
            # If we are already in legacy mode, or this is not a schema-like 400,
            # fail fast and surface the original error.
            if self._session_run_extended_fields_supported is False:
                raise
            if not self._is_bad_request(exc) and not self._is_missing_session_run_column_error(exc):
                raise

            fallback_payload = self._strip_optional_session_run_fields(payload)
            fallback_base_payload = self._strip_non_base_session_run_fields(payload)

            # Try progressively older payload shapes:
            # 1) no LangFuse/feedback fields
            # 2) base schema only (for older DBs missing run status/live progress columns)
            candidates: list[dict[str, Any]] = []
            if fallback_payload != request_payload:
                candidates.append(fallback_payload)
            if fallback_base_payload != request_payload and fallback_base_payload != fallback_payload:
                candidates.append(fallback_base_payload)
            if not candidates:
                raise

            self._session_run_extended_fields_supported = False
            last_exc: httpx.HTTPStatusError = exc
            for candidate in candidates:
                try:
                    return await self._request(
                        method,
                        "session_runs",
                        params=params,
                        json_body=candidate,
                        extra_headers=extra_headers,
                    )
                except httpx.HTTPStatusError as retry_exc:
                    last_exc = retry_exc
                    if not self._is_bad_request(retry_exc):
                        raise

            logger.warning(
                "[sessions] session_runs %s fallback retries exhausted; re-raising last 400 response.",
                method,
            )
            raise last_exc

    async def create_session(self, user_id: str, title: str) -> Session:
        from src.sessions import Session

        session_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        payload = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": created_at,
        }
        await self._request("POST", "research_sessions", json_body=payload)
        await self._invalidate_sessions_list_cache(user_id)
        return Session(
            session_id=session_id,
            title=title,
            runs=[],
            conversation=[],
            created_at=created_at,
        )

    async def list_sessions(self, user_id: str) -> list[dict[str, str]]:
        """List lightweight session summaries for a user."""
        cache_key = self._cache_key(self._CACHE_PREFIX_SESSIONS_LIST, user_id)
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "research_sessions",
            params={
                "select": "id,title,created_at",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
            },
        )
        rows = response.json()
        if not rows:
            return []

        session_ids = [row["id"] for row in rows]
        runs_resp = await self._request(
            "GET",
            "session_runs",
            params={
                "select": "session_id,status,created_at",
                "session_id": f"in.({','.join(session_ids)})",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
            },
        )
        latest_status_by_session: dict[str, str] = {}
        for run_row in runs_resp.json():
            latest_status_by_session.setdefault(run_row["session_id"], run_row.get("status", "completed"))

        result = [
            {
                "session_id": row["id"],
                "title": row.get("title") or "New session",
                "created_at": row.get("created_at", ""),
                "latest_run_status": latest_status_by_session.get(row["id"]),
            }
            for row in rows
        ]
        await self._cache_set_list(cache_key, result)
        return result

    async def get_session(self, session_id: str, user_id: str) -> Session | None:
        return await self._fetch_session_from_db(session_id, user_id)

    async def _fetch_session_from_db(
        self, session_id: str, user_id: str
    ) -> Session | None:
        from src.sessions import ConversationTurn, Session

        session_resp = await self._request(
            "GET",
            "research_sessions",
            params={
                "select": "id,title,created_at",
                "id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        session_rows = session_resp.json()
        if not session_rows:
            return None

        runs_resp = await self._request_session_runs(
            params={
                "session_id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "order": "created_at.asc",
            }
        )
        run_rows = runs_resp.json()
        runs = [self._session_run_from_row(row) for row in run_rows]

        turns_resp = await self._request(
            "GET",
            "conversation_turns",
            params={
                "select": "role,content,run_id,citations,suggestions,created_at",
                "session_id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "order": "created_at.asc",
            },
        )
        turn_rows = turns_resp.json()
        conversation = [
            ConversationTurn(
                role=row.get("role", "user"),
                content=row.get("content", ""),
                run_id=row.get("run_id"),
                citations=row.get("citations") or [],
                suggestions=row.get("suggestions") or [],
                created_at=row.get("created_at", ""),
            )
            for row in turn_rows
        ]

        session_row = session_rows[0]
        return Session(
            session_id=session_row["id"],
            title=session_row.get("title") or "New session",
            runs=runs,
            conversation=conversation,
            created_at=session_row.get("created_at", ""),
        )

    async def update_session_title(
        self,
        *,
        user_id: str,
        session_id: str,
        title: str,
    ) -> bool:
        response = await self._request(
            "PATCH",
            "research_sessions",
            params={
                "id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
            },
            json_body={"title": title},
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        await self._invalidate_sessions_list_cache(user_id)
        return bool(rows)

    async def delete_session(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "research_sessions",
            params={
                "id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        await self._invalidate_sessions_list_cache(user_id)
        return bool(rows)

    async def append_run(
        self,
        *,
        user_id: str,
        session_id: str,
        run: SessionRun,
    ) -> None:
        payload = {
            "id": run.run_id,
            "session_id": session_id,
            "user_id": user_id,
            "query": run.query,
            "source_urls": run.source_urls,
            "report": run.report,
            "status": run.status,
            "error_details": run.error_details,
            "latest_node": run.latest_node,
            "latest_event_at": run.latest_event_at,
            "partial_report": run.partial_report,
            "langfuse_trace_id": run.langfuse_trace_id,
            "langfuse_observation_id": run.langfuse_observation_id,
            "feedback_submitted_at": run.feedback_submitted_at,
            "feedback_helpful": run.feedback_helpful,
            "created_at": run.created_at,
        }
        await self._write_session_run_payload(
            method="POST",
            params=None,
            payload=payload,
        )
        await self._invalidate_sessions_list_cache(user_id)

    async def create_session_run(
        self,
        *,
        user_id: str,
        session_id: str,
        run: SessionRun,
    ) -> None:
        payload = {
            "id": run.run_id,
            "session_id": session_id,
            "user_id": user_id,
            "query": run.query,
            "source_urls": run.source_urls,
            "report": run.report,
            "status": run.status,
            "error_details": run.error_details,
            "latest_node": run.latest_node,
            "latest_event_at": run.latest_event_at,
            "partial_report": run.partial_report,
            "langfuse_trace_id": run.langfuse_trace_id,
            "langfuse_observation_id": run.langfuse_observation_id,
            "feedback_submitted_at": run.feedback_submitted_at,
            "feedback_helpful": run.feedback_helpful,
            "created_at": run.created_at,
        }
        await self._write_session_run_payload(
            method="POST",
            params=None,
            payload=payload,
        )
        await self._invalidate_sessions_list_cache(user_id)

    async def update_session_run(
        self,
        *,
        run_id: str,
        user_id: str,
        session_id: str,
        patch: dict[str, Any],
    ) -> bool:
        update_body = dict(patch)
        response = await self._write_session_run_payload(
            method="PATCH",
            params={
                "id": f"eq.{run_id}",
                "user_id": f"eq.{user_id}",
                "session_id": f"eq.{session_id}",
            },
            payload=update_body,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        await self._invalidate_sessions_list_cache(user_id)
        return bool(rows)

    async def get_session_run(
        self,
        *,
        run_id: str,
        user_id: str,
        session_id: str,
    ) -> SessionRun | None:
        response = await self._request_session_runs(
            params={
                "id": f"eq.{run_id}",
                "user_id": f"eq.{user_id}",
                "session_id": f"eq.{session_id}",
                "limit": "1",
            }
        )
        rows = response.json()
        if not rows:
            return None
        return self._session_run_from_row(rows[0])

    async def append_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
    ) -> None:
        payload = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "run_id": turn.run_id,
            "user_id": user_id,
            "role": turn.role,
            "content": turn.content,
            "citations": turn.citations,
            "suggestions": turn.suggestions,
            "created_at": turn.created_at,
        }
        await self._request("POST", "conversation_turns", json_body=payload)

    # ------------------------------------------------------------------
    # RAG resources + jobs
    # ------------------------------------------------------------------

    async def create_rag_resource(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["resource_id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "filename": payload["filename"],
            "mime_type": payload["mime_type"],
            "byte_size": payload["byte_size"],
            "storage_uri": payload["storage_uri"],
            "state": payload["state"],
            "error_details": payload.get("error_details"),
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        await self._request("POST", "rag_resources", json_body=body)
        await self._invalidate_rag_resources_list_cache(
            payload["owner_id"],
            payload["workspace_id"],
        )

    async def list_rag_resources(self, *, owner_id: str, workspace_id: str) -> list[dict[str, Any]]:
        cache_key = self._cache_key(
            self._CACHE_PREFIX_RAG_RESOURCES_LIST,
            f"{owner_id}:{workspace_id}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "rag_resources",
            params={
                "select": (
                    "id,owner_id,workspace_id,filename,mime_type,byte_size,storage_uri,state,"
                    "error_details,created_at,updated_at"
                ),
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "order": "created_at.desc",
            },
        )
        rows = response.json()
        result = [self._map_rag_resource_row(row) for row in rows]
        await self._cache_set_list(cache_key, result)
        return result

    async def get_rag_resource(
        self,
        *,
        resource_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "rag_resources",
            params={
                "select": (
                    "id,owner_id,workspace_id,filename,mime_type,byte_size,storage_uri,state,"
                    "error_details,created_at,updated_at"
                ),
                "id": f"eq.{resource_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return self._map_rag_resource_row(rows[0])

    async def get_rag_resources_by_ids(
        self,
        *,
        resource_ids: list[str],
        owner_id: str,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        if not resource_ids:
            return []
        joined = ",".join(resource_ids)
        response = await self._request(
            "GET",
            "rag_resources",
            params={
                "select": (
                    "id,owner_id,workspace_id,filename,mime_type,byte_size,storage_uri,state,"
                    "error_details,created_at,updated_at"
                ),
                "id": f"in.({joined})",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
        )
        rows = response.json()
        return [self._map_rag_resource_row(row) for row in rows]

    async def count_rag_resources_in_workspace(self, *, owner_id: str, workspace_id: str) -> int:
        response = await self._request(
            "GET",
            "rag_resources",
            params={
                "select": "id",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
        )
        rows = response.json()
        return len(rows)

    async def update_rag_resource(self, resource_id: str, patch: dict[str, Any]) -> bool:
        update_body = dict(patch)
        update_body["updated_at"] = datetime.now(UTC).isoformat()
        response = await self._request(
            "PATCH",
            "rag_resources",
            params={"id": f"eq.{resource_id}"},
            json_body=update_body,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            row = rows[0]
            owner_id = str(row.get("owner_id", ""))
            workspace_id = str(row.get("workspace_id", ""))
            if owner_id and workspace_id:
                await self._invalidate_rag_resources_list_cache(owner_id, workspace_id)
        return bool(rows)

    async def delete_rag_resource(
        self,
        *,
        resource_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "rag_resources",
            params={
                "id": f"eq.{resource_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_resources_list_cache(owner_id, workspace_id)
        return bool(rows)

    async def create_rag_ingestion_job(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["job_id"],
            "resource_id": payload["resource_id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "status": payload["status"],
            "stage": payload["stage"],
            "retries": payload["retries"],
            "max_retries": payload["max_retries"],
            "error_details": payload.get("error_details"),
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        await self._request("POST", "rag_ingestion_jobs", json_body=body)

    async def get_rag_ingestion_job(self, job_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "rag_ingestion_jobs",
            params={
                "select": (
                    "id,resource_id,owner_id,workspace_id,status,stage,retries,max_retries,"
                    "error_details,created_at,updated_at"
                ),
                "id": f"eq.{job_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return self._map_rag_ingestion_row(rows[0])

    async def get_latest_rag_ingestion_job_for_resource(
        self,
        *,
        resource_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "rag_ingestion_jobs",
            params={
                "select": (
                    "id,resource_id,owner_id,workspace_id,status,stage,retries,max_retries,"
                    "error_details,created_at,updated_at"
                ),
                "resource_id": f"eq.{resource_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "order": "created_at.desc",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return self._map_rag_ingestion_row(rows[0])

    async def claim_rag_ingestion_job(self, job_id: str) -> bool:
        """Atomically transition job status from 'queued' to 'running'.

        Returns True if the claim succeeded (job was queued), False if already claimed.
        """
        update_body = {
            "status": "running",
            "stage": "claimed",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        response = await self._request(
            "PATCH",
            "rag_ingestion_jobs",
            params={"id": f"eq.{job_id}", "status": "eq.queued"},
            json_body=update_body,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        return bool(rows)

    async def update_rag_ingestion_job(self, job_id: str, patch: dict[str, Any]) -> bool:
        update_body = dict(patch)
        update_body["updated_at"] = datetime.now(UTC).isoformat()
        response = await self._request(
            "PATCH",
            "rag_ingestion_jobs",
            params={"id": f"eq.{job_id}"},
            json_body=update_body,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        return bool(rows)

    # ------------------------------------------------------------------
    # Event outbox
    # ------------------------------------------------------------------

    async def create_resource_job_and_outbox(
        self,
        resource_payload: dict[str, Any],
        job_payload: dict[str, Any],
        outbox_payload: dict[str, Any],
    ) -> None:
        """Atomically insert resource + ingestion job + outbox event in one DB transaction."""
        await self._request(
            "POST",
            "rpc/create_resource_job_and_outbox",
            json_body={
                "p_resource": resource_payload,
                "p_job": job_payload,
                "p_outbox": outbox_payload,
            },
        )

    async def claim_outbox_event(self, event_id: str) -> bool:
        """Atomically transition an outbox event from pending -> dispatching.

        Returns True if the claim succeeded, False if another dispatcher already claimed it.
        """
        response = await self._request(
            "PATCH",
            "event_outbox",
            params={"id": f"eq.{event_id}", "status": "eq.pending"},
            json_body={
                "status": "dispatching",
                "dispatched_at": datetime.now(UTC).isoformat(),
            },
            extra_headers={"Prefer": "return=representation"},
        )
        return bool(response.json())

    async def reset_stuck_dispatching_events(self, older_than_seconds: int = 300) -> None:
        """Reset dispatching rows stuck longer than the threshold back to pending."""
        cutoff = (datetime.now(UTC) - timedelta(seconds=older_than_seconds)).isoformat()
        await self._request(
            "PATCH",
            "event_outbox",
            params={"status": "eq.dispatching", "dispatched_at": f"lt.{cutoff}"},
            json_body={"status": "pending"},
        )

    async def insert_outbox_event(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["id"],
            "event_name": payload["event_name"],
            "payload": payload["payload"],
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": payload.get("next_attempt_at", datetime.now(UTC).isoformat()),
            "created_at": payload.get("created_at", datetime.now(UTC).isoformat()),
        }
        await self._request("POST", "event_outbox", json_body=body)

    async def fetch_pending_outbox_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "event_outbox",
            params={
                "select": (
                    "id,event_name,payload,status,attempts,last_error,"
                    "next_attempt_at,created_at,sent_at"
                ),
                "status": "eq.pending",
                "next_attempt_at": f"lte.{datetime.now(UTC).isoformat()}",
                "order": "created_at.asc",
                "limit": str(limit),
            },
        )
        return response.json()

    async def update_outbox_event(self, event_id: str, patch: dict[str, Any]) -> None:
        await self._request(
            "PATCH",
            "event_outbox",
            params={"id": f"eq.{event_id}"},
            json_body=patch,
        )

    # ------------------------------------------------------------------
    # User memory
    # ------------------------------------------------------------------

    async def get_user_memory(
        self,
        *,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "user_memory",
            params={
                "select": "owner_id,workspace_id,content,updated_at,last_refreshed_at",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        return rows[0] if rows else None

    async def upsert_user_memory(self, *, payload: dict[str, Any]) -> None:
        await self._request(
            "POST",
            "user_memory",
            params={"on_conflict": "owner_id,workspace_id"},
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates"},
        )
        await self._invalidate_user_memory_cache(payload["owner_id"], payload["workspace_id"])

    async def delete_user_memory(
        self,
        *,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "user_memory",
            params={
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_user_memory_cache(owner_id, workspace_id)
        return bool(rows)

    async def claim_user_memory_refresh_event(self, payload: dict[str, Any]) -> bool:
        response = await self._request(
            "POST",
            "user_memory_refresh_events",
            params={"on_conflict": "event_key"},
            json_body=payload,
            extra_headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
        )
        return bool(response.json())

    async def upsert_rag_sidecar_artifact(
        self,
        *,
        resource_id: str,
        owner_id: str,
        workspace_id: str,
        source_locator: str,
        chunks: list[str],
    ) -> None:
        payload = {
            "resource_id": resource_id,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "source_locator": source_locator,
            "chunks": chunks,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        await self._request(
            "POST",
            "rag_sidecar_artifacts",
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates"},
        )

    async def get_rag_sidecar_artifact(self, *, resource_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "rag_sidecar_artifacts",
            params={
                "select": "resource_id,owner_id,workspace_id,source_locator,chunks,updated_at",
                "resource_id": f"eq.{resource_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def list_rag_sidecar_artifacts(
        self,
        *,
        resource_ids: list[str],
        owner_id: str,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        if not resource_ids:
            return []
        joined = ",".join(resource_ids)
        response = await self._request(
            "GET",
            "rag_sidecar_artifacts",
            params={
                "select": "resource_id,owner_id,workspace_id,source_locator,chunks,updated_at",
                "resource_id": f"in.({joined})",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
        )
        return response.json()

    async def delete_rag_sidecar_artifact(self, *, resource_id: str) -> bool:
        response = await self._request(
            "DELETE",
            "rag_sidecar_artifacts",
            params={"resource_id": f"eq.{resource_id}"},
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        return bool(rows)

    # ------------------------------------------------------------------
    # RAG agents + linking
    # ------------------------------------------------------------------

    async def create_rag_agent(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["agent_id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "name": payload["name"],
            "description": payload["description"],
            "system_instructions": payload["system_instructions"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        await self._request("POST", "rag_agents", json_body=body)
        await self._invalidate_rag_agents_list_cache(
            payload["owner_id"],
            payload["workspace_id"],
        )

    async def list_rag_agents(self, *, owner_id: str, workspace_id: str) -> list[dict[str, Any]]:
        cache_key = self._cache_key(
            self._CACHE_PREFIX_RAG_AGENTS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "rag_agents",
            params={
                "select": "id,owner_id,workspace_id,name,description,system_instructions,created_at,updated_at",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "order": "created_at.desc",
            },
        )
        agents = response.json()
        if not agents:
            return []

        agent_ids = [a["id"] for a in agents]
        links = await self._list_agent_links(agent_ids)
        by_agent: dict[str, list[str]] = {}
        for link in links:
            by_agent.setdefault(link["agent_id"], []).append(link["resource_id"])

        result = [
            self._map_rag_agent_row(row, by_agent.get(row["id"], []))
            for row in agents
        ]
        await self._cache_set_list(cache_key, result)
        return result

    async def get_rag_agent(
        self,
        *,
        agent_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "rag_agents",
            params={
                "select": "id,owner_id,workspace_id,name,description,system_instructions,created_at,updated_at",
                "id": f"eq.{agent_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        links = await self._list_agent_links([agent_id])
        resource_ids = [link["resource_id"] for link in links]
        return self._map_rag_agent_row(rows[0], resource_ids)

    async def update_rag_agent(
        self,
        *,
        agent_id: str,
        owner_id: str,
        workspace_id: str,
        patch: dict[str, Any],
    ) -> bool:
        update_body = dict(patch)
        update_body["updated_at"] = datetime.now(UTC).isoformat()
        response = await self._request(
            "PATCH",
            "rag_agents",
            params={
                "id": f"eq.{agent_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
            json_body=update_body,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_agents_list_cache(owner_id, workspace_id)
        return bool(rows)

    async def delete_rag_agent(
        self,
        *,
        agent_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "rag_agents",
            params={
                "id": f"eq.{agent_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_agents_list_cache(owner_id, workspace_id)
            await self._invalidate_rag_chat_sessions_list_cache(owner_id, agent_id)
        return bool(rows)

    async def replace_rag_agent_resources(
        self,
        *,
        agent_id: str,
        owner_id: str,
        workspace_id: str,
        resource_ids: list[str],
    ) -> None:
        await self._request(
            "DELETE",
            "rag_agent_resources",
            params={
                "agent_id": f"eq.{agent_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
        )
        if not resource_ids:
            return
        rows = [
            {
                "agent_id": agent_id,
                "resource_id": resource_id,
                "owner_id": owner_id,
                "workspace_id": workspace_id,
            }
            for resource_id in resource_ids
        ]
        await self._request("POST", "rag_agent_resources", json_body=rows)
        await self._invalidate_rag_agents_list_cache(owner_id, workspace_id)

    async def _list_agent_links(self, agent_ids: list[str]) -> list[dict[str, str]]:
        if not agent_ids:
            return []
        joined = ",".join(agent_ids)
        response = await self._request(
            "GET",
            "rag_agent_resources",
            params={
                "select": "agent_id,resource_id",
                "agent_id": f"in.({joined})",
            },
        )
        return response.json()

    # ------------------------------------------------------------------
    # PRD plans
    # ------------------------------------------------------------------

    async def create_prd_plan(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "prompt": payload["prompt"],
            "prompt_preview": payload["prompt_preview"],
            "title": payload["title"],
            "summary": payload["summary"],
            "suggested_filename": payload["suggested_filename"],
            "markdown": payload["markdown"],
            "plan_json": payload["plan_json"],
            "planning_brief_json": payload["planning_brief_json"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        await self._request("POST", "prds", json_body=body)
        await self._invalidate_prd_plans_list_cache(
            payload["owner_id"],
            payload["workspace_id"],
        )

    async def list_prd_plans(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        cache_key = self._cache_key(
            self._CACHE_PREFIX_PRD_PLANS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "prds",
            params={
                "select": "id,title,summary,prompt_preview,created_at,updated_at",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        rows = response.json()
        await self._cache_set_list(cache_key, rows)
        return rows

    async def get_prd_plan(
        self,
        *,
        plan_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "prds",
            params={
                "select": (
                    "id,owner_id,workspace_id,prompt,prompt_preview,title,summary,"
                    "suggested_filename,markdown,plan_json,planning_brief_json,"
                    "created_at,updated_at"
                ),
                "id": f"eq.{plan_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def delete_prd_plan(
        self,
        *,
        plan_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "prds",
            params={
                "id": f"eq.{plan_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
        )
        deleted = response.status_code in (200, 204)
        if deleted:
            await self._invalidate_prd_plans_list_cache(owner_id, workspace_id)
        return deleted

    # ------------------------------------------------------------------
    # Itinerary planner sessions
    # ------------------------------------------------------------------

    async def create_itinerary_session(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "title": payload["title"],
            "status": payload["status"],
            "requirements_json": payload["requirements_json"],
            "current_version_id": payload.get("current_version_id"),
            "prompt_preview": payload.get("prompt_preview") or "",
            "last_message_preview": payload.get("last_message_preview") or "",
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        await self._request("POST", "itinerary_sessions", json_body=body)
        await self._invalidate_itinerary_sessions_list_cache(
            payload["owner_id"],
            payload["workspace_id"],
        )

    async def list_itinerary_sessions(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        cache_key = self._cache_key(
            self._CACHE_PREFIX_ITINERARY_SESSIONS_LIST,
            f"{owner_id}:{workspace_id}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "itinerary_sessions",
            params={
                "select": (
                    "id,owner_id,workspace_id,title,status,current_version_id,"
                    "prompt_preview,last_message_preview,created_at,updated_at"
                ),
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "order": "updated_at.desc",
                "limit": str(limit),
            },
        )
        rows = response.json()
        await self._cache_set_list(cache_key, rows)
        return rows

    async def get_itinerary_session(
        self,
        *,
        session_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "itinerary_sessions",
            params={
                "select": (
                    "id,owner_id,workspace_id,title,status,requirements_json,current_version_id,"
                    "prompt_preview,last_message_preview,created_at,updated_at"
                ),
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def update_itinerary_session(
        self,
        *,
        session_id: str,
        owner_id: str,
        patch: dict[str, Any],
    ) -> bool:
        response = await self._request(
            "PATCH",
            "itinerary_sessions",
            params={
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
            },
            json_body=patch,
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            row = rows[0]
            await self._invalidate_itinerary_sessions_list_cache(
                owner_id,
                str(row.get("workspace_id") or ""),
            )
        return bool(rows)

    async def delete_itinerary_session(
        self,
        *,
        session_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        response = await self._request(
            "DELETE",
            "itinerary_sessions",
            params={
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "workspace_id": f"eq.{workspace_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_itinerary_sessions_list_cache(owner_id, workspace_id)
        return bool(rows)

    async def create_itinerary_message(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["id"],
            "session_id": payload["session_id"],
            "owner_id": payload["owner_id"],
            "role": payload["role"],
            "content": payload["content"],
            "metadata_json": payload.get("metadata_json") or {},
            "created_at": payload["created_at"],
        }
        await self._request("POST", "itinerary_messages", json_body=body)

    async def list_itinerary_messages(
        self,
        *,
        session_id: str,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "itinerary_messages",
            params={
                "select": "id,session_id,role,content,metadata_json,created_at",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "order": "created_at.asc",
            },
        )
        return response.json()

    async def create_itinerary_version(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["id"],
            "session_id": payload["session_id"],
            "owner_id": payload["owner_id"],
            "version_number": payload["version_number"],
            "revision_summary": payload["revision_summary"],
            "markdown": payload["markdown"],
            "itinerary_json": payload["itinerary_json"],
            "created_at": payload["created_at"],
        }
        await self._request("POST", "itinerary_versions", json_body=body)

    async def list_itinerary_versions(
        self,
        *,
        session_id: str,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "itinerary_versions",
            params={
                "select": (
                    "id,session_id,version_number,revision_summary,markdown,itinerary_json,created_at"
                ),
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "order": "version_number.asc",
            },
        )
        return response.json()

    # ------------------------------------------------------------------
    # RAG chat sessions + messages
    # ------------------------------------------------------------------

    async def create_rag_chat_session(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["session_id"],
            "owner_id": payload["owner_id"],
            "workspace_id": payload["workspace_id"],
            "agent_id": payload["agent_id"],
            "chat_scope": payload.get("chat_scope") or "agent",
            "title": payload.get("title") or "New chat",
            "created_at": datetime.now(UTC).isoformat(),
        }
        await self._request("POST", "rag_chat_sessions", json_body=body)
        await self._invalidate_rag_chat_sessions_list_cache(
            payload["owner_id"],
            payload["agent_id"],
            payload.get("chat_scope") or "agent",
        )

    async def get_rag_chat_session(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str | None,
        chat_scope: str = "agent",
    ) -> dict[str, Any] | None:
        agent_filter_key = "is" if agent_id is None else "eq"
        agent_filter_value = "null" if agent_id is None else agent_id
        response = await self._request(
            "GET",
            "rag_chat_sessions",
            params={
                "select": "id,owner_id,workspace_id,agent_id,chat_scope,title,created_at",
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "chat_scope": f"eq.{chat_scope}",
                "agent_id": f"{agent_filter_key}.{agent_filter_value}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        return {
            "session_id": row["id"],
            "owner_id": row["owner_id"],
            "workspace_id": row["workspace_id"],
            "agent_id": row["agent_id"],
            "chat_scope": row.get("chat_scope") or "agent",
            "title": row.get("title") or "New chat",
            "created_at": row.get("created_at"),
        }

    async def list_rag_chat_sessions(
        self,
        *,
        agent_id: str | None,
        owner_id: str,
        chat_scope: str = "agent",
    ) -> list[dict[str, Any]]:
        agent_key = agent_id or "__workspace__"
        cache_key = self._cache_key(
            self._CACHE_PREFIX_RAG_CHAT_SESSIONS_LIST,
            f"{owner_id}:{chat_scope}:{agent_key}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        agent_filter_key = "is" if agent_id is None else "eq"
        agent_filter_value = "null" if agent_id is None else agent_id
        response = await self._request(
            "GET",
            "rag_chat_sessions",
            params={
                "select": "id,owner_id,workspace_id,agent_id,chat_scope,title,created_at",
                "owner_id": f"eq.{owner_id}",
                "chat_scope": f"eq.{chat_scope}",
                "agent_id": f"{agent_filter_key}.{agent_filter_value}",
                "order": "created_at.desc",
            },
        )
        sessions = response.json()
        if not sessions:
            return []

        session_ids = [row["id"] for row in sessions]
        messages_response = await self._request(
            "GET",
            "rag_chat_messages",
            params={
                "select": "session_id,content,created_at",
                "session_id": f"in.({','.join(session_ids)})",
                "owner_id": f"eq.{owner_id}",
                "order": "created_at.desc",
            },
        )
        latest_by_session: dict[str, dict[str, Any]] = {}
        for message in messages_response.json():
            latest_by_session.setdefault(message["session_id"], message)

        summaries: list[dict[str, Any]] = []
        for row in sessions:
            latest = latest_by_session.get(row["id"], {})
            content = latest.get("content") or ""
            preview = content[:120] + "..." if len(content) > 120 else content
            summaries.append(
                {
                    "session_id": row["id"],
                    "owner_id": row["owner_id"],
                    "workspace_id": row["workspace_id"],
                    "agent_id": row["agent_id"],
                    "chat_scope": row.get("chat_scope") or "agent",
                    "title": row.get("title") or "New chat",
                    "created_at": row.get("created_at"),
                    "last_message_at": latest.get("created_at") or row.get("created_at"),
                    "last_message_preview": preview,
                }
            )

        result = sorted(
            summaries,
            key=lambda summary: summary.get("last_message_at") or summary.get("created_at") or "",
            reverse=True,
        )
        await self._cache_set_list(cache_key, result)
        return result

    async def update_rag_chat_session_title(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str | None,
        title: str,
        chat_scope: str = "agent",
    ) -> bool:
        agent_filter_key = "is" if agent_id is None else "eq"
        agent_filter_value = "null" if agent_id is None else agent_id
        response = await self._request(
            "PATCH",
            "rag_chat_sessions",
            params={
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "chat_scope": f"eq.{chat_scope}",
                "agent_id": f"{agent_filter_key}.{agent_filter_value}",
            },
            json_body={"title": title},
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_chat_sessions_list_cache(owner_id, agent_id, chat_scope)
        return bool(rows)

    async def delete_rag_chat_session(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str | None,
        chat_scope: str = "agent",
    ) -> bool:
        agent_filter_key = "is" if agent_id is None else "eq"
        agent_filter_value = "null" if agent_id is None else agent_id
        response = await self._request(
            "DELETE",
            "rag_chat_sessions",
            params={
                "id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "chat_scope": f"eq.{chat_scope}",
                "agent_id": f"{agent_filter_key}.{agent_filter_value}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_chat_sessions_list_cache(owner_id, agent_id, chat_scope)
            await self._invalidate_rag_chat_messages_list_cache(owner_id, session_id)
        return bool(rows)

    async def list_ready_rag_chat_session_attachment_resource_ids(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> list[str]:
        response = await self._request(
            "GET",
            "rag_chat_session_attachments",
            params={
                "select": "id,resource_id",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "agent_id": f"eq.{agent_id}",
                "state": "eq.ready",
                "order": "created_at.asc",
            },
        )
        return [
            row["resource_id"]
            for row in response.json()
            if isinstance(row.get("resource_id"), str) and row["resource_id"]
        ]

    async def create_rag_chat_session_attachment(self, payload: dict[str, Any]) -> None:
        await self._request(
            "POST",
            "rag_chat_session_attachments",
            json_body={
                "id": payload["attachment_id"],
                "session_id": payload["session_id"],
                "agent_id": payload["agent_id"],
                "owner_id": payload["owner_id"],
                "workspace_id": payload["workspace_id"],
                "resource_id": payload["resource_id"],
                "filename": payload["filename"],
                "mime_type": payload["mime_type"],
                "byte_size": payload["byte_size"],
                "storage_uri": payload["storage_uri"],
                "state": payload["state"],
                "error_details": payload.get("error_details"),
                "created_at": payload["created_at"],
                "updated_at": payload["updated_at"],
            },
            extra_headers={"Prefer": "resolution=ignore-duplicates"},
        )

    async def update_rag_chat_session_attachment(
        self,
        *,
        attachment_id: str,
        session_id: str,
        agent_id: str,
        owner_id: str,
        patch: dict[str, Any],
    ) -> None:
        payload = dict(patch)
        payload["updated_at"] = datetime.now(UTC).isoformat()
        await self._request(
            "PATCH",
            "rag_chat_session_attachments",
            params={
                "id": f"eq.{attachment_id}",
                "session_id": f"eq.{session_id}",
                "agent_id": f"eq.{agent_id}",
                "owner_id": f"eq.{owner_id}",
            },
            json_body=payload,
        )

    async def list_rag_chat_session_attachments(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "rag_chat_session_attachments",
            params={
                "select": (
                    "id,session_id,agent_id,owner_id,workspace_id,resource_id,filename,mime_type,"
                    "byte_size,storage_uri,state,error_details,created_at,updated_at"
                ),
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "agent_id": f"eq.{agent_id}",
                "order": "created_at.asc",
            },
        )
        return response.json()

    async def delete_rag_chat_session_attachments(
        self,
        *,
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> list[dict[str, str]]:
        response = await self._request(
            "DELETE",
            "rag_chat_session_attachments",
            params={
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "agent_id": f"eq.{agent_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        return [
            {
                "attachment_id": row["id"],
                "resource_id": row["resource_id"],
                "storage_uri": row["storage_uri"],
            }
            for row in response.json()
            if (
                isinstance(row.get("id"), str)
                and isinstance(row.get("resource_id"), str)
                and isinstance(row.get("storage_uri"), str)
            )
        ]

    async def delete_rag_chat_session_attachments_by_ids(
        self,
        *,
        attachment_ids: list[str],
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> list[dict[str, str]]:
        response = await self._request(
            "DELETE",
            "rag_chat_session_attachments",
            params={
                "id": f"in.({','.join(attachment_ids)})",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "agent_id": f"eq.{agent_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        return [
            {
                "attachment_id": row["id"],
                "resource_id": row["resource_id"],
                "storage_uri": row["storage_uri"],
            }
            for row in response.json()
            if (
                isinstance(row.get("id"), str)
                and isinstance(row.get("resource_id"), str)
                and isinstance(row.get("storage_uri"), str)
            )
        ]

    async def update_rag_chat_message_suggestions(
        self,
        *,
        message_id: str,
        session_id: str,
        owner_id: str,
        suggestions: list[str],
    ) -> bool:
        response = await self._request(
            "PATCH",
            "rag_chat_messages",
            params={
                "id": f"eq.{message_id}",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
            },
            json_body={"suggestions": suggestions},
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_chat_messages_list_cache(owner_id, session_id)
        return bool(rows)

    async def create_rag_chat_message(self, payload: dict[str, Any]) -> None:
        body = {
            "id": payload["message_id"],
            "session_id": payload["session_id"],
            "agent_id": payload["agent_id"],
            "owner_id": payload["owner_id"],
            "role": payload["role"],
            "content": payload["content"],
            "citations": payload.get("citations") or [],
            "suggestions": payload.get("suggestions") or [],
            "created_at": payload["created_at"],
        }
        await self._request("POST", "rag_chat_messages", json_body=body)
        await self._invalidate_rag_chat_messages_list_cache(
            payload["owner_id"],
            payload["session_id"],
        )
        await self._invalidate_rag_chat_sessions_list_cache(
            payload["owner_id"],
            payload["agent_id"],
            payload.get("chat_scope") or "agent",
        )

    async def delete_rag_chat_message(
        self, *, message_id: str, session_id: str, owner_id: str
    ) -> bool:
        response = await self._request(
            "DELETE",
            "rag_chat_messages",
            params={
                "id": f"eq.{message_id}",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
            },
            extra_headers={"Prefer": "return=representation"},
        )
        rows = response.json()
        if rows:
            await self._invalidate_rag_chat_messages_list_cache(owner_id, session_id)
        return bool(rows)

    async def delete_last_user_assistant_pair(
        self, *, session_id: str, owner_id: str
    ) -> tuple[bool, str | None]:
        """Delete the most recent assistant + user message in order.

        Returns (deleted, error_code). error_code is one of:
          None, "empty", "not_user_assistant_pair".
        """
        rows = await self.list_rag_chat_messages(session_id=session_id, owner_id=owner_id)
        if not rows:
            return False, "empty"
        if len(rows) < 2 or rows[-1]["role"] != "assistant" or rows[-2]["role"] != "user":
            return False, "not_user_assistant_pair"
        await self.delete_rag_chat_message(
            message_id=rows[-1]["message_id"], session_id=session_id, owner_id=owner_id
        )
        await self.delete_rag_chat_message(
            message_id=rows[-2]["message_id"], session_id=session_id, owner_id=owner_id
        )
        return True, None

    async def list_rag_chat_messages(self, *, session_id: str, owner_id: str) -> list[dict[str, Any]]:
        cache_key = self._cache_key(
            self._CACHE_PREFIX_RAG_CHAT_MESSAGES_LIST,
            f"{owner_id}:{session_id}",
        )
        cached = await self._cache_get_list(cache_key)
        if cached is not None:
            return cached

        response = await self._request(
            "GET",
            "rag_chat_messages",
            params={
                "select": "id,session_id,agent_id,owner_id,role,content,citations,suggestions,created_at",
                "session_id": f"eq.{session_id}",
                "owner_id": f"eq.{owner_id}",
                "order": "created_at.asc",
            },
        )
        rows = response.json()
        result = [
            {
                "message_id": row["id"],
                "session_id": row["session_id"],
                "agent_id": row["agent_id"],
                "owner_id": row["owner_id"],
                "role": row["role"],
                "content": row["content"],
                "citations": row.get("citations") or [],
                "suggestions": row.get("suggestions") or [],
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]
        await self._cache_set_list(cache_key, result)
        return result

    @staticmethod
    def _map_rag_resource_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "resource_id": row["id"],
            "owner_id": row["owner_id"],
            "workspace_id": row["workspace_id"],
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "byte_size": row["byte_size"],
            "storage_uri": row["storage_uri"],
            "state": row["state"],
            "error_details": row.get("error_details"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    @staticmethod
    def _map_rag_ingestion_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": row["id"],
            "resource_id": row["resource_id"],
            "owner_id": row["owner_id"],
            "workspace_id": row["workspace_id"],
            "status": row["status"],
            "stage": row["stage"],
            "retries": row["retries"],
            "max_retries": row["max_retries"],
            "error_details": row.get("error_details"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    @staticmethod
    def _map_rag_agent_row(row: dict[str, Any], resource_ids: list[str]) -> dict[str, Any]:
        return {
            "agent_id": row["id"],
            "owner_id": row["owner_id"],
            "workspace_id": row["workspace_id"],
            "name": row["name"],
            "description": row.get("description") or "",
            "system_instructions": row.get("system_instructions") or "",
            "linked_resource_ids": resource_ids,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    # ------------------------------------------------------------------
    # Billing subscriptions + usage
    # ------------------------------------------------------------------

    async def get_user_subscription(self, user_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "user_subscriptions",
            params={
                "select": (
                    "user_id,plan,status,stripe_customer_id,stripe_subscription_id,"
                    "current_period_start,current_period_end,cancel_at_period_end,cancel_at,canceled_at,"
                    "created_at,updated_at"
                ),
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def get_user_subscription_by_customer_id(self, customer_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "user_subscriptions",
            params={
                "select": (
                    "user_id,plan,status,stripe_customer_id,stripe_subscription_id,"
                    "current_period_start,current_period_end,cancel_at_period_end,cancel_at,canceled_at,"
                    "created_at,updated_at"
                ),
                "stripe_customer_id": f"eq.{customer_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def get_user_subscription_by_subscription_id(self, subscription_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "user_subscriptions",
            params={
                "select": (
                    "user_id,plan,status,stripe_customer_id,stripe_subscription_id,"
                    "current_period_start,current_period_end,cancel_at_period_end,cancel_at,canceled_at,"
                    "created_at,updated_at"
                ),
                "stripe_subscription_id": f"eq.{subscription_id}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def upsert_user_subscription(self, payload: dict[str, Any]) -> None:
        await self._request(
            "POST",
            "user_subscriptions",
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates"},
        )

    async def get_daily_usage_counter(self, *, user_id: str, usage_date: date) -> dict[str, Any] | None:
        response = await self._request(
            "GET",
            "daily_usage_counters",
            params={
                "select": (
                    "user_id,usage_date,research_queries_count,total_questions_count,created_at,updated_at"
                ),
                "user_id": f"eq.{user_id}",
                "usage_date": f"eq.{usage_date.isoformat()}",
                "limit": "1",
            },
        )
        rows = response.json()
        if not rows:
            return None
        return rows[0]

    async def increment_daily_usage_counter(
        self,
        *,
        user_id: str,
        usage_date: date,
        add_research_queries: int,
        add_total_questions: int,
    ) -> dict[str, Any]:
        try:
            response = await self._request(
                "POST",
                "rpc/increment_daily_usage_counters",
                json_body={
                    "p_user_id": user_id,
                    "p_usage_date": usage_date.isoformat(),
                    "p_research_queries": add_research_queries,
                    "p_total_questions": add_total_questions,
                },
            )
            rows = response.json()
            if not rows:
                return {
                    "user_id": user_id,
                    "usage_date": usage_date.isoformat(),
                    "research_queries_count": 0,
                    "total_questions_count": 0,
                }
            return rows[0]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            try:
                payload = exc.response.json()
            except Exception:
                payload = {"message": exc.response.text}
            logger.warning(
                "billing rpc increment_daily_usage_counters failed; using non-atomic fallback. "
                "status=%s payload=%s",
                exc.response.status_code,
                payload,
            )
            return await self._increment_daily_usage_counter_fallback(
                user_id=user_id,
                usage_date=usage_date,
                add_research_queries=add_research_queries,
                add_total_questions=add_total_questions,
            )

    async def _increment_daily_usage_counter_fallback(
        self,
        *,
        user_id: str,
        usage_date: date,
        add_research_queries: int,
        add_total_questions: int,
    ) -> dict[str, Any]:
        current = await self.get_daily_usage_counter(user_id=user_id, usage_date=usage_date)
        current_research = int((current or {}).get("research_queries_count") or 0)
        current_total = int((current or {}).get("total_questions_count") or 0)

        next_research = current_research + max(0, add_research_queries)
        next_total = current_total + max(0, add_total_questions)
        now_iso = datetime.now(UTC).isoformat()

        response = await self._request(
            "POST",
            "daily_usage_counters",
            json_body={
                "user_id": user_id,
                "usage_date": usage_date.isoformat(),
                "research_queries_count": next_research,
                "total_questions_count": next_total,
                "updated_at": now_iso,
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        rows = response.json()
        if rows:
            return rows[0]
        return {
            "user_id": user_id,
            "usage_date": usage_date.isoformat(),
            "research_queries_count": next_research,
            "total_questions_count": next_total,
            "updated_at": now_iso,
        }
