"""RAG Agent domain models and orchestration helpers."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from src.config import settings
from src.db.supabase_store import SupabaseSessionStore
from src.rag_engine import (
    RagQueryResult,
    delete_resource_artifacts,
    ingest_resource_from_locator,
    query_resource_context,
)
from src.storage import SupabaseStorageAdapter


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}

_RESOURCE_STATES = {"uploaded", "processing", "ready", "failed"}
CHAT_SCOPE_AGENT = "agent"
CHAT_SCOPE_WORKSPACE = "workspace"


@dataclass
class RagResource:
    resource_id: str
    owner_id: str
    workspace_id: str
    filename: str
    mime_type: str
    byte_size: int
    storage_uri: str
    state: str = "uploaded"
    error_details: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "byte_size": self.byte_size,
            "storage_uri": self.storage_uri,
            "state": self.state,
            "error_details": self.error_details,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RagIngestionJob:
    job_id: str
    resource_id: str
    owner_id: str
    workspace_id: str
    status: str = "queued"
    stage: str = "queued"
    retries: int = 0
    max_retries: int = 2
    error_details: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "status": self.status,
            "stage": self.stage,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "error_details": self.error_details,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RagAgent:
    agent_id: str
    owner_id: str
    workspace_id: str
    name: str
    description: str
    system_instructions: str
    linked_resource_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "system_instructions": self.system_instructions,
            "linked_resource_ids": self.linked_resource_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RagChatMessage:
    message_id: str
    session_id: str
    agent_id: str | None
    owner_id: str
    role: str
    content: str
    citations: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    chat_scope: str = CHAT_SCOPE_AGENT
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "suggestions": self.suggestions,
            "chat_scope": self.chat_scope,
            "created_at": self.created_at,
        }


class RagValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_store: SupabaseSessionStore | None = None
_storage: SupabaseStorageAdapter | None = None


def _workspace_id_for_user(user_id: str) -> str:
    return user_id


def _get_store() -> SupabaseSessionStore:
    global _store
    if _store is None:
        _store = SupabaseSessionStore()
    return _store


def _get_storage() -> SupabaseStorageAdapter:
    global _storage
    if _storage is None:
        _storage = SupabaseStorageAdapter()
    return _storage


def _validate_upload(file: UploadFile, content: bytes) -> None:
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise RagValidationError(
            "unsupported_type",
            "Unsupported file type. Allowed: pdf, docx, txt, md.",
        )

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise RagValidationError(
            "unsupported_type",
            "Unsupported MIME type. Allowed: pdf, docx, txt, md.",
        )

    max_bytes = settings.rag_max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise RagValidationError(
            "size_exceeded",
            f"File too large. Max size is {settings.rag_max_file_size_mb} MB.",
        )


def _normalize_state(value: str) -> str:
    return value if value in _RESOURCE_STATES else "failed"


def _fallback_chat_title(message: str | None) -> str:
    if not message or not message.strip():
        return "New chat"
    cleaned = " ".join(message.strip().split())
    if not cleaned:
        return "New chat"
    words = cleaned.split(" ")
    if len(words) > 6:
        cleaned = " ".join(words[:6])
    return cleaned[:120] or "New chat"


def _suggest_chat_session_title_sync(message: str | None) -> str:
    fallback = _fallback_chat_title(message)
    if not message or not message.strip():
        return fallback
    prompt = (
        "Create a concise title (max 5 words) for this agent chat session.\n"
        "Return plain text only, no quotes, no punctuation at the end.\n"
        f"First user message: {message.strip()}"
    )
    try:
        from src.llm.factory import get_llm

        llm = get_llm(temperature=0.1)
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        candidate = " ".join(text.strip().split())
        if not candidate:
            return fallback
        words = candidate.split(" ")
        if len(words) > 6:
            candidate = " ".join(words[:6])
        return candidate[:120] or fallback
    except Exception:
        return fallback


async def suggest_chat_session_title(message: str | None) -> str:
    # LLM calls can block; keep title generation off the event loop.
    return await asyncio.to_thread(_suggest_chat_session_title_sync, message)


async def list_resources(user_id: str) -> list[RagResource]:
    workspace_id = _workspace_id_for_user(user_id)
    rows = await _get_store().list_rag_resources(
        owner_id=user_id, workspace_id=workspace_id
    )
    return [RagResource(**row) for row in rows]


async def get_resource(resource_id: str, user_id: str) -> RagResource | None:
    workspace_id = _workspace_id_for_user(user_id)
    row = await _get_store().get_rag_resource(
        resource_id=resource_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not row:
        return None
    return RagResource(**row)


async def create_resource_and_ingest(
    file: UploadFile, user_id: str
) -> tuple[RagResource, RagIngestionJob]:
    content = await file.read()
    _validate_upload(file, content)

    workspace_id = _workspace_id_for_user(user_id)
    current_count = await _get_store().count_rag_resources_in_workspace(
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if current_count >= settings.rag_max_resources_per_workspace:
        raise RagValidationError(
            "workspace_limit_exceeded",
            "Workspace resource limit exceeded.",
        )

    resource_id = str(uuid.uuid4())
    filename = file.filename or f"resource-{resource_id}.txt"
    storage_key = f"{workspace_id}/{user_id}/{resource_id}/{filename}"
    storage_uri = await _get_storage().upload_bytes(
        key=storage_key,
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )

    now = datetime.now(UTC).isoformat()
    resource = RagResource(
        resource_id=resource_id,
        owner_id=user_id,
        workspace_id=workspace_id,
        filename=filename,
        mime_type=file.content_type or "application/octet-stream",
        byte_size=len(content),
        storage_uri=storage_uri,
        state="uploaded",
        created_at=now,
        updated_at=now,
    )
    job = RagIngestionJob(
        job_id=str(uuid.uuid4()),
        resource_id=resource_id,
        owner_id=user_id,
        workspace_id=workspace_id,
        status="queued",
        stage="queued",
    )
    outbox_id = str(uuid.uuid4())
    outbox_now = datetime.now(UTC).isoformat()
    await _get_store().create_resource_job_and_outbox(
        resource_payload=resource.to_dict(),
        job_payload=job.to_dict(),
        outbox_payload={
            "id": outbox_id,
            "event_name": "rag/ingestion.requested",
            "payload": {
                "job_id": job.job_id,
                "resource_id": job.resource_id,
                "owner_id": job.owner_id,
                "workspace_id": job.workspace_id,
            },
            "next_attempt_at": outbox_now,
            "created_at": outbox_now,
        },
    )

    return resource, job


async def _run_ingestion_job(job_id: str) -> None:
    store = _get_store()

    job_row = await store.get_rag_ingestion_job(job_id)
    if not job_row:
        return
    job = RagIngestionJob(**job_row)

    resource_row = await store.get_rag_resource(
        resource_id=job.resource_id,
        owner_id=job.owner_id,
        workspace_id=job.workspace_id,
    )
    if not resource_row:
        await store.update_rag_ingestion_job(
            job_id,
            {
                "status": "failed",
                "stage": "resource_lookup",
                "error_details": "Resource not found for ingestion.",
            },
        )
        return

    resource = RagResource(**resource_row)

    max_attempts = job.max_retries + 1
    for attempt in range(max_attempts):
        await store.update_rag_ingestion_job(
            job.job_id,
            {
                "status": "running",
                "stage": "ingesting",
                "retries": attempt,
                "error_details": None,
            },
        )
        await store.update_rag_resource(
            resource.resource_id,
            {
                "state": "processing",
                "error_details": None,
            },
        )

        try:
            signed_file_url = await _get_storage().create_signed_download_url(
                storage_uri=resource.storage_uri,
                expires_in=settings.rag_signed_url_ttl_seconds,
            )
            await ingest_resource_from_locator(
                store=store,
                resource_id=resource.resource_id,
                file_locator=signed_file_url,
                owner_id=resource.owner_id,
                workspace_id=resource.workspace_id,
            )
            await store.update_rag_resource(
                resource.resource_id,
                {
                    "state": "ready",
                    "error_details": None,
                },
            )
            await store.update_rag_ingestion_job(
                job.job_id,
                {
                    "status": "succeeded",
                    "stage": "completed",
                    "retries": attempt,
                    "error_details": None,
                },
            )
            return
        except Exception as exc:
            if attempt < max_attempts - 1:
                await store.update_rag_ingestion_job(
                    job.job_id,
                    {
                        "status": "queued",
                        "stage": "retrying",
                        "retries": attempt + 1,
                        "error_details": str(exc),
                    },
                )
                continue

            await store.update_rag_resource(
                resource.resource_id,
                {
                    "state": "failed",
                    "error_details": str(exc),
                },
            )
            await store.update_rag_ingestion_job(
                job.job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "retries": attempt,
                    "error_details": str(exc),
                },
            )
            return


async def get_resource_status(resource_id: str, user_id: str) -> dict:
    workspace_id = _workspace_id_for_user(user_id)
    resource_row = await _get_store().get_rag_resource(
        resource_id=resource_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not resource_row:
        return {}

    job = await _get_store().get_latest_rag_ingestion_job_for_resource(
        resource_id=resource_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    resource = RagResource(**resource_row)
    payload = {"resource": resource.to_dict()}
    if job:
        payload["job"] = RagIngestionJob(**job).to_dict()
    return payload


async def delete_resource(resource_id: str, user_id: str) -> bool:
    resource = await get_resource(resource_id, user_id)
    if resource is None:
        return False

    try:
        await delete_resource_artifacts(
            store=_get_store(),
            resource_id=resource.resource_id,
            owner_id=resource.owner_id,
            workspace_id=resource.workspace_id,
        )
    except Exception:
        # Sidecar cleanup is best-effort; resource deletion still proceeds.
        pass

    if resource.storage_uri:
        try:
            await _get_storage().delete_object(storage_uri=resource.storage_uri)
        except Exception:
            # Object cleanup is best-effort; DB deletion should still proceed.
            pass

    return await _get_store().delete_rag_resource(
        resource_id=resource.resource_id,
        owner_id=user_id,
        workspace_id=resource.workspace_id,
    )


async def list_agents(user_id: str) -> list[RagAgent]:
    workspace_id = _workspace_id_for_user(user_id)
    rows = await _get_store().list_rag_agents(
        owner_id=user_id, workspace_id=workspace_id
    )
    return [RagAgent(**row) for row in rows]


async def create_agent(
    *,
    user_id: str,
    name: str,
    description: str,
    system_instructions: str,
    linked_resource_ids: list[str],
) -> RagAgent:
    workspace_id = _workspace_id_for_user(user_id)
    if len(linked_resource_ids) > settings.rag_max_resources_per_agent:
        raise RagValidationError(
            "agent_resource_limit_exceeded",
            "Too many resources linked to this agent.",
        )

    await _validate_resources_linkable(
        owner_id=user_id,
        workspace_id=workspace_id,
        resource_ids=linked_resource_ids,
    )

    now = datetime.now(UTC).isoformat()
    agent = RagAgent(
        agent_id=str(uuid.uuid4()),
        owner_id=user_id,
        workspace_id=workspace_id,
        name=name,
        description=description,
        system_instructions=system_instructions,
        linked_resource_ids=linked_resource_ids,
        created_at=now,
        updated_at=now,
    )
    await _get_store().create_rag_agent(agent.to_dict())
    if linked_resource_ids:
        await _get_store().replace_rag_agent_resources(
            agent_id=agent.agent_id,
            owner_id=user_id,
            workspace_id=workspace_id,
            resource_ids=linked_resource_ids,
        )
    return agent


async def update_agent(
    *,
    agent_id: str,
    user_id: str,
    name: str | None,
    description: str | None,
    system_instructions: str | None,
    linked_resource_ids: list[str] | None,
) -> RagAgent | None:
    workspace_id = _workspace_id_for_user(user_id)
    existing = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not existing:
        return None

    patch: dict[str, str] = {}
    if name is not None:
        patch["name"] = name
    if description is not None:
        patch["description"] = description
    if system_instructions is not None:
        patch["system_instructions"] = system_instructions
    if patch:
        await _get_store().update_rag_agent(
            agent_id=agent_id,
            owner_id=user_id,
            workspace_id=workspace_id,
            patch=patch,
        )

    if linked_resource_ids is not None:
        if len(linked_resource_ids) > settings.rag_max_resources_per_agent:
            raise RagValidationError(
                "agent_resource_limit_exceeded",
                "Too many resources linked to this agent.",
            )
        await _validate_resources_linkable(
            owner_id=user_id,
            workspace_id=workspace_id,
            resource_ids=linked_resource_ids,
        )
        await _get_store().replace_rag_agent_resources(
            agent_id=agent_id,
            owner_id=user_id,
            workspace_id=workspace_id,
            resource_ids=linked_resource_ids,
        )

    updated = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not updated:
        return None
    return RagAgent(**updated)


async def delete_agent(agent_id: str, user_id: str) -> bool:
    workspace_id = _workspace_id_for_user(user_id)
    existing = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not existing:
        return False
    return await _get_store().delete_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )


async def link_resources(
    *,
    agent_id: str,
    user_id: str,
    resource_ids: list[str],
) -> RagAgent | None:
    workspace_id = _workspace_id_for_user(user_id)
    current = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not current:
        return None

    linked_ids = set(current.get("linked_resource_ids") or [])
    linked_ids.update(resource_ids)
    final_ids = list(linked_ids)

    if len(final_ids) > settings.rag_max_resources_per_agent:
        raise RagValidationError(
            "agent_resource_limit_exceeded",
            "Too many resources linked to this agent.",
        )

    await _validate_resources_linkable(
        owner_id=user_id,
        workspace_id=workspace_id,
        resource_ids=final_ids,
    )

    await _get_store().replace_rag_agent_resources(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
        resource_ids=final_ids,
    )

    updated = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not updated:
        return None
    return RagAgent(**updated)


async def _validate_resources_linkable(
    *,
    owner_id: str,
    workspace_id: str,
    resource_ids: list[str],
) -> None:
    if not resource_ids:
        return

    resources = await _get_store().get_rag_resources_by_ids(
        resource_ids=resource_ids,
        owner_id=owner_id,
        workspace_id=workspace_id,
    )
    existing_ids = {r["resource_id"] for r in resources}
    missing = [rid for rid in resource_ids if rid not in existing_ids]
    if missing:
        raise RagValidationError(
            "unauthorized_linkage",
            "One or more resources are not available in your workspace.",
        )

    non_ready = [
        r["resource_id"]
        for r in resources
        if _normalize_state(r.get("state", "")) != "ready"
    ]
    if non_ready:
        raise RagValidationError(
            "processing_failed",
            "Only resources in ready state can be linked.",
        )


async def get_agent_for_chat(
    agent_id: str, user_id: str
) -> tuple[RagAgent, list[str]] | None:
    workspace_id = _workspace_id_for_user(user_id)
    row = await _get_store().get_rag_agent(
        agent_id=agent_id,
        owner_id=user_id,
        workspace_id=workspace_id,
    )
    if not row:
        return None
    linked = row.get("linked_resource_ids") or []
    return RagAgent(**row), linked


async def create_or_get_chat_session(
    *,
    user_id: str,
    agent_id: str,
    session_id: str | None,
    initial_message: str | None = None,
) -> str:
    if session_id:
        valid = await _get_store().get_rag_chat_session(
            session_id=session_id,
            owner_id=user_id,
            agent_id=agent_id,
        )
        if valid:
            return session_id

    new_session = str(uuid.uuid4())
    await _get_store().create_rag_chat_session(
        {
            "session_id": new_session,
            "owner_id": user_id,
            "agent_id": agent_id,
            "workspace_id": _workspace_id_for_user(user_id),
            "title": await suggest_chat_session_title(initial_message),
        }
    )
    return new_session


async def create_or_get_workspace_chat_session(
    *,
    user_id: str,
    session_id: str | None,
    initial_message: str | None = None,
) -> str:
    if session_id:
        valid = await _get_store().get_rag_chat_session(
            session_id=session_id,
            owner_id=user_id,
            agent_id=None,
            chat_scope=CHAT_SCOPE_WORKSPACE,
        )
        if valid:
            return session_id

    new_session = str(uuid.uuid4())
    await _get_store().create_rag_chat_session(
        {
            "session_id": new_session,
            "owner_id": user_id,
            "agent_id": None,
            "workspace_id": _workspace_id_for_user(user_id),
            "title": await suggest_chat_session_title(initial_message),
            "chat_scope": CHAT_SCOPE_WORKSPACE,
        }
    )
    return new_session


async def list_chat_sessions(
    agent_id: str | None, user_id: str, chat_scope: str = CHAT_SCOPE_AGENT
) -> list[dict[str, str | None]]:
    return await _get_store().list_rag_chat_sessions(
        agent_id=agent_id, owner_id=user_id, chat_scope=chat_scope
    )


async def get_chat_session(
    *,
    session_id: str,
    agent_id: str | None,
    user_id: str,
    chat_scope: str = CHAT_SCOPE_AGENT,
) -> dict[str, str | None] | None:
    return await _get_store().get_rag_chat_session(
        session_id=session_id,
        owner_id=user_id,
        agent_id=agent_id,
        chat_scope=chat_scope,
    )


async def update_chat_session_title(
    *, session_id: str, agent_id: str | None, user_id: str, title: str, chat_scope: str = CHAT_SCOPE_AGENT
) -> bool:
    return await _get_store().update_rag_chat_session_title(
        session_id=session_id,
        owner_id=user_id,
        agent_id=agent_id,
        title=title,
        chat_scope=chat_scope,
    )


async def delete_chat_session(
    *, session_id: str, agent_id: str | None, user_id: str, chat_scope: str = CHAT_SCOPE_AGENT
) -> bool:
    return await _get_store().delete_rag_chat_session(
        session_id=session_id,
        owner_id=user_id,
        agent_id=agent_id,
        chat_scope=chat_scope,
    )


async def append_chat_message(message: RagChatMessage) -> None:
    await _get_store().create_rag_chat_message(message.to_dict())


async def delete_last_exchange(
    *, session_id: str, user_id: str
) -> tuple[bool, str | None]:
    return await _get_store().delete_last_user_assistant_pair(
        session_id=session_id, owner_id=user_id
    )


async def list_chat_messages(session_id: str, user_id: str) -> list[RagChatMessage]:
    rows = await _get_store().list_rag_chat_messages(
        session_id=session_id, owner_id=user_id
    )
    return [RagChatMessage(**row) for row in rows]


async def retrieve_context_for_query(
    *,
    user_id: str,
    resource_ids: list[str],
    question: str,
) -> RagQueryResult:
    if not resource_ids:
        return RagQueryResult(context="", chunks=[], entities=None)
    return await query_resource_context(
        store=_get_store(),
        resource_ids=resource_ids,
        query=question,
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )


async def list_workspace_ready_resource_ids(user_id: str) -> list[str]:
    workspace_id = _workspace_id_for_user(user_id)
    rows = await _get_store().list_rag_resources(owner_id=user_id, workspace_id=workspace_id)
    return [
        row["resource_id"]
        for row in rows
        if _normalize_state(str(row.get("state", ""))) == "ready"
    ]
