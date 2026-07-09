"""RAG resource routes: upload, list, delete, and ingestion status."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src import outbox
from src.api.deps import _raise_rag_validation_error
from src.auth import AuthenticatedUser, get_authenticated_user
from src.rag import (
    RagValidationError,
    create_resource_and_ingest,
    delete_resource as delete_rag_resource_record,
    get_resource_status,
    list_resources as list_rag_resources_records,
)

router = APIRouter()


@router.post("/api/rag/resources/upload", tags=["RAG"])
async def rag_upload_resource(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        resource, job = await create_resource_and_ingest(file, current_user.user_id)
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    background_tasks.add_task(outbox.dispatch_outbox_events, limit=10)
    return {
        "resource": resource.to_dict(),
        "job": job.to_dict(),
    }


@router.get("/api/rag/resources", tags=["RAG"])
async def rag_list_resources(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    resources = await list_rag_resources_records(current_user.user_id)
    return {"resources": [r.to_dict() for r in resources]}


@router.delete("/api/rag/resources/{resource_id}", tags=["RAG"])
async def rag_delete_resource(
    resource_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    deleted = await delete_rag_resource_record(resource_id, current_user.user_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Resource '{resource_id}' not found."
        )
    return {"resource_id": resource_id, "deleted": True}


@router.get("/api/rag/resources/{resource_id}/status", tags=["RAG"])
async def rag_resource_status(
    resource_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    status_payload = await get_resource_status(resource_id, current_user.user_id)
    if not status_payload:
        raise HTTPException(
            status_code=404, detail=f"Resource '{resource_id}' not found."
        )
    return status_payload
