"""User memory routes: read, update, and delete the user's memory document."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.auth import AuthenticatedUser, get_authenticated_user
from src.user_memory import delete_user_memory, get_user_memory, update_user_memory

router = APIRouter()


class MemoryUpdateRequest(BaseModel):
    content: str


@router.get("/api/memory", tags=["Memory"])
async def get_memory_endpoint(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    return await get_user_memory(current_user.user_id)


@router.put("/api/memory", tags=["Memory"])
async def update_memory_endpoint(
    body: MemoryUpdateRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content cannot be empty.")
    return await update_user_memory(current_user.user_id, content)


@router.delete("/api/memory", tags=["Memory"])
async def delete_memory_endpoint(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    return await delete_user_memory(current_user.user_id)
