"""Planner chat API router — interactive, multi-turn planning endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.auth import AuthenticatedUser, get_authenticated_user
from src.planner import save_software_dev_plan
from src.planner_graph.graph import planner_graph
from src.planner_graph.thread_store import planner_thread_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Planner Chat"])


class PlannerChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_planner_turn(
    message: str,
    thread_id: str,
    user_id: str,
    access_token: str | None = None,
) -> AsyncGenerator[str, None]:
    # Emit session event immediately
    yield _sse_line({"type": "session", "thread_id": thread_id})

    config = {"configurable": {"thread_id": thread_id}}

    # Get current conversation history from graph state (may be empty on first turn)
    existing_state = planner_graph.get_state(config)
    existing_history: list = []
    if existing_state and existing_state.values:
        existing_history = list(existing_state.values.get("conversation_history") or [])

    # Append the new human message
    new_history = existing_history + [HumanMessage(content=message)]

    # Append the user message to the thread store display history
    now = datetime.now(UTC).isoformat()
    planner_thread_store.append_message(
        thread_id,
        {
            "message_id": str(uuid.uuid4()),
            "role": "user",
            "content": message,
            "created_at": now,
        },
    )

    # Invoke the graph (blocking — run in thread pool)
    try:
        await asyncio.to_thread(
            planner_graph.invoke,
            {"conversation_history": new_history},
            config,
        )
    except Exception as exc:
        logger.error("Planner graph invocation failed: %s", exc)
        yield _sse_line({"type": "error", "error": "Graph execution failed."})
        yield _sse_line({"type": "done"})
        return

    # Read the resulting state
    result_state = planner_graph.get_state(config)
    if not result_state or not result_state.values:
        yield _sse_line({"type": "error", "error": "No state returned from graph."})
        yield _sse_line({"type": "done"})
        return

    state_values = result_state.values
    error = state_values.get("error")
    clarification_question = state_values.get("clarification_question")
    final_plan = state_values.get("final_plan")

    if error:
        ai_content = "I encountered an error while processing your request. Please try again."
        yield _sse_line({"type": "chunk", "text": ai_content})
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": ai_content,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        yield _sse_line({"type": "error", "error": error})
        yield _sse_line({"type": "done"})
        return

    # Determine AI response content and update LangGraph conversation history
    ai_history_content: str | None = None

    if final_plan is not None:
        # Stream the plan markdown in chunks for a streaming feel
        markdown = final_plan.markdown
        chunk_size = 200
        for i in range(0, len(markdown), chunk_size):
            yield _sse_line({"type": "chunk", "text": markdown[i : i + chunk_size]})

        plan_event = {
            "type": "plan",
            "plan": final_plan.plan.model_dump(mode="json"),
            "markdown": final_plan.markdown,
            "suggested_filename": final_plan.suggested_filename,
            "planning_brief": final_plan.planning_brief.model_dump(mode="json"),
            "repo_analysis": final_plan.repo_analysis.model_dump(mode="json"),
            "planning_options": final_plan.planning_options.model_dump(mode="json"),
        }
        yield _sse_line(plan_event)

        ai_message_dict = {
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": markdown,
            "plan_event": plan_event,
            "created_at": datetime.now(UTC).isoformat(),
        }
        planner_thread_store.append_message(thread_id, ai_message_dict)

        # Persist to Supabase (non-fatal if it fails)
        try:
            await save_software_dev_plan(user_id, message, final_plan)
        except Exception as persist_exc:
            logger.warning("Failed to persist planner output: %s", persist_exc)

        # Store a compact summary in history (not the full markdown) to keep
        # the context manageable for future clarification/refinement turns.
        ai_history_content = (
            "I have generated a complete software development plan based on our discussion. "
            "If you'd like to refine it, describe what you'd like to change."
        )

    elif clarification_question:
        yield _sse_line({"type": "chunk", "text": clarification_question})
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": clarification_question,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        ai_history_content = clarification_question
    else:
        # Graph ended without producing a question or plan (unexpected)
        msg = "I'm ready to generate your plan. What else would you like to clarify?"
        yield _sse_line({"type": "chunk", "text": msg})
        planner_thread_store.append_message(
            thread_id,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": msg,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        ai_history_content = msg

    # Persist the AI response back into the LangGraph conversation history so
    # future turns (refinements) have full context about what was said.
    if ai_history_content:
        updated_history = new_history + [AIMessage(content=ai_history_content)]
        try:
            await asyncio.to_thread(
                planner_graph.update_state,
                config,
                {"conversation_history": updated_history},
            )
        except Exception as update_exc:
            logger.warning("Failed to update planner graph state: %s", update_exc)

    yield _sse_line({"type": "done"})


@router.post("/api/planner/chat")
async def planner_chat(
    body: PlannerChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> StreamingResponse:
    if body.thread_id:
        entry = planner_thread_store.get_thread(body.thread_id, current_user.user_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Thread not found or expired.")
        thread_id = body.thread_id
    else:
        thread_id = planner_thread_store.create_thread(user_id=current_user.user_id)

    return StreamingResponse(
        _stream_planner_turn(
            message=body.message,
            thread_id=thread_id,
            user_id=current_user.user_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/planner/chat/{thread_id}/messages")
async def get_planner_chat_messages(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict:
    entry = planner_thread_store.get_thread(thread_id, current_user.user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Thread not found or expired.")
    return {"thread_id": thread_id, "messages": entry.messages}


@router.delete("/api/planner/chat/{thread_id}/last")
async def delete_planner_chat_last_exchange(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict:
    entry = planner_thread_store.get_thread(thread_id, current_user.user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Thread not found or expired.")
    # v1 limitation: only removes display history; does not roll back the LangGraph MemorySaver
    # checkpoint because the MemorySaver API does not expose a rollback operation.
    planner_thread_store.delete_last_exchange(thread_id)
    return {"thread_id": thread_id, "deleted": True}
