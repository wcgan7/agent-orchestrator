"""Collaboration feedback/comment route registration."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Query

from .router_state import (
    _load_comment_records,
    _load_feedback_records,
    _save_comment_records,
    _save_feedback_records,
)
from .schemas import AddCommentRequest, AddFeedbackRequest
from ..domain.models import now_iso


def register_collaboration_feedback_comment_routes(
    router: APIRouter,
    ctx: Callable[[Optional[str]], tuple[Any, Any, Any]],
) -> None:
    """Register collaboration feedback/comment CRUD endpoints."""

    @router.get("/collaboration/feedback/{task_id}")
    async def get_collaboration_feedback(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = ctx(project_dir)
        items = [item for item in _load_feedback_records(container) if item.get("task_id") == task_id]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"feedback": items}

    @router.post("/collaboration/feedback")
    async def add_collaboration_feedback(body: AddFeedbackRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = ctx(project_dir)
        task = container.tasks.get(body.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        item = {
            "id": f"fb-{uuid.uuid4().hex[:10]}",
            "task_id": body.task_id,
            "feedback_type": body.feedback_type,
            "priority": body.priority,
            "status": "active",
            "summary": body.summary,
            "details": body.details,
            "target_file": body.target_file,
            "action": f"{body.feedback_type}: {body.summary}",
            "created_by": "human",
            "created_at": now_iso(),
            "agent_response": None,
        }
        items = _load_feedback_records(container)
        items.append(item)
        _save_feedback_records(container, items)
        bus.emit(channel="review", event_type="feedback.added", entity_id=body.task_id, payload={"feedback_id": item["id"]})
        return {"feedback": item}

    @router.post("/collaboration/feedback/{feedback_id}/dismiss")
    async def dismiss_collaboration_feedback(feedback_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = ctx(project_dir)
        items = _load_feedback_records(container)
        for item in items:
            if item.get("id") == feedback_id:
                item["status"] = "addressed"
                item["agent_response"] = item.get("agent_response") or "Dismissed by reviewer"
                _save_feedback_records(container, items)
                bus.emit(channel="review", event_type="feedback.dismissed", entity_id=str(item.get("task_id") or ""), payload={"feedback_id": feedback_id})
                return {"feedback": item}
        raise HTTPException(status_code=404, detail="Feedback not found")

    @router.get("/collaboration/comments/{task_id}")
    async def get_collaboration_comments(task_id: str, project_dir: Optional[str] = Query(None), file_path: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = ctx(project_dir)
        items = []
        for item in _load_comment_records(container):
            if item.get("task_id") != task_id:
                continue
            if file_path and item.get("file_path") != file_path:
                continue
            items.append(item)
        items.sort(key=lambda item: str(item.get("created_at") or ""))
        return {"comments": items}

    @router.post("/collaboration/comments")
    async def add_collaboration_comment(body: AddCommentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = ctx(project_dir)
        task = container.tasks.get(body.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        item = {
            "id": f"cm-{uuid.uuid4().hex[:10]}",
            "task_id": body.task_id,
            "file_path": body.file_path,
            "line_number": body.line_number,
            "line_type": body.line_type,
            "body": body.body,
            "author": "human",
            "created_at": now_iso(),
            "resolved": False,
            "parent_id": body.parent_id,
        }
        items = _load_comment_records(container)
        items.append(item)
        _save_comment_records(container, items)
        bus.emit(channel="review", event_type="comment.added", entity_id=body.task_id, payload={"comment_id": item["id"], "file_path": body.file_path})
        return {"comment": item}

    @router.post("/collaboration/comments/{comment_id}/resolve")
    async def resolve_collaboration_comment(comment_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = ctx(project_dir)
        items = _load_comment_records(container)
        for item in items:
            if item.get("id") == comment_id:
                item["resolved"] = True
                _save_comment_records(container, items)
                bus.emit(channel="review", event_type="comment.resolved", entity_id=str(item.get("task_id") or ""), payload={"comment_id": comment_id})
                return {"comment": item}
        raise HTTPException(status_code=404, detail="Comment not found")
