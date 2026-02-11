"""Collaboration API — feedback, review comments, HITL modes, and activity timeline."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..collaboration.feedback import (
    Feedback,
    FeedbackPriority,
    FeedbackStore,
    FeedbackType,
    ReviewComment,
)
from ..collaboration.modes import HITLMode, MODE_CONFIGS, get_mode_config
from ..collaboration.reasoning import ReasoningStore
from ..collaboration.timeline import StateChangeStore, TimelineAggregator
from .users import PresenceTracker, UserProfile, UserRole, UserStore


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AddFeedbackRequest(BaseModel):
    task_id: str
    feedback_type: str = "general"
    priority: str = "should"
    summary: str
    details: str = ""
    target_file: Optional[str] = None
    target_lines: Optional[str] = None
    action: str = ""
    original_value: str = ""
    replacement_value: str = ""
    created_by: str = ""


class AddCommentRequest(BaseModel):
    task_id: str
    file_path: str
    line_number: int
    body: str
    line_type: str = "context"
    author: str = ""
    parent_id: Optional[str] = None


class SetModeRequest(BaseModel):
    mode: str


class CreateUserRequest(BaseModel):
    username: str
    display_name: str = ""
    role: str = "developer"


class UpdatePresenceRequest(BaseModel):
    username: str
    viewing: Optional[str] = None
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_collaboration_router(
    get_feedback_store: Callable[[], FeedbackStore],
    get_reasoning_store: Optional[Callable[[], ReasoningStore]] = None,
    get_user_store: Optional[Callable[[], UserStore]] = None,
    get_presence: Optional[Callable[[], PresenceTracker]] = None,
    get_state_change_store: Optional[Callable[[], StateChangeStore]] = None,
    get_web_notifications: Optional[Callable] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v2/collaboration", tags=["collaboration"])

    # Mutable mode state: {"project": "autopilot", "task:<id>": "supervised", ...}
    _mode_state: dict[str, str] = {"project": "autopilot"}

    # -- Feedback endpoints --------------------------------------------------

    @router.post("/feedback")
    async def add_feedback(req: AddFeedbackRequest) -> dict[str, Any]:
        store = get_feedback_store()
        fb = Feedback(
            task_id=req.task_id,
            feedback_type=FeedbackType(req.feedback_type) if req.feedback_type else FeedbackType.GENERAL,
            priority=FeedbackPriority(req.priority) if req.priority else FeedbackPriority.SHOULD,
            summary=req.summary,
            details=req.details,
            target_file=req.target_file,
            target_lines=req.target_lines,
            action=req.action,
            original_value=req.original_value,
            replacement_value=req.replacement_value,
            created_by=req.created_by,
        )
        store.add_feedback(fb)
        return fb.to_dict()

    @router.get("/feedback/{task_id}")
    async def get_feedback(
        task_id: str,
        active_only: bool = False,
        feedback_type: Optional[str] = None,
    ) -> dict[str, Any]:
        store = get_feedback_store()
        ft = FeedbackType(feedback_type) if feedback_type else None
        items = store.get_feedback(task_id, active_only=active_only, feedback_type=ft)
        return {"feedback": [f.to_dict() for f in items]}

    @router.post("/feedback/{feedback_id}/address")
    async def address_feedback(feedback_id: str, agent_response: str = "") -> dict[str, str]:
        store = get_feedback_store()
        if not store.address_feedback(feedback_id, agent_response):
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"status": "addressed"}

    @router.post("/feedback/{feedback_id}/dismiss")
    async def dismiss_feedback(feedback_id: str) -> dict[str, str]:
        store = get_feedback_store()
        if not store.dismiss_feedback(feedback_id):
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"status": "dismissed"}

    @router.get("/feedback/{task_id}/prompt")
    async def get_feedback_prompt(task_id: str) -> dict[str, str]:
        store = get_feedback_store()
        instructions = store.get_prompt_instructions(task_id)
        return {"instructions": instructions}

    @router.get("/feedback/{task_id}/effectiveness")
    async def get_feedback_effectiveness(task_id: str) -> dict[str, Any]:
        store = get_feedback_store()
        return store.get_effectiveness_report(task_id)

    # -- Review Comments endpoints -------------------------------------------

    @router.post("/comments")
    async def add_comment(req: AddCommentRequest) -> dict[str, Any]:
        store = get_feedback_store()
        comment = ReviewComment(
            task_id=req.task_id,
            file_path=req.file_path,
            line_number=req.line_number,
            body=req.body,
            line_type=req.line_type,
            author=req.author,
            parent_id=req.parent_id,
        )
        store.add_comment(comment)
        if get_web_notifications:
            try:
                get_web_notifications().review_requested(req.task_id)
            except Exception:
                pass
        return comment.to_dict()

    @router.get("/comments/{comment_id}/replies")
    async def get_comment_replies(comment_id: str) -> dict[str, Any]:
        store = get_feedback_store()
        replies = store.get_replies(comment_id)
        return {"replies": [c.to_dict() for c in replies]}

    @router.get("/comments/{task_id}")
    async def get_comments(
        task_id: str,
        file_path: Optional[str] = None,
        unresolved_only: bool = False,
    ) -> dict[str, Any]:
        store = get_feedback_store()
        items = store.get_comments(task_id, file_path=file_path, unresolved_only=unresolved_only)
        return {"comments": [c.to_dict() for c in items]}

    @router.post("/comments/{comment_id}/resolve")
    async def resolve_comment(comment_id: str) -> dict[str, str]:
        store = get_feedback_store()
        if not store.resolve_comment(comment_id):
            raise HTTPException(status_code=404, detail="Comment not found")
        return {"status": "resolved"}

    # -- HITL Mode endpoints -------------------------------------------------

    @router.get("/modes")
    async def list_modes() -> dict[str, Any]:
        return {"modes": [c.to_dict() for c in MODE_CONFIGS.values()]}

    @router.get("/modes/{mode}")
    async def get_mode(mode: str) -> dict[str, Any]:
        config = get_mode_config(mode)
        return config.to_dict()

    @router.get("/modes/current")
    async def get_current_mode() -> dict[str, Any]:
        current = _mode_state.get("project", "autopilot")
        config = get_mode_config(current)
        return {"current_mode": current, "config": config.to_dict()}

    @router.put("/modes")
    async def set_mode(req: SetModeRequest) -> dict[str, Any]:
        try:
            validated = HITLMode(req.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}. Valid: {[m.value for m in HITLMode]}")
        _mode_state["project"] = validated.value
        config = get_mode_config(validated.value)
        if get_web_notifications:
            try:
                get_web_notifications().mode_changed(validated.value)
            except Exception:
                pass
        return {"mode": validated.value, "config": config.to_dict()}

    @router.put("/modes/task/{task_id}")
    async def set_task_mode(task_id: str, req: SetModeRequest) -> dict[str, Any]:
        try:
            validated = HITLMode(req.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}. Valid: {[m.value for m in HITLMode]}")
        _mode_state[f"task:{task_id}"] = validated.value
        config = get_mode_config(validated.value)
        return {"task_id": task_id, "mode": validated.value, "config": config.to_dict()}

    @router.get("/modes/task/{task_id}")
    async def get_task_mode(task_id: str) -> dict[str, Any]:
        task_mode = _mode_state.get(f"task:{task_id}")
        project_mode = _mode_state.get("project", "autopilot")
        effective = task_mode or project_mode
        config = get_mode_config(effective)
        return {
            "task_id": task_id,
            "task_mode": task_mode,
            "project_mode": project_mode,
            "effective_mode": effective,
            "config": config.to_dict(),
        }

    @router.delete("/modes/task/{task_id}")
    async def clear_task_mode(task_id: str) -> dict[str, str]:
        _mode_state.pop(f"task:{task_id}", None)
        return {"status": "cleared"}

    # -- Timeline endpoint ---------------------------------------------------

    @router.get("/timeline/{task_id}")
    async def get_timeline(task_id: str, limit: int = 100) -> dict[str, Any]:
        store = get_feedback_store()
        reasoning = get_reasoning_store() if get_reasoning_store else None
        state_changes = get_state_change_store() if get_state_change_store else None
        aggregator = TimelineAggregator(store, reasoning, state_changes)
        events = aggregator.get_timeline(task_id, limit=limit)
        return {"events": [e.to_dict() for e in events]}

    # -- User management endpoints -------------------------------------------

    @router.get("/users")
    async def list_users() -> dict[str, Any]:
        if not get_user_store:
            return {"users": []}
        store = get_user_store()
        return {"users": [u.to_dict() for u in store.list_users()]}

    @router.post("/users")
    async def create_user(req: CreateUserRequest) -> dict[str, Any]:
        if not get_user_store:
            raise HTTPException(status_code=501, detail="User store not configured")
        store = get_user_store()
        try:
            role = UserRole(req.role)
        except ValueError:
            role = UserRole.DEVELOPER
        user = store.create_user(req.username, role=role, display_name=req.display_name)
        return user.to_dict()

    @router.get("/users/{username}")
    async def get_user(username: str) -> dict[str, Any]:
        if not get_user_store:
            raise HTTPException(status_code=501, detail="User store not configured")
        store = get_user_store()
        user = store.get_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user.to_dict()

    # -- Presence endpoints --------------------------------------------------

    @router.get("/presence")
    async def get_online_users() -> dict[str, Any]:
        if not get_presence:
            return {"users": []}
        tracker = get_presence()
        return {"users": tracker.get_online_users()}

    @router.post("/presence")
    async def update_presence(req: UpdatePresenceRequest) -> dict[str, str]:
        if get_presence:
            tracker = get_presence()
            tracker.set_online(req.username, {
                "viewing": req.viewing or "",
                "task_id": req.task_id or "",
            })
        if get_user_store:
            get_user_store().update_last_seen(req.username)
        return {"status": "ok"}

    return router
