"""Collaboration route registration for the runtime API."""

from __future__ import annotations

from typing import Any, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query

from ...collaboration.modes import MODE_CONFIGS
from ..domain.models import now_iso
from .deps import RouteDeps
from . import router_impl as impl


AddCommentRequest = impl.AddCommentRequest
AddFeedbackRequest = impl.AddFeedbackRequest
_normalize_human_blocking_issues = impl._normalize_human_blocking_issues


def register_collab_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register collaboration mode, timeline, feedback, and comment routes."""
    @router.get("/collaboration/modes")
    async def get_collaboration_modes() -> dict[str, Any]:
        """Return configured collaboration mode options.
        
        Returns:
            A payload containing all collaboration mode definitions.
        """
        return {"modes": [config.to_dict() for config in MODE_CONFIGS.values()]}

    @router.get("/collaboration/presence")
    async def get_collaboration_presence() -> dict[str, Any]:
        """Return active collaboration presence information.
        
        Returns:
            A payload containing currently active users.
        """
        return {"users": []}

    @router.get("/collaboration/timeline/{task_id}")
    async def get_collaboration_timeline(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Assemble timeline events for task collaboration activity.
        
        Args:
            task_id: Identifier of the task whose timeline is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing timeline entries sorted by recency.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            return {"events": []}

        task_issues = _normalize_human_blocking_issues(
            task.metadata.get("human_blocking_issues") if isinstance(task.metadata, dict) else None
        )
        task_details = task.description or ""
        if task_issues:
            issue_summary = "; ".join(issue.get("summary", "") for issue in task_issues if issue.get("summary"))
            if issue_summary:
                task_details = (f"{task_details}\n\n" if task_details else "") + f"Human blockers: {issue_summary}"

        events: list[dict[str, Any]] = [
            {
                "id": f"task-{task.id}",
                "type": "status_change",
                "timestamp": task.updated_at or task.created_at,
                "actor": "system",
                "actor_type": "system",
                "summary": f"Task status: {task.status}",
                "details": task_details,
                "human_blocking_issues": task_issues,
            }
        ]
        # Inject human review actions from task metadata so they survive event-log eviction.
        # Only dedup approve/request_changes (clean 1:1 with history entries).
        # Retry events are NOT deduped because task.retry fires for all retries
        # but human_review_actions only records guided retries.
        injected_review_types: set[str] = set()
        action_type_map = {"approve": "task.approved", "request_changes": "task.changes_requested", "retry": "task.retry_with_guidance"}
        dedup_types = {"task.approved", "task.changes_requested"}
        raw_review_actions = task.metadata.get("human_review_actions") if isinstance(task.metadata, dict) else None
        if isinstance(raw_review_actions, list):
            for idx, entry in enumerate(raw_review_actions):
                if not isinstance(entry, dict):
                    continue
                action = str(entry.get("action") or "")
                ts = str(entry.get("ts") or task.created_at)
                guidance = str(entry.get("guidance") or "")
                previous_error = str(entry.get("previous_error") or "")
                event_type_label = action_type_map.get(action, f"task.{action}")
                if event_type_label in dedup_types:
                    injected_review_types.add(event_type_label)
                details = guidance
                if previous_error:
                    details = f"{guidance}\n\nPrevious error: {previous_error}" if guidance else f"Previous error: {previous_error}"
                events.append(
                    {
                        "id": f"review-action-{task.id}-{idx}",
                        "type": event_type_label,
                        "timestamp": ts,
                        "actor": "human",
                        "actor_type": "human",
                        "summary": {"approve": "Approved", "request_changes": "Changes requested", "retry": "Retry with guidance"}.get(action, action),
                        "details": details,
                    }
                )

        for event in container.events.list_recent(limit=2000):
            if event.get("entity_id") != task_id:
                continue
            # Deduplicate: skip event-log entries whose type was already injected
            # from human_review_actions (the authoritative source).
            event_type = str(event.get("type") or "")
            if event_type in injected_review_types:
                continue
            payload = event.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            issues = _normalize_human_blocking_issues(
                payload_dict.get("issues") if "issues" in payload_dict else payload_dict.get("human_blocking_issues")
            )
            details = str(payload_dict.get("error") or payload_dict.get("guidance") or "")
            if not details and issues:
                details = "; ".join(issue.get("summary", "") for issue in issues if issue.get("summary"))
            events.append(
                {
                    "id": str(event.get("id") or f"evt-{uuid.uuid4().hex[:10]}"),
                    "type": str(event.get("type") or "event"),
                    "timestamp": str(event.get("ts") or task.created_at),
                    "actor": "system",
                    "actor_type": "system",
                    "summary": str(event.get("type") or "event"),
                    "details": details,
                    "human_blocking_issues": issues,
                }
            )
        for item in deps.load_feedback_records(container):
            if item.get("task_id") != task_id:
                continue
            events.append(
                {
                    "id": f"feedback-{item.get('id')}",
                    "type": "feedback",
                    "timestamp": item.get("created_at") or task.created_at,
                    "actor": str(item.get("created_by") or "human"),
                    "actor_type": "human",
                    "summary": str(item.get("summary") or "Feedback added"),
                    "details": str(item.get("details") or ""),
                }
            )
        for item in deps.load_comment_records(container):
            if item.get("task_id") != task_id:
                continue
            events.append(
                {
                    "id": f"comment-{item.get('id')}",
                    "type": "comment",
                    "timestamp": item.get("created_at") or task.created_at,
                    "actor": str(item.get("author") or "human"),
                    "actor_type": "human",
                    "summary": str(item.get("body") or "Comment added"),
                    "details": "",
                }
            )
        events.sort(key=lambda event: str(event.get("timestamp") or ""), reverse=True)
        return {"events": events}

    @router.get("/collaboration/feedback/{task_id}")
    async def get_collaboration_feedback(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List feedback records linked to a task.
        
        Args:
            task_id: Identifier of the task whose feedback is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing feedback entries sorted by creation time.
        """
        container, _, _ = deps.ctx(project_dir)
        items = [item for item in deps.load_feedback_records(container) if item.get("task_id") == task_id]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"feedback": items}

    @router.post("/collaboration/feedback")
    async def add_collaboration_feedback(body: AddFeedbackRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create a new collaboration feedback entry for a task.
        
        Args:
            body: Feedback creation payload with summary and target metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the newly created feedback entry.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, bus, _ = deps.ctx(project_dir)
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
        items = deps.load_feedback_records(container)
        items.append(item)
        deps.save_feedback_records(container, items)
        bus.emit(channel="review", event_type="feedback.added", entity_id=body.task_id, payload={"feedback_id": item["id"]})
        return {"feedback": item}

    @router.post("/collaboration/feedback/{feedback_id}/dismiss")
    async def dismiss_collaboration_feedback(feedback_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Mark a feedback item as addressed.
        
        Args:
            feedback_id: Identifier of the feedback item to dismiss.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated feedback entry.
        
        Raises:
            HTTPException: If the feedback item cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        items = deps.load_feedback_records(container)
        for item in items:
            if item.get("id") == feedback_id:
                item["status"] = "addressed"
                item["agent_response"] = item.get("agent_response") or "Dismissed by reviewer"
                deps.save_feedback_records(container, items)
                bus.emit(channel="review", event_type="feedback.dismissed", entity_id=str(item.get("task_id") or ""), payload={"feedback_id": feedback_id})
                return {"feedback": item}
        raise HTTPException(status_code=404, detail="Feedback not found")

    @router.get("/collaboration/comments/{task_id}")
    async def get_collaboration_comments(task_id: str, project_dir: Optional[str] = Query(None), file_path: Optional[str] = Query(None)) -> dict[str, Any]:
        """List collaboration comments for a task, optionally by file path.
        
        Args:
            task_id: Identifier of the task whose comments are requested.
            project_dir: Optional project directory used to resolve runtime state.
            file_path: Optional file path filter.
        
        Returns:
            A payload containing sorted comment records.
        """
        container, _, _ = deps.ctx(project_dir)
        items = []
        for item in deps.load_comment_records(container):
            if item.get("task_id") != task_id:
                continue
            if file_path and item.get("file_path") != file_path:
                continue
            items.append(item)
        items.sort(key=lambda item: str(item.get("created_at") or ""))
        return {"comments": items}

    @router.post("/collaboration/comments")
    async def add_collaboration_comment(body: AddCommentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create a collaboration comment linked to a task or file location.
        
        Args:
            body: Comment creation payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created comment record.
        
        Raises:
            HTTPException: If the target task does not exist.
        """
        container, bus, _ = deps.ctx(project_dir)
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
        items = deps.load_comment_records(container)
        items.append(item)
        deps.save_comment_records(container, items)
        bus.emit(channel="review", event_type="comment.added", entity_id=body.task_id, payload={"comment_id": item["id"], "file_path": body.file_path})
        return {"comment": item}

    @router.post("/collaboration/comments/{comment_id}/resolve")
    async def resolve_collaboration_comment(comment_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Mark an existing collaboration comment as resolved.
        
        Args:
            comment_id: Identifier of the comment to resolve.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the resolved comment record.
        
        Raises:
            HTTPException: If the comment cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        items = deps.load_comment_records(container)
        for item in items:
            if item.get("id") == comment_id:
                item["resolved"] = True
                deps.save_comment_records(container, items)
                bus.emit(channel="review", event_type="comment.resolved", entity_id=str(item.get("task_id") or ""), payload={"comment_id": comment_id})
                return {"comment": item}
        raise HTTPException(status_code=404, detail="Comment not found")

