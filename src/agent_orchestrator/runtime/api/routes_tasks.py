"""Task-focused route registration for the runtime API."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Optional, cast

from fastapi import APIRouter, HTTPException, Query

from ...pipelines.registry import PipelineRegistry
from ..domain.models import (
    ApprovalMode,
    DependencyPolicy,
    Priority,
    Task,
    TaskStatus,
    now_iso,
)
from ..storage.bootstrap import archive_state_root, ensure_state_root
from .deps import RouteDeps
from . import router_impl as impl


AddDependencyRequest = impl.AddDependencyRequest
ApproveGateRequest = impl.ApproveGateRequest
CommitPlanRequest = impl.CommitPlanRequest
CreatePlanRevisionRequest = impl.CreatePlanRevisionRequest
CreateTaskRequest = impl.CreateTaskRequest
GenerateTasksRequest = impl.GenerateTasksRequest
PipelineClassificationRequest = impl.PipelineClassificationRequest
PipelineClassificationResponse = impl.PipelineClassificationResponse
PlanRefineRequest = impl.PlanRefineRequest
RetryTaskRequest = impl.RetryTaskRequest
TransitionRequest = impl.TransitionRequest
UpdateTaskRequest = impl.UpdateTaskRequest
VALID_TRANSITIONS = impl.VALID_TRANSITIONS
_canonical_task_type_for_pipeline = impl._canonical_task_type_for_pipeline
_execution_batches = impl._execution_batches
_has_unresolved_blockers = impl._has_unresolved_blockers
_logs_snapshot_id = impl._logs_snapshot_id
_normalize_pipeline_classification_output = impl._normalize_pipeline_classification_output
_priority_rank = impl._priority_rank
_read_from_offset = impl._read_from_offset
_read_tail = impl._read_tail
_safe_state_path = impl._safe_state_path
_task_payload = impl._task_payload


def _remove_task_relationship_refs(*, task_id: str, container: Any) -> None:
    """Remove references to a deleted task from all remaining tasks."""
    for existing in container.tasks.list():
        changed = False
        if task_id in existing.blocked_by:
            existing.blocked_by = [dep_id for dep_id in existing.blocked_by if dep_id != task_id]
            changed = True
        if task_id in existing.blocks:
            existing.blocks = [dep_id for dep_id in existing.blocks if dep_id != task_id]
            changed = True
        if existing.parent_id == task_id:
            existing.parent_id = None
            changed = True
        if task_id in existing.children_ids:
            existing.children_ids = [child_id for child_id in existing.children_ids if child_id != task_id]
            changed = True
        if changed:
            existing.updated_at = now_iso()
            container.tasks.upsert(existing)


def register_task_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register task, planning, execution, and logs routes."""
    @router.post("/tasks")
    async def create_task(body: CreateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create and persist a task with resolved pipeline configuration.
        
        Args:
            body: Request payload describing task metadata and pipeline preferences.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created task details.
        
        Raises:
            HTTPException: If task type auto-resolution inputs are invalid.
        """
        container, bus, _ = deps.ctx(project_dir)
        registry = PipelineRegistry()
        allowed_pipelines = sorted(template.id for template in registry.list_templates())
        classifier_pipeline_id = str(body.classifier_pipeline_id or "").strip()
        classifier_confidence = str(body.classifier_confidence or "").strip().lower()
        classifier_reason = str(body.classifier_reason or "").strip()
        classifier_pipeline_valid = classifier_pipeline_id in allowed_pipelines
        classifier_confidence_valid = classifier_confidence in {"high", "low"}
        requested_task_type = str(body.task_type or "feature").strip() or "feature"

        if requested_task_type == "auto":
            if classifier_confidence != "high" or not classifier_pipeline_valid:
                raise HTTPException(
                    status_code=400,
                    detail="Auto task type requires a high-confidence classifier result.",
                )
            resolved_task_type = _canonical_task_type_for_pipeline(registry, classifier_pipeline_id)
        else:
            resolved_task_type = requested_task_type

        template = registry.resolve_for_task_type(resolved_task_type)
        final_pipeline_id = template.id
        pipeline_steps = body.pipeline_template
        if pipeline_steps is None:
            pipeline_steps = template.step_names()
        dep_policy = body.dependency_policy.strip() if body.dependency_policy else ""
        if dep_policy not in ("permissive", "prudent", "strict"):
            cfg = container.config.load()
            dep_policy = str((cfg.get("defaults") or {}).get("dependency_policy") or "prudent")
            if dep_policy not in ("permissive", "prudent", "strict"):
                dep_policy = "prudent"
        priority = body.priority if body.priority in ("P0", "P1", "P2", "P3") else "P2"
        approval_mode = body.approval_mode if body.approval_mode in ("human_review", "auto_approve") else "human_review"
        task = Task(
            title=body.title,
            description=body.description,
            task_type=resolved_task_type,
            priority=cast(Priority, priority),
            labels=body.labels,
            blocked_by=body.blocked_by,
            parent_id=body.parent_id,
            pipeline_template=pipeline_steps,
            approval_mode=cast(ApprovalMode, approval_mode),
            hitl_mode=body.hitl_mode,
            dependency_policy=cast(DependencyPolicy, dep_policy),
            source=body.source,
            worker_model=(str(body.worker_model).strip() if body.worker_model else None),
            metadata=dict(body.metadata or {}),
            project_commands=(body.project_commands if body.project_commands else None),
        )
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        if classifier_pipeline_valid and classifier_confidence_valid:
            task.metadata["classifier_pipeline_id"] = classifier_pipeline_id
            task.metadata["classifier_confidence"] = classifier_confidence
            task.metadata["classifier_reason"] = classifier_reason
            task.metadata["was_user_override"] = bool(body.was_user_override)
        task.metadata["final_pipeline_id"] = final_pipeline_id
        if body.status in ("backlog", "queued"):
            task.status = cast(TaskStatus, body.status)
        if task.parent_id:
            parent = container.tasks.get(task.parent_id)
            if parent and task.id not in parent.children_ids:
                parent.children_ids.append(task.id)
                container.tasks.upsert(parent)
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.created", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task)}

    @router.post("/tasks/classify-pipeline", response_model=PipelineClassificationResponse)
    async def classify_pipeline(
        body: PipelineClassificationRequest,
        project_dir: Optional[str] = Query(None),
    ) -> PipelineClassificationResponse:
        """Classify the best pipeline template for a proposed task.
        
        Args:
            body: Request payload with title, description, and optional metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A normalized pipeline classification result with confidence and rationale.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        registry = PipelineRegistry()
        allowed_pipelines = sorted(template.id for template in registry.list_templates())
        synthetic = Task(
            title=str(body.title or "").strip(),
            description=str(body.description or "").strip(),
            task_type="feature",
            status="queued",
            metadata=dict(body.metadata or {}),
        )
        if not isinstance(synthetic.metadata, dict):
            synthetic.metadata = {}
        synthetic.metadata["classification_allowed_pipelines"] = allowed_pipelines
        try:
            result = orchestrator.worker_adapter.run_step_ephemeral(task=synthetic, step="pipeline_classify", attempt=1)
        except Exception:
            result = None
        if result is None or result.status != "ok":
            return PipelineClassificationResponse(
                pipeline_id="feature",
                task_type="feature",
                confidence="low",
                reason="Pipeline auto-classification failed. Please select a pipeline.",
                allowed_pipelines=allowed_pipelines,
            )
        return _normalize_pipeline_classification_output(
            summary=result.summary,
            allowed_pipelines=allowed_pipelines,
            registry=registry,
        )

    @router.get("/tasks")
    async def list_tasks(
        project_dir: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        priority: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """List tasks filtered by lifecycle, type, or priority.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
            status: Optional status filter.
            task_type: Optional task type filter.
            priority: Optional priority filter.
        
        Returns:
            A payload with sorted task summaries and total count.
        """
        container, _, _ = deps.ctx(project_dir)
        tasks = container.tasks.list()
        filtered = []
        for task in tasks:
            if status and task.status != status:
                continue
            if task_type and task.task_type != task_type:
                continue
            if priority and task.priority != priority:
                continue
            filtered.append(task)
        filtered.sort(key=lambda t: (_priority_rank(t.priority), t.created_at))
        return {"tasks": [_task_payload(task) for task in filtered], "total": len(filtered)}

    @router.get("/tasks/board")
    async def board(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return task data grouped by Kanban-style status columns.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload keyed by task status with sorted task cards.
        """
        container, _, _ = deps.ctx(project_dir)
        columns: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ["backlog", "queued", "in_progress", "in_review", "blocked", "done", "cancelled"]
        }
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(_task_payload(task))
        for key, items in columns.items():
            items.sort(key=lambda x: (_priority_rank(str(x.get("priority") or "P3")), str(x.get("created_at") or "")))
        return {"columns": columns}

    @router.post("/tasks/clear")
    async def clear_tasks(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Clear all tasks by archiving runtime state and reinitializing storage.

        Args:
            project_dir: Optional project directory used to resolve runtime state.

        Returns:
            A payload indicating clear status and archive destination path.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        # Pause intake and stop scheduler/workers before mutating state files.
        orchestrator.control("pause")
        orchestrator.shutdown(timeout=10.0)

        archived_to = archive_state_root(container.project_dir)
        ensure_state_root(container.project_dir)
        deps.job_store.clear()
        archive_path = str(archived_to) if archived_to else ""
        message = (
            f"Cleared all tasks. Archived previous runtime state to {archive_path}."
            if archive_path
            else "Cleared all tasks. No existing runtime state archive was needed."
        )
        payload = {"archived_to": archive_path, "message": message, "cleared_at": now_iso()}
        bus.emit(channel="tasks", event_type="tasks.cleared", entity_id=container.project_id, payload=payload)
        bus.emit(channel="notifications", event_type="tasks.cleared", entity_id=container.project_id, payload=payload)
        return {"cleared": True, **payload}

    @router.get("/tasks/execution-order")
    async def execution_order(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Compute execution batches for non-terminal tasks.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing ready-to-run batches and recently completed tasks.
        """
        container, _, _ = deps.ctx(project_dir)
        tasks = container.tasks.list()
        terminal = {"done", "cancelled"}
        pending = [t for t in tasks if t.status not in terminal]
        completed = [t for t in tasks if t.status in terminal]
        completed.sort(key=lambda t: t.updated_at, reverse=True)
        return {
            "batches": _execution_batches(pending),
            "completed": [t.id for t in completed],
        }

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Load one task and return its expanded payload.
        
        Args:
            task_id: Identifier of the task to fetch.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the requested task.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": _task_payload(task, container)}

    @router.delete("/tasks/{task_id}")
    async def delete_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete a terminal task and clean stale task relationship references.

        Args:
            task_id: Identifier of the task to delete.
            project_dir: Optional project directory used to resolve runtime state.

        Returns:
            A payload indicating successful deletion.

        Raises:
            HTTPException: If the task is missing or non-terminal.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status not in {"done", "cancelled"}:
            raise HTTPException(status_code=400, detail="Only terminal tasks (done/cancelled) can be deleted.")

        _remove_task_relationship_refs(task_id=task_id, container=container)
        if not container.tasks.delete(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        bus.emit(channel="tasks", event_type="task.deleted", entity_id=task_id, payload={"status": task.status})
        return {"deleted": True, "task_id": task_id}

    @router.get("/tasks/{task_id}/diff")
    async def get_task_diff(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Load git diff metadata for the most recent commit step on a task.

        Args:
            task_id: Identifier of the task whose commit diff should be loaded.
            project_dir: Optional project directory used to resolve runtime
                state and git context.

        Returns:
            A payload with ``commit``, ``files``, ``diff``, and ``stat`` keys.
            When no commit step exists, returns
            ``{"commit": None, "files": [], "diff": "", "stat": ""}``.

        Raises:
            HTTPException: If the task does not exist (404) or git diff/stat
                retrieval fails (500).
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Find the commit SHA from the latest run's steps.
        commit_sha: str | None = None
        for run_id in reversed(task.run_ids):
            for run in container.runs.list():
                if run.id == run_id:
                    for step in reversed(run.steps or []):
                        if isinstance(step, dict) and step.get("step") == "commit" and step.get("commit"):
                            commit_sha = str(step["commit"])
                            break
                    if commit_sha:
                        break
            if commit_sha:
                break

        if not commit_sha:
            return {"commit": None, "files": [], "diff": "", "stat": ""}

        git_dir = container.project_dir
        try:
            stat_result = subprocess.run(
                ["git", "show", "--stat", "--format=", commit_sha],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
            diff_result = subprocess.run(
                ["git", "diff", f"{commit_sha}~1..{commit_sha}"],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read diff: {exc}") from exc

        # Parse stat lines into structured file list.
        files: list[dict[str, str]] = []
        for line in stat_result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("Showing") or " | " not in line:
                continue
            parts = line.split(" | ", 1)
            file_path = parts[0].strip()
            change_info = parts[1].strip() if len(parts) > 1 else ""
            files.append({"path": file_path, "changes": change_info})

        return {
            "commit": commit_sha,
            "files": files,
            "diff": diff_result.stdout,
            "stat": stat_result.stdout,
        }

    @router.get("/tasks/{task_id}/logs")
    async def get_task_logs(
        task_id: str,
        project_dir: Optional[str] = Query(None),
        max_chars: int = Query(12000, ge=200, le=2000000),
        stdout_offset: int = Query(0, ge=0),
        stderr_offset: int = Query(0, ge=0),
        backfill: bool = Query(False),
        stdout_read_to: int = Query(0, ge=0),
        stderr_read_to: int = Query(0, ge=0),
        step: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Return active, historical, or incremental task log output.
        
        Args:
            task_id: Identifier of the task whose logs are requested.
            project_dir: Optional project directory used to resolve runtime state.
            max_chars: Maximum number of characters returned from each stream read.
            stdout_offset: Byte offset for incremental stdout reads.
            stderr_offset: Byte offset for incremental stderr reads.
            backfill: Whether to align incremental reads to full-line boundaries.
            stdout_read_to: Optional upper byte boundary for stdout reads.
            stderr_read_to: Optional upper byte boundary for stderr reads.
            step: Optional step name to load logs from historical run metadata.
        
        Returns:
            A payload containing log text, offsets, and progress metadata.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        # When a specific step is requested, look up its log paths from run.steps.
        step_logs_meta: Optional[dict[str, Any]] = None
        if step:
            for run_id in reversed(task.run_ids):
                for run in container.runs.list():
                    if run.id == run_id:
                        # Find the last matching step entry (latest attempt).
                        for entry in reversed(run.steps or []):
                            if isinstance(entry, dict) and entry.get("step") == step and entry.get("stdout_path"):
                                step_logs_meta = entry
                                break
                        break
                if step_logs_meta:
                    break

        if step_logs_meta:
            logs_meta = step_logs_meta
            mode = "history"
        else:
            active_meta = metadata.get("active_logs") if isinstance(metadata.get("active_logs"), dict) else None
            last_meta = metadata.get("last_logs") if isinstance(metadata.get("last_logs"), dict) else None
            logs_meta = active_meta or last_meta or {}
            mode = "active" if active_meta else ("last" if last_meta else "none")

        stdout_path = _safe_state_path(logs_meta.get("stdout_path"), container.state_root)
        stderr_path = _safe_state_path(logs_meta.get("stderr_path"), container.state_root)
        progress_path = _safe_state_path(logs_meta.get("progress_path"), container.state_root)

        progress_payload: dict[str, Any] = {}
        if progress_path and progress_path.exists():
            try:
                raw = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    progress_payload = raw
            except Exception:
                progress_payload = {}

        use_incremental = backfill or stdout_offset > 0 or stderr_offset > 0
        stdout_tail_start = 0
        stderr_tail_start = 0
        stdout_chunk_start = 0
        stderr_chunk_start = 0
        if use_incremental:
            stdout_text, new_stdout_offset, stdout_chunk_start = _read_from_offset(
                stdout_path,
                stdout_offset,
                max_chars,
                read_to=stdout_read_to,
                align_start_to_line=backfill,
            )
            stderr_text, new_stderr_offset, stderr_chunk_start = _read_from_offset(
                stderr_path,
                stderr_offset,
                max_chars,
                read_to=stderr_read_to,
                align_start_to_line=backfill,
            )
        else:
            stdout_text, stdout_tail_start = _read_tail(stdout_path, max_chars)
            stderr_text, stderr_tail_start = _read_tail(stderr_path, max_chars)
            new_stdout_offset = stdout_path.stat().st_size if stdout_path and stdout_path.exists() else 0
            new_stderr_offset = stderr_path.stat().st_size if stderr_path and stderr_path.exists() else 0
            stdout_chunk_start = stdout_tail_start
            stderr_chunk_start = stderr_tail_start

        # Collect which steps have stored logs and count executions.
        available_steps: list[str] = []
        step_execution_counts: dict[str, int] = {}
        seen: set[str] = set()
        for run_id in task.run_ids[-1:]:
            for run in container.runs.list():
                if run.id == run_id:
                    for entry in (run.steps or []):
                        if isinstance(entry, dict) and entry.get("stdout_path"):
                            s = str(entry.get("step") or "")
                            if s:
                                step_execution_counts[s] = step_execution_counts.get(s, 0) + 1
                                if s not in seen:
                                    available_steps.append(s)
                                    seen.add(s)
                    break
        # Include the currently running step so users can switch between
        # earlier completed steps and the live step while a task is in progress.
        active_step = (metadata.get("active_logs") or {}).get("step") if isinstance(metadata.get("active_logs"), dict) else None
        if not active_step:
            active_step = task.current_step
        if active_step and active_step not in seen:
            available_steps.append(str(active_step))

        return {
            "mode": mode,
            "task_status": task.status,
            "step": str(logs_meta.get("step") or task.current_step or ""),
            "current_step": str(task.current_step or ""),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_offset": new_stdout_offset,
            "stderr_offset": new_stderr_offset,
            "stdout_chunk_start": stdout_chunk_start,
            "stderr_chunk_start": stderr_chunk_start,
            "stdout_tail_start": stdout_tail_start,
            "stderr_tail_start": stderr_tail_start,
            "started_at": logs_meta.get("started_at"),
            "finished_at": logs_meta.get("finished_at"),
            "log_id": _logs_snapshot_id(logs_meta),
            "progress": progress_payload,
            "available_steps": available_steps,
            "step_execution_counts": step_execution_counts,
        }

    @router.patch("/tasks/{task_id}")
    async def patch_task(task_id: str, body: UpdateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Apply mutable field updates to an existing task.
        
        Args:
            task_id: Identifier of the task to update.
            body: Partial task update payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task is missing or if status mutation is requested.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        updates = body.model_dump(exclude_none=True)
        if "status" in updates:
            raise HTTPException(
                status_code=400,
                detail="Task status cannot be changed via PATCH. Use /transition, /retry, /cancel, or review actions.",
            )
        if "project_commands" in updates and not updates["project_commands"]:
            updates["project_commands"] = None
        for key, value in updates.items():
            setattr(task, key, value)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.updated", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/{task_id}/transition")
    async def transition_task(task_id: str, body: TransitionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Transition a task to a valid next status.
        
        Args:
            task_id: Identifier of the task to transition.
            body: Requested target status payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the transitioned task.
        
        Raises:
            HTTPException: If the task is missing, the transition is invalid, or blockers remain.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        target = body.status
        valid = VALID_TRANSITIONS.get(task.status, set())
        if target not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid transition {task.status} -> {target}")
        if target == "queued":
            unresolved = _has_unresolved_blockers(container, task)
            if unresolved is not None:
                raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        task.status = cast(TaskStatus, target)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.transitioned", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/{task_id}/run")
    async def run_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Trigger orchestrator execution for a queued task.
        
        Args:
            task_id: Identifier of the task to execute.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the task after run dispatch.
        
        Raises:
            HTTPException: If the task cannot be run.
        """
        _, _, orchestrator = deps.ctx(project_dir)
        try:
            task = orchestrator.run_task(task_id)
        except ValueError as exc:
            if "Task not found" in str(exc):
                raise HTTPException(status_code=404, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        return {"task": _task_payload(task)}

    @router.post("/tasks/{task_id}/retry")
    async def retry_task(task_id: str, body: Optional[RetryTaskRequest] = None, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Retry a task after clearing error and gate state.
        
        Args:
            task_id: Identifier of the task to retry.
            body: Optional retry payload containing guidance and restart step.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the retried task.
        
        Raises:
            HTTPException: If the task is missing or unresolved blockers remain.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        unresolved = _has_unresolved_blockers(container, task)
        if unresolved is not None:
            raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        # Capture previous error before clearing it.
        previous_error = task.error or ""
        task.retry_count += 1
        task.status = "queued"
        task.error = None
        task.pending_gate = None
        task.metadata = task.metadata if isinstance(task.metadata, dict) else {}
        task.metadata.pop("human_blocking_issues", None)
        guidance = (body.guidance if body else None) or ""
        ts = now_iso()
        if guidance.strip() or previous_error.strip():
            task.metadata["retry_guidance"] = {
                "ts": ts,
                "guidance": guidance.strip(),
                "previous_error": previous_error.strip(),
            }
        if guidance.strip():
            history_list: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            history_list.append({"action": "retry", "ts": ts, "guidance": guidance.strip(), "previous_error": previous_error.strip()})
        start_from = (body.start_from_step if body else None) or ""
        if start_from.strip():
            task.metadata["retry_from_step"] = start_from.strip()
        else:
            task.metadata.pop("retry_from_step", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.retry", entity_id=task.id, payload={"retry_count": task.retry_count})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Cancel a task and emit a cancellation event.
        
        Args:
            task_id: Identifier of the task to cancel.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the cancelled task.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.status = "cancelled"
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.cancelled", entity_id=task.id, payload={})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/{task_id}/approve-gate")
    async def approve_gate(task_id: str, body: ApproveGateRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Clear a pending human gate on a task.
        
        Args:
            task_id: Identifier of the task awaiting gate approval.
            body: Gate approval payload, including optional gate name validation.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with the updated task and cleared gate value.
        
        Raises:
            HTTPException: If the task is missing or no matching pending gate exists.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.pending_gate:
            raise HTTPException(status_code=400, detail="No pending gate on this task")
        if body.gate and body.gate != task.pending_gate:
            raise HTTPException(status_code=400, detail=f"Gate mismatch: pending={task.pending_gate}, requested={body.gate}")
        cleared_gate = task.pending_gate
        task.pending_gate = None
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.gate_approved", entity_id=task.id, payload={"gate": cleared_gate})
        return {"task": _task_payload(task, container), "cleared_gate": cleared_gate}

    @router.post("/tasks/{task_id}/dependencies")
    async def add_dependency(task_id: str, body: AddDependencyRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Link a task to another task as a blocker.
        
        Args:
            task_id: Identifier of the blocked task.
            body: Dependency payload identifying the blocker task.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated blocked task.
        
        Raises:
            HTTPException: If either task cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        blocker = container.tasks.get(body.depends_on)
        if not task or not blocker:
            raise HTTPException(status_code=404, detail="Task or dependency not found")
        if body.depends_on not in task.blocked_by:
            task.blocked_by.append(body.depends_on)
        if task.id not in blocker.blocks:
            blocker.blocks.append(task.id)
        container.tasks.upsert(task)
        container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_added", entity_id=task.id, payload={"depends_on": body.depends_on})
        return {"task": _task_payload(task, container)}

    @router.delete("/tasks/{task_id}/dependencies/{dep_id}")
    async def remove_dependency(task_id: str, dep_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Remove an existing dependency edge between two tasks.
        
        Args:
            task_id: Identifier of the task being unblocked.
            dep_id: Identifier of the blocker task to unlink.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.blocked_by = [item for item in task.blocked_by if item != dep_id]
        container.tasks.upsert(task)
        blocker = container.tasks.get(dep_id)
        if blocker:
            blocker.blocks = [item for item in blocker.blocks if item != task.id]
            container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_removed", entity_id=task.id, payload={"dep_id": dep_id})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/analyze-dependencies")
    async def analyze_dependencies(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Run dependency inference and return inferred edges.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing inferred dependency edges across tasks.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        orchestrator._maybe_analyze_dependencies()
        # Collect all inferred edges across tasks
        edges: list[dict[str, str]] = []
        for task in container.tasks.list():
            inferred = task.metadata.get("inferred_deps") if isinstance(task.metadata, dict) else None
            if isinstance(inferred, list):
                for dep in inferred:
                    if isinstance(dep, dict):
                        edges.append({"from": dep.get("from", ""), "to": task.id, "reason": dep.get("reason", "")})
        return {"edges": edges}

    @router.post("/tasks/{task_id}/reset-dep-analysis")
    async def reset_dep_analysis(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Remove dependency-analysis metadata from a task.
        
        Args:
            task_id: Identifier of the task whose analysis metadata is reset.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if isinstance(task.metadata, dict):
            # Remove inferred blocked_by entries and corresponding blocks on blockers
            inferred = task.metadata.get("inferred_deps")
            if isinstance(inferred, list):
                inferred_from_ids = {
                    dep_id
                    for dep in inferred
                    if isinstance(dep, dict)
                    for dep_id in [dep.get("from")]
                    if isinstance(dep_id, str) and dep_id
                }
                task.blocked_by = [bid for bid in task.blocked_by if bid not in inferred_from_ids]
                for blocker_id in inferred_from_ids:
                    blocker = container.tasks.get(blocker_id)
                    if blocker:
                        blocker.blocks = [bid for bid in blocker.blocks if bid != task.id]
                        container.tasks.upsert(blocker)
            task.metadata.pop("deps_analyzed", None)
            task.metadata.pop("inferred_deps", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.dep_analysis_reset", entity_id=task.id, payload={})
        return {"task": _task_payload(task, container)}

    @router.get("/tasks/{task_id}/workdoc")
    async def get_task_workdoc(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the orchestrator work document for a task.
        
        Args:
            task_id: Identifier of the task whose work document is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The serialized work document payload.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = deps.resolve_orchestrator(project_dir)
        return orchestrator.get_workdoc(task_id)

    @router.get("/tasks/{task_id}/plan")
    async def get_task_plan(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the current plan document for a task.
        
        Args:
            task_id: Identifier of the task whose plan is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The serialized plan document payload.
        
        Raises:
            HTTPException: If the task is missing or plan retrieval fails validation.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = deps.resolve_orchestrator(project_dir)
        try:
            return orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/tasks/{task_id}/plan/refine")
    async def refine_task_plan(
        task_id: str,
        body: PlanRefineRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Queue a plan-refinement job for a task.
        
        Args:
            task_id: Identifier of the task whose plan is being refined.
            body: Refinement request with feedback, optional instructions, and priority.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the queued refinement job.
        
        Raises:
            HTTPException: If the task is missing, feedback is empty, or queueing fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.feedback or "").strip():
            raise HTTPException(status_code=400, detail="feedback is required")
        try:
            job = orchestrator.queue_plan_refine_job(
                task_id=task_id,
                base_revision_id=body.base_revision_id,
                feedback=body.feedback,
                instructions=body.instructions,
                priority=body.priority,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job": job.to_dict()}

    @router.get("/tasks/{task_id}/plan/jobs")
    async def list_plan_refine_jobs(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List plan-refinement jobs associated with a task.
        
        Args:
            task_id: Identifier of the task whose refinement jobs are requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing serialized refinement jobs.
        
        Raises:
            HTTPException: If the task is missing or job lookup fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            jobs = orchestrator.list_plan_refine_jobs(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"jobs": [job.to_dict() for job in jobs]}

    @router.get("/tasks/{task_id}/plan/jobs/{job_id}")
    async def get_plan_refine_job(task_id: str, job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Fetch a specific plan-refinement job for a task.
        
        Args:
            task_id: Identifier of the owning task.
            job_id: Identifier of the refinement job.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the requested refinement job.
        
        Raises:
            HTTPException: If the task or job cannot be found.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            job = orchestrator.get_plan_refine_job(task_id, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"job": job.to_dict()}

    @router.post("/tasks/{task_id}/plan/commit")
    async def commit_plan_revision(
        task_id: str,
        body: CommitPlanRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Commit a plan revision as the task's accepted plan.
        
        Args:
            task_id: Identifier of the task whose plan is being committed.
            body: Commit payload containing the revision identifier.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with committed and latest revision identifiers.
        
        Raises:
            HTTPException: If the task is missing or revision commit fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            committed_revision_id = orchestrator.commit_plan_revision(task_id, body.revision_id)
            plan_doc = orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "committed_revision_id": committed_revision_id,
            "latest_revision_id": plan_doc.get("latest_revision_id"),
            "committed_plan_revision_id": plan_doc.get("committed_revision_id"),
        }

    @router.post("/tasks/{task_id}/plan/revisions")
    async def create_plan_revision(
        task_id: str,
        body: CreatePlanRevisionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Create a manual plan revision for a task.
        
        Args:
            task_id: Identifier of the task whose plan revision is created.
            body: Revision payload containing content and optional parent metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created plan revision.
        
        Raises:
            HTTPException: If the task is missing or content is invalid.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.content or "").strip():
            raise HTTPException(status_code=400, detail="content is required")
        try:
            revision = orchestrator.create_plan_revision(
                task_id=task_id,
                content=body.content,
                source="human_edit",
                parent_revision_id=body.parent_revision_id,
                feedback_note=body.feedback_note,
                step=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"revision": revision.to_dict()}

    @router.post("/tasks/{task_id}/generate-tasks")
    async def generate_tasks_from_plan(
        task_id: str, body: GenerateTasksRequest, project_dir: Optional[str] = Query(None)
    ) -> dict[str, Any]:
        """Generate child tasks from selected or override plan text.
        
        Args:
            task_id: Identifier of the parent task receiving generated children.
            body: Generation payload selecting plan source and dependency inference mode.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with generated child identifiers and resolved source metadata.
        
        Raises:
            HTTPException: If the task is missing or generation inputs are inconsistent.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        source = body.source
        if source is None:
            # Backward compatibility: previous API accepted only optional plan_override.
            source = "override" if str(body.plan_override or "").strip() else "latest"
        assert source is not None
        if source == "revision" and not body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is required when source=revision")
        if source == "override" and not str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is required when source=override")
        if source != "revision" and body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is only valid when source=revision")
        if source != "override" and str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is only valid when source=override")
        try:
            plan_text, resolved_revision_id = orchestrator.resolve_plan_text_for_generation(
                task_id=task_id,
                source=source,
                revision_id=body.revision_id,
                plan_override=body.plan_override,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            created_ids = orchestrator.generate_tasks_from_plan(task_id, plan_text, infer_deps=body.infer_deps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Re-fetch parent to get updated children_ids
        updated_task = container.tasks.get(task_id)
        children: list[dict[str, Any]] = []
        for child_id in created_ids:
            child_task = container.tasks.get(child_id)
            if child_task is not None:
                children.append(_task_payload(child_task))
        return {
            "task": _task_payload(updated_task) if updated_task else None,
            "created_task_ids": created_ids,
            "children": children,
            "source": source,
            "source_revision_id": resolved_revision_id,
        }
