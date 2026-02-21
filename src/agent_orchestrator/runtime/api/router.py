"""FastAPI routes for orchestrator runtime control and task management."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional, cast

from fastapi import APIRouter, HTTPException, Query

from ...collaboration.modes import MODE_CONFIGS
from ...pipelines.registry import PipelineRegistry
from .collaboration_routes import register_collaboration_feedback_comment_routes
from .helpers import (
    _normalize_human_blocking_issues,
    _priority_rank,
)
from .logs_utils import (
    _logs_snapshot_id,
    _read_from_offset,
    _read_tail,
    _safe_state_path,
)
from .prd_ingestion import (
    _apply_generated_dep_links,
    _generated_tasks_from_parsed_prd,
    _ingest_prd,
)
from .pipeline_classification import (
    _canonical_task_type_for_pipeline,
    _normalize_pipeline_classification_output,
)
from .project_routes import register_project_routes
from .router_state import (
    _fetch_import_job,
    _load_comment_records,
    _load_feedback_records,
    _prune_in_memory_jobs,
    _upsert_import_job,
)
from .settings_workers import (
    _settings_payload,
    _workers_health_payload,
    _workers_routing_payload,
)
from .task_helpers import _execution_batches, _has_unresolved_blockers, _task_payload
from .schemas import (
    AddDependencyRequest,
    AgentRoutingSettingsRequest,
    ApproveGateRequest,
    CommitPlanRequest,
    CreatePlanRevisionRequest,
    CreateTaskRequest,
    DefaultsSettingsRequest,
    GenerateTasksRequest,
    LanguageCommandsRequest,
    OrchestratorControlRequest,
    OrchestratorSettingsRequest,
    PipelineClassificationRequest,
    PipelineClassificationResponse,
    PlanRefineRequest,
    PrdCommitRequest,
    PrdPreviewRequest,
    ProjectSettingsRequest,
    RetryTaskRequest,
    ReviewActionRequest,
    SpawnAgentRequest,
    StartTerminalSessionRequest,
    StopTerminalSessionRequest,
    TerminalInputRequest,
    TerminalResizeRequest,
    TransitionRequest,
    UpdateSettingsRequest,
    UpdateTaskRequest,
    WorkerProviderSettingsRequest,
    WorkersSettingsRequest,
)
from ..domain.models import (
    AgentRecord,
    ApprovalMode,
    DependencyPolicy,
    Priority,
    Task,
    TaskStatus,
    now_iso,
)
from ..events.bus import EventBus
from ..orchestrator.service import OrchestratorService
from ..storage.container import Container
from ..terminal.service import TerminalService


VALID_TRANSITIONS: dict[str, set[str]] = {
    "backlog": {"queued", "cancelled"},
    "queued": {"backlog", "cancelled"},
    "in_progress": {"cancelled"},
    "in_review": {"done", "blocked", "cancelled"},
    "blocked": {"queued", "in_review", "cancelled"},
    "done": set(),
    "cancelled": {"backlog"},
}

def create_router(
    resolve_container: Any,
    resolve_orchestrator: Any,
    job_store: dict[str, dict[str, Any]],
) -> APIRouter:
    """Create the runtime API router.

    Args:
        resolve_container (Any): Callable that resolves and returns the
            project-scoped ``Container`` for an optional ``project_dir`` value.
        resolve_orchestrator (Any): Callable that resolves and returns the
            project-scoped ``OrchestratorService`` for an optional
            ``project_dir`` value.
        job_store (dict[str, dict[str, Any]]): Shared in-memory map used to keep
            asynchronous import and plan-refinement job state visible across API
            requests.

    Returns:
        APIRouter: Router exposing runtime endpoints for task lifecycle,
        orchestration operations, collaboration metadata, terminal sessions, and
        import workflows.
    """
    router = APIRouter(prefix="/api", tags=["api"])
    terminal_services: dict[str, TerminalService] = {}

    def _ctx(project_dir: Optional[str]) -> tuple[Container, EventBus, OrchestratorService]:
        container: Container = resolve_container(project_dir)
        bus = EventBus(container.events, container.project_id)
        orchestrator: OrchestratorService = resolve_orchestrator(project_dir)
        return container, bus, orchestrator

    def _terminal_ctx(project_dir: Optional[str]) -> tuple[Container, EventBus, TerminalService]:
        container: Container = resolve_container(project_dir)
        bus = EventBus(container.events, container.project_id)
        key = str(container.project_dir)
        service = terminal_services.get(key)
        if service is None:
            service = TerminalService(container, bus)
            terminal_services[key] = service
        return container, bus, service

    register_project_routes(router, _ctx)
    register_collaboration_feedback_comment_routes(router, _ctx)

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
        container, bus, _ = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, _ = _ctx(project_dir)
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
        container, _, _ = _ctx(project_dir)
        columns: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ["backlog", "queued", "in_progress", "in_review", "blocked", "done", "cancelled"]
        }
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(_task_payload(task))
        for key, items in columns.items():
            items.sort(key=lambda x: (_priority_rank(str(x.get("priority") or "P3")), str(x.get("created_at") or "")))
        return {"columns": columns}

    @router.get("/tasks/execution-order")
    async def execution_order(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Compute execution batches for non-terminal tasks.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing ready-to-run batches and recently completed tasks.
        """
        container, _, _ = _ctx(project_dir)
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
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": _task_payload(task, container)}

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
        container, _, _ = _ctx(project_dir)
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
        container, _, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        _, _, orchestrator = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, bus, _ = _ctx(project_dir)
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
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = resolve_orchestrator(project_dir)
        return cast(dict[str, Any], orchestrator.get_workdoc(task_id))

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
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = resolve_orchestrator(project_dir)
        try:
            return cast(dict[str, Any], orchestrator.get_plan_document(task_id))
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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
        container, _, orchestrator = _ctx(project_dir)
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

    @router.post("/import/prd/preview")
    async def preview_import(body: PrdPreviewRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Ingest PRD content and return a preview graph of candidate tasks.
        
        Args:
            body: PRD preview request with raw content and default priority.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the temporary import job id and preview graph.
        """
        container, _, _ = _ctx(project_dir)
        _prune_in_memory_jobs(job_store)
        ingestion = _ingest_prd(body.content, body.default_priority)
        parsed_prd = ingestion["parsed_prd"]
        original_prd = ingestion["original_prd"]
        items = list(parsed_prd.get("task_candidates") or [])
        nodes = [
            {
                "id": f"n{idx + 1}",
                "title": str(item.get("title") or "Imported task"),
                "priority": str(item.get("priority") or body.default_priority),
            }
            for idx, item in enumerate(items)
        ]
        edges = [{"from": nodes[idx]["id"], "to": nodes[idx + 1]["id"]} for idx in range(len(nodes) - 1)]
        job_id = f"imp-{uuid.uuid4().hex[:10]}"
        job = {
            "id": job_id,
            "project_id": container.project_id,
            "title": body.title or "Imported PRD",
            "status": "preview_ready",
            "created_at": now_iso(),
            "tasks": items,
            "original_prd": original_prd,
            "parsed_prd": parsed_prd,
        }
        job_store[job_id] = job
        _upsert_import_job(container, job)
        return {
            "job_id": job_id,
            "preview": {
                "nodes": nodes,
                "edges": edges,
                "chunk_count": int(parsed_prd.get("chunk_count") or 0),
                "strategy": str(parsed_prd.get("strategy") or "unknown"),
                "ambiguity_warnings": list(parsed_prd.get("ambiguity_warnings") or []),
            },
        }

    @router.post("/import/prd/commit")
    async def commit_import(body: PrdCommitRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Commit a previewed PRD import and run initiative generation.
        
        Args:
            body: PRD commit request identifying the preview job to finalize.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with created task identifiers and the parent initiative task id.
        
        Raises:
            HTTPException: If the import job is missing, invalid, or execution fails.
        """
        container, bus, orchestrator = _ctx(project_dir)
        _prune_in_memory_jobs(job_store)
        job = job_store.get(body.job_id)
        if not job:
            job = _fetch_import_job(container, body.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        if not isinstance(job.get("parsed_prd"), dict):
            raise HTTPException(status_code=400, detail="Import job is missing parsed PRD artifacts.")
        parsed_prd = dict(job.get("parsed_prd") or {})
        generated_tasks = _generated_tasks_from_parsed_prd(parsed_prd)
        if not generated_tasks:
            raise HTTPException(status_code=400, detail="Parsed PRD produced no task candidates to generate.")

        # Execute the existing initiative pipeline end-to-end using the ingested PRD artifacts.
        prd_parent = Task(
            title=str(job.get("title") or "Imported PRD"),
            description=str((job.get("original_prd") or {}).get("normalized_content") or ""),
            task_type="initiative_plan",
            priority="P2",
            source="prd_import",
            metadata={
                "prd_import_job_id": body.job_id,
                "original_prd": job.get("original_prd"),
                "parsed_prd": parsed_prd,
            },
        )
        prd_parent.status = "queued"
        container.tasks.upsert(prd_parent)
        bus.emit(
            channel="tasks",
            event_type="task.created",
            entity_id=prd_parent.id,
            payload={"source": "prd_import", "import_job_id": body.job_id, "task_type": "initiative_plan"},
        )

        try:
            orchestrator.run_task(prd_parent.id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        refreshed_parent = container.tasks.get(prd_parent.id)
        created = list(refreshed_parent.children_ids if refreshed_parent else [])
        _apply_generated_dep_links(container, created)
        for child_id in created:
            child = container.tasks.get(child_id)
            if not child:
                continue
            child.source = "prd_import"
            container.tasks.upsert(child)

        job["status"] = "committed"
        job["created_task_ids"] = created
        job["parent_task_id"] = prd_parent.id
        job_store[body.job_id] = job
        _upsert_import_job(container, job)
        bus.emit(channel="tasks", event_type="import.committed", entity_id=body.job_id, payload={"created_task_ids": created})
        return {"job_id": body.job_id, "created_task_ids": created, "parent_task_id": prd_parent.id}

    @router.get("/import/{job_id}")
    async def get_import_job(job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the current state of a PRD import job.
        
        Args:
            job_id: Identifier of the import job.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the import job record.
        
        Raises:
            HTTPException: If the job cannot be found.
        """
        _prune_in_memory_jobs(job_store)
        job = job_store.get(job_id)
        if not job:
            container: Container = resolve_container(project_dir)
            job = _fetch_import_job(container, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        return {"job": job}

    @router.get("/metrics")
    async def get_metrics(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Compute runtime metrics from tasks, runs, and recent events.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A metrics payload for UI dashboard summaries.
        """
        container, _, orchestrator = _ctx(project_dir)
        status = orchestrator.status()
        tasks = container.tasks.list()
        runs = container.runs.list()
        events = container.events.list_recent(limit=2000)
        phases_completed = sum(len(list(run.steps or [])) for run in runs)
        phases_total = sum(max(1, len(list(task.pipeline_template or []))) for task in tasks)
        wall_time_seconds = 0.0
        for run in runs:
            if not run.started_at:
                continue
            try:
                start = datetime.fromisoformat(str(run.started_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if run.finished_at:
                try:
                    end = datetime.fromisoformat(str(run.finished_at).replace("Z", "+00:00"))
                except ValueError:
                    continue
            else:
                end = datetime.now(timezone.utc)
            wall_time_seconds += max((end - start).total_seconds(), 0.0)
        api_calls = len(events)
        return {
            "tokens_used": 0,
            "api_calls": api_calls,
            "estimated_cost_usd": 0.0,
            "wall_time_seconds": int(wall_time_seconds),
            "phases_completed": phases_completed,
            "phases_total": phases_total,
            "files_changed": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "queue_depth": int(status.get("queue_depth", 0)),
            "in_progress": int(status.get("in_progress", 0)),
        }

    @router.get("/phases")
    async def get_phases(project_dir: Optional[str] = Query(None)) -> list[dict[str, Any]]:
        """Return phase progress entries for each task.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A list of task phase records with status and progress estimates.
        """
        container, _, _ = _ctx(project_dir)
        phases: list[dict[str, Any]] = []
        for task in container.tasks.list():
            total_steps = max(1, len(list(task.pipeline_template or [])))
            completed_steps = 0
            if task.status == "done":
                completed_steps = total_steps
            elif task.status == "in_review":
                completed_steps = max(total_steps - 1, 1)
            elif task.status == "in_progress":
                completed_steps = 2
            elif task.status in {"queued", "blocked"}:
                completed_steps = 1
            progress = {
                "backlog": 0.0,
                "queued": 0.1,
                "blocked": 0.1,
                "in_progress": 0.6,
                "in_review": 0.9,
                "done": 1.0,
                "cancelled": 1.0,
            }.get(task.status, min(completed_steps / total_steps, 1.0))
            phases.append(
                {
                    "id": task.id,
                    "name": task.title,
                    "description": task.description,
                    "status": task.status,
                    "deps": list(task.blocked_by),
                    "progress": progress,
                }
            )
        return phases

    @router.get("/agents/types")
    async def get_agent_types(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List supported agent roles and routing affinities.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload describing available agent role definitions.
        """
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        task_role_map = dict(routing.get("task_type_roles") or {})
        role_affinity: dict[str, list[str]] = {}
        for task_type, role in task_role_map.items():
            role_name = str(role or "")
            if not role_name:
                continue
            role_affinity.setdefault(role_name, []).append(str(task_type))
        roles = ["general", "implementer", "reviewer", "researcher", "tester", "planner", "debugger"]
        return {
            "types": [
                {
                    "role": role,
                    "display_name": role.replace("_", " ").title(),
                    "description": f"{role.replace('_', ' ').title()} agent",
                    "task_type_affinity": sorted(role_affinity.get(role, [])),
                    "allowed_steps": ["plan", "implement", "verify", "review"],
                    "limits": {"max_tokens": 0, "max_time_seconds": 0, "max_cost_usd": 0.0},
                }
                for role in roles
            ]
        }

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
        container, _, _ = _ctx(project_dir)
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
        for item in _load_feedback_records(container):
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
        for item in _load_comment_records(container):
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

    @router.post("/terminal/session")
    async def start_terminal_session(
        body: StartTerminalSessionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Start a managed terminal session for the current project.
        
        Args:
            body: Terminal session configuration payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created session and project identifier.
        """
        container, _, terminal = _terminal_ctx(project_dir)
        session = terminal.start_session(shell=body.shell, cols=body.cols or 120, rows=body.rows or 36)
        return {"session": session.to_dict(), "project_id": container.project_id}

    @router.get("/terminal/session")
    async def get_terminal_session(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the currently active terminal session for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the active session, if any.
        """
        container, _, terminal = _terminal_ctx(project_dir)
        session = terminal.get_active_session(container.project_id)
        return {"session": session.to_dict() if session else None}

    @router.post("/terminal/session/{session_id}/input")
    async def write_terminal_input(
        session_id: str,
        body: TerminalInputRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Write input bytes to a managed terminal session.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Input payload containing data to write.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = _terminal_ctx(project_dir)
        try:
            session = terminal.write_input(session_id=session_id, data=body.data)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.post("/terminal/session/{session_id}/resize")
    async def resize_terminal_session(
        session_id: str,
        body: TerminalResizeRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Resize a managed terminal session pseudo-terminal.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Resize payload with column and row counts.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = _terminal_ctx(project_dir)
        try:
            session = terminal.resize(session_id=session_id, cols=body.cols, rows=body.rows)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.post("/terminal/session/{session_id}/stop")
    async def stop_terminal_session(
        session_id: str,
        body: StopTerminalSessionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Stop a managed terminal session with an optional signal.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Stop payload containing the signal name.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the stopped session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = _terminal_ctx(project_dir)
        try:
            session = terminal.stop_session(session_id=session_id, signal_name=body.signal)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.get("/terminal/session/{session_id}/logs")
    async def get_terminal_session_logs(
        session_id: str,
        project_dir: Optional[str] = Query(None),
        offset: int = Query(0),
        max_bytes: int = Query(65536),
    ) -> dict[str, Any]:
        """Read terminal session output from a byte offset.
        
        Args:
            session_id: Identifier of the terminal session.
            project_dir: Optional project directory used to resolve runtime state.
            offset: Starting byte offset in the output stream.
            max_bytes: Maximum number of bytes to read.
        
        Returns:
            A payload containing output data, new offset, and session status.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = _terminal_ctx(project_dir)
        try:
            output, new_offset = terminal.read_output(session_id=session_id, offset=offset, max_bytes=max_bytes)
            session = terminal.get_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "output": output,
            "offset": new_offset,
            "status": session.status if session else "unknown",
            "finished_at": session.finished_at if session else None,
        }

    @router.get("/review-queue")
    async def review_queue(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List tasks currently awaiting human review.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing in-review tasks and total count.
        """
        container, _, _ = _ctx(project_dir)
        items = [_task_payload(task) for task in container.tasks.list() if task.status == "in_review"]
        return {"tasks": items, "total": len(items)}

    @router.post("/review/{task_id}/approve")
    async def approve_review(task_id: str, body: ReviewActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Approve a reviewed task and optionally merge preserved branch work.
        
        Args:
            task_id: Identifier of the task being approved.
            body: Review action payload containing optional guidance.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the approved task.
        
        Raises:
            HTTPException: If task lookup fails, status is invalid, or merge conflicts occur.
        """
        container, bus, orchestrator = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status not in ("in_review", "blocked"):
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review or blocked")

        # If there's a preserved branch, merge it before marking done
        if task.metadata.get("preserved_branch"):
            merge_result = orchestrator.approve_and_merge(task)
            if merge_result.get("status") == "merge_conflict":
                raise HTTPException(status_code=409, detail="Merge conflict could not be resolved")
            # Re-fetch task after merge (approve_and_merge upserts)
            task = container.tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found after merge")

        task.status = "done"
        task.error = None
        ts = now_iso()
        task.metadata["last_review_approval"] = {"ts": ts, "guidance": body.guidance}
        history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
        history.append({"action": "approve", "ts": ts, "guidance": body.guidance or ""})
        container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.approved", entity_id=task.id, payload={"guidance": body.guidance or ""})
        return {"task": _task_payload(task, container)}

    @router.post("/review/{task_id}/request-changes")
    async def request_review_changes(task_id: str, body: ReviewActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return a task from review to implementation with reviewer guidance.
        
        Args:
            task_id: Identifier of the task requiring changes.
            body: Review action payload containing change guidance.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task is missing or not in review.
        """
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")
        task.status = "queued"
        ts = now_iso()
        task.metadata["requested_changes"] = {"ts": ts, "guidance": body.guidance}
        task.metadata["retry_from_step"] = "implement"
        history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
        history.append({"action": "request_changes", "ts": ts, "guidance": body.guidance or ""})
        container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.changes_requested", entity_id=task.id, payload={"guidance": body.guidance})
        return {"task": _task_payload(task, container)}

    @router.get("/orchestrator/status")
    async def orchestrator_status(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return current orchestrator queue and worker status metrics.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The orchestrator status payload.
        """
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.status()

    @router.post("/orchestrator/control")
    async def orchestrator_control(body: OrchestratorControlRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Apply a control action to the orchestrator runtime.
        
        Args:
            body: Control request specifying the desired orchestrator action.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The orchestrator control action result payload.
        """
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.control(body.action)

    @router.get("/settings")
    async def get_settings(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return normalized runtime settings for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The settings payload consumed by the UI.
        """
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _settings_payload(cfg)

    @router.get("/workers/health")
    async def workers_health(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return worker health summary derived from persisted configuration.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A worker health payload.
        """
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _workers_health_payload(cfg)

    @router.get("/workers/routing")
    async def workers_routing(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return worker routing configuration details.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A worker routing payload.
        """
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _workers_routing_payload(cfg)

    @router.patch("/settings")
    async def patch_settings(body: UpdateSettingsRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Apply partial settings updates across orchestrator configuration sections.
        
        Args:
            body: Settings patch payload with optional section updates.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The normalized settings payload after persistence.
        """
        container, bus, _ = _ctx(project_dir)
        cfg = container.config.load()
        touched_sections: list[str] = []

        if body.orchestrator is not None:
            orchestrator_cfg = dict(cfg.get("orchestrator") or {})
            orchestrator_cfg.update(body.orchestrator.model_dump())
            cfg["orchestrator"] = orchestrator_cfg
            touched_sections.append("orchestrator")

        if body.agent_routing is not None:
            routing_cfg = dict(cfg.get("agent_routing") or {})
            routing_cfg.update(body.agent_routing.model_dump())
            cfg["agent_routing"] = routing_cfg
            touched_sections.append("agent_routing")

        if body.defaults is not None:
            defaults_cfg = dict(cfg.get("defaults") or {})
            incoming_defaults = body.defaults.model_dump()
            incoming_quality_gate = dict(incoming_defaults.get("quality_gate") or {})
            quality_gate_cfg = dict(defaults_cfg.get("quality_gate") or {})
            quality_gate_cfg.update(incoming_quality_gate)
            defaults_cfg["quality_gate"] = quality_gate_cfg
            dep_policy = str(incoming_defaults.get("dependency_policy") or "").strip()
            if dep_policy in ("permissive", "prudent", "strict"):
                defaults_cfg["dependency_policy"] = dep_policy
            cfg["defaults"] = defaults_cfg
            touched_sections.append("defaults")

        if body.workers is not None:
            workers_cfg = dict(cfg.get("workers") or {})
            incoming_workers = body.workers.model_dump(exclude_none=True, exclude_unset=True)

            if "default" in incoming_workers:
                workers_cfg["default"] = str(incoming_workers.get("default") or "codex")
            if "default_model" in incoming_workers:
                default_model = str(incoming_workers.get("default_model") or "").strip()
                if default_model:
                    workers_cfg["default_model"] = default_model
                else:
                    workers_cfg.pop("default_model", None)
            if "heartbeat_seconds" in incoming_workers:
                workers_cfg["heartbeat_seconds"] = incoming_workers.get("heartbeat_seconds")
            if "heartbeat_grace_seconds" in incoming_workers:
                workers_cfg["heartbeat_grace_seconds"] = incoming_workers.get("heartbeat_grace_seconds")
            if "routing" in incoming_workers:
                workers_cfg["routing"] = dict(incoming_workers.get("routing") or {})
            if "providers" in incoming_workers:
                workers_cfg["providers"] = dict(incoming_workers.get("providers") or {})

            normalized_workers = _settings_payload({"workers": workers_cfg})["workers"]
            cfg["workers"] = normalized_workers
            touched_sections.append("workers")

        if body.project is not None and body.project.commands is not None:
            project_cfg = dict(cfg.get("project") or {})
            existing_commands = dict(project_cfg.get("commands") or {})
            for raw_lang, lang_req in body.project.commands.items():
                lang = raw_lang.strip().lower()
                if not lang:
                    continue
                lang_entry = dict(existing_commands.get(lang) or {})
                for field in ("test", "lint", "typecheck", "format"):
                    value = getattr(lang_req, field)
                    if value is None:
                        continue
                    if value == "":
                        lang_entry.pop(field, None)
                    else:
                        lang_entry[field] = value
                if lang_entry:
                    existing_commands[lang] = lang_entry
                else:
                    existing_commands.pop(lang, None)
            project_cfg["commands"] = existing_commands
            cfg["project"] = project_cfg
            touched_sections.append("project.commands")

        container.config.save(cfg)
        bus.emit(
            channel="system",
            event_type="settings.updated",
            entity_id=container.project_id,
            payload={"sections": touched_sections},
        )
        return _settings_payload(cfg)

    @router.get("/agents")
    async def list_agents(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List all persisted agent records for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing serialized agent records.
        """
        container, _, _ = _ctx(project_dir)
        return {"agents": [agent.to_dict() for agent in container.agents.list()]}

    @router.post("/agents/spawn")
    async def spawn_agent(body: SpawnAgentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create a new agent record and publish an agent-created event.
        
        Args:
            body: Agent spawn payload with role, capacity, and provider override.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created agent record.
        """
        container, bus, _ = _ctx(project_dir)
        agent = AgentRecord(role=body.role, capacity=body.capacity, override_provider=body.override_provider)
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.spawned", entity_id=agent.id, payload=agent.to_dict())
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Pause an existing agent record.
        
        Args:
            agent_id: Identifier of the agent to pause.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "paused"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.paused", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Resume a previously paused agent record.
        
        Args:
            agent_id: Identifier of the agent to resume.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "running"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.resumed", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/terminate")
    async def terminate_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Mark an agent record as terminated.
        
        Args:
            agent_id: Identifier of the agent to terminate.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "terminated"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.terminated", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.delete("/agents/{agent_id}")
    async def remove_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete an agent record by identifier.
        
        Args:
            agent_id: Identifier of the agent to remove.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating successful removal.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = _ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}

    @router.post("/agents/{agent_id}/remove")
    async def remove_agent_post(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete an agent record through the POST compatibility endpoint.
        
        Args:
            agent_id: Identifier of the agent to remove.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating successful removal.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = _ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}

    return router
