"""Miscellaneous route registration for the runtime API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..domain.models import now_iso
from .deps import RouteDeps
from . import router_impl as impl


OrchestratorControlRequest = impl.OrchestratorControlRequest
ReviewActionRequest = impl.ReviewActionRequest
UpdateSettingsRequest = impl.UpdateSettingsRequest
_settings_payload = impl._settings_payload
_task_payload = impl._task_payload
_workers_health_payload = impl._workers_health_payload
_workers_routing_payload = impl._workers_routing_payload


def register_misc_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register metrics, review, orchestrator control, and settings routes."""
    @router.get("/metrics")
    async def get_metrics(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Compute runtime metrics from tasks, runs, and recent events.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A metrics payload for UI dashboard summaries.
        """
        container, _, orchestrator = deps.ctx(project_dir)
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
        container, _, _ = deps.ctx(project_dir)
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


    @router.get("/review-queue")
    async def review_queue(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List tasks currently awaiting human review.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing in-review tasks and total count.
        """
        container, _, _ = deps.ctx(project_dir)
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
        container, bus, orchestrator = deps.ctx(project_dir)
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

        latest_run = None
        for run_id in reversed(task.run_ids):
            latest_run = container.runs.get(run_id)
            if latest_run:
                break
        if latest_run and (latest_run.status != "done" or not latest_run.finished_at):
            latest_run.status = "done"
            latest_run.finished_at = latest_run.finished_at or ts
            if not latest_run.summary:
                latest_run.summary = "Completed after human approval"
            container.runs.upsert(latest_run)

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
        container, bus, _ = deps.ctx(project_dir)
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
        _, _, orchestrator = deps.ctx(project_dir)
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
        _, _, orchestrator = deps.ctx(project_dir)
        return orchestrator.control(body.action)

    @router.get("/settings")
    async def get_settings(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return normalized runtime settings for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The settings payload consumed by the UI.
        """
        container, _, _ = deps.ctx(project_dir)
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
        container, _, _ = deps.ctx(project_dir)
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
        container, _, _ = deps.ctx(project_dir)
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
        container, bus, _ = deps.ctx(project_dir)
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
