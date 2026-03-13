"""Miscellaneous route registration for the runtime API."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ...collaboration.modes import normalize_hitl_mode
from ..orchestrator.env_resolver import resolved_env_vars_view
from ..orchestrator.venv_detector import detect_python_venv
from ..orchestrator.human_guidance import (
    clear_active_human_guidance,
    set_active_human_guidance,
)
from ..domain.models import Task, now_iso
from .deps import RouteDeps
from . import router_impl as impl


OrchestratorControlRequest = impl.OrchestratorControlRequest
ReviewActionRequest = impl.ReviewActionRequest
UpdateSettingsRequest = impl.UpdateSettingsRequest
_settings_payload = impl._settings_payload
_mask_settings_env_vars = impl._mask_settings_env_vars
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

        # Aggregate token usage from persisted step logs.
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0
        has_cost_data = False
        for run in runs:
            for step_entry in (run.steps or []):
                if not isinstance(step_entry, dict):
                    continue
                tu = step_entry.get("token_usage")
                if not isinstance(tu, dict):
                    continue
                inp = tu.get("input_tokens")
                if isinstance(inp, (int, float)):
                    total_input_tokens += int(inp)
                out = tu.get("output_tokens")
                if isinstance(out, (int, float)):
                    total_output_tokens += int(out)
                cost = tu.get("cost_usd")
                if isinstance(cost, (int, float)):
                    total_cost_usd += float(cost)
                    has_cost_data = True

        return {
            "tokens_used": total_input_tokens + total_output_tokens,
            "api_calls": api_calls,
            "estimated_cost_usd": total_cost_usd,
            "cost_available": has_cost_data,
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
        container, _, orchestrator = deps.ctx(project_dir)
        items = [
            _task_payload(task, orchestrator=orchestrator)
            for task in container.tasks.list()
            if task.status == "in_review"
        ]
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
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")

        is_precommit_review = bool(task.metadata.get("pending_precommit_approval"))

        if is_precommit_review:
            if (container.project_dir / ".git").exists():
                preserved_branch = str(task.metadata.get("preserved_branch") or "").strip()
                if not preserved_branch:
                    raise HTTPException(
                        status_code=409,
                        detail="Pre-commit context missing; request changes to regenerate implementation context.",
                    )
                try:
                    has_preserved_branch = (
                        subprocess.run(
                            ["git", "rev-parse", "--verify", f"refs/heads/{preserved_branch}"],
                            cwd=container.project_dir,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=10,
                        ).returncode
                        == 0
                    )
                except Exception:
                    has_preserved_branch = False
                if not has_preserved_branch:
                    raise HTTPException(
                        status_code=409,
                        detail="Pre-commit context missing; request changes to regenerate implementation context.",
                    )
                review_context = task.metadata.get("review_context") if isinstance(task.metadata, dict) else None
                if not isinstance(review_context, dict):
                    raise HTTPException(
                        status_code=409,
                        detail="Pre-commit context missing; request changes to regenerate implementation context.",
                    )
                context_branch = str(review_context.get("preserved_branch") or "").strip()
                if context_branch != preserved_branch:
                    raise HTTPException(
                        status_code=409,
                        detail="Pre-commit context missing; request changes to regenerate implementation context.",
                    )
                base_branch = str(review_context.get("base_branch") or "").strip() or "HEAD"
                base_sha = str(review_context.get("base_sha") or "").strip()
                head_sha = str(review_context.get("head_sha") or "").strip()
                if not base_sha:
                    try:
                        base_result = subprocess.run(
                            ["git", "rev-parse", "--verify", f"{base_branch}^{{commit}}"],
                            cwd=container.project_dir,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=10,
                        )
                        if base_result.returncode == 0:
                            base_sha = base_result.stdout.strip()
                    except Exception:
                        base_sha = ""
                if not head_sha:
                    try:
                        head_result = subprocess.run(
                            ["git", "rev-parse", "--verify", f"refs/heads/{preserved_branch}^{{commit}}"],
                            cwd=container.project_dir,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=10,
                        )
                        if head_result.returncode == 0:
                            head_sha = head_result.stdout.strip()
                    except Exception:
                        head_sha = ""
                diff_range = f"{base_sha}..{head_sha}" if base_sha and head_sha else f"{base_branch}..{preserved_branch}"
                expected_fingerprint = str(review_context.get("diff_fingerprint") or "").strip()
                try:
                    diff_result = subprocess.run(
                        ["git", "diff", "--binary", "--no-color", diff_range],
                        cwd=container.project_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                    actual_fingerprint = hashlib.sha256(
                        (diff_result.stdout or "").encode("utf-8", errors="replace")
                    ).hexdigest() if diff_result.returncode == 0 else ""
                except Exception:
                    actual_fingerprint = ""
                if not expected_fingerprint or not actual_fingerprint or expected_fingerprint != actual_fingerprint:
                    raise HTTPException(
                        status_code=409,
                        detail="Pre-commit context missing; request changes to regenerate implementation context.",
                    )
            ts = now_iso()
            task.status = "queued"
            task.error = None
            task.metadata.pop("pending_precommit_approval", None)
            task.metadata.pop("review_stage", None)
            task.metadata.pop("review_context", None)
            task.metadata["retry_from_step"] = "commit"
            task.metadata["last_review_approval"] = {"ts": ts, "guidance": body.guidance}
            precommit_history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            precommit_history.append({"action": "approve", "ts": ts, "guidance": body.guidance or ""})
            with container.transaction():
                latest_run = None
                for run_id in reversed(task.run_ids):
                    latest_run = container.runs.get(run_id)
                    if latest_run:
                        break
                if latest_run and latest_run.status in {"in_review", "in_progress", "waiting_gate"}:
                    latest_run.status = "interrupted"
                    latest_run.finished_at = latest_run.finished_at or ts
                    latest_run.summary = latest_run.summary or "Approved pre-commit; queued to run commit"
                    container.runs.upsert(latest_run)
                container.tasks.upsert(task)
            bus.emit(channel="review", event_type="task.approved", entity_id=task.id, payload={"stage": "pre_commit", "guidance": body.guidance or ""})
            return {"task": _task_payload(task, container, orchestrator)}

        # If there's a preserved branch, merge it before marking done
        if task.metadata.get("preserved_branch"):
            merge_result = orchestrator.approve_and_merge(task)
            merge_status = str(merge_result.get("status") or "")
            if merge_status in {"merge_conflict", "dirty_overlapping", "git_error"}:
                # Transition to blocked and surface a precise merge failure reason,
                # mirroring what the task executor does for the pre-commit path.
                task = container.tasks.get(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                ts = now_iso()
                task.status = "blocked"
                task.current_step = "commit"
                task.metadata["pipeline_phase"] = "commit"
                reason_code = str(merge_result.get("reason_code") or merge_status or "").strip() or None
                if merge_status == "merge_conflict":
                    task.error = "Merge conflict could not be resolved automatically"
                else:
                    task.error = str(merge_result.get("error") or "").strip() or "Git merge failed before conflict resolution"
                review_history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
                review_history.append({"action": "approve", "ts": ts, "guidance": body.guidance or ""})
                with container.transaction():
                    latest_run = None
                    for run_id in reversed(task.run_ids):
                        latest_run = container.runs.get(run_id)
                        if latest_run:
                            break
                    if latest_run and latest_run.status not in {"done", "blocked"}:
                        latest_run.status = "blocked"
                        latest_run.finished_at = latest_run.finished_at or ts
                        if reason_code == "dirty_overlapping":
                            latest_run.summary = "Blocked due to overlapping local integration changes"
                        elif reason_code == "git_error":
                            latest_run.summary = "Blocked due to git merge error"
                        else:
                            latest_run.summary = "Blocked due to unresolved merge conflict"
                        container.runs.upsert(latest_run)
                    container.tasks.upsert(task)
                orchestrator._emit_task_blocked(task, payload={"error": task.error, "reason_code": reason_code})
                return {"task": _task_payload(task, container, orchestrator)}
            # Re-fetch task after merge (approve_and_merge upserts)
            task = container.tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found after merge")

        task.status = "done"
        task.error = None
        ts = now_iso()
        task.metadata["last_review_approval"] = {"ts": ts, "guidance": body.guidance}
        approve_history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
        approve_history.append({"action": "approve", "ts": ts, "guidance": body.guidance or ""})

        with container.transaction():
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
        return {"task": _task_payload(task, container, orchestrator)}

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
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")
        task.metadata = task.metadata if isinstance(task.metadata, dict) else {}
        task.status = "queued"
        ts = now_iso()
        task.metadata.pop("pending_precommit_approval", None)
        task.metadata.pop("review_stage", None)
        task.metadata.pop("review_context", None)
        guidance = str(body.guidance or "").strip()
        task.metadata["requested_changes"] = {"ts": ts, "guidance": guidance}
        task.metadata["retry_from_step"] = "implement"
        if guidance:
            set_active_human_guidance(
                task,
                source="review_request_changes",
                guidance=guidance,
                created_at=ts,
                target_step="implement",
                fallback_step="implement_fix",
            )
        else:
            clear_active_human_guidance(task, cleared_at=ts)
        history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
        history.append({"action": "request_changes", "ts": ts, "guidance": guidance})
        with container.transaction():
            latest_run = None
            for run_id in reversed(task.run_ids):
                latest_run = container.runs.get(run_id)
                if latest_run:
                    break
            if latest_run and latest_run.status in {"in_review", "in_progress", "waiting_gate"}:
                latest_run.status = "interrupted"
                latest_run.finished_at = latest_run.finished_at or ts
                latest_run.summary = latest_run.summary or "Changes requested before commit"
                container.runs.upsert(latest_run)
            container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.changes_requested", entity_id=task.id, payload={"guidance": guidance})
        return {"task": _task_payload(task, container, orchestrator)}

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

    @router.post("/orchestrator/reconcile")
    async def orchestrator_reconcile(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Run one manual runtime reconciliation pass and return summary + status."""
        _, _, orchestrator = deps.ctx(project_dir)
        summary = orchestrator.reconcile(source="manual")
        payload = orchestrator.status()
        payload["reconcile"] = summary
        return payload

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
        payload = _mask_settings_env_vars(_settings_payload(cfg))
        try:
            payload["resolved_env_vars"] = resolved_env_vars_view(
                project_dir=container.project_dir, cfg=cfg,
            )
        except Exception:
            payload["resolved_env_vars"] = []
        try:
            venv = detect_python_venv(container.project_dir)
            payload["detected_python_venv"] = (
                {"path": str(venv.path), "bin_dir": str(venv.bin_dir), "source": venv.source}
                if venv is not None else None
            )
        except Exception:
            payload["detected_python_venv"] = None
        return payload

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
        container, bus, orchestrator = deps.ctx(project_dir)
        cfg = container.config.load()
        touched_sections: list[str] = []

        if body.orchestrator is not None:
            orchestrator_cfg = dict(cfg.get("orchestrator") or {})
            orchestrator_cfg.update(body.orchestrator.model_dump(exclude_unset=True))
            cfg["orchestrator"] = orchestrator_cfg
            touched_sections.append("orchestrator")

        if body.agent_routing is not None:
            routing_cfg = dict(cfg.get("agent_routing") or {})
            routing_cfg.update(body.agent_routing.model_dump(exclude_unset=True))
            cfg["agent_routing"] = routing_cfg
            touched_sections.append("agent_routing")

        if body.defaults is not None:
            defaults_cfg = dict(cfg.get("defaults") or {})
            incoming_defaults = body.defaults.model_dump(exclude_unset=True)
            incoming_quality_gate = dict(incoming_defaults.get("quality_gate") or {})
            quality_gate_cfg = dict(defaults_cfg.get("quality_gate") or {})
            quality_gate_cfg.update(incoming_quality_gate)
            defaults_cfg["quality_gate"] = quality_gate_cfg
            dep_policy = str(incoming_defaults.get("dependency_policy") or "").strip()
            if dep_policy in ("permissive", "prudent", "strict"):
                defaults_cfg["dependency_policy"] = dep_policy
            if "hitl_mode" in incoming_defaults:
                defaults_cfg["hitl_mode"] = normalize_hitl_mode(incoming_defaults.get("hitl_mode"))
            incoming_task_generation = incoming_defaults.get("task_generation")
            if isinstance(incoming_task_generation, dict):
                task_generation_cfg = orchestrator.resolve_task_generation_policy(
                    Task(hitl_mode=defaults_cfg.get("hitl_mode") or "autopilot"),
                    request_overrides=incoming_task_generation,
                )
                defaults_cfg["task_generation"] = {
                    "child_status": task_generation_cfg.get("child_status"),
                    "child_hitl_mode": task_generation_cfg.get("child_hitl_mode_selection"),
                    "infer_deps": bool(task_generation_cfg.get("infer_deps", True)),
                }
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
            if "environment" in incoming_workers:
                existing_env = dict(workers_cfg.get("environment") or {})
                existing_env.update(dict(incoming_workers.get("environment") or {}))
                workers_cfg["environment"] = existing_env

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

        if body.project is not None and body.project.prompt_overrides is not None:
            project_cfg = dict(cfg.get("project") or {})
            existing_overrides = impl._normalize_prompt_overrides(project_cfg.get("prompt_overrides"))
            for raw_step, raw_prompt in body.project.prompt_overrides.items():
                step = str(raw_step or "").strip().lower()
                if not step:
                    continue
                prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt)
                if prompt.strip():
                    existing_overrides[step] = prompt
                else:
                    existing_overrides.pop(step, None)
            project_cfg["prompt_overrides"] = existing_overrides
            cfg["project"] = project_cfg
            touched_sections.append("project.prompt_overrides")

        if body.project is not None and body.project.prompt_injections is not None:
            project_cfg = dict(cfg.get("project") or {})
            existing_injections = impl._normalize_prompt_injections(project_cfg.get("prompt_injections"))
            for raw_step, raw_prompt in body.project.prompt_injections.items():
                step = str(raw_step or "").strip().lower()
                if not step:
                    continue
                prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt)
                if prompt.strip():
                    existing_injections[step] = prompt
                else:
                    existing_injections.pop(step, None)
            project_cfg["prompt_injections"] = existing_injections
            cfg["project"] = project_cfg
            touched_sections.append("project.prompt_injections")

        container.config.save(cfg)
        bus.emit(
            channel="system",
            event_type="settings.updated",
            entity_id=container.project_id,
            payload={"sections": touched_sections},
        )
        payload = _mask_settings_env_vars(_settings_payload(cfg))
        try:
            payload["resolved_env_vars"] = resolved_env_vars_view(
                project_dir=container.project_dir, cfg=cfg,
            )
        except Exception:
            payload["resolved_env_vars"] = []
        try:
            venv = detect_python_venv(container.project_dir)
            payload["detected_python_venv"] = (
                {"path": str(venv.path), "bin_dir": str(venv.bin_dir), "source": venv.source}
                if venv is not None else None
            )
        except Exception:
            payload["detected_python_venv"] = None
        return payload
