"""Core orchestration service for task lifecycle execution."""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal, Optional, cast

from ...collaboration.modes import should_gate
from ...pipelines.registry import PipelineRegistry
from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ..domain.models import (
    PlanRefineJob,
    PlanRevision,
    PlanRevisionStatus,
    Priority,
    ReviewCycle,
    ReviewFinding,
    RunRecord,
    Task,
    now_iso,
)
from ..events.bus import EventBus
from ..storage.container import Container
from ...worker import WorkerCancelledError
from .dependency_manager import DependencyManager
from .live_worker_adapter import _VERIFY_STEPS
from .plan_manager import PlanManager
from .task_executor import TaskExecutor
from .workdoc_manager import WorkdocManager
from .worktree_manager import WorktreeManager
from .worker_adapter import DefaultWorkerAdapter, StepResult, WorkerAdapter

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Coordinate task scheduling, execution, review, and commit flow."""

    # Compatibility shim for tests and callers that referenced the old class constant.
    _GENERIC_WORKDOC_TEMPLATE = WorkdocManager._GENERIC_WORKDOC_TEMPLATE

    _GATE_MAPPING: dict[str, str] = {
        "plan": "before_plan",
        "implement": "before_implement",
        "review": "after_implement",
        "commit": "before_commit",
    }
    _HUMAN_INTERVENTION_GATE = "human_intervention"

    def __init__(
        self,
        container: Container,
        bus: EventBus,
        *,
        worker_adapter: WorkerAdapter | None = None,
    ) -> None:
        """Initialize the OrchestratorService.

        Args:
            container (Container): Container for this call.
            bus (EventBus): Bus for this call.
            worker_adapter (WorkerAdapter | None): Worker adapter for this call.
        """
        self.container = container
        self.bus = bus
        self.worker_adapter = worker_adapter or DefaultWorkerAdapter()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._drain = False
        self._run_branch: Optional[str] = None
        self._pool: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future[Any]] = {}
        self._futures_lock = threading.Lock()
        self._merge_lock = threading.Lock()
        self._branch_lock = threading.Lock()
        self._dependency_manager = DependencyManager(
            container,
            bus,
            worker_adapter_getter=lambda: self.worker_adapter,
        )
        self._workdoc_manager = WorkdocManager(
            container,
            bus,
            pipeline_id_resolver=self._pipeline_id_for_task,
        )
        self._plan_manager = PlanManager(self)
        self._worktree_manager = WorktreeManager(self)
        self._task_executor = TaskExecutor(self)

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            cfg = self.container.config.load()
            max_workers = int(dict(cfg.get("orchestrator") or {}).get("concurrency", 2) or 2)
            self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="orchestrator-task")
        return self._pool

    def status(self) -> dict[str, Any]:
        """Build a status snapshot of queue depth and active worker usage.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        tasks = self.container.tasks.list()
        queue_depth = len([task for task in tasks if task.status == "queued"])
        in_progress = len([task for task in tasks if task.status == "in_progress"])
        with self._futures_lock:
            active_workers = len(self._futures)
        return {
            "status": orchestrator_cfg.get("status", "running"),
            "queue_depth": queue_depth,
            "in_progress": in_progress,
            "active_workers": active_workers,
            "draining": self._drain,
            "run_branch": self._run_branch,
        }

    def control(self, action: str) -> dict[str, Any]:
        """Apply a control action and return updated orchestrator status.

        Args:
            action (str): Action for this call.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if action == "pause":
            orchestrator_cfg["status"] = "paused"
        elif action == "resume":
            orchestrator_cfg["status"] = "running"
        elif action == "drain":
            self._drain = True
            orchestrator_cfg["status"] = "running"
        elif action == "stop":
            self._stop.set()
            orchestrator_cfg["status"] = "stopped"
        else:
            raise ValueError(f"Unsupported control action: {action}")
        cfg["orchestrator"] = orchestrator_cfg
        self.container.config.save(cfg)
        self.bus.emit(channel="system", event_type="orchestrator.control", entity_id=self.container.project_id, payload={"action": action})
        return self.status()

    def ensure_worker(self) -> None:
        """Start the background scheduling loop when not already running."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._recover_in_progress_tasks()
            self._cleanup_orphaned_worktrees()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="orchestrator")
            self._thread.start()

    def shutdown(self, *, timeout: float = 10.0) -> None:
        """Stop the scheduler and wait for in-flight work up to ``timeout``.

        Args:
            timeout (float): Timeout for this call.
        """
        with self._lock:
            self._stop.set()
            thread = self._thread

        if thread and thread.is_alive():
            thread.join(timeout=max(timeout, 0.0))

        with self._futures_lock:
            inflight = list(self._futures.values())
        if inflight and timeout > 0:
            wait(inflight, timeout=timeout)

        pool = self._pool
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)
            self._pool = None

        with self._futures_lock:
            self._futures.clear()
        self._thread = None

    def _recover_in_progress_tasks(self) -> None:
        tasks = self.container.tasks.list()
        in_progress_ids = {task.id for task in tasks if task.status == "in_progress"}
        if not in_progress_ids:
            return

        for run in self.container.runs.list():
            if run.task_id in in_progress_ids and run.status == "in_progress" and not run.finished_at:
                run.status = "interrupted"
                run.finished_at = now_iso()
                run.summary = run.summary or "Interrupted by orchestrator restart"
                self.container.runs.upsert(run)

        for task in tasks:
            if task.id not in in_progress_ids:
                continue
            task.status = "queued"
            task.current_step = None
            task.current_agent_id = None
            task.pending_gate = None
            task.error = "Recovered from interrupted run"
            if isinstance(task.metadata, dict):
                task.metadata.pop("pipeline_phase", None)
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.recovered",
                entity_id=task.id,
                payload={"reason": "orchestrator_restart"},
            )

    def _sweep_futures(self) -> None:
        """Remove completed futures and log any unexpected errors."""
        with self._futures_lock:
            done_ids = [tid for tid, f in self._futures.items() if f.done()]
            for tid in done_ids:
                fut = self._futures.pop(tid)
                exc = fut.exception()
                if exc:
                    logger.error("Task %s raised unexpected error: %s", tid, exc, exc_info=exc)

    def tick_once(self) -> bool:
        """Run one scheduler iteration and return whether work was dispatched.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        self._sweep_futures()

        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if orchestrator_cfg.get("status", "running") != "running":
            return False

        self._maybe_analyze_dependencies()

        max_in_progress = int(orchestrator_cfg.get("concurrency", 2) or 2)
        claimed = self.container.tasks.claim_next_runnable(max_in_progress=max_in_progress)
        if not claimed:
            return False

        self.bus.emit(channel="queue", event_type="task.claimed", entity_id=claimed.id, payload={"status": claimed.status})
        future = self._get_pool().submit(self._execute_task, claimed)
        with self._futures_lock:
            self._futures[claimed.id] = future
        return True

    def run_task(self, task_id: str) -> Task:
        """Synchronously execute one task by id and return the final record.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            Task: Result produced by this call.
        """
        wait_existing = False
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            if task.pending_gate and task.status != "in_progress":
                raise ValueError(f"Task {task_id} is waiting for gate approval: {task.pending_gate}")
            # Make explicit run idempotent when a worker already started or finished
            # the same task; this avoids request races with the background loop.
            if task.status in {"in_review", "done"}:
                return task
            if task.status == "in_progress":
                wait_existing = True
            if task.status in {"cancelled"}:
                raise ValueError(f"Task {task_id} cannot be run from status={task.status}")

            if not wait_existing:
                terminal = {"done", "cancelled"}
                for dep_id in task.blocked_by:
                    dep = self.container.tasks.get(dep_id)
                    if dep is None or dep.status not in terminal:
                        raise ValueError(f"Task {task_id} has unresolved blocker {dep_id}")
                task.status = "queued"
                self.container.tasks.upsert(task)

        if wait_existing:
            with self._futures_lock:
                existing_future = self._futures.get(task_id)
            if existing_future:
                existing_future.result()
            updated = self.container.tasks.get(task_id)
            if not updated:
                raise ValueError(f"Task disappeared during execution: {task_id}")
            return updated

        future = self._get_pool().submit(self._execute_task, task)
        with self._futures_lock:
            self._futures[task_id] = future
        try:
            future.result()
        finally:
            with self._futures_lock:
                self._futures.pop(task_id, None)
        updated = self.container.tasks.get(task_id)
        if not updated:
            raise ValueError(f"Task disappeared during execution: {task_id}")
        return updated

    def _resolve_worker_lineage(self, task: Task, step: str) -> tuple[str | None, str | None]:
        try:
            cfg = self.container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, step)
        except Exception:
            return None, None
        return spec.name, spec.model

    def _active_plan_refine_job(self, task_id: str) -> PlanRefineJob | None:
        return self._plan_manager.active_plan_refine_job(task_id)

    def get_plan_document(self, task_id: str) -> dict[str, Any]:
        """Build plan-revision state plus active refine-job metadata for a task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        return self._plan_manager.get_plan_document(task_id)

    def create_plan_revision(
        self,
        *,
        task_id: str,
        content: str,
        source: Literal["worker_plan", "worker_refine", "human_edit", "import"],
        parent_revision_id: str | None = None,
        step: str | None = None,
        feedback_note: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        status: Literal["draft", "committed"] = "draft",
        created_at: str | None = None,
    ) -> PlanRevision:
        """Create and persist a plan revision for a task.

        Args:
            task_id (str): Identifier for the target task.
            content (str): Content for this call.
            source (Literal['worker_plan', 'worker_refine', 'human_edit', 'import']): Source for this call.
            parent_revision_id (str | None): Identifier for the related parent revision.
            step (str | None): Step for this call.
            feedback_note (str | None): Feedback note for this call.
            provider (str | None): Provider for this call.
            model (str | None): Model for this call.
            status (Literal['draft', 'committed']): Status for this call.
            created_at (str | None): Created at for this call.

        Returns:
            PlanRevision: Result produced by this call.
        """
        return self._plan_manager.create_plan_revision(
            task_id=task_id,
            content=content,
            source=source,
            parent_revision_id=parent_revision_id,
            step=step,
            feedback_note=feedback_note,
            provider=provider,
            model=model,
            status=status,
            created_at=created_at,
        )

    def queue_plan_refine_job(
        self,
        *,
        task_id: str,
        feedback: str,
        instructions: str | None = None,
        base_revision_id: str | None = None,
        priority: str = "normal",
    ) -> PlanRefineJob:
        """Queue a plan refinement job and schedule background processing.

        Args:
            task_id (str): Identifier for the target task.
            feedback (str): Feedback for this call.
            instructions (str | None): Instructions for this call.
            base_revision_id (str | None): Identifier for the related base revision.
            priority (str): Priority for this call.

        Returns:
            PlanRefineJob: Result produced by this call.
        """
        return self._plan_manager.queue_plan_refine_job(
            task_id=task_id,
            feedback=feedback,
            instructions=instructions,
            base_revision_id=base_revision_id,
            priority=priority,
        )

    def process_plan_refine_job(self, job_id: str) -> PlanRefineJob | None:
        """Execute one queued plan-refine job to completion.

        Args:
            job_id (str): Identifier for the target job.

        Returns:
            PlanRefineJob | None: Result produced by this call.
        """
        return self._plan_manager.process_plan_refine_job(job_id)

    def list_plan_refine_jobs(self, task_id: str) -> list[PlanRefineJob]:
        """List refine jobs for a task.

        Args:
            task_id (str): Task identifier whose refine-job history should be returned.

        Returns:
            list[PlanRefineJob]: Refine jobs for the task, ordered by repository behavior.

        Raises:
            ValueError: If the task does not exist.
        """
        return self._plan_manager.list_plan_refine_jobs(task_id)

    def get_plan_refine_job(self, task_id: str, job_id: str) -> PlanRefineJob:
        """Fetch one refine job and verify it belongs to the given task.

        Args:
            task_id (str): Parent task identifier expected for the refine job.
            job_id (str): Refine-job identifier to load.

        Returns:
            PlanRefineJob: The matching refine job owned by ``task_id``.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If the job does not exist or belongs to a different task.
        """
        return self._plan_manager.get_plan_refine_job(task_id, job_id)

    def commit_plan_revision(self, task_id: str, revision_id: str) -> str:
        """Mark one plan revision as committed and sync task/workdoc metadata.

        Args:
            task_id (str): Task identifier that owns the plan revision.
            revision_id (str): Revision identifier to mark as committed.

        Returns:
            str: The committed revision identifier.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If the revision does not exist for the task.
        """
        return self._plan_manager.commit_plan_revision(task_id, revision_id)

    def resolve_plan_text_for_generation(
        self,
        *,
        task_id: str,
        source: Literal["committed", "revision", "override", "latest"],
        revision_id: str | None = None,
        plan_override: str | None = None,
    ) -> tuple[str, str | None]:
        """Resolve plan text and optional revision id for task generation.

        Args:
            task_id (str): Task identifier whose plan should be resolved.
            source (Literal['committed', 'revision', 'override', 'latest']): Source
                strategy to use when selecting plan text.
            revision_id (str | None): Required when ``source='revision'``; ignored
                otherwise.
            plan_override (str | None): Required non-empty text when
                ``source='override'``.

        Returns:
            tuple[str, str | None]: A tuple of ``(plan_text, revision_id)``.
            ``revision_id`` is ``None`` only when ``source='override'``.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If source-specific required inputs are missing.
            ValueError: If the requested revision/committed plan cannot be found.
        """
        return self._plan_manager.resolve_plan_text_for_generation(
            task_id=task_id,
            source=source,
            revision_id=revision_id,
            plan_override=plan_override,
        )

    def _resolve_task_plan_excerpt(self, task: Task, *, max_chars: int = 800) -> str:
        """Extract a bounded plan snippet for prompts and conflict resolution context."""
        return self._worktree_manager.resolve_task_plan_excerpt(task, max_chars=max_chars)

    def _format_task_objective_summary(self, task: Task, *, max_chars: int = 1600) -> str:
        """Build concise objective context for merge-conflict resolution."""
        return self._worktree_manager.format_task_objective_summary(task, max_chars=max_chars)

    def _loop(self) -> None:
        while not self._stop.is_set():
            handled = self.tick_once()
            with self._futures_lock:
                has_inflight = bool(self._futures)
            if self._drain and not handled and not has_inflight:
                self.control("pause")
                self._drain = False
                break
            time.sleep(1 if handled else 2)

    def _create_worktree(self, task: Task) -> Optional[Path]:
        return self._worktree_manager.create_worktree(task)

    # ------------------------------------------------------------------

    def _pipeline_id_for_task(self, task: Task) -> str:
        """Resolve pipeline id for a task; fall back safely on failure."""
        try:
            return PipelineRegistry().resolve_for_task_type(task.task_type).id
        except Exception:
            return "feature"

    def _workdoc_template_for_task(self, task: Task) -> str:
        """Select the default workdoc template for the task's pipeline."""
        return self._workdoc_manager.workdoc_template_for_task(task)

    def _workdoc_section_for_step(self, task: Task, step: str) -> tuple[str, str | None] | None:
        """Resolve section heading/placeholder mapping for a step and task pipeline."""
        return self._workdoc_manager.workdoc_section_for_step(task, step)

    def _workdoc_canonical_path(self, task_id: str) -> Path:
        return self._workdoc_manager.workdoc_canonical_path(task_id)

    @staticmethod
    def _workdoc_worktree_path(project_dir: Path) -> Path:
        return WorkdocManager.workdoc_worktree_path(project_dir)

    def _init_workdoc(self, task: Task, project_dir: Path) -> Path:
        """Render the workdoc template, write canonical + worktree copies."""
        return self._workdoc_manager.init_workdoc(task, project_dir)

    @staticmethod
    def _cleanup_workdoc_for_commit(project_dir: Path) -> None:
        """Remove the worktree .workdoc.md before commit so git add -A won't stage it."""
        WorkdocManager.cleanup_workdoc_for_commit(project_dir)

    def _refresh_workdoc(self, task: Task, project_dir: Path) -> None:
        """Copy canonical workdoc to worktree so the worker sees the latest version."""
        self._workdoc_manager.refresh_workdoc(task, project_dir)

    def _sync_workdoc(
        self, task: Task, step: str, project_dir: Path, summary: str | None, attempt: int | None = None
    ) -> None:
        """Post-step sync: accept worker changes or fallback-append summary."""
        self._workdoc_manager.sync_workdoc(task, step, project_dir, summary, attempt)

    def _sync_workdoc_review(self, task: Task, cycle: ReviewCycle, project_dir: Path) -> None:
        """Append review cycle findings to the workdoc."""
        self._workdoc_manager.sync_workdoc_review(task, cycle, project_dir)

    def _validate_task_workdoc(self, task: Task) -> Path | None:
        """Return canonical workdoc path when present, else ``None``."""
        canonical = self._workdoc_canonical_path(task.id)
        if canonical.exists():
            return canonical
        return None

    def _block_for_missing_workdoc(self, task: Task, run: RunRecord, *, step: str) -> None:
        """Fail fast when a task loses its required canonical workdoc."""
        canonical = self._workdoc_canonical_path(task.id)
        task.status = "blocked"
        task.error = f"Missing required workdoc: {canonical.name}"
        task.pending_gate = None
        task.current_step = step
        task.metadata["pipeline_phase"] = step
        task.metadata["missing_workdoc_path"] = str(canonical)
        self.container.tasks.upsert(task)
        self._finalize_run(task, run, status="blocked", summary=f"Blocked during {step}: missing required workdoc")
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": task.error},
        )

    def _step_project_dir(self, task: Task) -> Path:
        """Resolve task worktree directory, falling back to the main project root."""
        worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
        return Path(worktree_path) if worktree_path else self.container.project_dir

    def get_workdoc(self, task_id: str) -> dict[str, Any]:
        """Read canonical workdoc for a task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        canonical = self._workdoc_canonical_path(task_id)
        if not canonical.exists():
            raise FileNotFoundError(f"Missing required workdoc for task {task_id}")
        return {"task_id": task_id, "content": canonical.read_text(encoding="utf-8"), "exists": True}

    def _merge_and_cleanup(self, task: Task, worktree_dir: Path) -> None:
        self._worktree_manager.merge_and_cleanup(task, worktree_dir)

    def approve_and_merge(self, task: Task) -> dict[str, Any]:
        """Merge a preserved branch to the run branch on user approval.

        Called when a blocked task is approved and its work was preserved on a
        branch (for example, after review-attempt limits were hit).

        Args:
            task (Task): Task carrying preserved branch metadata.

        Returns:
            dict[str, Any]: Merge outcome payload with ``status``. Includes
            ``commit_sha`` on successful merge and ``status='merge_conflict'``
            when automatic conflict resolution fails.
        """
        return self._worktree_manager.approve_and_merge(task)

    def _resolve_merge_conflict(self, task: Task, branch: str) -> bool:
        return self._worktree_manager.resolve_merge_conflict(task, branch)

    def _cleanup_orphaned_worktrees(self) -> None:
        self._worktree_manager.cleanup_orphaned_worktrees()

    def _role_for_task(self, task: Task) -> str:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        by_type = dict(routing.get("task_type_roles") or {})
        default_role = str(routing.get("default_role") or "general")
        return str(by_type.get(task.task_type) or default_role)

    def _provider_override_for_role(self, role: str) -> Optional[str]:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        overrides = dict(routing.get("role_provider_overrides") or {})
        raw = overrides.get(role)
        return str(raw) if raw else None

    def _choose_agent_for_task(self, task: Task) -> Optional[str]:
        desired_role = self._role_for_task(task)
        running = [agent for agent in self.container.agents.list() if agent.status == "running"]
        exact = [agent for agent in running if agent.role == desired_role]
        pool = exact or running
        if not pool:
            return None
        pool.sort(key=lambda agent: agent.last_seen_at)
        chosen = pool[0]
        override_provider = self._provider_override_for_role(chosen.role)
        if override_provider:
            task.metadata["provider_override"] = override_provider
        return chosen.id

    def _ensure_branch(self) -> Optional[str]:
        return self._worktree_manager.ensure_branch()

    def _commit_for_task(self, task: Task, working_dir: Optional[Path] = None) -> Optional[str]:
        return self._worktree_manager.commit_for_task(task, working_dir)

    def _has_uncommitted_changes(self, cwd: Path) -> bool:
        """Check whether the working tree has staged or unstaged changes.

        Returns True on git failure (no repo, etc.) to avoid false blocking.
        """
        return self._worktree_manager.has_uncommitted_changes(cwd)

    def _preserve_worktree_work(self, task: Task, worktree_dir: Path) -> bool:
        """Commit agent edits in the worktree, remove the worktree but keep the branch.

        Returns True if a branch was preserved with work on it.
        """
        return self._worktree_manager.preserve_worktree_work(task, worktree_dir)

    def _exceeds_quality_gate(self, task: Task, findings: list[ReviewFinding]) -> bool:
        gate = dict(task.quality_gate or {})
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            if finding.status != "open":
                continue
            sev = finding.severity if finding.severity in counts else "low"
            counts[sev] += 1
        return any(counts[sev] > int(gate.get(sev, 0)) for sev in counts)

    def _build_review_history_summary(
        self, task_id: str, *, max_cycles: int = 5, max_chars: int = 4000
    ) -> list[dict[str, Any]]:
        """Build a compact summary of prior review cycles for prompt injection."""
        cycles = self.container.reviews.for_task(task_id)
        cycles.sort(key=lambda c: c.attempt)
        if len(cycles) > max_cycles:
            cycles = cycles[-max_cycles:]
        result: list[dict[str, Any]] = []
        total_chars = 0
        for cycle in cycles:
            entry: dict[str, Any] = {
                "attempt": cycle.attempt,
                "decision": cycle.decision,
                "findings": [],
            }
            for f in cycle.findings:
                f_entry = {"severity": f.severity, "summary": f.summary, "status": f.status}
                total_chars += len(f.summary) + 30
                if total_chars > max_chars:
                    break
                entry["findings"].append(f_entry)
            result.append(entry)
            if total_chars > max_chars:
                break
        return result

    _VERIFY_OUTPUT_TAIL_CHARS = 8000
    _NON_ACTIONABLE_VERIFY_REASON_CODES = {
        "tool_missing",
        "config_missing",
        "no_tests",
        "permission_denied",
        "resource_limit",
        "os_incompatibility",
        "infrastructure",
    }
    _NON_ACTIONABLE_VERIFY_SUMMARY_PATTERNS = (
        r"command not found",
        r"not installed",
        r"no tests? found",
        r"missing (config|configuration)",
        r"no such file or directory",
        r"permission denied",
        r"operation not permitted",
        r"network is unreachable",
        r"connection refused",
        r"timed out connecting",
        r"docker.*not (available|running)",
        r"requires .* not available",
    )

    @staticmethod
    def _should_gate(mode: str, gate_name: str) -> bool:
        """Compatibility wrapper for HITL gate checks used by delegated executors."""
        return should_gate(mode, gate_name)

    def _capture_verify_output(self, task: Task) -> None:
        """Read the tail of the verify step's stdout/stderr and stash it in metadata.

        This gives ``implement_fix`` concrete test/lint output to work with,
        not just the terse ``task.error`` summary.
        """
        last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
        if not isinstance(last_logs, dict):
            return
        limit = self._VERIFY_OUTPUT_TAIL_CHARS
        parts: list[str] = []
        for key, label in (("stdout_path", "stdout"), ("stderr_path", "stderr")):
            raw = last_logs.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                p = Path(raw)
                if not p.is_file():
                    continue
                size = p.stat().st_size
                if size <= 0:
                    continue
                # Read only the tail to keep prompt size bounded.
                read_bytes = min(size, limit * 4)
                with open(p, "rb") as fh:
                    if read_bytes < size:
                        fh.seek(size - read_bytes)
                    text = fh.read(read_bytes).decode("utf-8", errors="replace")
                tail = text[-limit:] if len(text) > limit else text
                if tail.strip():
                    parts.append(f"--- {label} (last {len(tail)} chars) ---")
                    parts.append(tail.strip())
            except Exception:
                continue
        if parts:
            task.metadata["verify_output"] = "\n".join(parts)

    def _is_non_actionable_verify_failure(self, task: Task, summary: str | None) -> bool:
        """Classify whether verify failed because of environment or tooling issues."""
        if isinstance(task.metadata, dict):
            reason_code = str(task.metadata.get("verify_reason_code") or "").strip().lower()
            if reason_code in self._NON_ACTIONABLE_VERIFY_REASON_CODES:
                return True
        text = str(summary or "").strip().lower()
        if not text:
            return False
        return any(re.search(pattern, text) for pattern in self._NON_ACTIONABLE_VERIFY_SUMMARY_PATTERNS)

    def _mark_verify_degraded(
        self,
        task: Task,
        run: RunRecord,
        *,
        step: str,
        summary: str | None,
    ) -> None:
        """Persist and emit degraded verification context, then continue best-effort flow."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        entry = {
            "ts": now_iso(),
            "step": step,
            "reason_code": str(task.metadata.get("verify_reason_code") or "unknown"),
            "summary": str(summary or "Verification degraded due to environment/tooling constraints."),
        }
        issues = task.metadata.get("verify_degraded_issues")
        if not isinstance(issues, list):
            issues = []
        issues.append(entry)
        task.metadata["verify_degraded_issues"] = issues
        task.metadata["verify_degraded"] = entry
        task.metadata["verify_non_actionable_failure"] = True
        task.status = "in_progress"
        task.error = None
        task.pending_gate = None
        self.container.tasks.upsert(task)
        run.status = "in_progress"
        run.finished_at = None
        run.summary = None
        self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.verify_degraded",
            entity_id=task.id,
            payload=entry,
        )

    def _consume_verify_non_actionable_flag(self, task: Task) -> bool:
        if not isinstance(task.metadata, dict):
            return False
        return bool(task.metadata.pop("verify_non_actionable_failure", False))

    def _select_post_fix_validation_step(self, steps: list[str]) -> str | None:
        """Pick the validation step that should run after implement_fix in review loops."""
        if "verify" in steps:
            return "verify"
        if "benchmark" in steps:
            return "benchmark"
        return None

    def _run_non_review_step(self, task: Task, run: RunRecord, step: str, attempt: int = 1) -> bool:
        if self._validate_task_workdoc(task) is None:
            self._block_for_missing_workdoc(task, run, step=step)
            return False
        # Refresh workdoc before each step so the worker sees the latest version.
        step_project_dir = self._step_project_dir(task)
        self._refresh_workdoc(task, step_project_dir)

        step_started = now_iso()
        try:
            result = self.worker_adapter.run_step(task=task, step=step, attempt=attempt)
        except WorkerCancelledError:
            raise self._Cancelled()

        # Post-step workdoc sync (before other bookkeeping that may upsert task).
        self._sync_workdoc(task, step, step_project_dir, result.summary, attempt=attempt)

        step_log: dict[str, Any] = {"step": step, "status": result.status, "ts": now_iso(), "started_at": step_started, "summary": result.summary}
        if result.human_blocking_issues:
            step_log["human_blocking_issues"] = result.human_blocking_issues
        # Preserve log file paths so historical step logs can be retrieved.
        last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
        if isinstance(last_logs, dict):
            for key in ("stdout_path", "stderr_path", "progress_path"):
                if last_logs.get(key):
                    step_log[key] = last_logs[key]
        run.steps.append(step_log)
        self.container.runs.upsert(run)
        self.container.tasks.upsert(task)
        if result.human_blocking_issues:
            self._block_for_human_issues(task, run, step, result.summary, result.human_blocking_issues)
            return False
        if result.status != "ok":
            if step in _VERIFY_STEPS and self._is_non_actionable_verify_failure(task, result.summary):
                self._mark_verify_degraded(task, run, step=step, summary=result.summary)
                return False
            task.status = "blocked"
            task.error = result.summary or f"{step} failed"
            task.pending_gate = None
            task.current_step = step
            self.container.tasks.upsert(task)
            run.status = "blocked"
            run.finished_at = now_iso()
            run.summary = f"Blocked during {step}"
            self.container.runs.upsert(run)
            self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
            return False

        task.metadata.pop("human_blocking_issues", None)

        # Store plan output as first-class immutable plan revisions.
        if step in {"plan", "initiative_plan"} and result.summary:
            provider, model = self._resolve_worker_lineage(task, step)
            self.create_plan_revision(
                task_id=task.id,
                content=result.summary,
                source="worker_plan",
                step=step,
                provider=provider,
                model=model,
            )
            # Keep in-memory task metadata aligned so later upserts do not overwrite
            # the stored revision pointers/history.
            refreshed = self.container.tasks.get(task.id)
            if refreshed and isinstance(refreshed.metadata, dict):
                task.metadata = dict(refreshed.metadata)

        # Store step output for downstream prompt injection.
        if result.summary:
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            so = task.metadata.setdefault("step_outputs", {})
            # Plan text already bounded at 20KB by _normalize_planning_text.
            # Other outputs truncated to 4KB to prevent metadata bloat.
            max_len = 20_000 if step in {"plan", "initiative_plan"} else 4_000
            so[step] = result.summary[:max_len]

        # Handle generate_tasks: prefer generated tasks, but avoid silent no-op by recording warning metadata.
        if step == "generate_tasks":
            generated = list(result.generated_tasks or [])
            if not generated:
                if not isinstance(task.metadata, dict):
                    task.metadata = {}
                task.metadata["generate_tasks_warning"] = result.summary or "generate_tasks produced no structured tasks"
                self.container.tasks.upsert(task)
                self.bus.emit(
                    channel="tasks",
                    event_type="task.generate_tasks_empty",
                    entity_id=task.id,
                    payload={"warning": task.metadata["generate_tasks_warning"]},
                )
                return True
            self._create_child_tasks(task, generated)

        return True

    def _create_child_tasks(
        self, parent: Task, task_defs: list[dict[str, Any]], *, apply_deps: bool = False
    ) -> list[str]:
        return self._plan_manager.create_child_tasks(parent, task_defs, apply_deps=apply_deps)

    def generate_tasks_from_plan(
        self, task_id: str, plan_text: str, *, infer_deps: bool = True
    ) -> list[str]:
        """Generate child tasks from an explicit plan text.

        This supports a two-phase workflow: run a plan step, review the output,
        then explicitly trigger task generation from the plan.

        Args:
            task_id (str): Parent task identifier that will receive generated children.
            plan_text (str): Plan content to pass into the ``generate_tasks`` worker step.
            infer_deps (bool): Whether to apply ``depends_on`` links returned by
                the worker to the created child tasks.

        Returns:
            list[str]: Identifiers of newly created child tasks.

        Raises:
            ValueError: If the parent task does not exist.
            ValueError: If worker execution fails or yields no valid tasks.
        """
        return self._plan_manager.generate_tasks_from_plan(task_id, plan_text, infer_deps=infer_deps)

    def _findings_from_result(self, task: Task, review_attempt: int) -> tuple[list[ReviewFinding], StepResult]:
        try:
            result = self.worker_adapter.run_step(task=task, step="review", attempt=review_attempt)
        except WorkerCancelledError:
            raise self._Cancelled()
        raw_findings = list(result.findings or [])
        findings: list[ReviewFinding] = []
        for idx, finding in enumerate(raw_findings):
            if not isinstance(finding, dict):
                continue
            findings.append(
                ReviewFinding(
                    id=f"{task.id}-a{review_attempt}-{idx}",
                    task_id=task.id,
                    severity=str(finding.get("severity") or "medium"),
                    category=str(finding.get("category") or "quality"),
                    summary=str(finding.get("summary") or "Issue"),
                    file=finding.get("file"),
                    line=finding.get("line"),
                    suggested_fix=finding.get("suggested_fix"),
                    status=str(finding.get("status") or "open"),
                )
            )
        return findings, result

    def _block_for_human_issues(
        self,
        task: Task,
        run: RunRecord,
        step: str,
        summary: str | None,
        issues: list[dict[str, str]],
    ) -> None:
        task.status = "blocked"
        task.current_step = step
        task.pending_gate = self._HUMAN_INTERVENTION_GATE
        task.error = summary or "Human intervention required to continue"
        task.metadata["human_blocking_issues"] = issues
        self.container.tasks.upsert(task)

        self._finalize_run(task, run, status="blocked", summary=f"Blocked during {step}: human intervention required")

        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": self._HUMAN_INTERVENTION_GATE, "step": step, "issues": issues},
        )
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={
                "error": task.error,
                "gate": self._HUMAN_INTERVENTION_GATE,
                "step": step,
                "issues": issues,
            },
        )

    def _wait_for_gate(self, task: Task, gate_name: str, timeout: int = 3600) -> bool:
        """Block until ``pending_gate`` is cleared (via approve-gate API).

        Returns True if gate was approved, False on timeout/stop/cancel.
        """
        task.pending_gate = gate_name
        task.updated_at = now_iso()
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": gate_name},
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            fresh = self.container.tasks.get(task.id)
            if fresh is None or fresh.status == "cancelled":
                return False
            if fresh.pending_gate is None:
                return True
            time.sleep(1)
        return False

    def _abort_for_gate(self, task: Task, run: RunRecord, gate_name: str) -> None:
        """Mark task as blocked because a gate was not approved."""
        task.status = "blocked"
        task.error = f"Gate '{gate_name}' was not approved in time"
        task.pending_gate = None
        self.container.tasks.upsert(task)
        run.status = "blocked"
        run.finished_at = now_iso()
        run.summary = f"Blocked at gate: {gate_name}"
        self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": task.error},
        )

    def _run_summarize_step(self, task: Task, run: RunRecord) -> None:
        """Auto-inject a summarize step using a lightweight LLM call."""
        fn = getattr(self.worker_adapter, "generate_run_summary", None)
        if fn is None:
            return
        try:
            worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
            project_dir = Path(worktree_path) if worktree_path else self.container.project_dir
            if not project_dir.is_dir():
                project_dir = self.container.project_dir
            summary_started = now_iso()
            summary_text = fn(task=task, run=run, project_dir=project_dir)
            if isinstance(summary_text, str) and summary_text.strip():
                run.steps.append({
                    "step": "summary",
                    "status": "ok",
                    "ts": now_iso(),
                    "started_at": summary_started,
                    "summary": summary_text,
                })
                self.container.runs.upsert(run)
        except Exception:
            logger.debug("Summarize step failed for task %s; skipping", task.id)

    def _finalize_run(self, task: Task, run: RunRecord, *, status: str, summary: str) -> None:
        """Run the summarize step, then set run status/summary/finished_at."""
        self._run_summarize_step(task, run)
        run.status = status
        run.finished_at = now_iso()
        run.summary = summary
        self.container.runs.upsert(run)

    def _maybe_analyze_dependencies(self) -> None:
        """Run automatic dependency analysis on unanalyzed ready tasks."""
        self._dependency_manager.maybe_analyze_dependencies()

    def _apply_dependency_edges(
        self,
        candidates: list[Task],
        edges: list[dict[str, str]],
        all_tasks: list[Task],
    ) -> None:
        """Apply inferred dependency edges with cycle detection."""
        self._dependency_manager.apply_dependency_edges(candidates, edges, all_tasks)

    class _Cancelled(Exception):
        """Raised when a task is cancelled mid-execution."""

    def _check_cancelled(self, task: Task) -> None:
        """Re-read task from storage and raise _Cancelled if user cancelled it."""
        fresh = self.container.tasks.get(task.id)
        if fresh and fresh.status == "cancelled":
            raise self._Cancelled()

    def _execute_task(self, task: Task) -> None:
        self._task_executor.execute_task(task)

    def _execute_task_inner(self, task: Task) -> None:
        self._task_executor.execute_task_inner(task)


def create_orchestrator(
    container: Container,
    bus: EventBus,
    *,
    worker_adapter: WorkerAdapter | None = None,
) -> OrchestratorService:
    """Build, start, and return an orchestrator instance for a container.

    Args:
        container (Container): Container for this call.
        bus (EventBus): Bus for this call.
        worker_adapter (WorkerAdapter | None): Worker adapter for this call.

    Returns:
        OrchestratorService: Result produced by this call.
    """
    if worker_adapter is None:
        from .live_worker_adapter import LiveWorkerAdapter

        worker_adapter = LiveWorkerAdapter(container)
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
