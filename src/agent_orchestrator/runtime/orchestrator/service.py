"""Core orchestration service for task lifecycle execution."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
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
from .live_worker_adapter import _VERIFY_STEPS
from .worker_adapter import DefaultWorkerAdapter, StepResult, WorkerAdapter

logger = logging.getLogger(__name__)


def _has_cycle(adj: dict[str, list[str]], from_id: str, to_id: str) -> bool:
    """Return True if adding an edge from_id→to_id would create a cycle.

    Checks whether to_id can already reach from_id via existing edges.
    """
    visited: set[str] = set()
    stack = [to_id]
    while stack:
        node = stack.pop()
        if node == from_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))
    return False


class OrchestratorService:
    """Represents OrchestratorService."""
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

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            cfg = self.container.config.load()
            max_workers = int(dict(cfg.get("orchestrator") or {}).get("concurrency", 2) or 2)
            self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="orchestrator-task")
        return self._pool

    def status(self) -> dict[str, Any]:
        """Return status."""
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
        """Return control."""
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
        """Return ensure worker."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._recover_in_progress_tasks()
            self._cleanup_orphaned_worktrees()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="orchestrator")
            self._thread.start()

    def shutdown(self, *, timeout: float = 10.0) -> None:
        """Return shutdown."""
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
        """Return tick once."""
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
        """Return run task."""
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
        for job in self.container.plan_refine_jobs.for_task(task_id):
            if job.status in {"queued", "running"}:
                return job
        return None

    def get_plan_document(self, task_id: str) -> dict[str, Any]:
        """Return get plan document."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        revisions = self.container.plan_revisions.for_task(task_id)
        revisions.sort(key=lambda item: item.created_at)
        latest_revision_id = revisions[-1].id if revisions else None
        committed_revision_id = task.metadata.get("committed_plan_revision_id")
        if committed_revision_id and not any(item.id == committed_revision_id for item in revisions):
            committed_revision_id = None
        if not committed_revision_id:
            committed = [item for item in revisions if str(item.status or "").lower() == "committed"]
            if committed:
                committed_revision_id = committed[-1].id
                task.metadata["committed_plan_revision_id"] = committed_revision_id
                task.metadata["latest_plan_revision_id"] = latest_revision_id or committed_revision_id
                self.container.tasks.upsert(task)
        active_job = self._active_plan_refine_job(task_id)
        return {
            "task_id": task_id,
            "latest_revision_id": latest_revision_id,
            "committed_revision_id": committed_revision_id,
            "revisions": [item.to_dict() for item in revisions],
            "active_refine_job": active_job.to_dict() if active_job else None,
        }

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
        """Create and persist a plan revision for a task."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        body = str(content or "").strip()
        if not body:
            raise ValueError("Plan revision content cannot be empty")
        revisions = self.container.plan_revisions.for_task(task_id)
        revisions.sort(key=lambda item: item.created_at)
        if parent_revision_id:
            parent = self.container.plan_revisions.get(parent_revision_id)
            if not parent or parent.task_id != task_id:
                raise ValueError("parent_revision_id does not belong to task")
        else:
            parent_revision_id = revisions[-1].id if revisions else None
        revision = PlanRevision(
            task_id=task_id,
            created_at=created_at or now_iso(),
            source=source,
            parent_revision_id=parent_revision_id,
            step=step,
            feedback_note=feedback_note,
            provider=provider,
            model=model,
            content=body,
            status=status,
        )
        self.container.plan_revisions.upsert(revision)
        task.metadata["latest_plan_revision_id"] = revision.id
        if status == "committed":
            task.metadata["committed_plan_revision_id"] = revision.id
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="plan.revision.created",
            entity_id=task_id,
            payload={"revision_id": revision.id, "source": source},
        )
        return revision

    def queue_plan_refine_job(
        self,
        *,
        task_id: str,
        feedback: str,
        instructions: str | None = None,
        base_revision_id: str | None = None,
        priority: str = "normal",
    ) -> PlanRefineJob:
        """Queue a plan refinement job and schedule background processing."""
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError("Task not found")
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            if self._active_plan_refine_job(task_id):
                raise RuntimeError("A plan refine job is already active for this task")
            revisions = self.container.plan_revisions.for_task(task_id)
            revisions.sort(key=lambda item: item.created_at)
            if not revisions:
                raise ValueError("No plan revision exists for this task")
            normalized_feedback = str(feedback or "").strip()
            if not normalized_feedback:
                raise ValueError("feedback is required")
            if base_revision_id:
                base_revision = self.container.plan_revisions.get(base_revision_id)
                if not base_revision or base_revision.task_id != task_id:
                    raise ValueError("base_revision_id not found for task")
            else:
                base_revision = revisions[-1]
            normalized_priority = str(priority or "normal").strip().lower()
            if normalized_priority not in {"normal", "high"}:
                normalized_priority = "normal"
            job = PlanRefineJob(
                task_id=task_id,
                base_revision_id=base_revision.id,
                status="queued",
                feedback=normalized_feedback,
                instructions=(str(instructions).strip() if instructions else None),
                priority=normalized_priority,
            )
            self.container.plan_refine_jobs.upsert(job)
            self.bus.emit(
                channel="tasks",
                event_type="plan.refine.queued",
                entity_id=task_id,
                payload={"job_id": job.id, "base_revision_id": job.base_revision_id},
            )
        future = self._get_pool().submit(self.process_plan_refine_job, job.id)
        with self._futures_lock:
            self._futures[job.id] = future
        return job

    def process_plan_refine_job(self, job_id: str) -> PlanRefineJob | None:
        """Return process plan refine job."""
        job = self.container.plan_refine_jobs.get(job_id)
        if not job:
            return None
        if job.status not in {"queued", "running"}:
            return job
        job.status = "running"
        job.started_at = now_iso()
        self.container.plan_refine_jobs.upsert(job)
        try:
            self.bus.emit(
                channel="tasks",
                event_type="plan.refine.started",
                entity_id=job.task_id,
                payload={"job_id": job.id},
            )
        except Exception:
            logger.exception("Failed to emit plan.refine.started for job %s", job.id)
        task = self.container.tasks.get(job.task_id)
        base_revision = self.container.plan_revisions.get(job.base_revision_id)
        if not task or not base_revision or base_revision.task_id != job.task_id:
            job.status = "failed"
            job.finished_at = now_iso()
            job.error = "Task or base revision not found"
            self.container.plan_refine_jobs.upsert(job)
            self.bus.emit(
                channel="tasks",
                event_type="plan.refine.failed",
                entity_id=job.task_id,
                payload={"job_id": job.id, "error": job.error},
            )
            return job

        live_task = self.container.tasks.get(job.task_id)
        if not live_task:
            job.status = "failed"
            job.finished_at = now_iso()
            job.error = "Task not found"
            self.container.plan_refine_jobs.upsert(job)
            self.bus.emit(
                channel="tasks",
                event_type="plan.refine.failed",
                entity_id=job.task_id,
                payload={"job_id": job.id, "error": job.error},
            )
            return job
        refine_step = "initiative_plan_refine" if live_task.task_type == "initiative_plan" else "plan_refine"
        refine_key = "initiative_plan_refine" if refine_step == "initiative_plan_refine" else "plan_refine"

        if not isinstance(live_task.metadata, dict):
            live_task.metadata = {}
        live_task.metadata[f"{refine_key}_base"] = base_revision.content
        live_task.metadata[f"{refine_key}_feedback"] = job.feedback
        if job.instructions:
            live_task.metadata[f"{refine_key}_instructions"] = job.instructions
        self.container.tasks.upsert(live_task)

        try:
            try:
                result = self.worker_adapter.run_step(task=live_task, step=refine_step, attempt=1)
                if result.status != "ok":
                    raise ValueError(result.summary or f"{refine_step} failed")
                revised_plan = str(result.summary or "").strip()
                if not revised_plan:
                    raise ValueError("Worker returned empty refined plan")
                provider, model = self._resolve_worker_lineage(live_task, refine_step)
                revision = self.create_plan_revision(
                    task_id=job.task_id,
                    content=revised_plan,
                    source="worker_refine",
                    parent_revision_id=base_revision.id,
                    step=refine_step,
                    feedback_note=job.feedback,
                    provider=provider,
                    model=model,
                )
            except Exception as exc:
                job.status = "failed"
                job.finished_at = now_iso()
                job.error = str(exc)
                self.container.plan_refine_jobs.upsert(job)
                try:
                    self.bus.emit(
                        channel="tasks",
                        event_type="plan.refine.failed",
                        entity_id=job.task_id,
                        payload={"job_id": job.id, "error": job.error},
                    )
                except Exception:
                    logger.exception("Failed to emit plan.refine.failed for job %s", job.id)
                return job

            refreshed = self.container.tasks.get(live_task.id)
            if refreshed and isinstance(refreshed.metadata, dict):
                live_task.metadata = dict(refreshed.metadata)
            job.status = "completed"
            job.finished_at = now_iso()
            job.result_revision_id = revision.id
            job.error = None
            self.container.plan_refine_jobs.upsert(job)
            try:
                self.bus.emit(
                    channel="tasks",
                    event_type="plan.refine.completed",
                    entity_id=job.task_id,
                    payload={"job_id": job.id, "result_revision_id": revision.id},
                )
            except Exception:
                logger.exception("Failed to emit plan.refine.completed for job %s", job.id)
            return job
        finally:
            cleanup_task = self.container.tasks.get(job.task_id)
            if cleanup_task and isinstance(cleanup_task.metadata, dict):
                cleanup_task.metadata.pop(f"{refine_key}_base", None)
                cleanup_task.metadata.pop(f"{refine_key}_feedback", None)
                cleanup_task.metadata.pop(f"{refine_key}_instructions", None)
                self.container.tasks.upsert(cleanup_task)

    def list_plan_refine_jobs(self, task_id: str) -> list[PlanRefineJob]:
        """Return list plan refine jobs."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        return self.container.plan_refine_jobs.for_task(task_id)

    def get_plan_refine_job(self, task_id: str, job_id: str) -> PlanRefineJob:
        """Return get plan refine job."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        job = self.container.plan_refine_jobs.get(job_id)
        if not job or job.task_id != task_id:
            raise ValueError("Plan refine job not found")
        return job

    def commit_plan_revision(self, task_id: str, revision_id: str) -> str:
        """Return commit plan revision."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        target = self.container.plan_revisions.get(revision_id)
        if not target or target.task_id != task_id:
            raise ValueError("Revision not found for task")
        for revision in self.container.plan_revisions.for_task(task_id):
            next_status = "committed" if revision.id == revision_id else "draft"
            if revision.status != next_status:
                revision.status = cast(PlanRevisionStatus, next_status)
                self.container.plan_revisions.upsert(revision)
        task.metadata["latest_plan_revision_id"] = revision_id
        task.metadata["committed_plan_revision_id"] = revision_id
        # Sync committed plan text to step_outputs so implement step uses it.
        so = task.metadata.setdefault("step_outputs", {})
        so["plan"] = (target.content or "")[:20_000]
        self.container.tasks.upsert(task)

        # Keep workdoc ## Plan in sync with the committed plan text so manual edits
        # become the canonical implementation guide immediately.
        workdoc_path = task.metadata.get("workdoc_path") if isinstance(task.metadata, dict) else None
        canonical = (
            Path(workdoc_path)
            if isinstance(workdoc_path, str) and workdoc_path.strip()
            else self._workdoc_canonical_path(task.id)
        )
        if canonical.exists():
            text = canonical.read_text(encoding="utf-8")
            heading = "## Plan"
            idx = text.find(heading)
            if idx != -1:
                after_heading = text.find("\n", idx)
                if after_heading != -1:
                    rest = text[after_heading + 1 :]
                    next_heading = re.search(r"^## ", rest, re.MULTILINE)
                    section_end = after_heading + 1 + next_heading.start() if next_heading else len(text)
                    plan_body = (target.content or "").strip() or "_(empty committed plan)_"
                    updated = text[: after_heading + 1] + plan_body + "\n\n" + text[section_end:]
                    canonical.write_text(updated, encoding="utf-8")
                    # Mirror canonical into the active worktree copy if present.
                    step_project_dir = self._step_project_dir(task)
                    worktree_copy = self._workdoc_worktree_path(step_project_dir)
                    if worktree_copy.exists():
                        worktree_copy.write_text(updated, encoding="utf-8")
                    self.bus.emit(
                        channel="tasks",
                        event_type="workdoc.updated",
                        entity_id=task.id,
                        payload={"step": "plan", "source": "plan_commit"},
                    )

        self.bus.emit(
            channel="tasks",
            event_type="plan.revision.committed",
            entity_id=task_id,
            payload={"revision_id": revision_id},
        )
        return revision_id

    def resolve_plan_text_for_generation(
        self,
        *,
        task_id: str,
        source: Literal["committed", "revision", "override", "latest"],
        revision_id: str | None = None,
        plan_override: str | None = None,
    ) -> tuple[str, str | None]:
        """Resolve plan text and optional revision id for task generation."""
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        revisions = self.container.plan_revisions.for_task(task_id)
        revisions.sort(key=lambda item: item.created_at)

        if source == "override":
            body = str(plan_override or "").strip()
            if not body:
                raise ValueError("plan_override is required for source=override")
            return body, None

        if source == "revision":
            if not revision_id:
                raise ValueError("revision_id is required for source=revision")
            revision = self.container.plan_revisions.get(revision_id)
            if not revision or revision.task_id != task_id:
                raise ValueError("Revision not found for task")
            return revision.content, revision.id

        if source == "committed":
            committed_id = str(task.metadata.get("committed_plan_revision_id") or "").strip()
            revision = self.container.plan_revisions.get(committed_id) if committed_id else None
            if not revision or revision.task_id != task_id:
                committed = [item for item in revisions if str(item.status or "").lower() == "committed"]
                revision = committed[-1] if committed else None
                if revision:
                    task.metadata["committed_plan_revision_id"] = revision.id
                    task.metadata["latest_plan_revision_id"] = revisions[-1].id if revisions else revision.id
                    self.container.tasks.upsert(task)
            if not revision or revision.task_id != task_id:
                raise ValueError("No committed plan revision exists for this task")
            return revision.content, revision.id

        if not revisions:
            raise ValueError("No plan revision exists for this task")
        return revisions[-1].content, revisions[-1].id

    def _resolve_task_plan_excerpt(self, task: Task, *, max_chars: int = 800) -> str:
        """Return a short best-effort plan excerpt for objective context."""
        if not isinstance(task.metadata, dict):
            return ""

        # Prefer committed plan revision, then latest plan revision.
        for key in ("committed_plan_revision_id", "latest_plan_revision_id"):
            rev_id = str(task.metadata.get(key) or "").strip()
            if not rev_id:
                continue
            revision = self.container.plan_revisions.get(rev_id)
            if revision and revision.task_id == task.id and str(revision.content or "").strip():
                return str(revision.content).strip()[:max_chars]

        # Fallback to plan text stashed in step outputs.
        step_outputs = task.metadata.get("step_outputs")
        if isinstance(step_outputs, dict):
            plan_text = str(step_outputs.get("plan") or "").strip()
            if plan_text:
                return plan_text[:max_chars]
        return ""

    def _format_task_objective_summary(self, task: Task, *, max_chars: int = 1600) -> str:
        """Build concise objective context for merge-conflict resolution."""
        lines = [f"- Task: {task.title}"]
        if task.description:
            lines.append(f"  Description: {task.description}")
        plan_excerpt = self._resolve_task_plan_excerpt(task)
        if plan_excerpt:
            lines.append("  Plan excerpt:")
            lines.append("  ---")
            lines.append(plan_excerpt)
            lines.append("  ---")
        return "\n".join(lines)[:max_chars]

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
        git_dir = self.container.project_dir / ".git"
        if not git_dir.exists():
            return None
        self._ensure_branch()  # ensure run branch exists as merge target
        worktree_dir = self.container.state_root / "worktrees" / task.id
        branch = f"task-{task.id}"
        subprocess.run(
            ["git", "worktree", "add", str(worktree_dir), "-b", branch],
            cwd=self.container.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return worktree_dir

    # ------------------------------------------------------------------
    # Working Document (workdoc) helpers
    # ------------------------------------------------------------------

    # Maps step names to (heading, placeholder_step) pairs.
    # placeholder_step is the step name used in the template placeholder text.
    _WORKDOC_SECTION_MAP: dict[str, tuple[str, str | None]] = {
        "plan": ("## Plan", "plan"),
        "initiative_plan": ("## Plan", "plan"),
        "analyze": ("## Analysis", "analyze"),
        "diagnose": ("## Analysis", "analyze"),
        "scan_deps": ("## Dependency Scan Findings", "scan_deps"),
        "scan_code": ("## Code Scan Findings", "scan_code"),
        "generate_tasks": ("## Generated Tasks", "generate_tasks"),
        "profile": ("## Profiling Baseline", "profile"),
        "implement": ("## Implementation Log", "implement"),
        "prototype": ("## Implementation Log", "implement"),
        "implement_fix": ("## Fix Log", None),  # placeholder uses different wording
        "verify": ("## Verification Results", "verify"),
        "benchmark": ("## Verification Results", "verify"),
        "reproduce": ("## Verification Results", "verify"),
        "report": ("## Final Report", "report"),
    }

    _FEATURE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Plan

_Pending: will be populated by the plan step._

## Implementation Log

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _GENERIC_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Plan

_Pending: will be populated by the plan step._

## Analysis

_Pending: will be populated by the analyze step._

## Profiling Baseline

_Pending: will be populated by the profile step._

## Implementation Log

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Final Report

_Pending: will be populated by the report step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _VERIFY_ONLY_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Verification Results

_Pending: will be populated by the verify step._

## Final Report

_Pending: will be populated by the report step._
"""

    _SECURITY_AUDIT_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Dependency Scan Findings

_Pending: will be populated by the scan_deps step._

## Code Scan Findings

_Pending: will be populated by the scan_code step._

## Security Report

_Pending: will be populated by the report step._

## Generated Remediation Tasks

_Pending: will be populated by the generate_tasks step._
"""

    _REPO_REVIEW_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Repository Analysis

_Pending: will be populated by the analyze step._

## Initiative Plan

_Pending: will be populated by the plan step._

## Generated Tasks

_Pending: will be populated by the generate_tasks step._
"""

    _RESEARCH_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Research Analysis

_Pending: will be populated by the analyze step._

## Final Report

_Pending: will be populated by the report step._
"""

    _REVIEW_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Review Analysis

_Pending: will be populated by the analyze step._

## Review Findings

_Pending: will be populated by the review step._

## Final Report

_Pending: will be populated by the report step._
"""

    _BUG_FIX_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Reproduction Evidence

_Pending: will be populated by the reproduce step._

## Diagnosis

_Pending: will be populated by the diagnose step._

## Fix Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _REFACTOR_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Refactor Analysis

_Pending: will be populated by the analyze step._

## Refactor Plan

_Pending: will be populated by the plan step._

## Refactor Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _PERFORMANCE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Profiling Baseline

_Pending: will be populated by the profile step._

## Optimization Plan

_Pending: will be populated by the plan step._

## Optimization Implementation

_Pending: will be populated by the implement step._

## Benchmark Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _TEST_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Coverage Analysis

_Pending: will be populated by the analyze step._

## Test Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _DOCS_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Documentation Analysis

_Pending: will be populated by the analyze step._

## Documentation Updates

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _HOTFIX_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Hotfix Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _CHORE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Chore Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._
"""

    _SPIKE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Spike Analysis

_Pending: will be populated by the analyze step._

## Prototype Notes

_Pending: will be populated by the implement step._

## Final Report

_Pending: will be populated by the report step._
"""

    _PLAN_ONLY_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Analysis

_Pending: will be populated by the analyze step._

## Plan

_Pending: will be populated by the plan step._

## Generated Tasks

_Pending: will be populated by the generate_tasks step._

## Final Report

_Pending: will be populated by the report step._
"""

    def _pipeline_id_for_task(self, task: Task) -> str:
        """Resolve pipeline id for a task; fall back safely on failure."""
        try:
            return PipelineRegistry().resolve_for_task_type(task.task_type).id
        except Exception:
            return "feature"

    def _workdoc_template_for_task(self, task: Task) -> str:
        """Return the workdoc template body for the task's pipeline."""
        pipeline_id = self._pipeline_id_for_task(task)
        template_by_pipeline: dict[str, str] = {
            "feature": self._FEATURE_WORKDOC_TEMPLATE,
            "bug_fix": self._BUG_FIX_WORKDOC_TEMPLATE,
            "refactor": self._REFACTOR_WORKDOC_TEMPLATE,
            "research": self._RESEARCH_WORKDOC_TEMPLATE,
            "docs": self._DOCS_WORKDOC_TEMPLATE,
            "test": self._TEST_WORKDOC_TEMPLATE,
            "repo_review": self._REPO_REVIEW_WORKDOC_TEMPLATE,
            "security_audit": self._SECURITY_AUDIT_WORKDOC_TEMPLATE,
            "review": self._REVIEW_WORKDOC_TEMPLATE,
            "performance": self._PERFORMANCE_WORKDOC_TEMPLATE,
            "hotfix": self._HOTFIX_WORKDOC_TEMPLATE,
            "spike": self._SPIKE_WORKDOC_TEMPLATE,
            "chore": self._CHORE_WORKDOC_TEMPLATE,
            "plan_only": self._PLAN_ONLY_WORKDOC_TEMPLATE,
            "verify_only": self._VERIFY_ONLY_WORKDOC_TEMPLATE,
        }
        return template_by_pipeline.get(pipeline_id, self._GENERIC_WORKDOC_TEMPLATE)

    def _workdoc_section_for_step(self, task: Task, step: str) -> tuple[str, str | None] | None:
        """Resolve section heading/placeholder mapping for a step and task pipeline."""
        section = self._WORKDOC_SECTION_MAP.get(step)
        if not section:
            return None
        heading, placeholder_step = section
        pipeline_id = self._pipeline_id_for_task(task)
        section_overrides: dict[str, dict[str, tuple[str, str | None]]] = {
            "security_audit": {
                "report": ("## Security Report", "report"),
                "generate_tasks": ("## Generated Remediation Tasks", "generate_tasks"),
            },
            "repo_review": {
                "analyze": ("## Repository Analysis", "analyze"),
                "initiative_plan": ("## Initiative Plan", "plan"),
            },
            "research": {
                "analyze": ("## Research Analysis", "analyze"),
            },
            "review": {
                "analyze": ("## Review Analysis", "analyze"),
            },
            "bug_fix": {
                "reproduce": ("## Reproduction Evidence", "reproduce"),
                "diagnose": ("## Diagnosis", "diagnose"),
                "implement": ("## Fix Implementation", "implement"),
            },
            "refactor": {
                "analyze": ("## Refactor Analysis", "analyze"),
                "plan": ("## Refactor Plan", "plan"),
                "implement": ("## Refactor Implementation", "implement"),
            },
            "performance": {
                "plan": ("## Optimization Plan", "plan"),
                "implement": ("## Optimization Implementation", "implement"),
                "benchmark": ("## Benchmark Results", "verify"),
            },
            "test": {
                "analyze": ("## Coverage Analysis", "analyze"),
                "implement": ("## Test Implementation", "implement"),
            },
            "docs": {
                "analyze": ("## Documentation Analysis", "analyze"),
                "implement": ("## Documentation Updates", "implement"),
            },
            "hotfix": {
                "implement": ("## Hotfix Implementation", "implement"),
            },
            "chore": {
                "implement": ("## Chore Implementation", "implement"),
            },
            "spike": {
                "analyze": ("## Spike Analysis", "analyze"),
                "prototype": ("## Prototype Notes", "implement"),
            },
        }
        override = section_overrides.get(pipeline_id, {}).get(step)
        if override:
            heading, placeholder_step = override
        return heading, placeholder_step

    def _workdoc_canonical_path(self, task_id: str) -> Path:
        return self.container.state_root / "workdocs" / f"{task_id}.md"

    @staticmethod
    def _workdoc_worktree_path(project_dir: Path) -> Path:
        return project_dir / ".workdoc.md"

    def _init_workdoc(self, task: Task, project_dir: Path) -> Path:
        """Render the workdoc template, write canonical + worktree copies.

        .workdoc.md is already gitignored by bootstrap (_ensure_gitignored).
        """
        workdocs_dir = self.container.state_root / "workdocs"
        workdocs_dir.mkdir(parents=True, exist_ok=True)

        canonical = self._workdoc_canonical_path(task.id)
        # Use safe substitution to avoid KeyError if task fields contain braces.
        template = self._workdoc_template_for_task(task)
        content = (
            template
            .replace("{title}", task.title)
            .replace("{task_id}", task.id)
            .replace("{task_type}", task.task_type)
            .replace("{priority}", task.priority)
            .replace("{created_at}", task.created_at)
            .replace("{description}", task.description or "(no description)")
        )
        canonical.write_text(content, encoding="utf-8")

        worktree_copy = self._workdoc_worktree_path(project_dir)
        shutil.copy2(str(canonical), str(worktree_copy))

        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["workdoc_path"] = str(canonical)
        return canonical

    @staticmethod
    def _cleanup_workdoc_for_commit(project_dir: Path) -> None:
        """Remove the worktree .workdoc.md before commit so git add -A won't stage it."""
        workdoc = project_dir / ".workdoc.md"
        if workdoc.exists():
            workdoc.unlink()

    def _refresh_workdoc(self, task: Task, project_dir: Path) -> None:
        """Copy canonical workdoc to worktree so the worker sees the latest version."""
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return
        worktree_copy = self._workdoc_worktree_path(project_dir)
        shutil.copy2(str(canonical), str(worktree_copy))

    def _sync_workdoc(
        self, task: Task, step: str, project_dir: Path, summary: str | None, attempt: int | None = None
    ) -> None:
        """Post-step sync: accept worker changes or fallback-append summary."""
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return
        worktree_copy = self._workdoc_worktree_path(project_dir)
        if not worktree_copy.exists():
            return

        canonical_text = canonical.read_text(encoding="utf-8")
        worktree_text = worktree_copy.read_text(encoding="utf-8")

        changed = False
        orchestrator_managed_steps = {"verify", "benchmark", "reproduce", "implement_fix", "report", "profile"}
        allow_worker_workdoc_write = step not in orchestrator_managed_steps
        if worktree_text != canonical_text and allow_worker_workdoc_write:
            # Worker updated the file — accept as new canonical
            canonical.write_text(worktree_text, encoding="utf-8")
            changed = True
        elif summary and summary.strip():
            # Fallback: append summary under the step's heading
            section = self._workdoc_section_for_step(task, step)
            if not section:
                return
            heading, placeholder_step = section
            if placeholder_step:
                placeholder = f"_Pending: will be populated by the {placeholder_step} step._"
            else:
                placeholder = "_Pending: will be populated as needed._"
            trimmed = summary.strip()
            if step == "implement_fix":
                cycle_num = int(attempt or 1)
                trimmed = f"### Fix Cycle {cycle_num}\n{trimmed}"
            if placeholder in canonical_text:
                updated = canonical_text.replace(placeholder, trimmed, 1)
            else:
                # Append under the heading
                idx = canonical_text.find(heading)
                if idx == -1:
                    return
                # Find the end of the heading line
                newline_after = canonical_text.find("\n", idx)
                if newline_after == -1:
                    updated = canonical_text + "\n\n" + trimmed
                else:
                    # Find the next heading (## ) or end of file
                    rest = canonical_text[newline_after + 1:]
                    next_heading = re.search(r"^## ", rest, re.MULTILINE)
                    if next_heading:
                        insert_pos = newline_after + 1 + next_heading.start()
                        updated = canonical_text[:insert_pos] + trimmed + "\n\n" + canonical_text[insert_pos:]
                    else:
                        updated = canonical_text.rstrip() + "\n\n" + trimmed + "\n"
            canonical.write_text(updated, encoding="utf-8")
            worktree_copy.write_text(updated, encoding="utf-8")
            changed = True

        if changed:
            self.bus.emit(
                channel="tasks",
                event_type="workdoc.updated",
                entity_id=task.id,
                payload={"step": step},
            )

    def _sync_workdoc_review(self, task: Task, cycle: ReviewCycle, project_dir: Path) -> None:
        """Append review cycle findings to the workdoc."""
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return

        text = canonical.read_text(encoding="utf-8")

        lines: list[str] = [
            f"### Review Cycle {cycle.attempt} — {cycle.decision}",
        ]
        counts = cycle.open_counts or {}
        lines.append(
            "Open findings: "
            f"critical={int(counts.get('critical', 0))}, "
            f"high={int(counts.get('high', 0))}, "
            f"medium={int(counts.get('medium', 0))}, "
            f"low={int(counts.get('low', 0))}"
        )
        for f in cycle.findings:
            if f.file and f.line:
                loc = f" ({f.file}:{f.line})"
            elif f.file:
                loc = f" ({f.file})"
            else:
                loc = ""
            category = f"[{f.category}] " if f.category else ""
            lines.append(f"- **[{f.severity}]** {category}{f.summary}{loc}")
            if f.suggested_fix:
                lines.append(f"  - Suggested fix: {f.suggested_fix}")
            if f.status and f.status != "open":
                lines.append(f"  - Status: {f.status}")
        block = "\n".join(lines)

        placeholder = "_Pending: will be populated by the review step._"
        if placeholder in text:
            updated = text.replace(placeholder, block, 1)
        else:
            heading = "## Review Findings"
            idx = text.find(heading)
            if idx == -1:
                updated = text.rstrip() + "\n\n" + heading + "\n\n" + block + "\n"
            else:
                next_heading = re.search(r"^## ", text[idx + len(heading):], re.MULTILINE)
                if next_heading:
                    insert_pos = idx + len(heading) + next_heading.start()
                    updated = text[:insert_pos] + block + "\n\n" + text[insert_pos:]
                else:
                    updated = text.rstrip() + "\n\n" + block + "\n"

        canonical.write_text(updated, encoding="utf-8")
        worktree_copy = self._workdoc_worktree_path(project_dir)
        if worktree_copy.exists():
            worktree_copy.write_text(updated, encoding="utf-8")

        self.bus.emit(
            channel="tasks",
            event_type="workdoc.updated",
            entity_id=task.id,
            payload={"step": "review", "cycle": cycle.attempt},
        )

    def _step_project_dir(self, task: Task) -> Path:
        """Return the effective project dir for a task (worktree or main)."""
        worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
        return Path(worktree_path) if worktree_path else self.container.project_dir

    def get_workdoc(self, task_id: str) -> dict[str, Any]:
        """Read canonical workdoc for a task. Returns {task_id, content, exists}."""
        canonical = self._workdoc_canonical_path(task_id)
        if canonical.exists():
            return {"task_id": task_id, "content": canonical.read_text(encoding="utf-8"), "exists": True}
        return {"task_id": task_id, "content": None, "exists": False}

    def _merge_and_cleanup(self, task: Task, worktree_dir: Path) -> None:
        branch = f"task-{task.id}"
        merge_failed = False
        with self._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--no-edit"],
                    cwd=self.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                resolved = self._resolve_merge_conflict(task, branch)
                if not resolved:
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    merge_failed = True
                    task.metadata["merge_conflict"] = True
        # Always clean up worktree
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_dir), "--force"],
            cwd=self.container.project_dir,
            capture_output=True,
            text=True,
        )
        # Only delete branch if merge succeeded; preserve it for recovery on failure
        if not merge_failed:
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
            )

    def approve_and_merge(self, task: Task) -> dict[str, Any]:
        """Merge a preserved branch to the run branch on user approval.

        Called when a user approves a blocked task whose work was preserved
        on a git branch (e.g. after review-cap exceeded).
        """
        branch = task.metadata.get("preserved_branch")
        if not branch:
            return {"status": "ok"}

        # Verify the branch exists
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=self.container.project_dir,
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            task.metadata.pop("preserved_branch", None)
            self.container.tasks.upsert(task)
            return {"status": "ok"}

        self._ensure_branch()

        with self._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--no-edit"],
                    cwd=self.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                resolved = self._resolve_merge_conflict(task, branch)
                if not resolved:
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=self.container.project_dir,
                        capture_output=True, text=True,
                    )
                    task.metadata["merge_conflict"] = True
                    self.container.tasks.upsert(task)
                    return {"status": "merge_conflict"}

            # Capture SHA while still holding merge lock
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.container.project_dir,
                capture_output=True, text=True,
            ).stdout.strip()

        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=self.container.project_dir,
            capture_output=True, text=True,
        )
        task.metadata.pop("preserved_branch", None)
        task.metadata.pop("merge_conflict", None)
        self.container.tasks.upsert(task)
        return {"status": "ok", "commit_sha": sha}

    def _resolve_merge_conflict(self, task: Task, branch: str) -> bool:
        saved_worktree_dir = task.metadata.get("worktree_dir")
        try:
            # Get conflicted files
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            conflicted_files = [f for f in result.stdout.strip().split("\n") if f]
            if not conflicted_files:
                return False

            # Read conflicted file contents
            conflict_contents: dict[str, str] = {}
            for fpath in conflicted_files:
                full = self.container.project_dir / fpath
                if full.exists():
                    conflict_contents[fpath] = full.read_text(errors="replace")

            # Identify other recently completed tasks whose changes may conflict
            other_tasks_info: list[str] = []
            other_objectives: list[str] = []
            for other in self.container.tasks.list():
                if other.id != task.id and other.status == "done":
                    other_tasks_info.append(f"- {other.title}: {other.description}")
                    other_objectives.append(self._format_task_objective_summary(other))

            # Build resolve prompt and store in task metadata.
            # Temporarily clear worktree_dir so the worker runs in project_dir
            # (where the merge conflict lives), not the worktree.
            task.metadata.pop("worktree_dir", None)
            task.metadata["merge_conflict_files"] = conflict_contents
            task.metadata["merge_other_tasks"] = other_tasks_info
            task.metadata["merge_current_objective"] = self._format_task_objective_summary(task)
            task.metadata["merge_other_objectives"] = other_objectives
            self.container.tasks.upsert(task)

            # Dispatch worker to resolve
            step_result = self.worker_adapter.run_step(task=task, step="resolve_merge", attempt=1)

            if step_result.status != "ok":
                return False

            # Stage and commit the resolution
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--no-edit"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            logger.exception("Failed to resolve merge conflict for task %s", task.id)
            return False
        finally:
            # Always clean up conflict metadata and restore worktree_dir
            task.metadata.pop("merge_conflict_files", None)
            task.metadata.pop("merge_other_tasks", None)
            task.metadata.pop("merge_current_objective", None)
            task.metadata.pop("merge_other_objectives", None)
            if saved_worktree_dir:
                task.metadata["worktree_dir"] = saved_worktree_dir

    def _cleanup_orphaned_worktrees(self) -> None:
        worktrees_dir = self.container.state_root / "worktrees"
        if not worktrees_dir.exists():
            return
        if not (self.container.project_dir / ".git").exists():
            return
        # Collect branches referenced as preserved by any task
        preserved_branches: set[str] = set()
        for t in self.container.tasks.list():
            pb = t.metadata.get("preserved_branch") if isinstance(t.metadata, dict) else None
            if pb:
                preserved_branches.add(str(pb))
        for child in worktrees_dir.iterdir():
            if child.is_dir():
                branch_name = f"task-{child.name}"
                subprocess.run(
                    ["git", "worktree", "remove", str(child), "--force"],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                # Only delete branch if it's not preserved by a task
                if branch_name not in preserved_branches:
                    subprocess.run(
                        ["git", "branch", "-D", branch_name],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )

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
        if self._run_branch:
            return self._run_branch
        with self._branch_lock:
            # Double-check after acquiring lock
            if self._run_branch:
                return self._run_branch
            git_dir = self.container.project_dir / ".git"
            if not git_dir.exists():
                return None
            branch = f"orchestrator-run-{int(time.time())}"
            try:
                subprocess.run(["git", "checkout", "-B", branch], cwd=self.container.project_dir, check=True, capture_output=True, text=True)
                self._run_branch = branch
                return branch
            except subprocess.CalledProcessError:
                return None

    def _commit_for_task(self, task: Task, working_dir: Optional[Path] = None) -> Optional[str]:
        cwd = working_dir or self.container.project_dir
        if not (cwd / ".git").exists() and not (self.container.project_dir / ".git").exists():
            return None
        if working_dir is None:
            self._ensure_branch()
        try:
            subprocess.run(["git", "add", "-A"], cwd=cwd, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "commit", "-m", f"task({task.id}): {task.title[:60]}"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
            return sha
        except subprocess.CalledProcessError:
            return None

    def _has_uncommitted_changes(self, cwd: Path) -> bool:
        """Return True if the working tree has staged or unstaged changes.

        Returns True on git failure (no repo, etc.) to avoid false blocking.
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd, capture_output=True, text=True, check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return True

    def _preserve_worktree_work(self, task: Task, worktree_dir: Path) -> bool:
        """Commit agent edits in the worktree, remove the worktree but keep the branch.

        Returns True if a branch was preserved with work on it.
        """
        branch = f"task-{task.id}"
        try:
            self._cleanup_workdoc_for_commit(worktree_dir)
            self._commit_for_task(task, worktree_dir)

            # Check if the branch has any commits beyond the run branch base
            base_ref = self._run_branch or "HEAD"
            try:
                result = subprocess.run(
                    ["git", "log", f"{base_ref}..{branch}", "--oneline"],
                    cwd=self.container.project_dir,
                    capture_output=True, text=True, check=True,
                )
                if not result.stdout.strip():
                    return False
            except subprocess.CalledProcessError:
                # If we can't compare, assume there's work to preserve
                pass

            # Remove worktree directory but keep the branch
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_dir), "--force"],
                cwd=self.container.project_dir,
                capture_output=True, text=True,
            )
            task.metadata["preserved_branch"] = branch
            self.container.tasks.upsert(task)
            return True
        except Exception:
            logger.exception("Failed to preserve worktree work for task %s", task.id)
            return False

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
        """Return True when a verify failure is due to environment/tooling, not code."""
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
        created_ids: list[str] = []
        for item in task_defs:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority") or parent.priority)
            if priority not in {"P0", "P1", "P2", "P3"}:
                priority = parent.priority
            child = Task(
                title=str(item.get("title") or "Generated task"),
                description=str(item.get("description") or ""),
                task_type=str(item.get("task_type") or "feature"),
                priority=cast(Priority, priority),
                parent_id=parent.id,
                source="generated",
                labels=list(item.get("labels") or []),
                metadata=dict(item.get("metadata") or {}),
            )
            self.container.tasks.upsert(child)
            created_ids.append(child.id)
            self.bus.emit(
                channel="tasks",
                event_type="task.created",
                entity_id=child.id,
                payload={"parent_id": parent.id, "source": "generate_tasks"},
            )

        # Wire up depends_on task-ID references between generated tasks
        if apply_deps and created_ids:
            generated_ref_to_task_id: dict[str, str] = {}
            for idx, item in enumerate(task_defs):
                if not isinstance(item, dict) or idx >= len(created_ids):
                    continue
                ref = str(item.get("id") or "").strip()
                if ref and ref not in generated_ref_to_task_id:
                    generated_ref_to_task_id[ref] = created_ids[idx]
            for idx, item in enumerate(task_defs):
                if not isinstance(item, dict) or idx >= len(created_ids):
                    continue
                deps = item.get("depends_on")
                if not isinstance(deps, list):
                    continue
                child_id = created_ids[idx]
                child_task = self.container.tasks.get(child_id)
                if not child_task:
                    continue
                for dep_ref in deps:
                    dep_key = str(dep_ref or "").strip()
                    if not dep_key:
                        continue
                    dep_id = generated_ref_to_task_id.get(dep_key)
                    if not dep_id:
                        continue
                    if dep_id == child_id:
                        continue
                    if dep_id not in child_task.blocked_by:
                        child_task.blocked_by.append(dep_id)
                    dep_task = self.container.tasks.get(dep_id)
                    if dep_task and child_id not in dep_task.blocks:
                        dep_task.blocks.append(child_id)
                        self.container.tasks.upsert(dep_task)
                self.container.tasks.upsert(child_task)

        if created_ids:
            parent.children_ids.extend(created_ids)
            self.container.tasks.upsert(parent)
        return created_ids

    def generate_tasks_from_plan(
        self, task_id: str, plan_text: str, *, infer_deps: bool = True
    ) -> list[str]:
        """Generate child tasks from an explicit plan text.

        This supports a two-phase workflow: run a plan step, review the output,
        then explicitly trigger task generation from the plan.
        """
        task = self.container.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Inject the plan so the worker prompt can include it
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["plan_for_generation"] = plan_text
        self.container.tasks.upsert(task)

        try:
            result = self.worker_adapter.run_step(task=task, step="generate_tasks", attempt=1)
            if result.status != "ok":
                raise ValueError(f"generate_tasks step failed: {result.summary or result.status}")
            task_defs = list(result.generated_tasks or [])
            if not task_defs:
                detail = str(result.summary or "").strip()
                if detail:
                    raise ValueError(f"Worker returned no generated tasks: {detail}")
                raise ValueError("Worker returned no generated tasks for the selected plan source")
            created_ids = self._create_child_tasks(task, task_defs, apply_deps=infer_deps)
            if not created_ids:
                raise ValueError("Worker returned generated tasks, but none were valid task objects")
        finally:
            # Clean up the injected plan context
            task.metadata.pop("plan_for_generation", None)
            self.container.tasks.upsert(task)

        return created_ids

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
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if not orchestrator_cfg.get("auto_deps", True):
            return

        all_tasks = self.container.tasks.list()
        candidates = [
            t for t in all_tasks
            if t.status == "queued"
            and not (isinstance(t.metadata, dict) and t.metadata.get("deps_analyzed"))
            and t.source != "prd_import"
        ]

        # Mark all candidates analyzed regardless of outcome
        def _mark_analyzed(tasks: list[Task]) -> None:
            for t in tasks:
                if not isinstance(t.metadata, dict):
                    t.metadata = {}
                t.metadata["deps_analyzed"] = True
                self.container.tasks.upsert(t)

        if len(candidates) < 2:
            _mark_analyzed(candidates)
            return

        # Gather already-analyzed non-terminal tasks as context
        terminal = {"done", "cancelled"}
        existing = [
            t for t in all_tasks
            if isinstance(t.metadata, dict) and t.metadata.get("deps_analyzed")
            and t.status not in terminal
        ]

        # Build synthetic task with metadata for the worker
        candidate_data = [
            {
                "id": t.id,
                "title": t.title,
                "description": (t.description or "")[:200],
                "task_type": t.task_type,
                "labels": t.labels,
            }
            for t in candidates
        ]
        existing_data = [
            {"id": t.id, "title": t.title, "status": t.status}
            for t in existing
        ]

        synthetic = Task(
            title="Dependency analysis",
            description="Analyze task dependencies",
            task_type="research",
            source="system",
            metadata={
                "candidate_tasks": candidate_data,
                "existing_tasks": existing_data,
            },
        )

        try:
            result = self.worker_adapter.run_step_ephemeral(task=synthetic, step="analyze_deps", attempt=1)
            if result.status == "ok" and result.dependency_edges:
                self._apply_dependency_edges(candidates, result.dependency_edges, all_tasks)
        except Exception:
            logger.exception("Dependency analysis failed; tasks will run without inferred deps")
        finally:
            _mark_analyzed(candidates)

    def _apply_dependency_edges(
        self,
        candidates: list[Task],
        edges: list[dict[str, str]],
        all_tasks: list[Task],
    ) -> None:
        """Apply inferred dependency edges with cycle detection."""
        task_map: dict[str, Task] = {}
        # All tasks as context for resolving IDs outside candidate set
        for t in all_tasks:
            task_map[t.id] = t
        # Overlay candidate objects (same Python objects that _mark_analyzed will touch)
        for t in candidates:
            task_map[t.id] = t

        # Build adjacency list from existing blocked_by relationships
        adj: dict[str, list[str]] = {}
        for t in task_map.values():
            for dep_id in t.blocked_by:
                adj.setdefault(dep_id, []).append(t.id)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            reason = edge.get("reason", "")

            if not from_id or not to_id:
                continue
            if from_id not in task_map or to_id not in task_map:
                continue
            if from_id == to_id:
                continue

            # Cycle check
            if _has_cycle(adj, from_id, to_id):
                logger.warning("Skipping edge %s→%s: would create cycle", from_id, to_id)
                continue

            from_task = task_map[from_id]
            to_task = task_map[to_id]

            if from_id not in to_task.blocked_by:
                to_task.blocked_by.append(from_id)
            if to_id not in from_task.blocks:
                from_task.blocks.append(to_id)

            # Store inferred deps for traceability
            if not isinstance(to_task.metadata, dict):
                to_task.metadata = {}
            inferred = to_task.metadata.setdefault("inferred_deps", [])
            inferred.append({"from": from_id, "reason": reason})

            # Update adjacency for subsequent cycle checks
            adj.setdefault(from_id, []).append(to_id)

            self.container.tasks.upsert(from_task)
            self.container.tasks.upsert(to_task)
            self.bus.emit(
                channel="tasks",
                event_type="task.dependency_inferred",
                entity_id=to_id,
                payload={"from": from_id, "to": to_id, "reason": reason},
            )

    class _Cancelled(Exception):
        """Raised when a task is cancelled mid-execution."""

    def _check_cancelled(self, task: Task) -> None:
        """Re-read task from storage and raise _Cancelled if user cancelled it."""
        fresh = self.container.tasks.get(task.id)
        if fresh and fresh.status == "cancelled":
            raise self._Cancelled()

    def _execute_task(self, task: Task) -> None:
        try:
            self._execute_task_inner(task)
        except self._Cancelled:
            logger.info("Task %s was cancelled by user", task.id)
            # Ensure the persisted status stays cancelled
            fresh = self.container.tasks.get(task.id)
            if fresh:
                if fresh.status != "cancelled":
                    fresh.status = "cancelled"
                    self.container.tasks.upsert(fresh)
                # Finalize any active run
                for run_id in reversed(fresh.run_ids):
                    run = self.container.runs.get(run_id)
                    if run and run.status == "in_progress":
                        run.status = "cancelled"
                        run.finished_at = now_iso()
                        run.summary = "Cancelled by user"
                        self.container.runs.upsert(run)
                        break
                self.bus.emit(
                    channel="tasks",
                    event_type="task.cancelled",
                    entity_id=fresh.id,
                    payload={"status": "cancelled"},
                )
        except Exception:
            logger.exception("Unexpected error executing task %s", task.id)
            task.status = "blocked"
            task.error = "Internal error during execution"
            self.container.tasks.upsert(task)

    def _execute_task_inner(self, task: Task) -> None:
        worktree_dir: Optional[Path] = None
        try:
            # Clean up any preserved branch from a previous run (e.g. retry
            # after review-cap block or cancellation) so _create_worktree can
            # create a fresh branch with the same name.
            old_preserved = task.metadata.pop("preserved_branch", None)
            if old_preserved:
                subprocess.run(
                    ["git", "branch", "-D", str(old_preserved)],
                    cwd=self.container.project_dir,
                    capture_output=True, text=True,
                )
                task.metadata.pop("merge_conflict", None)
                self.container.tasks.upsert(task)

            worktree_dir = self._create_worktree(task)
            if worktree_dir:
                task.metadata["worktree_dir"] = str(worktree_dir)
                self.container.tasks.upsert(task)

            workdoc_dir = worktree_dir if worktree_dir else self.container.project_dir
            self._init_workdoc(task, workdoc_dir)

            task_branch = f"task-{task.id}" if worktree_dir else self._ensure_branch()
            run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=task_branch)
            run.steps = []
            self.container.runs.upsert(run)

            cfg = self.container.config.load()
            orch_cfg = dict(cfg.get("orchestrator") or {})
            max_review_attempts = int(orch_cfg.get("max_review_attempts", 10) or 10)
            max_verify_fix_attempts = int(orch_cfg.get("max_verify_fix_attempts", 3) or 3)

            # Resolve pipeline template from registry
            registry = PipelineRegistry()
            template = registry.resolve_for_task_type(task.task_type)
            steps = task.pipeline_template if task.pipeline_template else template.step_names()
            task.pipeline_template = steps
            has_review = "review" in steps
            has_commit = "commit" in steps

            # Determine resume point for retries.
            retry_from = ""
            if isinstance(task.metadata, dict):
                retry_from = str(task.metadata.pop("retry_from_step", "") or "").strip()

            task.run_ids.append(run.id)
            task.current_step = steps[0] if steps else None
            task.metadata["pipeline_phase"] = steps[0] if steps else None
            task.status = "in_progress"
            task.current_agent_id = self._choose_agent_for_task(task)
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.started",
                entity_id=task.id,
                payload={"run_id": run.id, "agent_id": task.current_agent_id},
            )

            mode = getattr(task, "hitl_mode", "autopilot") or "autopilot"

            # Phase 1: Run all pre-review/pre-commit steps
            # When retry_from is set, skip steps before the resume point.
            # If retry_from targets review or commit, skip all phase-1 steps.
            skip_phase1 = retry_from in ("review", "commit")
            reached_retry_step = not retry_from or skip_phase1
            last_phase1_step: str | None = None
            for step in steps:
                if step in ("review", "commit"):
                    continue
                if not reached_retry_step:
                    if step == retry_from:
                        reached_retry_step = True
                    else:
                        last_phase1_step = step
                        run.steps.append({"step": step, "status": "skipped", "ts": now_iso()})
                        self.container.runs.upsert(run)
                        continue
                self._check_cancelled(task)
                task.current_step = step
                task.metadata["pipeline_phase"] = step
                self.container.tasks.upsert(task)
                gate_name = self._GATE_MAPPING.get(step)
                if gate_name and should_gate(mode, gate_name):
                    if not self._wait_for_gate(task, gate_name):
                        self._abort_for_gate(task, run, gate_name)
                        return
                if not self._run_non_review_step(task, run, step, attempt=1):
                    # If a verify step failed, try implement_fix → verify loop.
                    if step in _VERIFY_STEPS:
                        if self._consume_verify_non_actionable_flag(task):
                            last_phase1_step = step
                            continue
                        fixed = False
                        for fix_attempt in range(1, max_verify_fix_attempts + 1):
                            # Stash the failure summary and log output so implement_fix sees it.
                            task.status = "in_progress"
                            task.metadata["verify_failure"] = task.error
                            self._capture_verify_output(task)
                            task.error = None
                            task.retry_count += 1
                            self.container.tasks.upsert(task)
                            run.status = "in_progress"
                            run.finished_at = None
                            run.summary = None
                            self.container.runs.upsert(run)

                            task.current_step = "implement_fix"
                            task.metadata["pipeline_phase"] = step
                            self.container.tasks.upsert(task)
                            if not self._run_non_review_step(task, run, "implement_fix", attempt=fix_attempt + 1):
                                return
                            task.metadata.pop("verify_failure", None)
                            task.metadata.pop("verify_output", None)
                            task.current_step = step
                            task.metadata["pipeline_phase"] = step
                            self.container.tasks.upsert(task)
                            if self._run_non_review_step(task, run, step, attempt=fix_attempt + 1):
                                fixed = True
                                break
                            # verify failed again — loop continues
                        if fixed:
                            last_phase1_step = step
                            continue
                        # All fix attempts exhausted — task stays blocked from last verify fail.
                    return
                last_phase1_step = step

            # Guard: block if implementation produced no file changes
            if has_commit:
                impl_dir = worktree_dir or self.container.project_dir
                # Remove worktree workdoc before checking — it's not a real change.
                self._cleanup_workdoc_for_commit(impl_dir)
                if not self._has_uncommitted_changes(impl_dir):
                    task.status = "blocked"
                    task.error = "No file changes detected after implementation"
                    task.current_step = last_phase1_step or "implement"
                    task.metadata["pipeline_phase"] = last_phase1_step or "implement"
                    self.container.tasks.upsert(task)
                    self._finalize_run(task, run, status="blocked", summary="Blocked: no changes produced by implementation steps")
                    self.bus.emit(
                        channel="tasks", event_type="task.blocked",
                        entity_id=task.id, payload={"error": task.error},
                    )
                    return

            # Phase 2: Review loop (only if template has "review")
            self._check_cancelled(task)
            if has_review and retry_from != "commit":
                post_fix_validation_step = self._select_post_fix_validation_step(steps)
                gate_name = self._GATE_MAPPING.get("review")
                if gate_name and should_gate(mode, gate_name):
                    if not self._wait_for_gate(task, gate_name):
                        self._abort_for_gate(task, run, gate_name)
                        return

                review_attempt = 0
                review_passed = False

                while review_attempt < max_review_attempts:
                    self._check_cancelled(task)
                    review_attempt += 1
                    task.current_step = "review"
                    task.metadata["pipeline_phase"] = "review"
                    if review_attempt > 1:
                        task.metadata["review_history"] = self._build_review_history_summary(task.id)
                    else:
                        task.metadata.pop("review_history", None)
                    self.container.tasks.upsert(task)
                    review_started = now_iso()
                    findings, review_result = self._findings_from_result(task, review_attempt)
                    # Build step log immediately so log paths are always stored
                    # in run.steps (same pattern as _run_non_review_step).
                    review_step_log: dict[str, Any] = {"step": "review", "status": "ok", "ts": now_iso(), "started_at": review_started}
                    review_last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
                    if isinstance(review_last_logs, dict):
                        for key in ("stdout_path", "stderr_path", "progress_path"):
                            if review_last_logs.get(key):
                                review_step_log[key] = review_last_logs[key]
                    if review_result.human_blocking_issues:
                        review_step_log["status"] = "blocked"
                        review_step_log["human_blocking_issues"] = review_result.human_blocking_issues
                        run.steps.append(review_step_log)
                        self.container.runs.upsert(run)
                        self._block_for_human_issues(
                            task,
                            run,
                            "review",
                            review_result.summary,
                            review_result.human_blocking_issues,
                        )
                        return
                    if review_result.status != "ok":
                        review_step_log["status"] = review_result.status or "error"
                        run.steps.append(review_step_log)
                        self.container.runs.upsert(run)
                        task.status = "blocked"
                        task.error = review_result.summary or "Review step failed"
                        task.pending_gate = None
                        task.current_step = "review"
                        task.metadata["pipeline_phase"] = "review"
                        self.container.tasks.upsert(task)
                        self._finalize_run(task, run, status="blocked", summary="Blocked during review")
                        self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
                        return
                    open_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                    for finding in findings:
                        if finding.status == "open" and finding.severity in open_counts:
                            open_counts[finding.severity] += 1
                    cycle = ReviewCycle(
                        task_id=task.id,
                        attempt=review_attempt,
                        findings=findings,
                        open_counts=open_counts,
                        decision="changes_requested" if self._exceeds_quality_gate(task, findings) else "approved",
                    )
                    self.container.reviews.append(cycle)
                    self._sync_workdoc_review(task, cycle, worktree_dir or self.container.project_dir)
                    review_step_log["status"] = cycle.decision
                    review_step_log["open_counts"] = open_counts
                    run.steps.append(review_step_log)
                    self.container.runs.upsert(run)
                    self.bus.emit(
                        channel="review",
                        event_type="task.reviewed",
                        entity_id=task.id,
                        payload={"attempt": review_attempt, "decision": cycle.decision, "open_counts": open_counts},
                    )

                    if cycle.decision == "approved":
                        review_passed = True
                        break

                    if review_attempt >= max_review_attempts:
                        break

                    # Attach open findings and review history so the worker knows what to fix
                    open_findings = [f.to_dict() for f in findings if f.status == "open"]
                    task.metadata["review_findings"] = open_findings
                    task.metadata["review_history"] = self._build_review_history_summary(task.id)

                    # Run implement_fix
                    task.retry_count += 1
                    task.current_step = "implement_fix"
                    task.metadata["pipeline_phase"] = "review"
                    self.container.tasks.upsert(task)
                    if not self._run_non_review_step(task, run, "implement_fix", attempt=review_attempt):
                        return

                    if post_fix_validation_step:
                        # Run post-fix validation with retry loop (same as Phase 1)
                        task.current_step = post_fix_validation_step
                        task.metadata["pipeline_phase"] = "review"
                        self.container.tasks.upsert(task)
                        if not self._run_non_review_step(task, run, post_fix_validation_step, attempt=review_attempt):
                            if self._consume_verify_non_actionable_flag(task):
                                task.current_step = "review"
                                task.metadata["pipeline_phase"] = "review"
                                self.container.tasks.upsert(task)
                                continue
                            validation_fixed = False
                            for vfix in range(1, max_verify_fix_attempts + 1):
                                task.status = "in_progress"
                                task.metadata["verify_failure"] = task.error
                                self._capture_verify_output(task)
                                task.error = None
                                task.retry_count += 1
                                self.container.tasks.upsert(task)
                                run.status = "in_progress"
                                run.finished_at = None
                                run.summary = None
                                self.container.runs.upsert(run)

                                task.current_step = "implement_fix"
                                task.metadata["pipeline_phase"] = "review"
                                self.container.tasks.upsert(task)
                                if not self._run_non_review_step(task, run, "implement_fix", attempt=vfix + 1):
                                    return
                                task.metadata.pop("verify_failure", None)
                                task.metadata.pop("verify_output", None)
                                task.current_step = post_fix_validation_step
                                task.metadata["pipeline_phase"] = "review"
                                self.container.tasks.upsert(task)
                                if self._run_non_review_step(task, run, post_fix_validation_step, attempt=vfix + 1):
                                    validation_fixed = True
                                    break
                            if not validation_fixed:
                                return

                    task.metadata.pop("review_findings", None)
                    task.metadata.pop("review_history", None)
                    task.metadata.pop("verify_environment_note", None)

                if not review_passed:
                    task.metadata.pop("review_history", None)
                    task.metadata.pop("verify_environment_note", None)
                    if worktree_dir and worktree_dir.exists():
                        if self._preserve_worktree_work(task, worktree_dir):
                            worktree_dir = None  # prevent double-cleanup in finally
                    task.status = "blocked"
                    task.error = "Review attempt cap exceeded"
                    task.current_step = "review"
                    task.metadata["pipeline_phase"] = "review"
                    self.container.tasks.upsert(task)
                    self._finalize_run(task, run, status="blocked", summary="Blocked due to unresolved review findings")
                    self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
                    return

            # Phase 3: Commit (only if template has "commit")
            self._check_cancelled(task)
            if has_commit:
                task.current_step = "commit"
                task.metadata["pipeline_phase"] = "commit"
                self.container.tasks.upsert(task)
                if should_gate(mode, "before_commit"):
                    if not self._wait_for_gate(task, "before_commit"):
                        self._abort_for_gate(task, run, "before_commit")
                        return

                commit_started = now_iso()
                self._cleanup_workdoc_for_commit(worktree_dir or self.container.project_dir)
                commit_sha = self._commit_for_task(task, worktree_dir)
                if not commit_sha:
                    # Only block when git is present — otherwise commit is a no-op
                    commit_cwd = worktree_dir or self.container.project_dir
                    git_present = (commit_cwd / ".git").exists() or (self.container.project_dir / ".git").exists()
                    if git_present:
                        task.status = "blocked"
                        task.error = "Commit failed (no changes to commit)"
                        self.container.tasks.upsert(task)
                        self._finalize_run(task, run, status="blocked", summary="Blocked: commit produced no changes")
                        self.bus.emit(
                            channel="tasks", event_type="task.blocked",
                            entity_id=task.id, payload={"error": task.error},
                        )
                        return
                run.steps.append({"step": "commit", "status": "ok", "ts": now_iso(), "started_at": commit_started, "commit": commit_sha})
                self.container.runs.upsert(run)

                # Merge worktree branch back to run branch
                if worktree_dir:
                    self._merge_and_cleanup(task, worktree_dir)
                    worktree_dir = None  # prevent double-cleanup in finally

                # If merge conflict couldn't be resolved, block the task
                if task.metadata.get("merge_conflict"):
                    task.status = "blocked"
                    task.error = "Merge conflict could not be resolved automatically"
                    task.metadata["preserved_branch"] = f"task-{task.id}"
                    self.container.tasks.upsert(task)
                    self._finalize_run(task, run, status="blocked", summary="Blocked due to unresolved merge conflict")
                    self.bus.emit(
                        channel="tasks",
                        event_type="task.blocked",
                        entity_id=task.id,
                        payload={"error": task.error},
                    )
                    return

                self._run_summarize_step(task, run)
                if task.approval_mode == "auto_approve":
                    task.status = "done"
                    task.current_step = None
                    task.metadata.pop("pipeline_phase", None)
                    run.status = "done"
                    run.summary = "Completed with auto-approve"
                    self.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={"commit": commit_sha})
                else:
                    task.status = "in_review"
                    task.current_step = None
                    task.metadata.pop("pipeline_phase", None)
                    run.status = "in_review"
                    run.summary = "Awaiting human review"
                    self.bus.emit(channel="review", event_type="task.awaiting_human", entity_id=task.id, payload={"commit": commit_sha})
            else:
                # Templates without commit (research, repo_review, security_audit, review)
                # Clean up worktree if present — no merge needed for non-commit pipelines
                if worktree_dir:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "branch", "-D", f"task-{task.id}"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    worktree_dir = None  # prevent double-cleanup in finally

                self._run_summarize_step(task, run)
                task.status = "done"
                task.current_step = None
                task.metadata.pop("pipeline_phase", None)
                run.status = "done"
                run.summary = "Pipeline completed"
                self.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={})

            task.error = None
            task.metadata.pop("step_outputs", None)
            task.metadata.pop("worktree_dir", None)
            self.container.tasks.upsert(task)
            run.finished_at = now_iso()
            self.container.runs.upsert(run)
        finally:
            if worktree_dir and worktree_dir.exists():
                preserved = task.metadata.get("preserved_branch")
                if not preserved:
                    # Safety net: try to commit and preserve work that would be lost
                    try:
                        if self._has_uncommitted_changes(worktree_dir):
                            self._preserve_worktree_work(task, worktree_dir)
                            preserved = task.metadata.get("preserved_branch")
                    except Exception:
                        pass  # best-effort; fall through to full cleanup
                if preserved:
                    # Branch preserved — remove worktree dir only, keep branch
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=self.container.project_dir,
                        capture_output=True, text=True,
                    )
                else:
                    # Nothing to preserve — full cleanup (worktree + branch)
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=self.container.project_dir,
                        capture_output=True, text=True,
                    )
                    subprocess.run(
                        ["git", "branch", "-D", f"task-{task.id}"],
                        cwd=self.container.project_dir,
                        capture_output=True, text=True,
                    )
            if task.metadata.pop("worktree_dir", None):
                self.container.tasks.upsert(task)


def create_orchestrator(
    container: Container,
    bus: EventBus,
    *,
    worker_adapter: WorkerAdapter | None = None,
) -> OrchestratorService:
    """Build, start, and return an orchestrator instance for a container."""
    if worker_adapter is None:
        from .live_worker_adapter import LiveWorkerAdapter

        worker_adapter = LiveWorkerAdapter(container)
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
