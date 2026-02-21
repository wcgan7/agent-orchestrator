"""Plan revision, refinement, and generation helpers for orchestrator tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ..domain.models import PlanRefineJob, PlanRevision, PlanRevisionStatus, Priority, now_iso
import logging

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)


class PlanManager:
    """Manage plan documents, refine jobs, and task generation from plans."""

    def __init__(self, service: OrchestratorService) -> None:
        """Initialize the manager with orchestrator dependencies."""
        self._service = service

    def active_plan_refine_job(self, task_id: str) -> PlanRefineJob | None:
        """Return the active queued/running refine job for a task if present."""
        svc = self._service
        for job in svc.container.plan_refine_jobs.for_task(task_id):
            if job.status in {"queued", "running"}:
                return job
        return None

    def get_plan_document(self, task_id: str) -> dict[str, Any]:
        """Build plan-revision state plus active refine-job metadata for a task."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        revisions = svc.container.plan_revisions.for_task(task_id)
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
                svc.container.tasks.upsert(task)
        active_job = self.active_plan_refine_job(task_id)
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
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        body = str(content or "").strip()
        if not body:
            raise ValueError("Plan revision content cannot be empty")
        revisions = svc.container.plan_revisions.for_task(task_id)
        revisions.sort(key=lambda item: item.created_at)
        if parent_revision_id:
            parent = svc.container.plan_revisions.get(parent_revision_id)
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
        svc.container.plan_revisions.upsert(revision)
        task.metadata["latest_plan_revision_id"] = revision.id
        if status == "committed":
            task.metadata["committed_plan_revision_id"] = revision.id
        svc.container.tasks.upsert(task)
        svc.bus.emit(
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
        svc = self._service
        with svc._lock:
            task = svc.container.tasks.get(task_id)
            if not task:
                raise ValueError("Task not found")
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            if self.active_plan_refine_job(task_id):
                raise RuntimeError("A plan refine job is already active for this task")
            revisions = svc.container.plan_revisions.for_task(task_id)
            revisions.sort(key=lambda item: item.created_at)
            if not revisions:
                raise ValueError("No plan revision exists for this task")
            normalized_feedback = str(feedback or "").strip()
            if not normalized_feedback:
                raise ValueError("feedback is required")
            if base_revision_id:
                base_revision = svc.container.plan_revisions.get(base_revision_id)
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
            svc.container.plan_refine_jobs.upsert(job)
            svc.bus.emit(
                channel="tasks",
                event_type="plan.refine.queued",
                entity_id=task_id,
                payload={"job_id": job.id, "base_revision_id": job.base_revision_id},
            )
        future = svc._get_pool().submit(self.process_plan_refine_job, job.id)
        with svc._futures_lock:
            svc._futures[job.id] = future
        return job

    def process_plan_refine_job(self, job_id: str) -> PlanRefineJob | None:
        """Execute one queued plan-refine job to completion."""
        svc = self._service
        job = svc.container.plan_refine_jobs.get(job_id)
        if not job:
            return None
        if job.status not in {"queued", "running"}:
            return job
        job.status = "running"
        job.started_at = now_iso()
        svc.container.plan_refine_jobs.upsert(job)
        try:
            svc.bus.emit(
                channel="tasks",
                event_type="plan.refine.started",
                entity_id=job.task_id,
                payload={"job_id": job.id},
            )
        except Exception:
            logger.exception("Failed to emit plan.refine.started for job %s", job.id)

        task = svc.container.tasks.get(job.task_id)
        base_revision = svc.container.plan_revisions.get(job.base_revision_id)
        if not task or not base_revision or base_revision.task_id != job.task_id:
            job.status = "failed"
            job.finished_at = now_iso()
            job.error = "Task or base revision not found"
            svc.container.plan_refine_jobs.upsert(job)
            svc.bus.emit(
                channel="tasks",
                event_type="plan.refine.failed",
                entity_id=job.task_id,
                payload={"job_id": job.id, "error": job.error},
            )
            return job

        live_task = svc.container.tasks.get(job.task_id)
        if not live_task:
            job.status = "failed"
            job.finished_at = now_iso()
            job.error = "Task not found"
            svc.container.plan_refine_jobs.upsert(job)
            svc.bus.emit(
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
        svc.container.tasks.upsert(live_task)

        try:
            try:
                result = svc.worker_adapter.run_step(task=live_task, step=refine_step, attempt=1)
                if result.status != "ok":
                    raise ValueError(result.summary or f"{refine_step} failed")
                revised_plan = str(result.summary or "").strip()
                if not revised_plan:
                    raise ValueError("Worker returned empty refined plan")
                provider, model = svc._resolve_worker_lineage(live_task, refine_step)
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
                svc.container.plan_refine_jobs.upsert(job)
                try:
                    svc.bus.emit(
                        channel="tasks",
                        event_type="plan.refine.failed",
                        entity_id=job.task_id,
                        payload={"job_id": job.id, "error": job.error},
                    )
                except Exception:
                    logger.exception("Failed to emit plan.refine.failed for job %s", job.id)
                return job

            refreshed = svc.container.tasks.get(live_task.id)
            if refreshed and isinstance(refreshed.metadata, dict):
                live_task.metadata = dict(refreshed.metadata)
            job.status = "completed"
            job.finished_at = now_iso()
            job.result_revision_id = revision.id
            job.error = None
            svc.container.plan_refine_jobs.upsert(job)
            try:
                svc.bus.emit(
                    channel="tasks",
                    event_type="plan.refine.completed",
                    entity_id=job.task_id,
                    payload={"job_id": job.id, "result_revision_id": revision.id},
                )
            except Exception:
                logger.exception("Failed to emit plan.refine.completed for job %s", job.id)
            return job
        finally:
            cleanup_task = svc.container.tasks.get(job.task_id)
            if cleanup_task and isinstance(cleanup_task.metadata, dict):
                cleanup_task.metadata.pop(f"{refine_key}_base", None)
                cleanup_task.metadata.pop(f"{refine_key}_feedback", None)
                cleanup_task.metadata.pop(f"{refine_key}_instructions", None)
                svc.container.tasks.upsert(cleanup_task)

    def list_plan_refine_jobs(self, task_id: str) -> list[PlanRefineJob]:
        """List refine jobs for a task."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        return svc.container.plan_refine_jobs.for_task(task_id)

    def get_plan_refine_job(self, task_id: str, job_id: str) -> PlanRefineJob:
        """Fetch one refine job and verify it belongs to the given task."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        job = svc.container.plan_refine_jobs.get(job_id)
        if not job or job.task_id != task_id:
            raise ValueError("Plan refine job not found")
        return job

    def commit_plan_revision(self, task_id: str, revision_id: str) -> str:
        """Mark one plan revision as committed and sync task/workdoc metadata."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        target = svc.container.plan_revisions.get(revision_id)
        if not target or target.task_id != task_id:
            raise ValueError("Revision not found for task")
        for revision in svc.container.plan_revisions.for_task(task_id):
            next_status = "committed" if revision.id == revision_id else "draft"
            if revision.status != next_status:
                revision.status = cast(PlanRevisionStatus, next_status)
                svc.container.plan_revisions.upsert(revision)
        task.metadata["latest_plan_revision_id"] = revision_id
        task.metadata["committed_plan_revision_id"] = revision_id
        so = task.metadata.setdefault("step_outputs", {})
        so["plan"] = (target.content or "")[:20_000]
        svc.container.tasks.upsert(task)

        workdoc_path = task.metadata.get("workdoc_path") if isinstance(task.metadata, dict) else None
        canonical = (
            Path(workdoc_path)
            if isinstance(workdoc_path, str) and workdoc_path.strip()
            else svc._workdoc_canonical_path(task.id)
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
                    step_project_dir = svc._step_project_dir(task)
                    worktree_copy = svc._workdoc_worktree_path(step_project_dir)
                    if worktree_copy.exists():
                        worktree_copy.write_text(updated, encoding="utf-8")
                    svc.bus.emit(
                        channel="tasks",
                        event_type="workdoc.updated",
                        entity_id=task.id,
                        payload={"step": "plan", "source": "plan_commit"},
                    )

        svc.bus.emit(
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
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        revisions = svc.container.plan_revisions.for_task(task_id)
        revisions.sort(key=lambda item: item.created_at)

        if source == "override":
            body = str(plan_override or "").strip()
            if not body:
                raise ValueError("plan_override is required for source=override")
            return body, None

        if source == "revision":
            if not revision_id:
                raise ValueError("revision_id is required for source=revision")
            revision = svc.container.plan_revisions.get(revision_id)
            if not revision or revision.task_id != task_id:
                raise ValueError("Revision not found for task")
            return revision.content, revision.id

        if source == "committed":
            committed_id = str(task.metadata.get("committed_plan_revision_id") or "").strip()
            revision = svc.container.plan_revisions.get(committed_id) if committed_id else None
            if not revision or revision.task_id != task_id:
                committed = [item for item in revisions if str(item.status or "").lower() == "committed"]
                revision = committed[-1] if committed else None
                if revision:
                    task.metadata["committed_plan_revision_id"] = revision.id
                    task.metadata["latest_plan_revision_id"] = revisions[-1].id if revisions else revision.id
                    svc.container.tasks.upsert(task)
            if not revision or revision.task_id != task_id:
                raise ValueError("No committed plan revision exists for this task")
            return revision.content, revision.id

        if not revisions:
            raise ValueError("No plan revision exists for this task")
        return revisions[-1].content, revisions[-1].id

    def create_child_tasks(
        self,
        parent: Any,
        task_defs: list[dict[str, Any]],
        *,
        apply_deps: bool = False,
    ) -> list[str]:
        """Create generated child tasks and optionally wire dependency edges."""
        svc = self._service
        created_ids: list[str] = []
        for item in task_defs:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority") or parent.priority)
            if priority not in {"P0", "P1", "P2", "P3"}:
                priority = parent.priority
            from ..domain.models import Task

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
            svc.container.tasks.upsert(child)
            created_ids.append(child.id)
            svc.bus.emit(
                channel="tasks",
                event_type="task.created",
                entity_id=child.id,
                payload={"parent_id": parent.id, "source": "generate_tasks"},
            )

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
                child_task = svc.container.tasks.get(child_id)
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
                    dep_task = svc.container.tasks.get(dep_id)
                    if dep_task and child_id not in dep_task.blocks:
                        dep_task.blocks.append(child_id)
                        svc.container.tasks.upsert(dep_task)
                svc.container.tasks.upsert(child_task)

        if created_ids:
            parent.children_ids.extend(created_ids)
            svc.container.tasks.upsert(parent)
        return created_ids

    def generate_tasks_from_plan(
        self,
        task_id: str,
        plan_text: str,
        *,
        infer_deps: bool = True,
    ) -> list[str]:
        """Generate child tasks from an explicit plan text."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["plan_for_generation"] = plan_text
        svc.container.tasks.upsert(task)

        try:
            result = svc.worker_adapter.run_step(task=task, step="generate_tasks", attempt=1)
            if result.status != "ok":
                raise ValueError(f"generate_tasks step failed: {result.summary or result.status}")
            task_defs = list(result.generated_tasks or [])
            if not task_defs:
                detail = str(result.summary or "").strip()
                if detail:
                    raise ValueError(f"Worker returned no generated tasks: {detail}")
                raise ValueError("Worker returned no generated tasks for the selected plan source")
            created_ids = self.create_child_tasks(task, task_defs, apply_deps=infer_deps)
            if not created_ids:
                raise ValueError("Worker returned generated tasks, but none were valid task objects")
        finally:
            task.metadata.pop("plan_for_generation", None)
            svc.container.tasks.upsert(task)

        return created_ids
