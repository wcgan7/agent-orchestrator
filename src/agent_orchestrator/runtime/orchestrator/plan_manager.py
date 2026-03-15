"""Plan revision, refinement, and generation helpers for orchestrator tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ...collaboration.modes import normalize_hitl_mode
from ..domain.models import PlanRefineJob, PlanRevision, PlanRevisionStatus, Priority, TaskStatus, now_iso
import logging

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)


class PlanManager:
    """Manage plan documents, refine jobs, and task generation from plans."""

    def __init__(self, service: OrchestratorService) -> None:
        """Initialize the manager with orchestrator dependencies."""
        self._service = service

    @staticmethod
    def _replace_plan_section_in_workdoc(text: str, plan_body: str, *, manager: Any) -> str:
        """Replace the canonical plan section body without corrupting sentinels."""
        state, bounds, _ = manager._sentinel_section_bounds(text, "plan")
        if state == "malformed":
            raise ValueError("Malformed canonical workdoc section markers for step 'plan'")
        replacement = f"{plan_body}\n\n"
        if state == "valid" and bounds is not None:
            start, end = bounds
            return text[:start] + replacement + text[end:]

        heading = "## Plan"
        heading_idx = text.find(heading)
        if heading_idx == -1:
            return text
        heading_line_end = text.find("\n", heading_idx)
        if heading_line_end == -1:
            return text
        section_start = heading_line_end + 1
        end_candidates = []
        for next_heading in (
            "\n## Implementation Log",
            "\n## Verification Results",
            "\n## Review Findings",
            "\n## Fix Log",
        ):
            candidate = text.find(next_heading, section_start)
            if candidate != -1:
                end_candidates.append(candidate + 1)
        section_end = min(end_candidates) if end_candidates else len(text)
        return text[:section_start] + replacement + text[section_end:]

    def is_plan_mutable(self, task_id: str) -> bool:
        """Return True when the plan is still open for edits/refines.

        Plan is locked once the pipeline has moved past the planning stage.
        """
        task = self._service.container.tasks.get(task_id)
        if not task:
            return False
        if task.status in ("done", "cancelled", "in_review"):
            return False
        if task.status in ("backlog", "queued"):
            return True
        # At a pre-implementation gate — plan is still open for changes
        if task.pending_gate in ("before_plan", "before_implement", "before_generate_tasks"):
            return True
        if task.current_step == "plan":
            return True
        steps = task.pipeline_template or []
        current = task.current_step
        if not steps or not current:
            return True  # can't determine, be permissive
        if "plan" not in steps:
            return False  # no plan step, task is executing
        try:
            return steps.index(current) <= steps.index("plan")
        except ValueError:
            return False  # virtual step (e.g. implement_fix), past plan

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
            "plan_mutable": self.is_plan_mutable(task_id),
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
        if source == "human_edit" and not self.is_plan_mutable(task_id):
            raise RuntimeError("Plan is locked — pipeline has moved past the planning stage")
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
            if not self.is_plan_mutable(task_id):
                raise RuntimeError("Plan is locked — pipeline has moved past the planning stage")
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
        if not self.is_plan_mutable(task_id):
            raise RuntimeError("Plan is locked — pipeline has moved past the planning stage")
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
            plan_body = (target.content or "").strip() or "_(empty committed plan)_"
            updated = self._replace_plan_section_in_workdoc(text, plan_body, manager=svc._workdoc_manager)
            if updated != text:
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
        generation_policy: dict[str, Any] | None = None,
    ) -> list[str]:
        """Create generated child tasks and optionally wire dependency edges."""
        svc = self._service
        policy = dict(generation_policy) if isinstance(generation_policy, dict) else {}
        child_status = str(policy.get("child_status") or "queued").strip().lower()
        if child_status not in {"backlog", "queued"}:
            child_status = "queued"
        child_hitl_mode = normalize_hitl_mode(str(policy.get("child_hitl_mode") or "supervised"))
        # Resolve initiative context once before the loop so every child
        # carries the parent objective and relevant plan excerpt.
        _initiative_context: dict[str, str] | None = None
        if isinstance(parent.metadata, dict) and parent.task_type in {
            "initiative_plan", "plan_only",
        }:
            _plan_text = (
                parent.metadata.get("plan_for_generation")
                or (parent.metadata.get("step_outputs") or {}).get("initiative_plan")
                or (parent.metadata.get("step_outputs") or {}).get("plan")
            )
            if isinstance(_plan_text, str) and _plan_text.strip():
                _initiative_context = {
                    "parent_id": parent.id,
                    "parent_title": getattr(parent, "title", ""),
                    "objective": getattr(parent, "description", "") or "",
                    "plan_excerpt": self._truncate_plan_excerpt(_plan_text.strip()),
                }

        created_ids: list[str] = []
        for item in task_defs:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority") or parent.priority)
            if priority not in {"P0", "P1", "P2", "P3"}:
                priority = parent.priority
            raw_dep_refs = item.get("depends_on")
            generated_dep_refs = [
                str(dep_ref or "").strip()
                for dep_ref in raw_dep_refs
                if str(dep_ref or "").strip()
            ] if isinstance(raw_dep_refs, list) else []
            child_metadata = dict(item.get("metadata") or {})
            # Generated-task dependency policy is resolved at generation time.
            # Marking as analyzed prevents redundant global auto-deps passes.
            child_metadata["deps_analyzed"] = True
            child_metadata["deps_analysis_source"] = "generate_tasks"
            child_metadata["generated_depends_on"] = generated_dep_refs
            if _initiative_context is not None:
                child_metadata["initiative_context"] = dict(_initiative_context)
            from ..domain.models import Task

            child = Task(
                title=str(item.get("title") or "Generated task"),
                description=str(item.get("description") or ""),
                task_type=self._normalize_generated_task_type(item.get("task_type")),
                priority=cast(Priority, priority),
                parent_id=parent.id,
                status=cast(TaskStatus, child_status),
                source="generated",
                labels=list(item.get("labels") or []),
                hitl_mode=child_hitl_mode,
                metadata=child_metadata,
            )
            svc.container.tasks.upsert(child)
            created_ids.append(child.id)
            svc.bus.emit(
                channel="tasks",
                event_type="task.created",
                entity_id=child.id,
                payload={
                    "parent_id": parent.id,
                    "source": "generate_tasks",
                    "generation_policy": {
                        "child_status": child_status,
                        "child_hitl_mode": child_hitl_mode,
                        "infer_deps": bool(policy.get("infer_deps", apply_deps)),
                    },
                },
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

    @staticmethod
    def _truncate_plan_excerpt(text: str, max_chars: int = 4000) -> str:
        """Truncate plan text at a paragraph boundary within *max_chars*.

        Looks for the last double-newline before the limit so the excerpt
        ends on a complete paragraph.  Appends a truncation marker when
        the text is shortened.
        """
        if len(text) <= max_chars:
            return text
        # Find the last paragraph break within the limit
        candidate = text.rfind("\n\n", 0, max_chars)
        if candidate <= 0:
            # No paragraph boundary — fall back to hard cut
            candidate = max_chars
        return text[:candidate].rstrip() + "\n\n[...plan truncated...]"

    @staticmethod
    def _normalize_generated_task_type(raw: Any) -> str:
        """Constrain generated child task types to supported execution paths.

        Generated tasks should use a narrow, stable set so pipeline selection
        stays deterministic. We currently support:
        - ``feature`` (default/fallback)
        - ``bug`` (includes common bugfix aliases)
        - ``chore``
        """
        value = str(raw or "").strip().lower()
        if value in {"bug", "bugfix", "bug_fix", "bug-fix", "bug fix"}:
            return "bug"
        if value == "chore":
            return "chore"
        if value == "feature":
            return "feature"
        return "feature"

    def generate_tasks_from_plan(
        self,
        task_id: str,
        plan_text: str,
        *,
        infer_deps: bool | None = True,
        generation_policy_overrides: dict[str, Any] | None = None,
        save_as_default: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        """Generate child tasks from an explicit plan text."""
        svc = self._service
        task = svc.container.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if not svc.supports_task_generation(task):
            raise ValueError("Task pipeline does not include generate_tasks")

        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["plan_for_generation"] = plan_text
        requested_policy = dict(generation_policy_overrides) if isinstance(generation_policy_overrides, dict) else {}
        if infer_deps is not None and "infer_deps" not in requested_policy:
            requested_policy["infer_deps"] = bool(infer_deps)
        effective_policy = svc.resolve_task_generation_policy(task, request_overrides=requested_policy)
        if save_as_default:
            persisted_defaults = svc.persist_task_generation_defaults(task, effective_policy=effective_policy)
            effective_policy["saved_defaults"] = persisted_defaults
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
            created_ids = self.create_child_tasks(
                task,
                task_defs,
                apply_deps=bool(effective_policy.get("infer_deps", True)),
                generation_policy=effective_policy,
            )
            if not created_ids:
                raise ValueError("Worker returned generated tasks, but none were valid task objects")
        finally:
            task.metadata.pop("plan_for_generation", None)
            svc.container.tasks.upsert(task)

        svc.bus.emit(
            channel="tasks",
            event_type="task.generated_from_plan",
            entity_id=task.id,
            payload={
                "created_task_ids": created_ids,
                "effective_policy": effective_policy,
            },
        )
        return created_ids, effective_policy
