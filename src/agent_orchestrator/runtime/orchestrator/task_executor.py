"""Task execution loop helpers for orchestrator tasks."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...collaboration.modes import normalize_hitl_mode
from ...pipelines.registry import PipelineRegistry
from ...worker import WorkerCancelledError
from ..domain.models import RunRecord, ReviewCycle, ReviewFinding, Task, now_iso
from .human_guidance import consume_human_guidance_for_step, promote_legacy_human_guidance
from .live_worker_adapter import _VERIFY_STEPS
from .worker_adapter import StepResult

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Drive end-to-end task execution while service remains the public facade."""

    def __init__(self, service: OrchestratorService) -> None:
        """Store orchestrator service dependencies used across execution phases."""
        self._service = service

    def _ensure_workdoc_or_block(self, task: Task, run: RunRecord, *, step: str) -> bool:
        """Verify canonical workdoc exists; otherwise block task/run for this step."""
        svc = self._service
        try:
            if svc._validate_task_workdoc(task) is not None:
                return True
            svc._block_for_missing_workdoc(task, run, step=step)
            return False
        except ValueError as exc:
            svc._block_for_invalid_workdoc(task, run, step=step, detail=str(exc))
            return False

    def _prepare_workdoc_for_run(
        self,
        task: Task,
        run: RunRecord,
        *,
        project_dir: Path,
        first_step: str,
        had_prior_runs: bool,
        append_retry_marker: bool,
        retry_from_step: str | None = None,
    ) -> bool:
        """Initialize on first run, refresh on retry, and block if retry lost canonical workdoc."""
        svc = self._service
        try:
            canonical = svc._validate_task_workdoc(task)
        except ValueError as exc:
            svc._block_for_invalid_workdoc(task, run, step=first_step, detail=str(exc))
            return False
        svc._clear_invalid_workdoc_markers(task)
        if canonical is not None:
            try:
                svc._refresh_workdoc_with_diagnostics(task, project_dir)
            except ValueError as exc:
                svc._block_for_invalid_workdoc(task, run, step=first_step, detail=str(exc))
                return False
            if append_retry_marker:
                attempt = max(1, len(task.run_ids))
                svc._append_retry_attempt_marker(
                    task,
                    project_dir=project_dir,
                    attempt=attempt,
                    start_from_step=retry_from_step or None,
                )
            return True
        if had_prior_runs:
            svc._block_for_missing_workdoc(task, run, step=first_step)
            return False
        svc._init_workdoc(task, project_dir)
        return True

    def _branch_exists(self, branch_name: str) -> bool:
        """Return whether a local branch exists in the project repository."""
        svc = self._service
        normalized = str(branch_name or "").strip()
        if not normalized:
            return False
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{normalized}"],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _prepare_precommit_review_context(self, task: Task, worktree_dir: Path | None) -> tuple[bool, str]:
        """Persist task-scoped changes for pre-commit human review.

        Returns:
            tuple[bool, str]: ``(ok, reason)`` where ``reason`` is non-empty when
            ``ok`` is ``False``.
        """
        svc = self._service
        # Non-git flows can legitimately run without task worktrees.
        if worktree_dir is None:
            return True, ""
        if not worktree_dir.exists() or not worktree_dir.is_dir():
            return False, "missing task worktree context"

        has_task_changes = svc._has_uncommitted_changes(worktree_dir) or svc._has_commits_ahead(worktree_dir)
        if not has_task_changes:
            return False, "no task-scoped changes available"

        if not svc._preserve_worktree_work(task, worktree_dir):
            return False, "failed to preserve task-scoped changes"

        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        preserved_branch = str(metadata.get("preserved_branch") or "").strip()
        if not preserved_branch:
            return False, "missing preserved branch metadata"
        if not self._branch_exists(preserved_branch):
            return False, "preserved branch is not available"
        base_branch = str(svc._run_branch or "HEAD").strip() or "HEAD"
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--binary", "--no-color", f"{base_branch}..{preserved_branch}"],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except Exception:
            return False, "failed to prepare review context fingerprint"
        if diff_result.returncode != 0:
            return False, "failed to prepare review context fingerprint"
        fingerprint = hashlib.sha256((diff_result.stdout or "").encode("utf-8", errors="replace")).hexdigest()
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["review_context"] = {
            "run_id": task.run_ids[-1] if task.run_ids else None,
            "preserved_branch": preserved_branch,
            "base_branch": base_branch,
            "prepared_at": now_iso(),
            "diff_fingerprint": fingerprint,
        }
        return True, ""

    def execute_task(self, task: Task) -> None:
        """Execute one task and normalize top-level cancellation/error outcomes.

        This wrapper ensures task/run status persistence and event emission are
        consistent even when the inner execution loop raises.
        """
        svc = self._service
        try:
            self.execute_task_inner(task)
        except svc._Cancelled:
            logger.info("Task %s was cancelled by user", task.id)
            fresh = svc.container.tasks.get(task.id)
            if fresh:
                if fresh.status != "cancelled":
                    fresh.status = "cancelled"
                    svc.container.tasks.upsert(fresh)
                for run_id in reversed(fresh.run_ids):
                    run = svc.container.runs.get(run_id)
                    if run and run.status == "in_progress":
                        run.status = "cancelled"
                        run.finished_at = now_iso()
                        run.summary = "Cancelled by user"
                        svc.container.runs.upsert(run)
                        break
                svc.bus.emit(
                    channel="tasks",
                    event_type="task.cancelled",
                    entity_id=fresh.id,
                    payload={"status": "cancelled"},
                )
        except Exception:
            logger.exception("Unexpected error executing task %s", task.id)
            fresh = svc.container.tasks.get(task.id) or task
            fresh.status = "blocked"
            fresh.current_agent_id = None
            fresh.error = "Internal error during execution"
            svc.container.tasks.upsert(fresh)

    def execute_task_inner(self, task: Task) -> None:
        """Run the full pipeline, including retries, gates, review, and merge.

        Coordinates worktree lifecycle, run-record updates, verify-fix loops,
        human-gate checks, review cycles, and final commit/merge behavior.
        """
        svc = self._service
        fresh_task = svc.container.tasks.get(task.id)
        if fresh_task is not None:
            task = fresh_task
        worktree_dir: Optional[Path] = None
        try:
            old_preserved = task.metadata.get("preserved_branch")
            if old_preserved:
                # Try to create worktree from the preserved branch
                try:
                    worktree_dir = svc._create_worktree_from_branch(task, old_preserved)
                except subprocess.CalledProcessError:
                    # Clean up stale worktree dir that may have caused the failure.
                    stale_dir = svc.container.state_root / "worktrees" / str(task.id)
                    if stale_dir.exists():
                        subprocess.run(
                            ["git", "worktree", "remove", str(stale_dir), "--force"],
                            cwd=svc.container.project_dir,
                            capture_output=True,
                            text=True,
                        )
                    # Retry from-branch after cleaning stale dir
                    try:
                        worktree_dir = svc._create_worktree_from_branch(task, old_preserved)
                    except subprocess.CalledProcessError:
                        # Branch truly missing or corrupted — fall back to fresh worktree
                        subprocess.run(
                            ["git", "branch", "-D", str(old_preserved)],
                            cwd=svc.container.project_dir,
                            capture_output=True,
                            text=True,
                        )
                        worktree_dir = svc._create_worktree(task)
                # Clear preserved metadata only after successful worktree creation
                task.metadata.pop("preserved_branch", None)
                task.metadata.pop("merge_conflict", None)
                svc.container.tasks.upsert(task)
            else:
                worktree_dir = svc._create_worktree(task)
            if worktree_dir:
                task.metadata["worktree_dir"] = str(worktree_dir)
                svc.container.tasks.upsert(task)

            registry = PipelineRegistry()
            template = registry.resolve_for_task_type(task.task_type)
            steps = task.pipeline_template if task.pipeline_template else template.step_names()
            task.pipeline_template = steps
            has_review = "review" in steps
            has_commit = "commit" in steps
            first_step = steps[0] if steps else "plan"

            workdoc_dir = worktree_dir if worktree_dir else svc.container.project_dir
            had_prior_runs = bool(task.run_ids)
            checkpoint = svc._execution_checkpoint(task)
            checkpoint_run_id = str(checkpoint.get("run_id") or "").strip()
            run: RunRecord | None = None
            created_new_run = False
            if checkpoint_run_id:
                existing_run = svc.container.runs.get(checkpoint_run_id)
                if existing_run and existing_run.task_id == task.id and existing_run.status in {"waiting_gate", "in_progress"}:
                    run = existing_run
                    run.status = "in_progress"
                    run.finished_at = None
                    if run.summary and str(run.summary).startswith("Paused at gate:"):
                        run.summary = None
            if run is None:
                created_new_run = True
                task_branch = f"task-{task.id}" if worktree_dir else svc._ensure_branch()
                run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=task_branch)
                run.steps = []
                if run.id not in task.run_ids:
                    task.run_ids.append(run.id)
            elif run.id not in task.run_ids:
                task.run_ids.append(run.id)
            svc.container.runs.upsert(run)

            # Gate resumes can restart execution on the same run; treat retry metadata
            # as "new run created due to retry" rather than "task has any retry_count".
            is_retry_run = created_new_run and int(task.retry_count or 0) > 0
            retry_from = ""
            has_retry_guidance = False
            checkpoint_resume_step = str(checkpoint.get("resume_step") or "").strip()
            if isinstance(task.metadata, dict):
                retry_from = str(task.metadata.get("retry_from_step", "") or "").strip()
                retry_guidance = task.metadata.get("retry_guidance")
                if isinstance(retry_guidance, dict):
                    has_retry_guidance = bool(str(retry_guidance.get("guidance") or "").strip())
            if promote_legacy_human_guidance(task):
                svc.container.tasks.upsert(task)
            if not retry_from and checkpoint_resume_step:
                retry_from = checkpoint_resume_step
            run_attempt = max(1, len(task.run_ids))
            append_retry_marker = created_new_run and is_retry_run
            if not self._prepare_workdoc_for_run(
                task,
                run,
                project_dir=workdoc_dir,
                first_step=first_step,
                had_prior_runs=had_prior_runs,
                append_retry_marker=append_retry_marker,
                retry_from_step=retry_from,
            ):
                return

            cfg = svc.container.config.load()
            orch_cfg = dict(cfg.get("orchestrator") or {})
            max_review_attempts = int(orch_cfg.get("max_review_attempts", 10) or 10)
            max_verify_fix_attempts = int(orch_cfg.get("max_verify_fix_attempts", 3) or 3)

            if isinstance(task.metadata, dict):
                retry_from = str(task.metadata.pop("retry_from_step", "") or "").strip()

            start_step: str | None
            if retry_from in steps:
                start_step = retry_from
            elif retry_from in {"review", "commit"}:
                start_step = retry_from
            elif retry_from == svc._BEFORE_DONE_RESUME_STEP:
                start_step = None
            else:
                start_step = steps[0] if steps else None
            task.current_step = start_step
            task.metadata["pipeline_phase"] = start_step
            task.status = "in_progress"
            task.current_agent_id = svc._choose_agent_for_task(task)
            svc.container.tasks.upsert(task)
            svc.bus.emit(
                channel="tasks",
                event_type="task.started",
                entity_id=task.id,
                payload={
                    "run_id": run.id,
                    "agent_id": task.current_agent_id,
                    "run_attempt": run_attempt,
                    "is_retry": is_retry_run,
                    "start_from_step": retry_from or None,
                    "has_retry_guidance": has_retry_guidance,
                    "retry_count": int(task.retry_count),
                },
            )

            def _consume_human_guidance(step_name: str) -> None:
                if consume_human_guidance_for_step(task, step=step_name, run_id=run.id):
                    svc.container.tasks.upsert(task)

            mode = normalize_hitl_mode(getattr(task, "hitl_mode", "autopilot"))
            # Retry flows resumed from implement/review/commit should not require
            # re-approval of the original plan gate.
            skip_before_implement_gate = bool(is_retry_run and retry_from and retry_from != "plan")

            skip_phase1 = retry_from in ("review", "commit")
            resume_from_done_gate = retry_from == svc._BEFORE_DONE_RESUME_STEP
            reached_retry_step = not retry_from
            last_phase1_step: str | None = None
            for step in steps:
                if step in ("review", "commit"):
                    continue
                if resume_from_done_gate:
                    continue
                if skip_phase1:
                    last_phase1_step = step
                    run.steps.append({"step": step, "status": "skipped", "ts": now_iso()})
                    svc.container.runs.upsert(run)
                    continue
                if not reached_retry_step:
                    if step == retry_from:
                        reached_retry_step = True
                    else:
                        last_phase1_step = step
                        run.steps.append({"step": step, "status": "skipped", "ts": now_iso()})
                        svc.container.runs.upsert(run)
                        continue
                svc._check_cancelled(task)
                gate_name = svc._gate_for_step(
                    task=task,
                    mode=mode,
                    steps=steps,
                    step=step,
                    skip_before_implement_gate=skip_before_implement_gate,
                )
                if gate_name and svc._should_gate(mode, gate_name):
                    if not svc._wait_for_gate(task, gate_name):
                        return
                task.current_step = step
                task.metadata["pipeline_phase"] = step
                svc.container.tasks.upsert(task)
                if not svc._run_non_review_step(task, run, step, attempt=1, workdoc_attempt=run_attempt):
                    if step in _VERIFY_STEPS:
                        if svc._consume_verify_non_actionable_flag(task):
                            last_phase1_step = step
                            continue
                        fixed = False
                        for fix_attempt in range(1, max_verify_fix_attempts + 1):
                            task.status = "in_progress"
                            task.metadata["verify_failure"] = task.error
                            svc._capture_verify_output(task)
                            task.error = None
                            task.retry_count += 1
                            svc.container.tasks.upsert(task)
                            run.status = "in_progress"
                            run.finished_at = None
                            run.summary = None
                            svc.container.runs.upsert(run)

                            task.current_step = "implement_fix"
                            task.metadata["pipeline_phase"] = step
                            svc.container.tasks.upsert(task)
                            if not svc._run_non_review_step(task, run, "implement_fix", attempt=fix_attempt + 1):
                                return
                            _consume_human_guidance("implement_fix")
                            task.metadata.pop("verify_failure", None)
                            task.metadata.pop("verify_output", None)
                            task.current_step = step
                            task.metadata["pipeline_phase"] = step
                            svc.container.tasks.upsert(task)
                            if svc._run_non_review_step(task, run, step, attempt=fix_attempt + 1):
                                _consume_human_guidance(step)
                                fixed = True
                                break
                        if fixed:
                            last_phase1_step = step
                            continue
                        # All verify-fix attempts exhausted — ensure task is blocked.
                        task.status = "blocked"
                        task.error = task.error or f"Could not fix {step} after {max_verify_fix_attempts} attempts"
                        task.current_step = step
                        svc.container.tasks.upsert(task)
                        svc._finalize_run(task, run, status="blocked", summary=f"Blocked: {step} failed after {max_verify_fix_attempts} fix attempts")
                        svc.bus.emit(
                            channel="tasks",
                            event_type="task.blocked",
                            entity_id=task.id,
                            payload={"error": task.error},
                        )
                    return
                _consume_human_guidance(step)
                last_phase1_step = step

            next_phase = "review" if has_review and retry_from != "commit" else "commit"
            if (has_review and retry_from != "commit") or has_commit:
                if not self._ensure_workdoc_or_block(task, run, step=next_phase):
                    return

            if has_commit:
                impl_dir = worktree_dir or svc.container.project_dir
                svc._cleanup_workdoc_for_commit(impl_dir)
                if not svc._has_uncommitted_changes(impl_dir) and not svc._has_commits_ahead(impl_dir):
                    task.status = "blocked"
                    task.error = "No file changes detected after implementation"
                    task.current_step = last_phase1_step or "implement"
                    task.metadata["pipeline_phase"] = last_phase1_step or "implement"
                    svc.container.tasks.upsert(task)
                    svc._finalize_run(task, run, status="blocked", summary="Blocked: no changes produced by implementation steps")
                    svc.bus.emit(
                        channel="tasks",
                        event_type="task.blocked",
                        entity_id=task.id,
                        payload={"error": task.error},
                    )
                    return

            svc._check_cancelled(task)
            if has_review and retry_from not in {"commit", svc._BEFORE_DONE_RESUME_STEP}:
                post_fix_validation_step = svc._select_post_fix_validation_step(steps)

                review_attempt = 0
                review_passed = False

                while review_attempt < max_review_attempts:
                    svc._check_cancelled(task)
                    review_attempt += 1
                    task.current_step = "review"
                    task.metadata["pipeline_phase"] = "review"
                    if review_attempt > 1:
                        task.metadata["review_history"] = svc._build_review_history_summary(task.id)
                    else:
                        task.metadata.pop("review_history", None)
                    svc.container.tasks.upsert(task)
                    if not self._ensure_workdoc_or_block(task, run, step="review"):
                        return
                    review_project_dir = worktree_dir or svc.container.project_dir
                    try:
                        svc._refresh_workdoc_with_diagnostics(task, review_project_dir)
                    except ValueError as exc:
                        svc._block_for_invalid_workdoc(task, run, step="review", detail=str(exc))
                        return
                    review_started = now_iso()
                    svc._heartbeat_execution_lease(task)
                    svc.container.tasks.upsert(task)
                    findings, review_result = svc._findings_from_result(task, review_attempt)
                    svc._heartbeat_execution_lease(task)
                    svc.container.tasks.upsert(task)
                    review_step_log: dict[str, object] = {
                        "step": "review",
                        "status": "ok",
                        "ts": now_iso(),
                        "started_at": review_started,
                    }
                    review_last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
                    if isinstance(review_last_logs, dict):
                        for key in ("stdout_path", "stderr_path", "progress_path"):
                            if review_last_logs.get(key):
                                review_step_log[key] = review_last_logs[key]
                    if review_result.human_blocking_issues:
                        review_step_log["status"] = "blocked"
                        review_step_log["human_blocking_issues"] = review_result.human_blocking_issues
                        run.steps.append(review_step_log)
                        svc.container.runs.upsert(run)
                        svc._block_for_human_issues(
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
                        svc.container.runs.upsert(run)
                        task.status = "blocked"
                        task.error = review_result.summary or "Review step failed"
                        task.pending_gate = None
                        task.current_step = "review"
                        task.metadata["pipeline_phase"] = "review"
                        svc.container.tasks.upsert(task)
                        svc._finalize_run(task, run, status="blocked", summary="Blocked during review")
                        svc.bus.emit(
                            channel="tasks",
                            event_type="task.blocked",
                            entity_id=task.id,
                            payload={"error": task.error},
                        )
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
                        decision="changes_requested" if svc._exceeds_quality_gate(task, findings) else "approved",
                    )
                    if not self._ensure_workdoc_or_block(task, run, step="review"):
                        return
                    svc.container.reviews.append(cycle)
                    svc._sync_workdoc_review(task, cycle, worktree_dir or svc.container.project_dir)
                    review_step_log["status"] = cycle.decision
                    review_step_log["open_counts"] = open_counts
                    run.steps.append(review_step_log)
                    svc.container.runs.upsert(run)
                    svc.bus.emit(
                        channel="review",
                        event_type="task.reviewed",
                        entity_id=task.id,
                        payload={"attempt": review_attempt, "decision": cycle.decision, "open_counts": open_counts},
                    )
                    _consume_human_guidance("review")

                    if cycle.decision == "approved":
                        review_passed = True
                        break

                    if review_attempt >= max_review_attempts:
                        break

                    open_findings = [f.to_dict() for f in findings if f.status == "open"]
                    task.metadata["review_findings"] = open_findings
                    task.metadata["review_history"] = svc._build_review_history_summary(task.id)

                    task.retry_count += 1
                    task.current_step = "implement_fix"
                    task.metadata["pipeline_phase"] = "review"
                    svc.container.tasks.upsert(task)
                    if not svc._run_non_review_step(task, run, "implement_fix", attempt=review_attempt):
                        return
                    _consume_human_guidance("implement_fix")

                    if post_fix_validation_step:
                        task.current_step = post_fix_validation_step
                        task.metadata["pipeline_phase"] = "review"
                        svc.container.tasks.upsert(task)
                        if not svc._run_non_review_step(task, run, post_fix_validation_step, attempt=review_attempt):
                            if svc._consume_verify_non_actionable_flag(task):
                                task.current_step = "review"
                                task.metadata["pipeline_phase"] = "review"
                                svc.container.tasks.upsert(task)
                                continue
                            validation_fixed = False
                            for vfix in range(1, max_verify_fix_attempts + 1):
                                task.status = "in_progress"
                                task.metadata["verify_failure"] = task.error
                                svc._capture_verify_output(task)
                                task.error = None
                                task.retry_count += 1
                                svc.container.tasks.upsert(task)
                                run.status = "in_progress"
                                run.finished_at = None
                                run.summary = None
                                svc.container.runs.upsert(run)

                                task.current_step = "implement_fix"
                                task.metadata["pipeline_phase"] = "review"
                                svc.container.tasks.upsert(task)
                                if not svc._run_non_review_step(task, run, "implement_fix", attempt=vfix + 1):
                                    return
                                _consume_human_guidance("implement_fix")
                                task.metadata.pop("verify_failure", None)
                                task.metadata.pop("verify_output", None)
                                task.current_step = post_fix_validation_step
                                task.metadata["pipeline_phase"] = "review"
                                svc.container.tasks.upsert(task)
                                if svc._run_non_review_step(task, run, post_fix_validation_step, attempt=vfix + 1):
                                    _consume_human_guidance(post_fix_validation_step)
                                    validation_fixed = True
                                    break
                            if not validation_fixed:
                                task.status = "blocked"
                                task.error = task.error or f"Could not fix {post_fix_validation_step} after {max_verify_fix_attempts} attempts"
                                task.current_step = post_fix_validation_step
                                svc.container.tasks.upsert(task)
                                svc._finalize_run(task, run, status="blocked", summary=f"Blocked: {post_fix_validation_step} failed after {max_verify_fix_attempts} fix attempts")
                                svc.bus.emit(
                                    channel="tasks",
                                    event_type="task.blocked",
                                    entity_id=task.id,
                                    payload={"error": task.error},
                                )
                                return
                        else:
                            _consume_human_guidance(post_fix_validation_step)

                    task.metadata.pop("review_findings", None)
                    task.metadata.pop("review_history", None)
                    task.metadata.pop("verify_environment_note", None)

                if not review_passed:
                    task.metadata.pop("review_history", None)
                    task.metadata.pop("verify_environment_note", None)
                    if worktree_dir and worktree_dir.exists():
                        if svc._preserve_worktree_work(task, worktree_dir):
                            worktree_dir = None
                    task.status = "blocked"
                    task.error = "Review attempt cap exceeded"
                    task.current_step = "review"
                    task.metadata["pipeline_phase"] = "review"
                    svc.container.tasks.upsert(task)
                    svc._finalize_run(task, run, status="blocked", summary="Blocked due to unresolved review findings")
                    svc.bus.emit(
                        channel="tasks",
                        event_type="task.blocked",
                        entity_id=task.id,
                        payload={"error": task.error},
                    )
                    return

            svc._check_cancelled(task)
            if has_commit:
                # Supervised/review-only flows pause in in_review before commit so
                # footer review actions drive final approve/request-changes.
                precommit_review_modes = {"supervised", "review_only"}
                requires_precommit_review = mode in precommit_review_modes
                if requires_precommit_review and retry_from != "commit":
                    context_ok, context_reason = self._prepare_precommit_review_context(task, worktree_dir)
                    if not context_ok:
                        task.status = "blocked"
                        task.pending_gate = None
                        task.current_step = "review"
                        task.current_agent_id = None
                        task.metadata["pipeline_phase"] = "review"
                        task.metadata.pop("pending_precommit_approval", None)
                        task.metadata.pop("review_stage", None)
                        detail = context_reason.strip()
                        task.error = "Failed to preserve task-scoped changes for pre-commit review"
                        if detail:
                            task.error = f"{task.error}: {detail}"
                        svc.container.tasks.upsert(task)
                        svc._finalize_run(task, run, status="blocked", summary="Blocked: pre-commit review context unavailable")
                        svc.bus.emit(
                            channel="tasks",
                            event_type="task.blocked",
                            entity_id=task.id,
                            payload={"error": task.error},
                        )
                        return

                    # Context is now preserved on task branch; current worktree has
                    # already been removed by preserve_worktree_work.
                    worktree_dir = None
                    task.status = "in_review"
                    task.current_step = "review"
                    task.metadata["pipeline_phase"] = "review"
                    task.metadata["pending_precommit_approval"] = True
                    task.metadata["review_stage"] = "pre_commit"
                    svc.container.tasks.upsert(task)
                    run.status = "in_review"
                    run.summary = "Awaiting pre-commit approval"
                    svc.container.runs.upsert(run)
                    svc.bus.emit(
                        channel="review",
                        event_type="task.awaiting_human",
                        entity_id=task.id,
                        payload={"stage": "pre_commit"},
                    )
                    return

                task.metadata.pop("pending_precommit_approval", None)
                task.metadata.pop("review_stage", None)
                task.current_step = "commit"
                task.metadata["pipeline_phase"] = "commit"
                svc.container.tasks.upsert(task)
                if not self._ensure_workdoc_or_block(task, run, step="commit"):
                    return

                commit_started = now_iso()
                svc._cleanup_workdoc_for_commit(worktree_dir or svc.container.project_dir)
                commit_sha = svc._commit_for_task(task, worktree_dir)
                if not commit_sha:
                    # No new commit was created — check if the branch already
                    # carries prior committed work (e.g. from a preserved branch
                    # retry).  If so, use the branch HEAD as the commit ref so
                    # merge_and_cleanup can still merge it into the run branch.
                    commit_cwd = worktree_dir or svc.container.project_dir
                    if svc._has_commits_ahead(commit_cwd):
                        head_result = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=commit_cwd,
                            capture_output=True,
                            text=True,
                        )
                        commit_sha = head_result.stdout.strip() if head_result.returncode == 0 else None
                    if not commit_sha:
                        git_present = (commit_cwd / ".git").exists() or (svc.container.project_dir / ".git").exists()
                        if git_present:
                            task.status = "blocked"
                            task.error = "Commit failed (no changes to commit)"
                            svc.container.tasks.upsert(task)
                            svc._finalize_run(task, run, status="blocked", summary="Blocked: commit produced no changes")
                            svc.bus.emit(
                                channel="tasks",
                                event_type="task.blocked",
                                entity_id=task.id,
                                payload={"error": task.error},
                            )
                            return
                run.steps.append(
                    {
                        "step": "commit",
                        "status": "ok",
                        "ts": now_iso(),
                        "started_at": commit_started,
                        "commit": commit_sha,
                    }
                )
                svc.container.runs.upsert(run)
                _consume_human_guidance("commit")

                if worktree_dir:
                    svc._merge_and_cleanup(task, worktree_dir)
                    worktree_dir = None

                if task.metadata.get("merge_conflict"):
                    task.status = "blocked"
                    task.error = "Merge conflict could not be resolved automatically"
                    task.metadata["preserved_branch"] = f"task-{task.id}"
                    svc.container.tasks.upsert(task)
                    svc._finalize_run(task, run, status="blocked", summary="Blocked due to unresolved merge conflict")
                    svc.bus.emit(
                        channel="tasks",
                        event_type="task.blocked",
                        entity_id=task.id,
                        payload={"error": task.error},
                    )
                    return

                svc._run_summarize_step(task, run)
                task.status = "done"
                task.current_step = None
                task.metadata.pop("pipeline_phase", None)
                task.metadata.pop("pending_precommit_approval", None)
                task.metadata.pop("review_stage", None)
                run.status = "done"
                run.summary = "Pipeline completed"
                svc.bus.emit(
                    channel="tasks",
                    event_type="task.done",
                    entity_id=task.id,
                    payload={"commit": commit_sha},
                )
            else:
                requires_done_gate = svc._should_before_done_gate(task=task, mode=mode, has_commit=has_commit)
                if requires_done_gate and retry_from != svc._BEFORE_DONE_RESUME_STEP:
                    gate_resume_step = svc._BEFORE_DONE_RESUME_STEP
                    task.current_step = last_phase1_step
                    task.metadata["pipeline_phase"] = last_phase1_step
                    svc.container.tasks.upsert(task)
                    if not svc._wait_for_gate(task, "before_done", resume_step=gate_resume_step):
                        return

                if worktree_dir:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "branch", "-D", f"task-{task.id}"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    worktree_dir = None

                svc._run_summarize_step(task, run)
                task.status = "done"
                task.current_step = None
                task.metadata.pop("pipeline_phase", None)
                run.status = "done"
                run.summary = "Pipeline completed"
                svc.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={})

            task.error = None
            task.metadata.pop("execution_checkpoint", None)
            task.metadata.pop("step_outputs", None)
            task.metadata.pop("worktree_dir", None)
            run.finished_at = now_iso()
            svc.container.runs.upsert(run)
            svc.container.tasks.upsert(task)
        finally:
            if worktree_dir and worktree_dir.exists():
                preserved = task.metadata.get("preserved_branch")
                if not preserved:
                    try:
                        if svc._has_uncommitted_changes(worktree_dir) or svc._has_commits_ahead(worktree_dir):
                            svc._preserve_worktree_work(task, worktree_dir)
                            preserved = task.metadata.get("preserved_branch")
                    except Exception:
                        pass
                if preserved:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                else:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "branch", "-D", f"task-{task.id}"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
            worktree_removed = bool(task.metadata.pop("worktree_dir", None))
            lease_removed = svc._release_execution_lease(task)
            if worktree_removed or lease_removed:
                svc.container.tasks.upsert(task)
