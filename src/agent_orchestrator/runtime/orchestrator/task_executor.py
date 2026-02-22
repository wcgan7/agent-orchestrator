"""Task execution loop helpers for orchestrator tasks."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...pipelines.registry import PipelineRegistry
from ...worker import WorkerCancelledError
from ..domain.models import RunRecord, ReviewCycle, ReviewFinding, Task, now_iso
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
        if svc._validate_task_workdoc(task) is not None:
            return True
        svc._block_for_missing_workdoc(task, run, step=step)
        return False

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
            task.status = "blocked"
            task.error = "Internal error during execution"
            svc.container.tasks.upsert(task)

    def execute_task_inner(self, task: Task) -> None:
        """Run the full pipeline, including retries, gates, review, and merge.

        Coordinates worktree lifecycle, run-record updates, verify-fix loops,
        human-gate checks, review cycles, and final commit/merge behavior.
        """
        svc = self._service
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

            workdoc_dir = worktree_dir if worktree_dir else svc.container.project_dir
            svc._init_workdoc(task, workdoc_dir)

            task_branch = f"task-{task.id}" if worktree_dir else svc._ensure_branch()
            run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=task_branch)
            run.steps = []
            svc.container.runs.upsert(run)

            cfg = svc.container.config.load()
            orch_cfg = dict(cfg.get("orchestrator") or {})
            max_review_attempts = int(orch_cfg.get("max_review_attempts", 10) or 10)
            max_verify_fix_attempts = int(orch_cfg.get("max_verify_fix_attempts", 3) or 3)

            registry = PipelineRegistry()
            template = registry.resolve_for_task_type(task.task_type)
            steps = task.pipeline_template if task.pipeline_template else template.step_names()
            task.pipeline_template = steps
            has_review = "review" in steps
            has_commit = "commit" in steps

            retry_from = ""
            if isinstance(task.metadata, dict):
                retry_from = str(task.metadata.pop("retry_from_step", "") or "").strip()

            task.run_ids.append(run.id)
            task.current_step = steps[0] if steps else None
            task.metadata["pipeline_phase"] = steps[0] if steps else None
            task.status = "in_progress"
            task.current_agent_id = svc._choose_agent_for_task(task)
            svc.container.tasks.upsert(task)
            svc.bus.emit(
                channel="tasks",
                event_type="task.started",
                entity_id=task.id,
                payload={"run_id": run.id, "agent_id": task.current_agent_id},
            )

            mode = getattr(task, "hitl_mode", "autopilot") or "autopilot"

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
                        svc.container.runs.upsert(run)
                        continue
                svc._check_cancelled(task)
                task.current_step = step
                task.metadata["pipeline_phase"] = step
                svc.container.tasks.upsert(task)
                gate_name = svc._GATE_MAPPING.get(step)
                if gate_name and svc._should_gate(mode, gate_name):
                    if not svc._wait_for_gate(task, gate_name):
                        svc._abort_for_gate(task, run, gate_name)
                        return
                if not svc._run_non_review_step(task, run, step, attempt=1):
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
                            task.metadata.pop("verify_failure", None)
                            task.metadata.pop("verify_output", None)
                            task.current_step = step
                            task.metadata["pipeline_phase"] = step
                            svc.container.tasks.upsert(task)
                            if svc._run_non_review_step(task, run, step, attempt=fix_attempt + 1):
                                fixed = True
                                break
                        if fixed:
                            last_phase1_step = step
                            continue
                    return
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
            if has_review and retry_from != "commit":
                post_fix_validation_step = svc._select_post_fix_validation_step(steps)
                gate_name = svc._GATE_MAPPING.get("review")
                if gate_name and svc._should_gate(mode, gate_name):
                    if not svc._wait_for_gate(task, gate_name):
                        svc._abort_for_gate(task, run, gate_name)
                        return

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
                    svc._refresh_workdoc(task, review_project_dir)
                    review_started = now_iso()
                    findings, review_result = svc._findings_from_result(task, review_attempt)
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
                                task.metadata.pop("verify_failure", None)
                                task.metadata.pop("verify_output", None)
                                task.current_step = post_fix_validation_step
                                task.metadata["pipeline_phase"] = "review"
                                svc.container.tasks.upsert(task)
                                if svc._run_non_review_step(task, run, post_fix_validation_step, attempt=vfix + 1):
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
                task.current_step = "commit"
                task.metadata["pipeline_phase"] = "commit"
                svc.container.tasks.upsert(task)
                if not self._ensure_workdoc_or_block(task, run, step="commit"):
                    return
                if svc._should_gate(mode, "before_commit"):
                    if not svc._wait_for_gate(task, "before_commit"):
                        svc._abort_for_gate(task, run, "before_commit")
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
                if task.approval_mode == "auto_approve":
                    task.status = "done"
                    task.current_step = None
                    task.metadata.pop("pipeline_phase", None)
                    run.status = "done"
                    run.summary = "Completed with auto-approve"
                    svc.bus.emit(
                        channel="tasks",
                        event_type="task.done",
                        entity_id=task.id,
                        payload={"commit": commit_sha},
                    )
                else:
                    task.status = "in_review"
                    task.current_step = None
                    task.metadata.pop("pipeline_phase", None)
                    run.status = "in_review"
                    run.summary = "Awaiting human review"
                    svc.bus.emit(
                        channel="review",
                        event_type="task.awaiting_human",
                        entity_id=task.id,
                        payload={"commit": commit_sha},
                    )
            else:
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
            if task.metadata.pop("worktree_dir", None):
                svc.container.tasks.upsert(task)
