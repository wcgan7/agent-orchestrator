"""Phase executor for parallel execution of independent phases.

This module provides the PhaseExecutor class that can execute an entire
phase (all its tasks) in a thread-safe manner, enabling parallel phase
execution.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .actions.run_commit import run_commit_action
from .actions.run_verify import run_verify_action
from .actions.run_worker import run_worker_action
from .breakpoints import BreakpointManager
from .constants import (
    ERROR_TYPE_BLOCKING_ISSUES,
    ERROR_TYPE_PLAN_MISSING,
    IGNORED_REVIEW_PATH_PREFIXES,
    MAX_ALLOWLIST_EXPANSION_ATTEMPTS,
    MAX_IMPL_PLAN_ATTEMPTS,
    MAX_NO_PROGRESS_ATTEMPTS,
    MAX_TEST_FAIL_ATTEMPTS,
    STATE_DIR_NAME,
    TASK_STATUS_BLOCKED,
)
from .fsm import reduce_task
from .git_coordinator import get_git_coordinator
from .git_utils import _ensure_branch, _git_changed_files, _git_has_changes
from .io_utils import FileLock, _append_event, _load_data, _save_data, _update_progress
from .state_manager import state_transaction
from .models import ProgressHumanBlockers, TaskLifecycle, TaskState, TaskStep, WorkerFailed, WorkerSucceeded
from .parallel import PhaseResult
from .tasks import _normalize_phases, _phase_for_task, _sync_phase_status
from .signals import build_allowed_files
from .tasks import (
    _find_task,
    _impl_plan_path,
    _normalize_tasks,
    _record_blocked_intent,
    _save_queue,
)
from .utils import _now_iso


class PhaseExecutor:
    """Execute a complete phase (all tasks) in a thread-safe manner."""

    def __init__(
        self,
        project_dir: Path,
        prd_path: Path,
        paths: dict[str, Path],
        codex_command: str,
        heartbeat_seconds: int,
        heartbeat_grace_seconds: int,
        shift_minutes: int,
        max_attempts: int,
        max_review_attempts: int = 3,
        test_command: Optional[str] = None,
        format_command: Optional[str] = None,
        lint_command: Optional[str] = None,
        typecheck_command: Optional[str] = None,
        verify_profile: str = "none",
        ensure_ruff: str = "off",
        ensure_deps: str = "off",
        ensure_deps_command: Optional[str] = None,
        simple_review: bool = False,
        commit_enabled: bool = True,
        push_enabled: bool = True,
        language: str = "python",
    ):
        """Initialize phase executor.

        Args:
            project_dir: Project directory.
            prd_path: Path to PRD file.
            paths: Paths dictionary from orchestrator.
            codex_command: Codex CLI command.
            heartbeat_seconds: Heartbeat interval.
            heartbeat_grace_seconds: Heartbeat grace period.
            shift_minutes: Shift duration.
            max_attempts: Maximum task attempts.
            max_review_attempts: Maximum review attempts.
            test_command: Test command.
            format_command: Format command.
            lint_command: Lint command.
            typecheck_command: Typecheck command.
            verify_profile: Verification profile.
            ensure_ruff: Ruff helper behavior.
            ensure_deps: Dependency helper behavior.
            ensure_deps_command: Dependency install command.
            simple_review: Use simple review mode.
            commit_enabled: Enable commit step.
            push_enabled: Enable push step.
            language: Project language for language-aware verification.
        """
        self.project_dir = project_dir
        self.prd_path = prd_path
        self.paths = paths
        self.codex_command = codex_command
        self.heartbeat_seconds = heartbeat_seconds
        self.heartbeat_grace_seconds = heartbeat_grace_seconds
        self.shift_minutes = shift_minutes
        self.max_attempts = max_attempts
        self.max_review_attempts = max_review_attempts
        self.test_command = test_command
        self.format_command = format_command
        self.lint_command = lint_command
        self.typecheck_command = typecheck_command
        self.verify_profile = verify_profile
        self.ensure_ruff = ensure_ruff
        self.ensure_deps = ensure_deps
        self.ensure_deps_command = ensure_deps_command
        self.simple_review = simple_review
        self.commit_enabled = commit_enabled
        self.push_enabled = push_enabled
        self.language = language
        self.git_coordinator = get_git_coordinator()
        self.lock_path = paths["state_dir"] / "run.lock"
        self.breakpoint_manager = BreakpointManager(paths["state_dir"])

    def execute_phase(
        self,
        phase_id: str,
        phase_data: dict[str, Any],
    ) -> PhaseResult:
        """Execute all tasks for a phase.

        Args:
            phase_id: Phase identifier.
            phase_data: Phase data including tasks.

        Returns:
            PhaseResult with success/failure status.
        """
        start_time = time.time()
        thread_name = os.getpid()  # Use PID for identification in logs

        logger.info("[Phase {}] Starting execution (thread={})", phase_id, thread_name)

        try:
            # Get tasks for this phase
            tasks = phase_data.get("tasks", [])
            if not tasks:
                logger.warning("[Phase {}] No tasks found", phase_id)
                return PhaseResult(
                    phase_id=phase_id,
                    success=True,
                    error=None,
                    duration_seconds=time.time() - start_time,
                )

            # Execute each task in the phase sequentially
            # Tasks within a phase must run sequentially (plan_impl -> implement -> verify -> review -> commit)
            for task in tasks:
                task_id = task.get("id")
                if not task_id:
                    continue

                logger.info("[Phase {}] Executing task {}", phase_id, task_id)

                # Execute the task
                success, error = self._execute_task(task_id, task, phase_id)

                if not success:
                    logger.error("[Phase {}] Task {} failed: {}", phase_id, task_id, error)
                    return PhaseResult(
                        phase_id=phase_id,
                        success=False,
                        error=f"Task {task_id} failed: {error}",
                        duration_seconds=time.time() - start_time,
                    )

                logger.info("[Phase {}] Task {} completed successfully", phase_id, task_id)

            # All tasks completed successfully
            duration = time.time() - start_time
            logger.info("[Phase {}] Completed successfully in {:.2f}s", phase_id, duration)

            return PhaseResult(
                phase_id=phase_id,
                success=True,
                error=None,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("[Phase {}] Unexpected error: {}", phase_id, e)
            return PhaseResult(
                phase_id=phase_id,
                success=False,
                error=f"Unexpected error: {e}",
                duration_seconds=duration,
            )

    def _execute_task(
        self,
        task_id: str,
        task: dict[str, Any],
        phase_id: str,
    ) -> tuple[bool, Optional[str]]:
        """Execute a single task through its lifecycle.

        This method runs through the complete task lifecycle:
        plan_impl -> implement -> verify -> review -> commit

        Args:
            task_id: Task identifier.
            task: Task dictionary.
            phase_id: Phase identifier.

        Returns:
            Tuple of (success, error_message).
        """
        # Task execution runs in a loop until the task is either DONE or WAITING_HUMAN
        max_iterations = 20  # Safety limit to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Load current state (with lock)
            with FileLock(self.lock_path):
                queue = _load_data(self.paths["task_queue"], {})
                tasks = _normalize_tasks(queue)
                plan = _load_data(self.paths["phase_plan"], {})
                phases = _normalize_phases(plan)
                current_task = _find_task(tasks, task_id)

                if not current_task:
                    return False, f"Task {task_id} not found in queue"

                # Get phase info
                phase = _phase_for_task(phases, current_task)
                if not phase:
                    logger.error("[Task {}] Phase {} not found", task_id, phase_id)
                    current_task["lifecycle"] = TaskLifecycle.WAITING_HUMAN.value
                    current_task["block_reason"] = "PLAN_MISSING"
                    current_task["last_error"] = "Phase not found"
                    current_task["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                    current_task["status"] = TASK_STATUS_BLOCKED
                    _save_queue(self.paths["task_queue"], queue, tasks)
                    return False, "Phase not found"

                # Check lifecycle
                lifecycle = current_task.get("lifecycle")
                if lifecycle == TaskLifecycle.DONE.value:
                    logger.info("[Task {}] Already done", task_id)
                    return True, None
                elif lifecycle == TaskLifecycle.WAITING_HUMAN.value:
                    logger.warning("[Task {}] Blocked: {}", task_id, current_task.get("last_error"))
                    return False, current_task.get("last_error")
                elif lifecycle not in [TaskLifecycle.READY.value, TaskLifecycle.RUNNING.value]:
                    logger.info("[Task {}] Skipping (lifecycle={})", task_id, lifecycle)
                    return True, None

                # Mark as running
                task_step = current_task.get("step") or TaskStep.PLAN_IMPL.value
                current_task["lifecycle"] = TaskLifecycle.RUNNING.value
                current_task["status"] = task_step
                current_task["step"] = task_step
                branch = current_task.get("branch")

                _save_queue(self.paths["task_queue"], queue, tasks)

            # Create run ID
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{run_id}-{uuid.uuid4().hex[:8]}-parallel-{phase_id}-{task_step}"

            run_dir = self.paths["runs"] / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_path = run_dir / "progress.json"

            _update_progress(
                progress_path,
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "phase": phase_id,
                    "step": task_step,
                    "human_blocking_issues": [],
                    "human_next_steps": [],
                },
            )

            # Git operations must be coordinated
            if branch:
                def ensure_branch_op() -> None:
                    _ensure_branch(self.project_dir, branch)

                try:
                    self.git_coordinator.execute_git_operation(
                        ensure_branch_op,
                        operation_name=f"ensure_branch_{task_id}",
                    )
                except Exception as e:
                    logger.error("[Task {}] Branch setup failed: {}", task_id, e)
                    with state_transaction(
                        self.lock_path,
                        self.paths["task_queue"],
                    ) as state:
                        task_obj = state.find_task(task_id)
                        if task_obj:
                            task_obj["lifecycle"] = TaskLifecycle.WAITING_HUMAN.value
                            task_obj["block_reason"] = "GIT_CHECKOUT_FAILED"
                            task_obj["last_error"] = str(e)
                            task_obj["status"] = TASK_STATUS_BLOCKED
                    return False, f"Branch setup failed: {e}"

            # Execute the appropriate action based on step
            logger.info("[Task {}] Executing step: {}", task_id, task_step)

            step_enum = TaskStep.PLAN_IMPL
            if isinstance(task_step, TaskStep):
                step_enum = task_step
            elif isinstance(task_step, str):
                try:
                    step_enum = TaskStep(task_step)
                except ValueError:
                    step_enum = TaskStep.PLAN_IMPL

            event = None
            try:
                # Breakpoints: pause before executing a step.
                try:
                    bp_context = {
                        "files_changed": len(
                            _git_changed_files(
                                self.project_dir,
                                include_untracked=True,
                                ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
                            )
                        )
                    }
                    bp = self.breakpoint_manager.check_breakpoint(
                        trigger="before_step",
                        target=step_enum.value,
                        task_id=task_id,
                        context=bp_context,
                    )
                except Exception:
                    bp = None

                if bp and bp.action == "pause":
                    msg = f"Breakpoint hit: {bp.id} ({bp.trigger} {bp.target})"
                    with FileLock(self.lock_path):
                        queue = _load_data(self.paths["task_queue"], {})
                        tasks = _normalize_tasks(queue)
                        task_obj = _find_task(tasks, task_id)
                        if task_obj:
                            task_state = TaskState.from_dict(task_obj)
                            task_state.lifecycle = TaskLifecycle.WAITING_HUMAN
                            task_state.block_reason = "BREAKPOINT_HIT"
                            task_state.last_error_type = "breakpoint_hit"
                            task_state.last_error = msg
                            task_state.human_blocking_issues = [msg]
                            task_state.human_next_steps = ["Resume the task when ready."]
                            task_state.last_run_id = run_id
                            updated = task_state.to_dict()
                            task_obj.clear()
                            task_obj.update(updated)
                            _record_blocked_intent(
                                task_obj,
                                task_status=task_obj.get("status", TASK_STATUS_BLOCKED),
                                task_type=task_obj.get("type", "implement"),
                                phase_id=phase_id,
                                branch=task_obj.get("branch"),
                                test_command=task_obj.get("test_command"),
                                run_id=run_id,
                                step=task_obj.get("step"),
                                lifecycle=task_obj.get("lifecycle"),
                                prompt_mode=task_obj.get("prompt_mode"),
                            )
                            _save_queue(self.paths["task_queue"], queue, tasks)

                    return False, msg

                if step_enum in {TaskStep.PLAN_IMPL, TaskStep.IMPLEMENT, TaskStep.REVIEW}:
                    plan_path = _impl_plan_path(self.paths["artifacts"], phase_id)
                    event = run_worker_action(
                        step=step_enum,
                        task=current_task,
                        phase=phase,
                        prd_path=self.prd_path,
                        project_dir=self.project_dir,
                        artifacts_dir=self.paths["artifacts"],
                        phase_plan_path=self.paths["phase_plan"],
                        task_queue_path=self.paths["task_queue"],
                        run_dir=run_dir,
                        run_id=run_id,
                        codex_command=self.codex_command,
                        user_prompt=None,
                        progress_path=progress_path,
                        events_path=self.paths["events"],
                        heartbeat_seconds=self.heartbeat_seconds,
                        heartbeat_grace_seconds=self.heartbeat_grace_seconds,
                        shift_minutes=self.shift_minutes,
                        test_command=self.test_command,
                        on_spawn=lambda pid: None,
                        simple_review=self.simple_review,
                    )

                elif step_enum == TaskStep.VERIFY:
                    plan_path = _impl_plan_path(self.paths["artifacts"], phase_id)
                    plan_data = _load_data(plan_path, {})
                    event = run_verify_action(
                        project_dir=self.project_dir,
                        artifacts_dir=self.paths["artifacts"],
                        run_dir=run_dir,
                        phase=phase,
                        task=current_task,
                        run_id=run_id,
                        plan_data=plan_data,
                        default_test_command=self.test_command,
                        default_format_command=self.format_command,
                        default_lint_command=self.lint_command,
                        default_typecheck_command=self.typecheck_command,
                        verify_profile=self.verify_profile,
                        ensure_ruff=self.ensure_ruff,
                        ensure_deps=self.ensure_deps,
                        ensure_deps_command=self.ensure_deps_command,
                        timeout_seconds=self.shift_minutes * 60,
                        language=self.language,
                    )

                elif step_enum == TaskStep.COMMIT:
                    commit_message = f"{phase_id}: {phase.get('name') or 'phase'}"
                    event = run_commit_action(
                        project_dir=self.project_dir,
                        branch=branch or "",
                        commit_message=commit_message,
                        run_id=run_id,
                        commit_enabled=self.commit_enabled,
                        push_enabled=self.push_enabled,
                    )

                else:
                    event = WorkerFailed(
                        step=step_enum,
                        run_id=run_id,
                        error_type="invalid_step",
                        error_detail=f"Unknown step: {task_step}",
                        stderr_tail="",
                        timed_out=False,
                        no_heartbeat=False,
                    )

            except Exception as e:
                logger.exception("[Task {}] Action failed: {}", task_id, e)
                event = WorkerFailed(
                    step=step_enum,
                    run_id=run_id,
                    error_type="action_exception",
                    error_detail=str(e),
                    stderr_tail="",
                    timed_out=False,
                    no_heartbeat=False,
                )

            # Record event
            if event:
                _append_event(self.paths["events"], event.to_dict())

            # Process event through FSM (with lock)
            with FileLock(self.lock_path):
                queue = _load_data(self.paths["task_queue"], {})
                tasks = _normalize_tasks(queue)
                plan = _load_data(self.paths["phase_plan"], {})
                phases = _normalize_phases(plan)

                task_obj = _find_task(tasks, task_id)
                if not task_obj:
                    return False, "Task not found after action"

                # Apply FSM transition
                task_state = TaskState.from_dict(task_obj)
                previous_step = task_state.step

                caps = {
                    "worker_attempts": self.max_attempts,
                    "plan_attempts": MAX_IMPL_PLAN_ATTEMPTS,
                    "no_progress_attempts": MAX_NO_PROGRESS_ATTEMPTS,
                    "test_fail_attempts": MAX_TEST_FAIL_ATTEMPTS,
                    "review_gen_attempts": self.max_review_attempts,
                    "review_fix_attempts": self.max_review_attempts,
                    "allowlist_expansion_attempts": MAX_ALLOWLIST_EXPANSION_ATTEMPTS,
                }
                task_state = reduce_task(task_state, event, caps=caps)
                task_state.last_run_id = run_id

                # Breakpoints: pause after completing a step (before proceeding).
                try:
                    bp_context = {
                        "files_changed": len(
                            _git_changed_files(
                                self.project_dir,
                                include_untracked=True,
                                ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
                            )
                        )
                    }
                    bp = self.breakpoint_manager.check_breakpoint(
                        trigger="after_step",
                        target=step_enum.value,
                        task_id=task_id,
                        context=bp_context,
                    )
                except Exception:
                    bp = None

                if bp and bp.action == "pause":
                    msg = f"Breakpoint hit: {bp.id} ({bp.trigger} {bp.target})"
                    task_state.lifecycle = TaskLifecycle.WAITING_HUMAN
                    task_state.block_reason = "BREAKPOINT_HIT"
                    task_state.last_error_type = "breakpoint_hit"
                    task_state.last_error = msg
                    task_state.human_blocking_issues = [msg]
                    task_state.human_next_steps = ["Resume the task when ready."]

                logger.info(
                    "[Task {}] FSM: {} -> {} (lifecycle={})",
                    task_id,
                    previous_step,
                    task_state.step,
                    task_state.lifecycle,
                )

                # Update task in queue
                updated = task_state.to_dict()
                task_obj.clear()
                task_obj.update(updated)

                # Update phase status
                phase_entry = _phase_for_task(phases, task_obj)
                if phase_entry:
                    _sync_phase_status(phase_entry, task_obj)

                _save_queue(self.paths["task_queue"], queue, tasks)

                # Check if task is done or blocked
                if task_state.lifecycle == TaskLifecycle.DONE:
                    logger.info("[Task {}] Completed successfully", task_id)
                    return True, None
                elif task_state.lifecycle == TaskLifecycle.WAITING_HUMAN:
                    error_msg = task_state.last_error or "Task blocked"
                    logger.error(
                        "[Task {}] Blocked: reason={} error={}",
                        task_id,
                        task_state.block_reason,
                        error_msg,
                    )
                    _record_blocked_intent(
                        task_obj,
                        task_status=task_state.status or task_step,
                        task_type="implement",
                        phase_id=phase_id,
                        branch=branch,
                        test_command=self.test_command,
                        run_id=run_id,
                        step=task_state.step,
                        lifecycle=task_state.lifecycle,
                        prompt_mode=task_state.prompt_mode,
                    )
                    return False, error_msg

                # Continue to next step
                logger.debug("[Task {}] Continuing to next step: {}", task_id, task_state.step)

        # Safety limit reached
        logger.error("[Task {}] Maximum iterations ({}) reached", task_id, max_iterations)
        with state_transaction(
            self.lock_path,
            self.paths["task_queue"],
        ) as state:
            task_obj = state.find_task(task_id)
            if task_obj:
                task_obj["lifecycle"] = TaskLifecycle.WAITING_HUMAN.value
                task_obj["block_reason"] = "MAX_ITERATIONS"
                task_obj["last_error"] = f"Task execution exceeded {max_iterations} iterations"
                task_obj["status"] = TASK_STATUS_BLOCKED
        return False, "Maximum iterations reached"


# NOTE: PhaseExecutor now provides complete parallel execution support:
#
# ✅ Uses the complete FSM reduce_task logic for each step
# ✅ Calls run_worker_action, run_verify_action, run_commit_action
# ✅ Handles all error cases and state transitions properly
# ✅ Supports all configuration options
# ✅ Properly manages allowlists and verification
# ✅ Thread-safe git operations via GitCoordinator
# ✅ Thread-safe state updates via FileLock
#
# Each phase executor runs all tasks for a phase sequentially (following the FSM),
# while different phases can run in parallel using ParallelExecutor.
