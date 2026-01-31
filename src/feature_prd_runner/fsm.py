"""Reduce `TaskState` by applying emitted events in a finite-state machine."""

from __future__ import annotations

import shlex
from typing import Any

from .constants import (
    ERROR_TYPE_BLOCKING_ISSUES,
)
from .models import (
    AllowlistViolation,
    CommitResult,
    NoIntroducedChanges,
    ProgressHumanBlockers,
    PromptMode,
    ResumePromptResult,
    ReviewResult,
    TaskLifecycle,
    TaskState,
    TaskStep,
    VerificationResult,
    WorkerFailed,
    WorkerSucceeded,
)


def _set_ready(task: TaskState) -> None:
    task.lifecycle = TaskLifecycle.READY


def _set_waiting(task: TaskState, reason: str, error_type: str, error_detail: str) -> None:
    task.lifecycle = TaskLifecycle.WAITING_HUMAN
    task.block_reason = reason
    task.last_error_type = error_type
    task.last_error = error_detail
    task.prompt_mode = None


def _clear_blocking(task: TaskState) -> None:
    task.block_reason = None
    task.human_blocking_issues = []
    task.human_next_steps = []


def _set_step(task: TaskState, step: TaskStep, prompt_mode: PromptMode | None = None) -> None:
    task.step = step
    task.prompt_mode = prompt_mode


def _record_run_id(task: TaskState, run_id: str) -> None:
    task.last_run_id = run_id


def _infer_missing_tool(command: str | None, log_tail: str | None) -> str | None:
    """Infer a missing tool name from a verification command/log."""
    cmd = (command or "").strip()
    if cmd:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        tokens = [t for t in tokens if t]
        if tokens:
            # python -m <module> ...
            if tokens[0] == "python" and "-m" in tokens:
                for i, tok in enumerate(tokens[:-1]):
                    if tok == "-m":
                        mod = tokens[i + 1].strip()
                        if mod:
                            return mod
            # direct tool invocation
            tool = tokens[0].strip()
            if tool and tool not in {"python"}:
                return tool

    tail = (log_tail or "").strip().lower()
    if tail.endswith(" not found"):
        return tail.split()[0].strip() or None
    return None


def _select_fix_prompt_mode(error_type: str | None) -> PromptMode:
    """Select the appropriate granular PromptMode based on verification error type.

    Args:
        error_type: The error type from verification (e.g., "format_failed", "lint_failed").

    Returns:
        The most specific PromptMode for the error type.
    """
    error_type = (error_type or "").strip().lower()
    if error_type == "format_failed":
        return PromptMode.FIX_FORMAT
    if error_type == "lint_failed":
        return PromptMode.FIX_LINT
    if error_type == "typecheck_failed":
        return PromptMode.FIX_TYPECHECK
    if error_type == "tests_failed":
        return PromptMode.FIX_TESTS
    # Default fallback for unknown error types
    return PromptMode.FIX_VERIFY


def reduce_task(task: TaskState, event: Any, *, caps: dict[str, int]) -> TaskState:
    """Reduce a task state given an event.

    Args:
        task: Current task state to update.
        event: Event instance produced by a worker/verify/review/commit step.
        caps: Attempt caps used to decide when to block rather than retry.

    Returns:
        The updated task state (mutated in-place and also returned for convenience).
    """
    if isinstance(event, ProgressHumanBlockers):
        _record_run_id(task, event.run_id)
        task.human_blocking_issues = list(event.issues)
        task.human_next_steps = list(event.next_steps)
        issue_summary = "; ".join(event.issues).strip() or "Human intervention required"
        _set_waiting(task, "HUMAN_REQUIRED", ERROR_TYPE_BLOCKING_ISSUES, issue_summary)
        return task

    if isinstance(event, AllowlistViolation):
        _record_run_id(task, event.run_id)
        task.last_changed_files = list(event.changed_files)
        task.plan_expansion_request = list(event.disallowed_paths)
        task.allowlist_expansion_attempts += 1
        task.last_error_type = "allowlist_violation"
        task.last_error = "Changes outside allowed files"
        if task.allowlist_expansion_attempts >= caps.get("allowlist_expansion_attempts", 3):
            _set_waiting(task, "ALLOWLIST_EXPANSION_EXHAUSTED", "allowlist_violation", task.last_error)
            return task
        _clear_blocking(task)
        _set_step(task, TaskStep.PLAN_IMPL, PromptMode.EXPAND_ALLOWLIST)
        _set_ready(task)
        return task

    if isinstance(event, WorkerFailed):
        _record_run_id(task, event.run_id)
        task.last_changed_files = list(event.changed_files)
        task.last_error_type = event.error_type
        task.last_error = event.error_detail
        if event.step == TaskStep.REVIEW:
            task.review_gen_attempts += 1
            if task.review_gen_attempts >= caps.get("review_gen_attempts", 3):
                _set_waiting(task, "REVIEW_INVALID", event.error_type, event.error_detail)
                return task
            _set_step(task, TaskStep.REVIEW)
        else:
            task.worker_attempts += 1
            if task.worker_attempts >= caps.get("worker_attempts", 5):
                _set_waiting(task, "WORKER_FAILED", event.error_type, event.error_detail)
                return task
            _set_step(task, event.step, task.prompt_mode)
        _clear_blocking(task)
        _set_ready(task)
        return task

    if isinstance(event, WorkerSucceeded):
        _record_run_id(task, event.run_id)
        task.last_changed_files = list(event.changed_files)
        task.worker_attempts = 0
        if event.step == TaskStep.PLAN_IMPL:
            if event.plan_valid:
                task.plan_attempts = 0
                task.allowlist_expansion_attempts = 0
                task.plan_expansion_request = []
                task.impl_plan_path = event.impl_plan_path
                task.impl_plan_hash = event.impl_plan_hash
                task.last_error = None
                task.last_error_type = None
                _clear_blocking(task)
                _set_step(task, TaskStep.IMPLEMENT, PromptMode.IMPLEMENT)
                _set_ready(task)
                return task

            task.plan_attempts += 1
            issue = event.plan_issue or "Implementation plan invalid"
            task.last_error = issue
            task.last_error_type = "plan_invalid"
            if task.plan_attempts >= caps.get("plan_attempts", 3):
                _set_waiting(task, "PLAN_INVALID", "plan_invalid", issue)
                return task
            _set_step(task, TaskStep.PLAN_IMPL, PromptMode.EXPAND_ALLOWLIST if task.plan_expansion_request else None)
            _clear_blocking(task)
            _set_ready(task)
            return task

        if event.step == TaskStep.IMPLEMENT:
            if event.introduced_changes:
                task.no_progress_attempts = 0
                task.last_error = None
                task.last_error_type = None
                _clear_blocking(task)
                _set_step(task, TaskStep.VERIFY)
                _set_ready(task)
                return task

        _set_step(task, event.step)
        _set_ready(task)
        return task

    if isinstance(event, NoIntroducedChanges):
        _record_run_id(task, event.run_id)
        task.last_changed_files = list(event.changed_files)
        if event.repo_dirty:
            task.no_progress_attempts = 0
            task.last_error = None
            task.last_error_type = None
            _clear_blocking(task)
            _set_step(task, TaskStep.VERIFY)
            _set_ready(task)
            return task

        task.no_progress_attempts += 1
        task.last_error = "No changes detected"
        task.last_error_type = "no_progress"
        if task.no_progress_attempts >= caps.get("no_progress_attempts", 3):
            _set_waiting(task, "NO_PROGRESS", "no_progress", task.last_error)
            return task
        _set_step(task, TaskStep.IMPLEMENT, PromptMode.IMPLEMENT)
        _clear_blocking(task)
        _set_ready(task)
        return task

    if isinstance(event, VerificationResult):
        _record_run_id(task, event.run_id)
        task.last_verification = {
            "command": event.command,
            "exit_code": int(event.exit_code),
            "log_path": event.log_path,
            "log_tail": event.log_tail,
            "captured_at": event.captured_at,
            "error_type": event.error_type,
            "stage": event.stage,
            "all_failing_paths": list(event.all_failing_paths) if event.all_failing_paths else [],
        }
        if event.passed:
            task.test_fail_attempts = 0
            task.last_error = None
            task.last_error_type = None
            _clear_blocking(task)
            _set_step(task, TaskStep.REVIEW)
            _set_ready(task)
            return task

        task.last_error_type = event.error_type or "tests_failed"
        task.last_error = "Verification failed"
        error_type = (event.error_type or "").strip()
        if error_type in {"tool_missing", "deps_install_failed"} or int(event.exit_code or 0) == 127:
            detail = (event.log_tail or "").strip() or (
                "Dependency install failed" if error_type == "deps_install_failed" else "Verification tool not found"
            )
            tool = _infer_missing_tool(event.command, event.log_tail) if error_type != "deps_install_failed" else None
            task.human_blocking_issues = [detail]
            next_steps: list[str] = []
            if error_type == "deps_install_failed":
                if event.command:
                    next_steps.append(f"Retry dependency install: {event.command}")
            elif tool:
                next_steps.append(f"Install missing tool: python -m pip install {tool}")
            if error_type != "deps_install_failed" and event.command:
                next_steps.append(f"Re-run verification: {event.command}")
            next_steps.append(f"Resume runner after fixing: feature-prd-runner resume {task.id} --step verify")
            task.human_next_steps = next_steps
            if error_type == "deps_install_failed":
                _set_waiting(task, "DEPS_INSTALL_FAILED", "deps_install_failed", detail)
            else:
                _set_waiting(task, "TOOL_MISSING", "tool_missing", detail)
            return task
        if event.needs_allowlist_expansion:
            task.plan_expansion_request = list(event.failing_paths)
            task.allowlist_expansion_attempts += 1
            if task.allowlist_expansion_attempts >= caps.get("allowlist_expansion_attempts", 3):
                _set_waiting(task, "ALLOWLIST_EXPANSION_EXHAUSTED", task.last_error_type, task.last_error)
                return task
            _clear_blocking(task)
            _set_step(task, TaskStep.PLAN_IMPL, PromptMode.EXPAND_ALLOWLIST)
            _set_ready(task)
            return task

        task.test_fail_attempts += 1
        if task.test_fail_attempts >= caps.get("test_fail_attempts", 3):
            _set_waiting(task, "TESTS_STUCK", task.last_error_type, task.last_error)
            return task
        _clear_blocking(task)
        # Select granular prompt mode based on the failing verification stage
        prompt_mode = _select_fix_prompt_mode(error_type)
        _set_step(task, TaskStep.IMPLEMENT, prompt_mode)
        _set_ready(task)
        return task

    if isinstance(event, ReviewResult):
        _record_run_id(task, event.run_id)
        task.last_review_path = event.review_path or task.last_review_path
        task.last_review_mergeable = bool(event.mergeable)
        task.last_review_issues = list(event.issues)

        if event.mergeable:
            task.review_fix_attempts = 0
            task.review_blockers = []
            task.review_blocker_files = []
            task.last_error = None
            task.last_error_type = None
            _clear_blocking(task)
            _set_step(task, TaskStep.COMMIT)
            _set_ready(task)
            return task

        task.review_fix_attempts += 1
        task.last_error = "Review blockers found"
        task.last_error_type = "review_blockers"
        task.review_blockers = list(event.issues)
        task.review_blocker_files = []
        _clear_blocking(task)
        _set_step(task, TaskStep.IMPLEMENT, PromptMode.ADDRESS_REVIEW)
        _set_ready(task)
        return task

    if isinstance(event, CommitResult):
        _record_run_id(task, event.run_id)
        if event.committed or event.repo_clean or event.pushed or getattr(event, "skipped", False):
            task.lifecycle = TaskLifecycle.DONE
            task.commit_sha = event.commit_sha or task.commit_sha
            task.last_error = None
            task.last_error_type = None
            task.block_reason = None
            task.prompt_mode = None
            return task

        error = event.error or "Commit or push failed"
        _set_waiting(task, "GIT_PUSH_FAILED", "git_push_failed", error)
        return task

    if isinstance(event, ResumePromptResult):
        _record_run_id(task, event.run_id)
        if event.succeeded:
            # Resume prompt succeeded - continue with normal flow
            task.last_changed_files = list(event.changed_files)
            task.last_error = None
            task.last_error_type = None
            _clear_blocking(task)
            _set_ready(task)
            return task

        # Resume prompt failed
        error = event.error_detail or "Resume prompt failed"
        _set_waiting(task, "RESUME_PROMPT_FAILED", "resume_prompt_failed", error)
        return task

    return task
