#!/usr/bin/env python3
"""Provide the CLI entrypoint and subcommands for Feature PRD Runner.

Coordinates planning, implementing, verifying, reviewing, and committing changes using a step-based FSM.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .constants import (
    DEFAULT_HEARTBEAT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_AUTO_RESUMES,
    DEFAULT_SHIFT_MINUTES,
    DEFAULT_STOP_ON_BLOCKING_ISSUES,
    LOCK_FILE,
    MAX_REVIEW_ATTEMPTS,
    STATE_DIR_NAME,
)
from .orchestrator import run_feature_prd
from .prompts import _build_phase_prompt, _build_plan_prompt, _build_review_prompt
from .io_utils import FileLock, _load_data, _load_data_with_error, _save_data
from .custom_execution import execute_custom_prompt
from .messaging import ApprovalResponse, MessageBus, Message
from .approval_gates import ApprovalGateManager
from .debug import ErrorAnalyzer
from .tasks import (
    _blocking_tasks,
    _find_task,
    _normalize_phases,
    _normalize_tasks,
    _phase_for_task,
    _read_progress_human_blockers,
    _resolve_test_command,
    _select_next_task,
    _summarize_blocking_tasks,
    _task_summary,
 )
from .state import _active_run_is_stale
from .utils import _hash_file, _now_iso
from .validation import validate_phase_plan_schema, validate_task_queue_schema
from .git_utils import _git_is_repo, _git_status_porcelain, _git_changed_files, _git_is_ignored, _git_tracked_paths


__all__ = [
    "run_feature_prd",
    "_build_phase_prompt",
    "_build_plan_prompt",
    "_build_review_prompt",
    "_read_progress_human_blockers",
]

def _configure_logging(level: str = "INFO") -> None:
    """Configure loguru logger with the specified level."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{line}</cyan>\n"
            "{message}"
        ),
    )


# Initialize with default level; will be reconfigured in main() based on CLI args
_configure_logging()


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - autonomous feature implementation coordinator",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--prd-file",
        type=Path,
        required=True,
        help="Path to feature PRD file",
    )
    parser.add_argument(
        "--codex-command",
        type=str,
        default="codex exec -",
        help="Codex CLI command (default: codex exec -)",
    )
    parser.add_argument(
        "--test-command",
        type=str,
        default=None,
        help="Global test command to run after each phase",
    )
    parser.add_argument(
        "--format-command",
        type=str,
        default=None,
        help="Global format check command to run during VERIFY (before lint/tests)",
    )
    parser.add_argument(
        "--lint-command",
        type=str,
        default=None,
        help="Global lint command to run during VERIFY (before tests)",
    )
    parser.add_argument(
        "--typecheck-command",
        type=str,
        default=None,
        help="Global typecheck command to run during VERIFY (before tests)",
    )
    parser.add_argument(
        "--verify-profile",
        type=str,
        choices=["none", "python"],
        default="none",
        help="Verification preset (default: none)",
    )
    parser.add_argument(
        "--ensure-ruff",
        type=str,
        choices=["off", "warn", "install", "add-config"],
        default="off",
        help="Ruff helper behavior for python verify profile (default: off)",
    )
    parser.add_argument(
        "--ensure-deps",
        type=str,
        choices=["off", "install"],
        default="off",
        help=(
            "Dependency helper behavior during VERIFY (default: off). "
            "When enabled, runs an install command before verification."
        ),
    )
    parser.add_argument(
        "--ensure-deps-command",
        type=str,
        default=None,
        help=(
            "Command run when --ensure-deps install is enabled. "
            'Default: python -m pip install -e ".[test]" (fallback to python -m pip install -e .)'
        ),
    )
    parser.add_argument(
        "--new-branch",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Create/switch to a new git branch at the start of the run, then keep using it for all phases "
            "(default: True)"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations (default: unlimited)",
    )
    parser.add_argument(
        "--shift-minutes",
        type=int,
        default=DEFAULT_SHIFT_MINUTES,
        help=f"Timebox per Codex run in minutes (default: {DEFAULT_SHIFT_MINUTES})",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=f"Heartbeat interval in seconds (default: {DEFAULT_HEARTBEAT_SECONDS})",
    )
    parser.add_argument(
        "--heartbeat-grace-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_GRACE_SECONDS,
        help=(
            "Allowed heartbeat staleness before termination "
            f"(default: {DEFAULT_HEARTBEAT_GRACE_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max attempts per task before blocking (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--max-auto-resumes",
        type=int,
        default=DEFAULT_MAX_AUTO_RESUMES,
        help=f"Max auto-resumes for transient failures (default: {DEFAULT_MAX_AUTO_RESUMES})",
    )
    parser.add_argument(
        "--max-review-attempts",
        type=int,
        default=MAX_REVIEW_ATTEMPTS,
        help=f"Max review failure/reimplementation attempts (default: {MAX_REVIEW_ATTEMPTS})",
    )
    parser.add_argument(
        "--stop-on-blocking-issues",
        default=DEFAULT_STOP_ON_BLOCKING_ISSUES,
        action=argparse.BooleanOptionalAction,
        help=(
            "Stop when a task is blocked and requires human intervention "
            "(default: True)"
        ),
    )
    parser.add_argument(
        "--resume-blocked",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resume the most recent blocked task automatically (default: True)",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help=(
            "Standalone prompt to execute before continuing implementation. "
            "The agent must complete the instructions successfully; if blocked, "
            "requires human intervention before continuing."
        ),
    )
    parser.add_argument(
        "--override-agents",
        action="store_true",
        help="Enable superadmin mode for custom-prompt (bypass AGENTS.md rules)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set logging level (default: info)",
    )
    parser.add_argument(
        "--simple-review",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use simplified review output schema (mergeable + issues only, default: True)",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Archive and recreate .prd_runner state before starting",
    )
    parser.add_argument(
        "--require-clean",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Refuse to run if git working tree has changes outside .prd_runner (default: True)",
    )
    parser.add_argument(
        "--commit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable git commit step (default: True)",
    )
    parser.add_argument(
        "--push",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable git push during commit step (default: True)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode with step-by-step approval gates (default: False)",
    )
    return parser


def _build_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - show current .prd_runner state",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    return parser


def _build_dry_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - dry-run (no changes), show what would happen next",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--prd-file",
        type=Path,
        default=None,
        help="Optional PRD path for mismatch checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    return parser


def _build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - diagnostics (read-only)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--prd-file",
        type=Path,
        default=None,
        help="Optional PRD path to validate against run_state",
    )
    parser.add_argument(
        "--check-codex",
        action="store_true",
        help="Check that `codex` is available in PATH (no network calls)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    return parser


def _build_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - list phases and tasks",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="Show tasks (default: show both phases and tasks)",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Show phases (default: show both phases and tasks)",
    )
    parser.add_argument(
        "--blocked",
        action="store_true",
        help="Only show blocked tasks/phases",
    )
    return parser


def _build_resume_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - resume a task by id (update .prd_runner state)",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task id to resume (e.g., phase-1)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["plan_impl", "implement", "verify", "review", "commit"],
        help="Step to resume at (default: task's current step/blocked intent)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resume even if task is not blocked",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON result",
    )
    return parser


def _build_retry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - retry a task by id (clear errors and re-run current step)",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task id to retry (e.g., phase-1)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retry even if another run appears active",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON result",
    )
    return parser


def _build_rerun_step_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - rerun a specific step for a task (update .prd_runner state)",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task id (e.g., phase-1)",
    )
    parser.add_argument(
        "--step",
        required=True,
        type=str,
        choices=["plan_impl", "implement", "verify", "review", "commit"],
        help="Step to rerun",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if another run appears active",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON result",
    )
    return parser


def _build_skip_step_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - skip a step for a task (advance to the next step)",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task id (e.g., phase-1)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["plan_impl", "implement", "verify", "review", "commit"],
        help="Step to skip (default: task's current step/blocked intent)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip even if another run appears active, or if task is not at --step",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON result",
    )
    return parser


def _build_exec_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - execute a custom prompt or ad-hoc task",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Custom prompt/instructions to execute",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--codex-command",
        type=str,
        default="codex exec -",
        help="Codex CLI command (default: codex exec -)",
    )
    parser.add_argument(
        "--override-agents",
        action="store_true",
        help="Enable superadmin mode - bypass AGENTS.md rules",
    )
    parser.add_argument(
        "--then-continue",
        action="store_true",
        help="Continue to normal implementation cycle after completion",
    )
    parser.add_argument(
        "--context-task",
        type=str,
        help="Task ID for context (limits scope to task files)",
    )
    parser.add_argument(
        "--context-files",
        type=str,
        help="Comma-separated list of files to focus on",
    )
    parser.add_argument(
        "--shift-minutes",
        type=int,
        default=DEFAULT_SHIFT_MINUTES,
        help=f"Timebox for execution in minutes (default: {DEFAULT_SHIFT_MINUTES})",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=f"Heartbeat interval in seconds (default: {DEFAULT_HEARTBEAT_SECONDS})",
    )
    parser.add_argument(
        "--heartbeat-grace-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_GRACE_SECONDS,
        help=f"Heartbeat grace period in seconds (default: {DEFAULT_HEARTBEAT_GRACE_SECONDS})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging verbosity (default: info)",
    )
    return parser


def _exec_command(
    project_dir: Path,
    prompt: str,
    codex_command: str,
    override_agents: bool,
    then_continue: bool,
    context_task: Optional[str],
    context_files: Optional[str],
    shift_minutes: int,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
) -> int:
    """Execute a custom prompt."""
    project_dir = project_dir.resolve()

    # Build context dict
    context: dict[str, Any] = {}
    if context_task:
        context["task_id"] = context_task
    if context_files:
        context["files"] = [f.strip() for f in context_files.split(",")]

    # Execute
    success, error = execute_custom_prompt(
        user_prompt=prompt,
        project_dir=project_dir,
        codex_command=codex_command,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_grace_seconds=heartbeat_grace_seconds,
        shift_minutes=shift_minutes,
        override_agents=override_agents,
        context=context if context else None,
        then_continue=then_continue,
    )

    if success:
        logger.info("✓ Custom prompt executed successfully")
        return 0
    else:
        logger.error("✗ Custom prompt failed: {}", error or "Unknown error")
        return 1


def _build_approve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - approve a pending approval request",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID (auto-detect if not specified)",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        help="Optional feedback message",
    )
    return parser


def _build_reject_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - reject a pending approval request",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID (auto-detect if not specified)",
    )
    parser.add_argument(
        "--reason",
        type=str,
        required=True,
        help="Reason for rejection",
    )
    return parser


def _build_steer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - send steering messages to running worker",
    )
    parser.add_argument(
        "message",
        type=str,
        nargs="?",
        help="Steering message to send (interactive mode if not specified)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID (auto-detect if not specified)",
    )
    return parser


def _approve_command(project_dir: Path, run_id: Optional[str], feedback: Optional[str]) -> int:
    """Approve a pending approval request."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        sys.stdout.write("No .prd_runner state directory found\n")
        return 1

    # Auto-detect run_id if not specified
    if not run_id:
        run_state_path = state_dir / "run_state.yaml"
        run_state = _load_data(run_state_path, {})
        run_id = run_state.get("run_id")

    if not run_id:
        sys.stdout.write("No active run found\n")
        return 1

    # Find progress.json for this run
    runs_dir = state_dir / "runs"
    progress_path = None
    for run_dir in runs_dir.glob(f"{run_id}*"):
        candidate = run_dir / "progress.json"
        if candidate.exists():
            progress_path = candidate
            break

    if not progress_path:
        sys.stdout.write(f"No progress.json found for run {run_id}\n")
        return 1

    # Get pending approval
    bus = MessageBus(progress_path)
    pending = bus.get_pending_approval()

    if not pending:
        sys.stdout.write("No pending approval request\n")
        return 1

    # Respond with approval
    from datetime import datetime, timezone

    response = ApprovalResponse(
        request_id=pending.id,
        approved=True,
        feedback=feedback,
        responded_at=datetime.now(timezone.utc).isoformat(),
    )

    bus.respond_to_approval(response)
    sys.stdout.write(f"✓ Approved: {pending.message}\n")
    if feedback:
        sys.stdout.write(f"  Feedback: {feedback}\n")

    return 0


def _reject_command(project_dir: Path, run_id: Optional[str], reason: str) -> int:
    """Reject a pending approval request."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        sys.stdout.write("No .prd_runner state directory found\n")
        return 1

    # Auto-detect run_id if not specified
    if not run_id:
        run_state_path = state_dir / "run_state.yaml"
        run_state = _load_data(run_state_path, {})
        run_id = run_state.get("run_id")

    if not run_id:
        sys.stdout.write("No active run found\n")
        return 1

    # Find progress.json for this run
    runs_dir = state_dir / "runs"
    progress_path = None
    for run_dir in runs_dir.glob(f"{run_id}*"):
        candidate = run_dir / "progress.json"
        if candidate.exists():
            progress_path = candidate
            break

    if not progress_path:
        sys.stdout.write(f"No progress.json found for run {run_id}\n")
        return 1

    # Get pending approval
    bus = MessageBus(progress_path)
    pending = bus.get_pending_approval()

    if not pending:
        sys.stdout.write("No pending approval request\n")
        return 1

    # Respond with rejection
    from datetime import datetime, timezone

    response = ApprovalResponse(
        request_id=pending.id,
        approved=False,
        feedback=reason,
        responded_at=datetime.now(timezone.utc).isoformat(),
    )

    bus.respond_to_approval(response)
    sys.stdout.write(f"✗ Rejected: {pending.message}\n")
    sys.stdout.write(f"  Reason: {reason}\n")

    return 0


def _steer_command(project_dir: Path, run_id: Optional[str], message: Optional[str]) -> int:
    """Send steering message to running worker."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        sys.stdout.write("No .prd_runner state directory found\n")
        return 1

    # Auto-detect run_id if not specified
    if not run_id:
        run_state_path = state_dir / "run_state.yaml"
        run_state = _load_data(run_state_path, {})
        run_id = run_state.get("run_id")

    if not run_id:
        sys.stdout.write("No active run found\n")
        return 1

    # Find progress.json for this run
    runs_dir = state_dir / "runs"
    progress_path = None
    for run_dir in runs_dir.glob(f"{run_id}*"):
        candidate = run_dir / "progress.json"
        if candidate.exists():
            progress_path = candidate
            break

    if not progress_path:
        sys.stdout.write(f"No progress.json found for run {run_id}\n")
        return 1

    bus = MessageBus(progress_path)

    # Interactive mode if no message specified
    if not message:
        sys.stdout.write("=== Interactive Steering Mode ===\n")
        sys.stdout.write("Enter messages to send to the worker (Ctrl+C to exit)\n\n")

        try:
            while True:
                msg = input("> ")
                if msg.strip():
                    bus.send_guidance(msg.strip())
                    sys.stdout.write("✓ Message sent to worker\n")
        except KeyboardInterrupt:
            sys.stdout.write("\nExiting steering mode\n")
            return 0
    else:
        # Send single message
        bus.send_guidance(message)
        sys.stdout.write(f"✓ Message sent to worker: {message}\n")

    return 0


def _build_explain_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - explain why a task is blocked",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID to explain",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    return parser


def _build_inspect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - inspect task state in detail",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID to inspect",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    return parser


def _build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - trace event history for a task",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID to trace",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of events to show (default: 50)",
    )
    return parser


def _build_logs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - view detailed logs for a task",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Specific step to show logs for",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Number of lines to show (default: 100)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run ID (auto-detect if not specified)",
    )
    return parser


def _explain_command(project_dir: Path, task_id: str) -> int:
    """Explain why a task is blocked."""
    project_dir = project_dir.resolve()

    analyzer = ErrorAnalyzer(project_dir)
    explanation = analyzer.explain_blocking(task_id)

    sys.stdout.write(explanation)
    sys.stdout.write("\n")

    return 0


def _inspect_command(project_dir: Path, task_id: str, *, as_json: bool = False) -> int:
    """Inspect task state in detail."""
    project_dir = project_dir.resolve()

    analyzer = ErrorAnalyzer(project_dir)
    snapshot = analyzer.inspect_state(task_id)

    if not snapshot:
        sys.stdout.write(f"Task {task_id} not found\n")
        return 1

    if as_json:
        import json
        from dataclasses import asdict

        sys.stdout.write(json.dumps(asdict(snapshot), indent=2) + "\n")
    else:
        formatted = analyzer.format_state_snapshot(snapshot)
        sys.stdout.write(formatted)

    return 0


def _trace_command(
    project_dir: Path,
    task_id: str,
    *,
    as_json: bool = False,
    limit: int = 50,
) -> int:
    """Trace event history for a task."""
    project_dir = project_dir.resolve()

    analyzer = ErrorAnalyzer(project_dir)
    events = analyzer.trace_history(task_id)

    if not events:
        sys.stdout.write(f"No events found for task {task_id}\n")
        return 0

    # Limit events
    if limit > 0:
        events = events[-limit:]

    if as_json:
        import json

        sys.stdout.write(json.dumps(events, indent=2) + "\n")
    else:
        sys.stdout.write(f"Event History for {task_id} (showing {len(events)} events):\n")
        sys.stdout.write("=" * 80 + "\n\n")

        for i, event in enumerate(events, 1):
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("timestamp", "")
            run_id = event.get("run_id", "")

            sys.stdout.write(f"{i}. [{timestamp}] {event_type}\n")
            if run_id:
                sys.stdout.write(f"   Run ID: {run_id}\n")

            # Show key details based on event type
            if event_type == "worker_failed":
                error = event.get("error_type", "")
                detail = event.get("error_detail", "")
                if error:
                    sys.stdout.write(f"   Error: {error}\n")
                if detail:
                    detail_preview = detail[:100] + "..." if len(detail) > 100 else detail
                    sys.stdout.write(f"   Detail: {detail_preview}\n")

            elif event_type == "verification_result":
                passed = event.get("passed", False)
                status = "✓ PASSED" if passed else "✗ FAILED"
                sys.stdout.write(f"   Status: {status}\n")

            elif event_type == "task_blocked":
                reason = event.get("block_reason", "")
                if reason:
                    sys.stdout.write(f"   Reason: {reason}\n")

            sys.stdout.write("\n")

    return 0


def _logs_command(
    project_dir: Path,
    task_id: str,
    step: Optional[str],
    lines: int,
    run_id: Optional[str],
) -> int:
    """View detailed logs for a task."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        sys.stdout.write("No .prd_runner state directory found\n")
        return 1

    # Auto-detect run_id if not specified
    if not run_id:
        # Find most recent run for this task
        runs_dir = state_dir / "runs"
        if not runs_dir.exists():
            sys.stdout.write("No runs directory found\n")
            return 1

        # Look for runs matching this task
        matching_runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                # Check if this run contains this task
                progress_path = run_dir / "progress.json"
                if progress_path.exists():
                    progress = _load_data(progress_path, {})
                    if progress.get("task_id") == task_id:
                        matching_runs.append(run_dir.name)

        if not matching_runs:
            sys.stdout.write(f"No runs found for task {task_id}\n")
            return 1

        # Use most recent
        run_id = sorted(matching_runs)[-1]

    run_dir = state_dir / "runs" / run_id

    if not run_dir.exists():
        sys.stdout.write(f"Run directory not found: {run_dir}\n")
        return 1

    sys.stdout.write(f"Logs for task {task_id} (run: {run_id})\n")
    sys.stdout.write("=" * 80 + "\n\n")

    # Determine which logs to show
    log_files = []

    if step:
        # Show logs for specific step
        if step == "verify":
            log_files.append(("Test Output", run_dir / "verify_output.txt"))
            log_files.append(("Pytest Failures", run_dir / "pytest_failures.txt"))
        else:
            log_files.append(("stdout", run_dir / "stdout.log"))
            log_files.append(("stderr", run_dir / "stderr.log"))
    else:
        # Show all available logs
        log_files.append(("stdout", run_dir / "stdout.log"))
        log_files.append(("stderr", run_dir / "stderr.log"))
        log_files.append(("Test Output", run_dir / "verify_output.txt"))

    for label, log_path in log_files:
        if log_path.exists():
            sys.stdout.write(f"--- {label} ---\n")
            try:
                content = log_path.read_text()
                # Show last N lines
                log_lines = content.splitlines()
                if len(log_lines) > lines:
                    sys.stdout.write(f"(showing last {lines} lines)\n\n")
                    log_lines = log_lines[-lines:]
                sys.stdout.write("\n".join(log_lines))
                sys.stdout.write("\n\n")
            except Exception as e:
                sys.stdout.write(f"Error reading log: {e}\n\n")

    return 0


def _status_command(project_dir: Path, *, as_json: bool = False) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    phase_plan_path = state_dir / "phase_plan.yaml"
    runs_dir = state_dir / "runs"
    blocked_report_path = state_dir / "runner_blocked.json"

    if not state_dir.exists():
        if as_json:
            sys.stdout.write('{"status":"missing_state_dir"}\n')
        else:
            sys.stdout.write(f"No state directory found at {state_dir}\n")
        return 0

    run_state, run_state_err = _load_data_with_error(run_state_path, {})
    queue, queue_err = _load_data_with_error(task_queue_path, {})
    plan, plan_err = _load_data_with_error(phase_plan_path, {})
    errors = [e for e in [run_state_err, queue_err, plan_err] if e]

    blocked_report = _load_data(blocked_report_path, {}) if blocked_report_path.exists() else {}

    tasks = _normalize_tasks(queue)
    phases = _normalize_phases(plan)
    next_task = _select_next_task(tasks) if tasks else None
    blockers = _blocking_tasks(tasks)

    payload = {
        "project_dir": str(project_dir),
        "state_dir": str(state_dir),
        "errors": errors,
        "run_state": run_state,
        "task_summary": _task_summary(tasks),
        "next_task": next_task,
        "blocking_tasks": blockers,
        "blocking_summary": _summarize_blocking_tasks(blockers),
        "blocked_report": blocked_report or None,
    }
    if as_json:
        import json

        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 2 if errors else 0

    sys.stdout.write(f"Project: {project_dir}\n")
    sys.stdout.write(f"State:   {state_dir}\n")
    if errors:
        sys.stdout.write("State errors:\n")
        for err in errors:
            sys.stdout.write(f"- {err}\n")

    status = run_state.get("status") or "unknown"
    sys.stdout.write(f"Run status: {status}\n")
    last_error = run_state.get("last_error")
    if last_error:
        sys.stdout.write(f"Last error: {last_error}\n")
    current_task_id = run_state.get("current_task_id")
    current_phase_id = run_state.get("current_phase_id")
    run_id = run_state.get("run_id")
    last_run_id = run_state.get("last_run_id")
    branch = run_state.get("branch")
    if current_task_id or current_phase_id or run_id:
        sys.stdout.write(
            f"Current: task={current_task_id or '-'} phase={current_phase_id or '-'} run_id={run_id or '-'}\n"
        )
    if branch:
        sys.stdout.write(f"Branch:  {branch}\n")

    summary = _task_summary(tasks)
    sys.stdout.write(
        "Tasks: "
        + ", ".join(f"{k}={v}" for k, v in summary.items())
        + f" (total={len(tasks)})\n"
    )
    if next_task:
        phase = _phase_for_task(phases, next_task)
        phase_name = (phase or {}).get("name") or (phase or {}).get("id")
        sys.stdout.write(
            f"Next:   {next_task.get('id')} step={next_task.get('step')} "
            f"lifecycle={next_task.get('lifecycle')} phase={phase_name or '-'}\n"
        )

    if blockers:
        sys.stdout.write(f"Blocked: {len(blockers)}\n")
        sys.stdout.write(_summarize_blocking_tasks(blockers) + "\n")
        for task in blockers[:5]:
            sys.stdout.write(
                f"- {task.get('id')}: {task.get('last_error_type') or 'unknown'}: {task.get('last_error') or ''}\n"
            )
    if blocked_report:
        sys.stdout.write(
            f"Runner blocked report: {blocked_report.get('error_type')}: {blocked_report.get('summary')}\n"
        )
        sys.stdout.write(f"Details: {blocked_report_path}\n")

    if last_run_id and (runs_dir / str(last_run_id)).exists():
        sys.stdout.write(f"Last run dir: {runs_dir / str(last_run_id)}\n")
    elif last_run_id:
        sys.stdout.write(f"Last run id: {last_run_id}\n")

    return 2 if errors else 0


def _load_state_for_control_plane(
    project_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    phase_plan_path = state_dir / "phase_plan.yaml"

    run_state, run_state_err = _load_data_with_error(run_state_path, {})
    queue, queue_err = _load_data_with_error(task_queue_path, {})
    plan, plan_err = _load_data_with_error(phase_plan_path, {})
    errors = [e for e in [run_state_err, queue_err, plan_err] if e]
    return run_state, queue, plan, errors


def _list_command(
    project_dir: Path,
    *,
    show_tasks: bool,
    show_phases: bool,
    blocked_only: bool,
    as_json: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    if not state_dir.exists():
        if as_json:
            import json
            sys.stdout.write(json.dumps({"status": "missing_state_dir", "state_dir": str(state_dir)}) + "\n")
        else:
            sys.stdout.write(f"No state directory found at {state_dir}\n")
        return 0

    run_state, queue, plan, errors = _load_state_for_control_plane(project_dir)
    tasks = _normalize_tasks(queue)
    phases = _normalize_phases(plan)

    if not show_tasks and not show_phases:
        show_tasks = True
        show_phases = True

    if blocked_only:
        tasks = _blocking_tasks(tasks)
        phases = [p for p in phases if p.get("status") == "blocked"]

    payload = {
        "project_dir": str(project_dir),
        "state_dir": str(state_dir),
        "errors": errors,
        "run_state": run_state,
        "phases": phases if show_phases else None,
        "tasks": tasks if show_tasks else None,
    }

    if as_json:
        import json
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 2 if errors else 0

    if errors:
        sys.stdout.write("State errors:\n")
        for err in errors:
            sys.stdout.write(f"- {err}\n")

    if show_phases:
        sys.stdout.write("Phases:\n")
        for phase in phases:
            sys.stdout.write(
                f"- {phase.get('id')}: {phase.get('status') or 'todo'}"
                f" ({phase.get('name') or ''})\n"
            )
    if show_tasks:
        sys.stdout.write("Tasks:\n")
        for task in tasks:
            sys.stdout.write(
                f"- {task.get('id')}: {task.get('lifecycle')}/{task.get('step')}"
                f" status={task.get('status')} type={task.get('type')}\n"
            )

    return 2 if errors else 0


def _resume_command(
    project_dir: Path,
    task_id: str,
    *,
    step: str | None,
    force: bool,
    as_json: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    if not state_dir.exists():
        msg = f"No state directory found at {state_dir}"
        if as_json:
            import json
            sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
        else:
            sys.stdout.write(msg + "\n")
        return 2

    lock_path = state_dir / LOCK_FILE
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    runs_dir = state_dir / "runs"

    with FileLock(lock_path):
        run_state, run_state_err = _load_data_with_error(run_state_path, {})
        if run_state_err:
            msg = f"Unable to read run_state.yaml: {run_state_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        if run_state.get("status") == "running":
            stale = _active_run_is_stale(
                run_state,
                runs_dir,
                heartbeat_grace_seconds=DEFAULT_HEARTBEAT_GRACE_SECONDS,
                shift_minutes=DEFAULT_SHIFT_MINUTES,
            )
            if not stale and not force:
                msg = "Another run appears active; refusing to resume without --force"
                if as_json:
                    import json
                    sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
                else:
                    sys.stdout.write(msg + "\n")
                return 2

        queue, queue_err = _load_data_with_error(task_queue_path, {})
        if queue_err:
            msg = f"Unable to read task_queue.yaml: {queue_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2
        tasks = _normalize_tasks(queue)
        target = _find_task(tasks, task_id)
        if not target:
            msg = f"Task not found: {task_id}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        is_blocked = target.get("lifecycle") == "waiting_human" or target.get("status") == "blocked"
        if not force and not is_blocked:
            msg = f"Task {task_id} is not blocked; use --force to resume anyway"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg, "task": target}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        restore_step = step
        restore_prompt_mode = None
        if not restore_step:
            intent = target.get("blocked_intent") or {}
            restore_step = intent.get("step") or target.get("step") or "plan_impl"
            pm = intent.get("prompt_mode")
            restore_prompt_mode = str(pm).strip() if isinstance(pm, str) and pm.strip() else None
        restore_step = str(restore_step)

        target["lifecycle"] = "ready"
        target["step"] = restore_step
        target["status"] = restore_step
        if step:
            target["prompt_mode"] = None
        elif restore_prompt_mode is not None:
            target["prompt_mode"] = restore_prompt_mode
        target["last_error"] = None
        target["last_error_type"] = None
        target["block_reason"] = None
        target["human_blocking_issues"] = []
        target["human_next_steps"] = []
        # Keep timestamps coherent; orchestrator also updates queue.updated_at.
        target["last_updated_at"] = _now_iso()
        target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1

        queue["tasks"] = tasks
        queue["updated_at"] = target["last_updated_at"]
        _save_data(task_queue_path, queue)

    if as_json:
        import json
        sys.stdout.write(json.dumps({"ok": True, "task_id": task_id, "resumed_step": restore_step}) + "\n")
    else:
        sys.stdout.write(f"Resumed {task_id} at step={restore_step}\n")
    return 0


def _append_task_context(task: dict[str, Any], note: str) -> None:
    context = task.get("context", [])
    if not isinstance(context, list):
        context = [str(context)]
    note = str(note).strip()
    if note and note not in context:
        context.append(note)
    task["context"] = context


_CONTROL_PLANE_STEP_ORDER = ["plan_impl", "implement", "verify", "review", "commit"]
_CONTROL_PLANE_STEP_INDEX = {name: idx for idx, name in enumerate(_CONTROL_PLANE_STEP_ORDER)}


def _control_plane_run_active_guard(
    run_state: dict[str, Any],
    runs_dir: Path,
    *,
    force: bool,
) -> str | None:
    if run_state.get("status") != "running":
        return None
    stale = _active_run_is_stale(
        run_state,
        runs_dir,
        heartbeat_grace_seconds=DEFAULT_HEARTBEAT_GRACE_SECONDS,
        shift_minutes=DEFAULT_SHIFT_MINUTES,
    )
    if stale or force:
        return None
    return "Another run appears active; refusing to modify state without --force"


def _retry_command(
    project_dir: Path,
    task_id: str,
    *,
    force: bool,
    as_json: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    if not state_dir.exists():
        msg = f"No state directory found at {state_dir}"
        if as_json:
            import json
            sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
        else:
            sys.stdout.write(msg + "\n")
        return 2

    lock_path = state_dir / LOCK_FILE
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    runs_dir = state_dir / "runs"

    with FileLock(lock_path):
        run_state, run_state_err = _load_data_with_error(run_state_path, {})
        if run_state_err:
            msg = f"Unable to read run_state.yaml: {run_state_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        active_msg = _control_plane_run_active_guard(run_state, runs_dir, force=force)
        if active_msg:
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": active_msg}) + "\n")
            else:
                sys.stdout.write(active_msg + "\n")
            return 2

        queue, queue_err = _load_data_with_error(task_queue_path, {})
        if queue_err:
            msg = f"Unable to read task_queue.yaml: {queue_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        tasks = _normalize_tasks(queue)
        target = _find_task(tasks, task_id)
        if not target:
            msg = f"Task not found: {task_id}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        intent = target.get("blocked_intent") or {}
        restore_step = str(target.get("step") or intent.get("step") or "plan_impl").strip()
        if restore_step not in _CONTROL_PLANE_STEP_INDEX:
            msg = (
                f"Task {task_id} has unexpected step={restore_step!r}; "
                "use `rerun-step --step {plan_impl,implement,verify,review,commit}` to set a valid step"
            )
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2
        restore_pm = target.get("prompt_mode")
        if restore_pm is None:
            pm = intent.get("prompt_mode")
            restore_pm = str(pm).strip() if isinstance(pm, str) and pm.strip() else None

        target["lifecycle"] = "ready"
        target["step"] = restore_step
        target["status"] = restore_step
        target["prompt_mode"] = restore_pm
        target["last_error"] = None
        target["last_error_type"] = None
        target["block_reason"] = None
        target["human_blocking_issues"] = []
        target["human_next_steps"] = []
        target["last_updated_at"] = _now_iso()
        target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1
        _append_task_context(target, f"Manual retry requested (step={restore_step}).")

        queue["tasks"] = tasks
        queue["updated_at"] = target["last_updated_at"]
        _save_data(task_queue_path, queue)

    if as_json:
        import json
        sys.stdout.write(json.dumps({"ok": True, "task_id": task_id, "step": restore_step}) + "\n")
    else:
        sys.stdout.write(f"Retry queued for {task_id} at step={restore_step}\n")
    return 0


def _rerun_step_command(
    project_dir: Path,
    task_id: str,
    *,
    step: str,
    force: bool,
    as_json: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    if not state_dir.exists():
        msg = f"No state directory found at {state_dir}"
        if as_json:
            import json
            sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
        else:
            sys.stdout.write(msg + "\n")
        return 2

    lock_path = state_dir / LOCK_FILE
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    runs_dir = state_dir / "runs"

    with FileLock(lock_path):
        run_state, run_state_err = _load_data_with_error(run_state_path, {})
        if run_state_err:
            msg = f"Unable to read run_state.yaml: {run_state_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        active_msg = _control_plane_run_active_guard(run_state, runs_dir, force=force)
        if active_msg:
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": active_msg}) + "\n")
            else:
                sys.stdout.write(active_msg + "\n")
            return 2

        queue, queue_err = _load_data_with_error(task_queue_path, {})
        if queue_err:
            msg = f"Unable to read task_queue.yaml: {queue_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        tasks = _normalize_tasks(queue)
        target = _find_task(tasks, task_id)
        if not target:
            msg = f"Task not found: {task_id}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        step = str(step).strip()
        target["lifecycle"] = "ready"
        target["step"] = step
        target["status"] = step
        target["prompt_mode"] = None
        target["last_error"] = None
        target["last_error_type"] = None
        target["block_reason"] = None
        target["human_blocking_issues"] = []
        target["human_next_steps"] = []
        target["last_updated_at"] = _now_iso()
        target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1
        _append_task_context(target, f"Manual rerun requested (step={step}).")

        queue["tasks"] = tasks
        queue["updated_at"] = target["last_updated_at"]
        _save_data(task_queue_path, queue)

    if as_json:
        import json
        sys.stdout.write(json.dumps({"ok": True, "task_id": task_id, "step": step}) + "\n")
    else:
        sys.stdout.write(f"Rerun queued for {task_id} at step={step}\n")
    return 0


def _skip_step_next(step: str) -> tuple[str | None, str]:
    step = str(step).strip()
    idx = _CONTROL_PLANE_STEP_INDEX.get(step)
    if idx is None:
        return None, step
    if idx >= len(_CONTROL_PLANE_STEP_ORDER) - 1:
        return None, step
    return _CONTROL_PLANE_STEP_ORDER[idx + 1], step


def _skip_step_command(
    project_dir: Path,
    task_id: str,
    *,
    step: str | None,
    force: bool,
    as_json: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME
    if not state_dir.exists():
        msg = f"No state directory found at {state_dir}"
        if as_json:
            import json
            sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
        else:
            sys.stdout.write(msg + "\n")
        return 2

    lock_path = state_dir / LOCK_FILE
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    runs_dir = state_dir / "runs"

    with FileLock(lock_path):
        run_state, run_state_err = _load_data_with_error(run_state_path, {})
        if run_state_err:
            msg = f"Unable to read run_state.yaml: {run_state_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        active_msg = _control_plane_run_active_guard(run_state, runs_dir, force=force)
        if active_msg:
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": active_msg}) + "\n")
            else:
                sys.stdout.write(active_msg + "\n")
            return 2

        queue, queue_err = _load_data_with_error(task_queue_path, {})
        if queue_err:
            msg = f"Unable to read task_queue.yaml: {queue_err}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        tasks = _normalize_tasks(queue)
        target = _find_task(tasks, task_id)
        if not target:
            msg = f"Task not found: {task_id}"
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        intent = target.get("blocked_intent") or {}
        requested_step = step
        if not requested_step:
            requested_step = str(intent.get("step") or target.get("step") or "plan_impl")
        requested_step = str(requested_step).strip()
        if requested_step not in _CONTROL_PLANE_STEP_INDEX:
            msg = (
                f"Task {task_id} has unexpected step={requested_step!r}; "
                "pass `--step` explicitly to skip a valid step"
            )
            if as_json:
                import json
                sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
            else:
                sys.stdout.write(msg + "\n")
            return 2

        current_step = str(target.get("step") or "").strip()
        if not force and current_step and current_step != requested_step:
            msg = f"Task {task_id} is at step={current_step}; refusing to skip step={requested_step} without --force"
            if as_json:
                import json
                sys.stdout.write(
                    json.dumps(
                        {
                            "ok": False,
                            "error": msg,
                            "task_id": task_id,
                            "current_step": current_step,
                            "requested_step": requested_step,
                        }
                    )
                    + "\n"
                )
            else:
                sys.stdout.write(msg + "\n")
            return 2

        next_step, normalized_step = _skip_step_next(requested_step)
        if next_step is None:
            target["lifecycle"] = "done"
            target["step"] = "commit"
            target["status"] = "done"
            target["prompt_mode"] = None
            _append_task_context(target, f"Manual skip requested (step={normalized_step}); marking done.")
        else:
            target["lifecycle"] = "ready"
            target["step"] = next_step
            target["status"] = next_step
            target["prompt_mode"] = None
            _append_task_context(target, f"Manual skip requested (step={normalized_step}); advancing to {next_step}.")

        target["last_error"] = None
        target["last_error_type"] = None
        target["block_reason"] = None
        target["human_blocking_issues"] = []
        target["human_next_steps"] = []
        target["last_updated_at"] = _now_iso()
        target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1

        queue["tasks"] = tasks
        queue["updated_at"] = target["last_updated_at"]
        _save_data(task_queue_path, queue)

    if as_json:
        import json
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "task_id": task_id,
                    "skipped_step": normalized_step,
                    "next_step": next_step,
                    "marked_done": next_step is None,
                }
            )
            + "\n"
        )
    else:
        if next_step is None:
            sys.stdout.write(f"Skipped {normalized_step} for {task_id}; marked done\n")
        else:
            sys.stdout.write(f"Skipped {normalized_step} for {task_id}; next step={next_step}\n")
    return 0


def _dry_run_command(project_dir: Path, prd_path: Path | None, *, as_json: bool = False) -> int:
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve() if prd_path else None
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    phase_plan_path = state_dir / "phase_plan.yaml"
    runs_dir = state_dir / "runs"
    gitignore_path = project_dir / ".gitignore"

    result: dict[str, Any] = {
        "project_dir": str(project_dir),
        "state_dir": str(state_dir),
        "dry_run_guarantees": {
            "writes_repo_files": False,
            "writes_state_files": False,
            "spawns_codex": False,
            "runs_tests": False,
            "checks_out_branch": False,
        },
        "would_write_repo_files": False,
        "would_write_state_files": False,
        "would_spawn_codex": False,
        "would_run_tests": False,
        "would_checkout_branch": False,
        "next": None,
        "warnings": [],
        "errors": [],
    }

    if not state_dir.exists():
        result["next"] = {
            "action": "initialize_state",
            "details": f"Would create {state_dir} and initial state files (plan task).",
        }
        if as_json:
            import json
            sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return 0
        sys.stdout.write(f"Project: {project_dir}\n")
        sys.stdout.write(f"State:   {state_dir} (missing)\n")
        sys.stdout.write("Next:    would initialize state and start PLAN\n")
        return 0

    run_state, run_state_err = _load_data_with_error(run_state_path, {})
    queue, queue_err = _load_data_with_error(task_queue_path, {})
    plan, plan_err = _load_data_with_error(phase_plan_path, {})
    errors = [e for e in [run_state_err, queue_err, plan_err] if e]
    if errors:
        result["errors"] = errors

    phases = _normalize_phases(plan)
    tasks = _normalize_tasks(queue)
    phase_ids = {str(p.get("id")).strip() for p in phases if isinstance(p, dict) and str(p.get("id") or "").strip()}
    schema_issues = validate_phase_plan_schema(plan) + validate_task_queue_schema(queue, phase_ids=phase_ids)
    if schema_issues:
        existing = result.get("errors")
        existing_errors = existing if isinstance(existing, list) else []
        result["errors"] = [str(item) for item in existing_errors] + schema_issues

    # Best-effort: indicate whether a real run would modify .gitignore to ignore .prd_runner.
    if _git_is_repo(project_dir):
        status_text, _ = _git_status_porcelain(project_dir)
        repo_porcelain_clean = not bool(str(status_text or "").strip())
        ignored = _git_is_ignored(project_dir, STATE_DIR_NAME)
        if not ignored and repo_porcelain_clean:
            try:
                contents = gitignore_path.read_text() if gitignore_path.exists() else ""
                lines = [
                    line.strip()
                    for line in contents.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                has_entry = any(line.rstrip("/") == STATE_DIR_NAME for line in lines)
            except OSError:
                has_entry = False
            if not has_entry:
                result["would_write_repo_files"] = True
                existing = result.get("warnings")
                existing_warnings = existing if isinstance(existing, list) else []
                result["warnings"] = [str(item) for item in existing_warnings] + [
                    "Would add .prd_runner/ to .gitignore (repo appears clean)"
                ]

    if prd_path is not None:
        stored_prd_path = run_state.get("prd_path")
        stored_prd_hash = run_state.get("prd_hash")
        current_hash = _hash_file(str(prd_path))
        if stored_prd_path:
            try:
                stored_resolved = Path(str(stored_prd_path)).expanduser().resolve()
            except Exception:
                stored_resolved = None
            if stored_resolved and stored_resolved != prd_path:
                existing = result.get("errors")
                existing_errors = existing if isinstance(existing, list) else []
                result["errors"] = [str(item) for item in existing_errors] + [
                    f"PRD mismatch: stored={stored_resolved} requested={prd_path}"
                ]
        if stored_prd_hash and current_hash and str(stored_prd_hash).strip() != current_hash:
            existing = result.get("errors")
            existing_errors = existing if isinstance(existing, list) else []
            result["errors"] = [str(item) for item in existing_errors] + [
                "PRD content hash mismatch (would block without --reset-state)"
            ]

    status = str(run_state.get("status") or "")
    if status == "running":
        stale = _active_run_is_stale(
            run_state,
            runs_dir,
            heartbeat_grace_seconds=DEFAULT_HEARTBEAT_GRACE_SECONDS,
            shift_minutes=DEFAULT_SHIFT_MINUTES,
        )
        if not stale:
            result["next"] = {"action": "exit", "details": "Another run appears active; runner would exit."}
        else:
            result["warnings"] = ["run_state indicates running but appears stale; runner would resume"]

    next_task = _select_next_task(tasks) if tasks else None
    if not tasks and not result.get("next"):
        result["would_write_state_files"] = True
        result["next"] = {
            "action": "initialize_tasks",
            "details": "Would create a default plan task in .prd_runner/task_queue.yaml and start PLAN.",
        }
    if next_task and not result.get("next"):
        step = str(next_task.get("step") or "")
        task_type = str(next_task.get("type") or "implement")
        phase = _phase_for_task(phases, next_task) if task_type != "plan" else None

        action = "run_worker"
        will_spawn_codex = step in {"plan_impl", "implement", "review"} or task_type == "plan"
        will_run_tests = step == "verify"
        will_checkout_branch = task_type != "plan" and phase is not None

        test_command = _resolve_test_command(phase, next_task, None) if will_run_tests else None
        branch = None
        if will_checkout_branch and phase:
            phase_id = phase.get("id") or next_task.get("phase_id") or next_task.get("id")
            branch = phase.get("branch") or next_task.get("branch") or f"feature/{phase_id}"

        result["would_write_state_files"] = True
        result["would_spawn_codex"] = bool(will_spawn_codex)
        result["would_run_tests"] = bool(will_run_tests)
        result["would_checkout_branch"] = bool(will_checkout_branch)
        result["next"] = {
            "task_id": next_task.get("id"),
            "type": task_type,
            "step": step,
            "phase_id": (phase or {}).get("id") if phase else None,
            "branch": branch,
            "test_command": test_command,
            "action": action,
        }

    if as_json:
        import json
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        errors_value = result.get("errors")
        errors_list = errors_value if isinstance(errors_value, list) else []
        return 2 if errors_list else 0

    sys.stdout.write(f"Project: {project_dir}\n")
    sys.stdout.write(f"State:   {state_dir}\n")
    errors_value = result.get("errors")
    warnings_value = result.get("warnings")
    errs = errors_value if isinstance(errors_value, list) else []
    warns = warnings_value if isinstance(warnings_value, list) else []
    if errs:
        sys.stdout.write("Errors:\n")
        for e in errs:
            sys.stdout.write(f"- {e}\n")
    if warns:
        sys.stdout.write("Warnings:\n")
        for w in warns:
            sys.stdout.write(f"- {w}\n")
    nxt = result.get("next")
    sys.stdout.write(f"Next:    {nxt}\n")
    sys.stdout.write("Dry-run guarantees: no git changes, no state writes, no codex/test execution.\n")
    return 2 if errs else 0


def _doctor_command(project_dir: Path, prd_path: Path | None, *, check_codex: bool = False, as_json: bool = False) -> int:
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve() if prd_path else None
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    phase_plan_path = state_dir / "phase_plan.yaml"
    runs_dir = state_dir / "runs"
    blocked_report_path = state_dir / "runner_blocked.json"

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {"project_dir": str(project_dir), "state_dir": str(state_dir)}

    if not project_dir.exists():
        errors.append("project_dir does not exist")

    is_git = _git_is_repo(project_dir)
    checks["git_repo"] = bool(is_git)
    if not is_git:
        warnings.append("Not a git repository (some runner features will not work)")
    else:
        status_text, _ = _git_status_porcelain(project_dir)
        checks["git_status"] = status_text
        changed = _git_changed_files(project_dir, include_untracked=True, ignore_prefixes=[])
        checks["git_changed_files"] = changed[:50]
        tracked_runner = _git_tracked_paths(project_dir, STATE_DIR_NAME)
        checks["prd_runner_tracked"] = bool(tracked_runner)
        if tracked_runner:
            errors.append(".prd_runner is tracked in git (must be removed from history)")
        ignored = _git_is_ignored(project_dir, STATE_DIR_NAME)
        checks["prd_runner_ignored"] = bool(ignored)
        if not ignored:
            warnings.append(".prd_runner is not ignored by git (commit step will refuse to proceed)")

    if check_codex:
        codex_path = shutil.which("codex")
        checks["codex_path"] = codex_path
        if not codex_path:
            warnings.append("codex not found in PATH")

    if not state_dir.exists():
        warnings.append("No .prd_runner state directory found")
    else:
        run_state, run_state_err = _load_data_with_error(run_state_path, {})
        queue, queue_err = _load_data_with_error(task_queue_path, {})
        plan, plan_err = _load_data_with_error(phase_plan_path, {})
        parse_errors = [e for e in [run_state_err, queue_err, plan_err] if e]
        checks["state_parse_errors"] = parse_errors
        if parse_errors:
            errors.extend(parse_errors)

        phases = _normalize_phases(plan)
        tasks = _normalize_tasks(queue)
        phase_ids = {str(p.get("id")).strip() for p in phases if isinstance(p, dict) and str(p.get("id") or "").strip()}
        schema_issues = validate_phase_plan_schema(plan) + validate_task_queue_schema(queue, phase_ids=phase_ids)
        checks["state_schema_issues"] = schema_issues
        if schema_issues:
            errors.extend(schema_issues)

        if prd_path is not None and prd_path.exists():
            stored_prd_path = run_state.get("prd_path")
            stored_prd_hash = run_state.get("prd_hash")
            checks["prd_path"] = str(prd_path)
            checks["prd_hash"] = _hash_file(str(prd_path))
            if stored_prd_path:
                try:
                    stored_resolved = Path(str(stored_prd_path)).expanduser().resolve()
                except Exception:
                    stored_resolved = None
                if stored_resolved and stored_resolved != prd_path:
                    errors.append(f"PRD mismatch: stored={stored_resolved} requested={prd_path}")
            if stored_prd_hash and checks["prd_hash"] and str(stored_prd_hash).strip() != checks["prd_hash"]:
                errors.append("PRD content hash mismatch (use --reset-state)")
        elif prd_path is not None:
            warnings.append("Provided --prd-file does not exist")

        blocked_report = _load_data(blocked_report_path, {}) if blocked_report_path.exists() else None
        checks["blocked_report"] = blocked_report

        last_run_id = run_state.get("last_run_id")
        if last_run_id:
            last_dir = runs_dir / str(last_run_id)
            checks["last_run_dir"] = str(last_dir) if last_dir.exists() else None
            if not last_dir.exists():
                warnings.append("last_run_id set but run dir missing")

    exit_code = 2 if errors else (1 if warnings else 0)
    payload = {"checks": checks, "warnings": warnings, "errors": errors, "exit_code": exit_code}

    if as_json:
        import json
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return exit_code

    sys.stdout.write(f"Project: {project_dir}\n")
    sys.stdout.write(f"Git repo: {bool(is_git)}\n")
    sys.stdout.write(f"State:   {state_dir if state_dir.exists() else '(missing)'}\n")
    if warnings:
        sys.stdout.write("Warnings:\n")
        for w in warnings:
            sys.stdout.write(f"- {w}\n")
    if errors:
        sys.stdout.write("Errors:\n")
        for e in errors[:20]:
            sys.stdout.write(f"- {e}\n")
    sys.stdout.write(f"Exit code: {exit_code}\n")
    return exit_code


def main(argv: list[str] | None = None) -> None:
    """Run the `feature-prd-runner` CLI.

    Args:
        argv: Optional argument list (excluding the executable name). When omitted,
            uses `sys.argv[1:]`.

    Raises:
        SystemExit: Raised to return a process exit code for CLI subcommands.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        if argv[0] == "status":
            args = _build_status_parser().parse_args(argv[1:])
            raise SystemExit(_status_command(args.project_dir, as_json=bool(args.json)))
        if argv[0] == "dry-run":
            args = _build_dry_run_parser().parse_args(argv[1:])
            raise SystemExit(_dry_run_command(args.project_dir, args.prd_file, as_json=bool(args.json)))
        if argv[0] == "doctor":
            args = _build_doctor_parser().parse_args(argv[1:])
            raise SystemExit(
                _doctor_command(
                    args.project_dir,
                    args.prd_file,
                    check_codex=bool(args.check_codex),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "list":
            args = _build_list_parser().parse_args(argv[1:])
            raise SystemExit(
                _list_command(
                    args.project_dir,
                    show_tasks=bool(args.tasks),
                    show_phases=bool(args.phases),
                    blocked_only=bool(args.blocked),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "resume":
            args = _build_resume_parser().parse_args(argv[1:])
            raise SystemExit(
                _resume_command(
                    args.project_dir,
                    args.task_id,
                    step=args.step,
                    force=bool(args.force),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "retry":
            args = _build_retry_parser().parse_args(argv[1:])
            raise SystemExit(
                _retry_command(
                    args.project_dir,
                    args.task_id,
                    force=bool(args.force),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "rerun-step":
            args = _build_rerun_step_parser().parse_args(argv[1:])
            raise SystemExit(
                _rerun_step_command(
                    args.project_dir,
                    args.task_id,
                    step=args.step,
                    force=bool(args.force),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "skip-step":
            args = _build_skip_step_parser().parse_args(argv[1:])
            raise SystemExit(
                _skip_step_command(
                    args.project_dir,
                    args.task_id,
                    step=args.step,
                    force=bool(args.force),
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "exec":
            args = _build_exec_parser().parse_args(argv[1:])
            _configure_logging(args.log_level)
            raise SystemExit(
                _exec_command(
                    args.project_dir,
                    args.prompt,
                    args.codex_command,
                    bool(args.override_agents),
                    bool(args.then_continue),
                    args.context_task,
                    args.context_files,
                    args.shift_minutes,
                    args.heartbeat_seconds,
                    args.heartbeat_grace_seconds,
                )
            )
        if argv[0] == "approve":
            args = _build_approve_parser().parse_args(argv[1:])
            raise SystemExit(
                _approve_command(
                    args.project_dir,
                    args.run_id,
                    args.feedback,
                )
            )
        if argv[0] == "reject":
            args = _build_reject_parser().parse_args(argv[1:])
            raise SystemExit(
                _reject_command(
                    args.project_dir,
                    args.run_id,
                    args.reason,
                )
            )
        if argv[0] == "steer":
            args = _build_steer_parser().parse_args(argv[1:])
            raise SystemExit(
                _steer_command(
                    args.project_dir,
                    args.run_id,
                    args.message,
                )
            )
        if argv[0] == "explain":
            args = _build_explain_parser().parse_args(argv[1:])
            raise SystemExit(
                _explain_command(
                    args.project_dir,
                    args.task_id,
                )
            )
        if argv[0] == "inspect":
            args = _build_inspect_parser().parse_args(argv[1:])
            raise SystemExit(
                _inspect_command(
                    args.project_dir,
                    args.task_id,
                    as_json=bool(args.json),
                )
            )
        if argv[0] == "trace":
            args = _build_trace_parser().parse_args(argv[1:])
            raise SystemExit(
                _trace_command(
                    args.project_dir,
                    args.task_id,
                    as_json=bool(args.json),
                    limit=args.limit,
                )
            )
        if argv[0] == "logs":
            args = _build_logs_parser().parse_args(argv[1:])
            raise SystemExit(
                _logs_command(
                    args.project_dir,
                    args.task_id,
                    args.step,
                    args.lines,
                    args.run_id,
                )
            )

    parser = _build_run_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    run_feature_prd(
        project_dir=args.project_dir,
        prd_path=args.prd_file,
        codex_command=args.codex_command,
        max_iterations=args.max_iterations,
        shift_minutes=args.shift_minutes,
        heartbeat_seconds=args.heartbeat_seconds,
        heartbeat_grace_seconds=args.heartbeat_grace_seconds,
        max_attempts=args.max_task_attempts,
        max_auto_resumes=args.max_auto_resumes,
        max_review_attempts=args.max_review_attempts,
        test_command=args.test_command,
        format_command=args.format_command,
        lint_command=args.lint_command,
        typecheck_command=args.typecheck_command,
        verify_profile=args.verify_profile,
        ensure_ruff=args.ensure_ruff,
        ensure_deps=args.ensure_deps,
        ensure_deps_command=args.ensure_deps_command,
        new_branch=args.new_branch,
        custom_prompt=args.custom_prompt,
        override_agents=bool(args.override_agents),
        stop_on_blocking_issues=args.stop_on_blocking_issues,
        resume_blocked=args.resume_blocked,
        simple_review=args.simple_review,
        reset_state=args.reset_state,
        require_clean=args.require_clean,
        commit_enabled=args.commit,
        push_enabled=args.push,
        interactive=bool(args.interactive),
    )


if __name__ == "__main__":
    main()
