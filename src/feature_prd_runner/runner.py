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
from .orchestrator import _default_run_branch, run_feature_prd
from .prompts import _build_phase_prompt, _build_plan_prompt, _build_review_prompt
from .io_utils import FileLock, _load_data, _load_data_with_error, _save_data
from .config import load_runner_config, get_language_config, get_verify_profile_config
from .custom_execution import execute_custom_prompt
from .messaging import ApprovalResponse, MessageBus, Message
from .approval_gates import ApprovalGateManager
from .debug import ErrorAnalyzer
from .parallel import ParallelExecutor
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
from .language import detect_language, get_verify_profile_for_language


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
        "--worker",
        type=str,
        default=None,
        help="Worker provider name override (e.g., codex, ollama). Overrides .prd_runner/config.yaml routing.",
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
        "--language",
        type=str,
        choices=["auto", "python", "typescript", "javascript", "nextjs", "go", "rust"],
        default="auto",
        help="Project language. If 'auto' (default), detected from manifest files.",
    )
    parser.add_argument(
        "--verify-profile",
        type=str,
        choices=["none", "python", "typescript", "javascript", "nextjs", "go", "rust"],
        default="none",
        help="Verification preset (default: none). If 'none' and --language is set, uses language-appropriate profile.",
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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution of independent phases (default: False)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)",
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


def _build_workers_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - worker providers",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    list_parser = subparsers.add_parser("list", help="List configured worker providers")
    list_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    list_parser.add_argument(
        "--codex-command",
        type=str,
        default="codex exec -",
        help="Codex CLI command fallback (default: codex exec -)",
    )

    test_parser = subparsers.add_parser("test", help="Test a configured worker provider")
    test_parser.add_argument(
        "worker",
        type=str,
        help="Worker provider name (e.g., codex, ollama)",
    )
    test_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    test_parser.add_argument(
        "--codex-command",
        type=str,
        default="codex exec -",
        help="Codex CLI command fallback (default: codex exec -)",
    )

    return parser


def _workers_command(args: argparse.Namespace) -> int:
    from .config import load_runner_config
    from .workers import get_workers_runtime_config
    from .workers.diagnostics import test_worker

    project_dir = args.project_dir.resolve()
    config, config_err = load_runner_config(project_dir)
    runtime = get_workers_runtime_config(
        config=config,
        codex_command_fallback=str(args.codex_command),
        cli_worker=None,
    )

    if args.subcommand == "list":
        sys.stdout.write(f"Default: {runtime.default_worker}\n")
        if runtime.routing:
            sys.stdout.write("Routing:\n")
            for k in sorted(runtime.routing.keys()):
                sys.stdout.write(f"- {k}: {runtime.routing[k]}\n")
        sys.stdout.write("Providers:\n")
        for name in sorted(runtime.providers.keys()):
            spec = runtime.providers[name]
            detail = spec.type
            if spec.type == "ollama":
                detail = f"ollama model={spec.model} endpoint={spec.endpoint}"
            elif spec.type == "codex":
                detail = f"codex command={spec.command}"
            sys.stdout.write(f"- {name}: {detail}\n")
        if config_err:
            sys.stdout.write(f"\nWarning: config.yaml parse error: {config_err}\n")
        return 0

    if args.subcommand == "test":
        name = str(args.worker).strip()
        if name not in runtime.providers:
            sys.stderr.write(f"Unknown worker '{name}' (available: {', '.join(sorted(runtime.providers.keys()))})\n")
            return 2
        ok, msg = test_worker(runtime.providers[name])
        if ok:
            sys.stdout.write(f"✓ {name}: {msg}\n")
            return 0
        sys.stderr.write(f"✗ {name}: {msg}\n")
        return 1

    sys.stderr.write(f"Unknown workers subcommand: {args.subcommand}\n")
    return 2


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
        "--worker",
        type=str,
        default=None,
        help="Worker provider name override for this exec (e.g., codex, ollama).",
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
    worker: str | None,
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
        worker=worker,
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


def _build_correct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - send correction to running worker",
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID to correct",
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
        "--file",
        type=str,
        help="File path to correct",
    )
    parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="Description of the issue",
    )
    parser.add_argument(
        "--fix",
        type=str,
        help="Suggested fix or correction",
    )
    return parser


def _correct_command(
    task_id: str,
    project_dir: Path,
    run_id: Optional[str],
    file_path: Optional[str],
    issue: str,
    fix: Optional[str],
) -> int:
    """Send correction to running worker."""
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

    # Create correction message
    correction = {
        "task_id": task_id,
        "issue": issue,
    }
    if file_path:
        correction["file"] = file_path
    if fix:
        correction["suggested_fix"] = fix

    from datetime import datetime, timezone

    msg = Message(
        id=f"correction-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        type="correction",
        content=issue,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata=correction,
    )

    bus.send_to_worker(msg)
    sys.stdout.write(f"✓ Correction sent for task {task_id}\n")
    if file_path:
        sys.stdout.write(f"  File: {file_path}\n")
    sys.stdout.write(f"  Issue: {issue}\n")
    if fix:
        sys.stdout.write(f"  Suggested fix: {fix}\n")

    return 0


def _build_require_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - add requirement to running worker",
    )
    parser.add_argument(
        "requirement",
        type=str,
        help="Requirement to add",
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
        "--task-id",
        type=str,
        help="Task ID to apply requirement to",
    )
    parser.add_argument(
        "--priority",
        type=str,
        choices=["high", "medium", "low"],
        default="medium",
        help="Priority of requirement (default: medium)",
    )
    return parser


def _require_command(
    requirement: str,
    project_dir: Path,
    run_id: Optional[str],
    task_id: Optional[str],
    priority: str,
) -> int:
    """Add requirement to running worker."""
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

    # Create requirement message
    req_metadata = {"priority": priority}
    if task_id:
        req_metadata["task_id"] = task_id

    from datetime import datetime, timezone

    msg = Message(
        id=f"requirement-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        type="requirement",
        content=requirement,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata=req_metadata,
    )

    bus.send_to_worker(msg)
    sys.stdout.write(f"✓ Requirement added: {requirement}\n")
    if task_id:
        sys.stdout.write(f"  Applied to task: {task_id}\n")
    sys.stdout.write(f"  Priority: {priority}\n")

    return 0


def _build_breakpoint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - manage breakpoints",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # breakpoint set
    set_parser = subparsers.add_parser("set", help="Set a breakpoint")
    set_parser.add_argument(
        "--before",
        type=str,
        help="Break before this step (plan_impl, implement, verify, review, commit)",
    )
    set_parser.add_argument(
        "--after",
        type=str,
        help="Break after this step",
    )
    set_parser.add_argument(
        "--task-id",
        type=str,
        help="Apply to specific task only",
    )
    set_parser.add_argument(
        "--condition",
        type=str,
        help="Conditional expression (e.g., 'files_changed > 10')",
    )
    set_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )

    # breakpoint list
    list_parser = subparsers.add_parser("list", help="List breakpoints")
    list_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )

    # breakpoint remove
    remove_parser = subparsers.add_parser("remove", help="Remove a breakpoint")
    remove_parser.add_argument(
        "breakpoint_id",
        type=str,
        help="Breakpoint ID to remove",
    )
    remove_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )

    # breakpoint clear
    clear_parser = subparsers.add_parser("clear", help="Clear all breakpoints")
    clear_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )

    # breakpoint toggle
    toggle_parser = subparsers.add_parser("toggle", help="Enable/disable a breakpoint")
    toggle_parser.add_argument(
        "breakpoint_id",
        type=str,
        help="Breakpoint ID to toggle",
    )
    toggle_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )

    return parser


def _breakpoint_command(args: argparse.Namespace) -> int:
    """Manage breakpoints."""
    from .breakpoints import BreakpointManager

    project_dir = args.project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        state_dir.mkdir(parents=True, exist_ok=True)

    manager = BreakpointManager(state_dir)

    if args.subcommand == "set":
        # Determine trigger and target
        if args.before:
            trigger = "before_step"
            target = args.before
        elif args.after:
            trigger = "after_step"
            target = args.after
        else:
            sys.stdout.write("Error: Must specify --before or --after\n")
            return 1

        bp = manager.add_breakpoint(
            trigger=trigger,
            target=target,
            task_id=args.task_id,
            condition=args.condition,
            action="pause",
        )

        sys.stdout.write(f"✓ Breakpoint created: {bp.id}\n")
        sys.stdout.write(f"  Trigger: {bp.trigger}\n")
        sys.stdout.write(f"  Target: {bp.target}\n")
        if bp.task_id:
            sys.stdout.write(f"  Task: {bp.task_id}\n")
        if bp.condition:
            sys.stdout.write(f"  Condition: {bp.condition}\n")

    elif args.subcommand == "list":
        breakpoints = manager.list_breakpoints()

        if not breakpoints:
            sys.stdout.write("No breakpoints set\n")
            return 0

        sys.stdout.write(f"Breakpoints ({len(breakpoints)}):\n\n")
        for bp in breakpoints:
            status = "✓" if bp.enabled else "✗"
            sys.stdout.write(f"{status} {bp.id}: {bp.trigger} {bp.target}\n")
            if bp.task_id:
                sys.stdout.write(f"    Task: {bp.task_id}\n")
            if bp.condition:
                sys.stdout.write(f"    Condition: {bp.condition}\n")
            sys.stdout.write(f"    Hit count: {bp.hit_count}\n")
            sys.stdout.write("\n")

    elif args.subcommand == "remove":
        if manager.remove_breakpoint(args.breakpoint_id):
            sys.stdout.write(f"✓ Breakpoint {args.breakpoint_id} removed\n")
        else:
            sys.stdout.write(f"Breakpoint {args.breakpoint_id} not found\n")
            return 1

    elif args.subcommand == "clear":
        count = manager.clear_all()
        sys.stdout.write(f"✓ Cleared {count} breakpoint(s)\n")

    elif args.subcommand == "toggle":
        enabled = manager.toggle_breakpoint(args.breakpoint_id)
        if enabled is not None:
            status = "enabled" if enabled else "disabled"
            sys.stdout.write(f"✓ Breakpoint {args.breakpoint_id} {status}\n")
        else:
            sys.stdout.write(f"Breakpoint {args.breakpoint_id} not found\n")
            return 1

    return 0


def _build_view_changes_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - view current code changes (git diff)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Show changes for specific task (uses task branch if available)",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Show staged changes only",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of context lines (default: 3)",
    )
    return parser


def _view_changes_command(
    project_dir: Path,
    task_id: Optional[str],
    staged: bool,
    context: int,
) -> int:
    """View current code changes (git diff)."""
    import subprocess

    project_dir = project_dir.resolve()

    # Check if in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=project_dir,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        sys.stdout.write("Not in a git repository\n")
        return 1

    # Build git diff command
    cmd = ["git", "diff"]

    if staged:
        cmd.append("--staged")

    cmd.append(f"--unified={context}")

    # If task_id specified, try to find task branch
    if task_id:
        state_dir = project_dir / STATE_DIR_NAME
        if state_dir.exists():
            task_queue_path = state_dir / "task_queue.yaml"
            tasks = _load_data(task_queue_path, {})

            task_data = None
            for t in tasks.get("tasks", []):
                if t.get("id") == task_id:
                    task_data = t
                    break

            if task_data and task_data.get("branch"):
                branch = task_data["branch"]
                # Show diff from main to task branch
                cmd = ["git", "diff", f"main...{branch}", f"--unified={context}"]
                sys.stdout.write(f"Showing changes on branch: {branch}\n\n")

    # Run git diff
    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            sys.stdout.write(result.stdout)
        else:
            sys.stdout.write("No changes to display\n")

        return 0
    except Exception as e:
        sys.stdout.write(f"Error running git diff: {e}\n")
        return 1


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


def _build_metrics_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - view run metrics and statistics",
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
        "--export",
        type=str,
        choices=["csv", "html"],
        help="Export metrics to file format (csv or html)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: metrics.{csv|html} in current directory)",
    )
    return parser


def _metrics_command(
    project_dir: Path,
    *,
    as_json: bool = False,
    export_format: str | None = None,
    output_path: Path | None = None,
) -> int:
    """Display run metrics and statistics."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    if not state_dir.exists():
        sys.stdout.write(f"No .prd_runner state directory found in {project_dir}\n")
        sys.stdout.write("Run 'feature-prd-runner run' first to generate metrics.\n")
        return 1

    # Calculate metrics directly to avoid complex imports
    import json
    import subprocess
    import yaml
    from dataclasses import dataclass, asdict
    from datetime import datetime

    @dataclass
    class Metrics:
        tokens_used: int = 0
        api_calls: int = 0
        estimated_cost_usd: float = 0.0
        wall_time_seconds: float = 0.0
        worker_time_seconds: float = 0.0
        verification_time_seconds: float = 0.0
        review_time_seconds: float = 0.0
        phases_completed: int = 0
        phases_total: int = 0
        tasks_done: int = 0
        tasks_total: int = 0
        files_changed: int = 0
        lines_added: int = 0
        lines_removed: int = 0
        worker_failures: int = 0
        verification_failures: int = 0
        allowlist_violations: int = 0
        review_blockers: int = 0

    m = Metrics()

    # Load phase plan
    phase_plan_path = state_dir / "phase_plan.yaml"
    if phase_plan_path.exists():
        try:
            phase_plan = _load_data(phase_plan_path, {})
            phases = phase_plan.get("phases", [])
            m.phases_total = len(phases)
        except Exception:
            pass

    # Load task queue
    task_queue_path = state_dir / "task_queue.yaml"
    if task_queue_path.exists():
        try:
            task_queue = _load_data(task_queue_path, {})
            tasks = task_queue.get("tasks", [])
            m.tasks_total = len(tasks)
            done_phases = set()
            for task in tasks:
                if task.get("lifecycle") == "done":
                    m.tasks_done += 1
                    if task.get("phase_id"):
                        done_phases.add(task["phase_id"])
            m.phases_completed = len(done_phases)
        except Exception:
            pass

    # Get git stats
    try:
        result = subprocess.run(
            ["git", "diff", "--numstat", "origin/main...HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3 and not parts[2].startswith(".prd_runner"):
                    try:
                        added = int(parts[0]) if parts[0] != "-" else 0
                        removed = int(parts[1]) if parts[1] != "-" else 0
                        m.lines_added += added
                        m.lines_removed += removed
                        if added > 0 or removed > 0:
                            m.files_changed += 1
                    except ValueError:
                        continue
    except Exception:
        pass

    # Try to import rich for formatting, fallback to plain text
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        use_rich = True
    except ImportError:
        use_rich = False

    if as_json:
        sys.stdout.write(json.dumps(asdict(m), indent=2))
        sys.stdout.write("\n")
        return 0

    def format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    # Handle export formats
    if export_format:
        if output_path is None:
            output_path = Path.cwd() / f"metrics.{export_format}"

        if export_format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])

                # Progress
                if m.tasks_total > 0:
                    task_pct = int(m.tasks_done / m.tasks_total * 100)
                    writer.writerow(["Tasks", f"{m.tasks_done}/{m.tasks_total} ({task_pct}%)"])
                if m.phases_total > 0:
                    phase_pct = int(m.phases_completed / m.phases_total * 100)
                    writer.writerow(["Phases", f"{m.phases_completed}/{m.phases_total} ({phase_pct}%)"])

                # Resource usage
                if m.tokens_used > 0:
                    writer.writerow(["Tokens Used", m.tokens_used])
                if m.api_calls > 0:
                    writer.writerow(["API Calls", m.api_calls])
                if m.estimated_cost_usd > 0:
                    writer.writerow(["Estimated Cost (USD)", f"${m.estimated_cost_usd:.2f}"])

                # Timing
                if m.wall_time_seconds > 0:
                    writer.writerow(["Wall Time", format_time(m.wall_time_seconds)])
                if m.worker_time_seconds > 0:
                    writer.writerow(["Worker Time", format_time(m.worker_time_seconds)])
                if m.verification_time_seconds > 0:
                    writer.writerow(["Verification Time", format_time(m.verification_time_seconds)])
                if m.review_time_seconds > 0:
                    writer.writerow(["Review Time", format_time(m.review_time_seconds)])

                # Code changes
                if m.files_changed > 0:
                    writer.writerow(["Files Changed", m.files_changed])
                if m.lines_added > 0:
                    writer.writerow(["Lines Added", m.lines_added])
                if m.lines_removed > 0:
                    writer.writerow(["Lines Removed", m.lines_removed])

                # Issues
                if m.worker_failures > 0:
                    writer.writerow(["Worker Failures", m.worker_failures])
                if m.verification_failures > 0:
                    writer.writerow(["Verification Failures", m.verification_failures])
                if m.allowlist_violations > 0:
                    writer.writerow(["Allowlist Violations", m.allowlist_violations])
                if m.review_blockers > 0:
                    writer.writerow(["Review Blockers", m.review_blockers])

            sys.stdout.write(f"Metrics exported to: {output_path}\n")
            return 0

        elif export_format == "html":
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Feature PRD Runner Metrics</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #555;
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric-value {{
            font-weight: 500;
            color: #2196F3;
        }}
        .error-value {{
            color: #f44336;
        }}
        .success-value {{
            color: #4CAF50;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Run Metrics Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

            # Progress section
            if m.tasks_total > 0 or m.phases_total > 0:
                html_content += """
        <div class="section">
            <h2>Progress</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
                if m.tasks_total > 0:
                    task_pct = int(m.tasks_done / m.tasks_total * 100)
                    html_content += f"""                <tr><td>Tasks</td><td class="metric-value">{m.tasks_done}/{m.tasks_total} ({task_pct}%)</td></tr>
"""
                if m.phases_total > 0:
                    phase_pct = int(m.phases_completed / m.phases_total * 100)
                    html_content += f"""                <tr><td>Phases</td><td class="metric-value">{m.phases_completed}/{m.phases_total} ({phase_pct}%)</td></tr>
"""
                html_content += """            </table>
        </div>
"""

            # Resource usage section
            if m.tokens_used > 0 or m.api_calls > 0 or m.estimated_cost_usd > 0:
                html_content += """
        <div class="section">
            <h2>Resource Usage</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
                if m.tokens_used > 0:
                    html_content += f"""                <tr><td>Tokens Used</td><td class="metric-value">{m.tokens_used:,}</td></tr>
"""
                if m.api_calls > 0:
                    html_content += f"""                <tr><td>API Calls</td><td class="metric-value">{m.api_calls}</td></tr>
"""
                if m.estimated_cost_usd > 0:
                    html_content += f"""                <tr><td>Estimated Cost</td><td class="metric-value">${m.estimated_cost_usd:.2f}</td></tr>
"""
                html_content += """            </table>
        </div>
"""

            # Timing section
            if m.wall_time_seconds > 0 or m.worker_time_seconds > 0:
                html_content += """
        <div class="section">
            <h2>Timing</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
                if m.wall_time_seconds > 0:
                    html_content += f"""                <tr><td>Wall Time</td><td class="metric-value">{format_time(m.wall_time_seconds)}</td></tr>
"""
                if m.worker_time_seconds > 0:
                    html_content += f"""                <tr><td>Worker Time</td><td class="metric-value">{format_time(m.worker_time_seconds)}</td></tr>
"""
                if m.verification_time_seconds > 0:
                    html_content += f"""                <tr><td>Verification Time</td><td class="metric-value">{format_time(m.verification_time_seconds)}</td></tr>
"""
                if m.review_time_seconds > 0:
                    html_content += f"""                <tr><td>Review Time</td><td class="metric-value">{format_time(m.review_time_seconds)}</td></tr>
"""
                html_content += """            </table>
        </div>
"""

            # Code changes section
            if m.files_changed > 0 or m.lines_added > 0 or m.lines_removed > 0:
                html_content += """
        <div class="section">
            <h2>Code Changes</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
                if m.files_changed > 0:
                    html_content += f"""                <tr><td>Files Changed</td><td class="metric-value">{m.files_changed}</td></tr>
"""
                if m.lines_added > 0:
                    html_content += f"""                <tr><td>Lines Added</td><td class="success-value">+{m.lines_added}</td></tr>
"""
                if m.lines_removed > 0:
                    html_content += f"""                <tr><td>Lines Removed</td><td class="error-value">-{m.lines_removed}</td></tr>
"""
                html_content += """            </table>
        </div>
"""

            # Issues section
            if (m.worker_failures > 0 or m.verification_failures > 0 or
                m.allowlist_violations > 0 or m.review_blockers > 0):
                html_content += """
        <div class="section">
            <h2>Issues</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
                if m.worker_failures > 0:
                    html_content += f"""                <tr><td>Worker Failures</td><td class="error-value">{m.worker_failures}</td></tr>
"""
                if m.verification_failures > 0:
                    html_content += f"""                <tr><td>Verification Failures</td><td class="error-value">{m.verification_failures}</td></tr>
"""
                if m.allowlist_violations > 0:
                    html_content += f"""                <tr><td>Allowlist Violations</td><td class="error-value">{m.allowlist_violations}</td></tr>
"""
                if m.review_blockers > 0:
                    html_content += f"""                <tr><td>Review Blockers</td><td class="error-value">{m.review_blockers}</td></tr>
"""
                html_content += """            </table>
        </div>
"""

            html_content += """
        <div class="footer">
            Generated by Feature PRD Runner
        </div>
    </div>
</body>
</html>
"""

            with open(output_path, "w") as f:
                f.write(html_content)

            sys.stdout.write(f"Metrics exported to: {output_path}\n")
            return 0

    # Collect metrics to display
    has_data = False

    if use_rich:
        console = Console()
        progress_table = Table(show_header=False, box=None, padding=(0, 2))
        progress_table.add_column("Metric", style="cyan")
        progress_table.add_column("Value", style="green")

        if m.tasks_total > 0:
            task_pct = int(m.tasks_done / m.tasks_total * 100)
            progress_table.add_row("Tasks", f"{m.tasks_done}/{m.tasks_total} ({task_pct}%)")
        if m.phases_total > 0:
            phase_pct = int(m.phases_completed / m.phases_total * 100)
            progress_table.add_row("Phases", f"{m.phases_completed}/{m.phases_total} ({phase_pct}%)")

        resource_table = Table(show_header=False, box=None, padding=(0, 2))
        resource_table.add_column("Metric", style="cyan")
        resource_table.add_column("Value", style="yellow")
        if m.tokens_used > 0:
            resource_table.add_row("Tokens", f"{m.tokens_used:,}")
        if m.api_calls > 0:
            resource_table.add_row("API Calls", str(m.api_calls))
        if m.estimated_cost_usd > 0:
            resource_table.add_row("Estimated Cost", f"${m.estimated_cost_usd:.2f}")

        timing_table = Table(show_header=False, box=None, padding=(0, 2))
        timing_table.add_column("Metric", style="cyan")
        timing_table.add_column("Value", style="magenta")
        if m.wall_time_seconds > 0:
            timing_table.add_row("Wall Time", format_time(m.wall_time_seconds))
        if m.worker_time_seconds > 0:
            timing_table.add_row("Worker Time", format_time(m.worker_time_seconds))
        if m.verification_time_seconds > 0:
            timing_table.add_row("Verification", format_time(m.verification_time_seconds))
        if m.review_time_seconds > 0:
            timing_table.add_row("Review", format_time(m.review_time_seconds))

        changes_table = Table(show_header=False, box=None, padding=(0, 2))
        changes_table.add_column("Metric", style="cyan")
        changes_table.add_column("Value", style="green")
        if m.files_changed > 0:
            changes_table.add_row("Files Changed", str(m.files_changed))
        if m.lines_added > 0:
            changes_table.add_row("Lines Added", f"+{m.lines_added}")
        if m.lines_removed > 0:
            changes_table.add_row("Lines Removed", f"-{m.lines_removed}")

        errors_table = Table(show_header=False, box=None, padding=(0, 2))
        errors_table.add_column("Metric", style="cyan")
        errors_table.add_column("Value", style="red")
        has_errors = False
        if m.worker_failures > 0:
            errors_table.add_row("Worker Failures", str(m.worker_failures))
            has_errors = True
        if m.verification_failures > 0:
            errors_table.add_row("Verification Failures", str(m.verification_failures))
            has_errors = True
        if m.allowlist_violations > 0:
            errors_table.add_row("Allowlist Violations", str(m.allowlist_violations))
            has_errors = True
        if m.review_blockers > 0:
            errors_table.add_row("Review Blockers", str(m.review_blockers))
            has_errors = True

        console.print("\n")
        console.print(Panel("[bold]Run Metrics[/bold]", style="blue"))
        console.print("\n")

        if progress_table.row_count > 0:
            console.print("[bold]Progress:[/bold]")
            console.print(progress_table)
            console.print()
            has_data = True
        if resource_table.row_count > 0:
            console.print("[bold]Resource Usage:[/bold]")
            console.print(resource_table)
            console.print()
            has_data = True
        if timing_table.row_count > 0:
            console.print("[bold]Timing:[/bold]")
            console.print(timing_table)
            console.print()
            has_data = True
        if changes_table.row_count > 0:
            console.print("[bold]Code Changes:[/bold]")
            console.print(changes_table)
            console.print()
            has_data = True
        if has_errors:
            console.print("[bold]Issues:[/bold]")
            console.print(errors_table)
            console.print()
            has_data = True

        if not has_data:
            console.print("[yellow]No metrics data available yet.[/yellow]")
            console.print("[yellow]Metrics will be populated as the run progresses.[/yellow]")
            console.print()
    else:
        # Plain text fallback
        sys.stdout.write("\n=== Run Metrics ===\n\n")
        if m.tasks_total > 0 or m.phases_total > 0:
            sys.stdout.write("Progress:\n")
            if m.tasks_total > 0:
                task_pct = int(m.tasks_done / m.tasks_total * 100)
                sys.stdout.write(f"  Tasks: {m.tasks_done}/{m.tasks_total} ({task_pct}%)\n")
                has_data = True
            if m.phases_total > 0:
                phase_pct = int(m.phases_completed / m.phases_total * 100)
                sys.stdout.write(f"  Phases: {m.phases_completed}/{m.phases_total} ({phase_pct}%)\n")
                has_data = True
            sys.stdout.write("\n")

        if m.tokens_used > 0 or m.api_calls > 0 or m.estimated_cost_usd > 0:
            sys.stdout.write("Resource Usage:\n")
            if m.tokens_used > 0:
                sys.stdout.write(f"  Tokens: {m.tokens_used:,}\n")
                has_data = True
            if m.api_calls > 0:
                sys.stdout.write(f"  API Calls: {m.api_calls}\n")
                has_data = True
            if m.estimated_cost_usd > 0:
                sys.stdout.write(f"  Estimated Cost: ${m.estimated_cost_usd:.2f}\n")
                has_data = True
            sys.stdout.write("\n")

        if m.wall_time_seconds > 0 or m.worker_time_seconds > 0:
            sys.stdout.write("Timing:\n")
            if m.wall_time_seconds > 0:
                sys.stdout.write(f"  Wall Time: {format_time(m.wall_time_seconds)}\n")
                has_data = True
            if m.worker_time_seconds > 0:
                sys.stdout.write(f"  Worker Time: {format_time(m.worker_time_seconds)}\n")
                has_data = True
            if m.verification_time_seconds > 0:
                sys.stdout.write(f"  Verification: {format_time(m.verification_time_seconds)}\n")
                has_data = True
            if m.review_time_seconds > 0:
                sys.stdout.write(f"  Review: {format_time(m.review_time_seconds)}\n")
                has_data = True
            sys.stdout.write("\n")

        if m.files_changed > 0 or m.lines_added > 0 or m.lines_removed > 0:
            sys.stdout.write("Code Changes:\n")
            if m.files_changed > 0:
                sys.stdout.write(f"  Files Changed: {m.files_changed}\n")
                has_data = True
            if m.lines_added > 0:
                sys.stdout.write(f"  Lines Added: +{m.lines_added}\n")
                has_data = True
            if m.lines_removed > 0:
                sys.stdout.write(f"  Lines Removed: -{m.lines_removed}\n")
                has_data = True
            sys.stdout.write("\n")

        if (m.worker_failures > 0 or m.verification_failures > 0 or
            m.allowlist_violations > 0 or m.review_blockers > 0):
            sys.stdout.write("Issues:\n")
            if m.worker_failures > 0:
                sys.stdout.write(f"  Worker Failures: {m.worker_failures}\n")
                has_data = True
            if m.verification_failures > 0:
                sys.stdout.write(f"  Verification Failures: {m.verification_failures}\n")
                has_data = True
            if m.allowlist_violations > 0:
                sys.stdout.write(f"  Allowlist Violations: {m.allowlist_violations}\n")
                has_data = True
            if m.review_blockers > 0:
                sys.stdout.write(f"  Review Blockers: {m.review_blockers}\n")
                has_data = True
            sys.stdout.write("\n")

        if not has_data:
            sys.stdout.write("No metrics data available yet.\n")
            sys.stdout.write("Metrics will be populated as the run progresses.\n\n")

    return 0


def _build_plan_parallel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - visualize parallel execution plan",
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
        help="PRD file (required for fresh plan)",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Show dependency tree instead of execution plan",
    )
    return parser


def _plan_parallel_command(
    project_dir: Path,
    prd_file: Optional[Path],
    tree: bool,
) -> int:
    """Visualize parallel execution plan."""
    project_dir = project_dir.resolve()
    state_dir = project_dir / STATE_DIR_NAME

    # Load phases from phase_plan.yaml
    phase_plan_path = state_dir / "phase_plan.yaml"

    if not phase_plan_path.exists():
        if not prd_file:
            sys.stdout.write("No phase plan found and no PRD file specified.\n")
            sys.stdout.write("Either run the planner first or specify --prd-file.\n")
            return 1
        sys.stdout.write("Phase plan not found. This would require running the planner.\n")
        sys.stdout.write("For now, please run the planner first:\n")
        sys.stdout.write(f"  feature-prd-runner run --prd-file {prd_file}\n")
        return 1

    phase_plan = _load_data(phase_plan_path, {})
    phases = phase_plan.get("phases", [])

    if not phases:
        sys.stdout.write("No phases found in phase plan.\n")
        return 1

    executor = ParallelExecutor()

    if tree:
        # Show dependency tree
        output = executor.visualize_as_tree(phases)
        sys.stdout.write(output)
    else:
        # Show execution plan
        try:
            plan = executor.resolve_execution_order(phases)
            output = executor.visualize_execution_plan(phases, plan)
            sys.stdout.write(output)
        except ValueError as e:
            sys.stdout.write(f"Error creating execution plan: {e}\n")
            return 1

    return 0


def _build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - start web dashboard server",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--no-cors",
        action="store_true",
        help="Disable CORS (not recommended for development)",
    )
    return parser


def _server_command(
    project_dir: Path,
    host: str,
    port: int,
    reload: bool,
    enable_cors: bool,
) -> int:
    """Start the web dashboard server."""
    project_dir = project_dir.resolve()

    try:
        import uvicorn  # type: ignore
    except ImportError:
        sys.stdout.write("Error: uvicorn not installed.\n")
        sys.stdout.write("Install with: pip install 'feature-prd-runner[server]'\n")
        return 1

    try:
        from .server import create_app
    except ImportError as e:
        sys.stdout.write(f"Error importing server module: {e}\n")
        sys.stdout.write("Install with: pip install 'feature-prd-runner[server]'\n")
        return 1

    # Create app with project directory
    app = create_app(project_dir=project_dir, enable_cors=enable_cors)

    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write("Feature PRD Runner - Web Dashboard\n")
    sys.stdout.write(f"{'='*60}\n")
    sys.stdout.write(f"Project: {project_dir}\n")
    sys.stdout.write(f"Server: http://{host}:{port}\n")
    sys.stdout.write(f"API Docs: http://{host}:{port}/docs\n")
    sys.stdout.write(f"{'='*60}\n\n")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

    return 0


def _build_example_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature PRD Runner - create example project",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for example project",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["python", "typescript", "javascript", "nextjs", "go", "rust"],
        default="python",
        help="Project language (default: python)",
    )
    return parser


def _example_command(output_dir: Path, language: str) -> int:
    """Create an example project with sample PRD and configuration."""
    output_dir = output_dir.resolve()

    if output_dir.exists():
        sys.stdout.write(f"Error: Directory {output_dir} already exists\n")
        sys.stdout.write("Please choose a different directory or remove the existing one.\n")
        return 1

    try:
        # Create project structure
        output_dir.mkdir(parents=True)
        docs_dir = output_dir / "docs"
        docs_dir.mkdir()
        src_dir = output_dir / "src"
        src_dir.mkdir()
        tests_dir = output_dir / "tests"
        tests_dir.mkdir()
        prd_runner_dir = output_dir / ".prd_runner"
        prd_runner_dir.mkdir()

        # Create example PRD
        prd_content = """# Feature: User Authentication System

## Overview
Implement a secure user authentication system with login, logout, and session management.

## Goals
- Allow users to securely log in and out
- Maintain user sessions
- Protect sensitive routes

## Requirements

### Phase 1: Basic Authentication
**Phase ID**: `auth-basic`
**Dependencies**: None

1. Create user model with email and password fields
2. Implement password hashing using industry-standard algorithms
3. Create login endpoint that validates credentials
4. Create logout endpoint that clears sessions
5. Add input validation for email and password

**Success Criteria**:
- Users can register with email and password
- Users can log in with valid credentials
- Passwords are securely hashed and never stored in plain text
- Invalid login attempts are rejected with appropriate errors

**Test Requirements**:
- Test successful login with valid credentials
- Test login rejection with invalid credentials
- Test password hashing implementation
- Test logout functionality

### Phase 2: Session Management
**Phase ID**: `auth-sessions`
**Dependencies**: `auth-basic`

1. Implement session token generation
2. Create session storage mechanism
3. Add session validation middleware
4. Implement session expiration logic
5. Add refresh token support

**Success Criteria**:
- Sessions are created upon successful login
- Session tokens are validated on protected routes
- Sessions expire after configured timeout
- Refresh tokens allow extending sessions

**Test Requirements**:
- Test session creation and validation
- Test session expiration
- Test protected route access with valid/invalid sessions
- Test refresh token functionality

### Phase 3: Security Hardening
**Phase ID**: `auth-security`
**Dependencies**: `auth-sessions`

1. Add rate limiting for login attempts
2. Implement account lockout after failed attempts
3. Add CSRF protection
4. Implement secure password reset flow
5. Add logging for security events

**Success Criteria**:
- Login attempts are rate-limited
- Accounts lock after excessive failed attempts
- CSRF tokens protect against cross-site attacks
- Password reset emails are sent securely
- All security events are logged

**Test Requirements**:
- Test rate limiting enforcement
- Test account lockout mechanism
- Test CSRF protection
- Test password reset flow
- Verify security event logging

## Technical Notes
- Use bcrypt or Argon2 for password hashing
- Sessions should be stored server-side (Redis recommended)
- JWT tokens are acceptable for stateless authentication
- Follow OWASP authentication best practices

## Out of Scope
- OAuth integration (future phase)
- Multi-factor authentication (future phase)
- Social login (future phase)
"""

        prd_path = docs_dir / "authentication.md"
        prd_path.write_text(prd_content)

        # Create config based on language
        if language == "python":
            config_content = """# Feature PRD Runner Configuration

# Verification profile and commands
verify_profile: python
test_command: pytest -v
lint_command: ruff check .
format_command: ruff format --check .
typecheck_command: mypy src/ --strict

# Ensure dependencies
ensure_ruff: install
ensure_deps: install

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true

# Approval gates (optional - uncomment to enable)
# approval_gates:
#   enabled: true
#   gates:
#     before_implement:
#       enabled: true
#       message: "Review implementation plan?"
#     before_commit:
#       enabled: true
#       message: "Review and approve commit?"
#       required: true
"""

            # Create sample Python files
            src_init = src_dir / "__init__.py"
            src_init.write_text('"""User authentication package."""\n')

            src_main = src_dir / "main.py"
            src_main.write_text("""\"\"\"Main application entry point.\"\"\"

def main() -> None:
    \"\"\"Run the application.\"\"\"
    print("Hello from Feature PRD Runner example!")


if __name__ == "__main__":
    main()
""")

            tests_init = tests_dir / "__init__.py"
            tests_init.write_text("")

            tests_main = tests_dir / "test_main.py"
            tests_main.write_text("""\"\"\"Tests for main module.\"\"\"

from src.main import main


def test_main() -> None:
    \"\"\"Test main function runs without error.\"\"\"
    main()  # Should not raise
""")

            # Create pyproject.toml
            pyproject_content = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "auth-example"
version = "0.1.0"
description = "Example authentication system"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7",
    "mypy>=1.8",
    "ruff>=0.4",
]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.ruff]
line-length = 100
target-version = "py310"
"""
            pyproject_path = output_dir / "pyproject.toml"
            pyproject_path.write_text(pyproject_content)

        elif language == "typescript":
            config_content = """# Feature PRD Runner Configuration

# Project language
language: typescript
verify_profile: typescript

# Verification commands
test_command: npm test
lint_command: npm run lint
typecheck_command: npm run type-check
format_command: npm run format:check

# Ensure dependencies
ensure_deps: install

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true
"""

            # Create package.json
            package_json = {
                "name": "auth-example",
                "version": "0.1.0",
                "description": "Example authentication system",
                "scripts": {
                    "test": "jest",
                    "lint": "eslint .",
                    "type-check": "tsc --noEmit",
                    "format:check": "prettier --check ."
                }
            }
            import json
            (output_dir / "package.json").write_text(json.dumps(package_json, indent=2) + "\n")

            # Create sample TypeScript files
            (src_dir / "index.ts").write_text("""export function main(): void {
  console.log("Hello from Feature PRD Runner example!");
}

main();
""")

            (tests_dir / "index.test.ts").write_text("""import { main } from "../src/index";

describe("main", () => {
  it("should run without error", () => {
    expect(() => main()).not.toThrow();
  });
});
""")

        elif language == "go":
            config_content = """# Feature PRD Runner Configuration

# Project language
language: go
verify_profile: go

# Verification commands
test_command: go test ./... -v
lint_command: golangci-lint run
format_command: gofmt -l .

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true
"""

            # Create go.mod
            (output_dir / "go.mod").write_text("""module example.com/auth

go 1.21
""")

            # Create sample Go files
            (src_dir / "main.go").write_text("""package main

import "fmt"

func main() {
	fmt.Println("Hello from Feature PRD Runner example!")
}
""")

            (tests_dir / "main_test.go").write_text("""package main

import "testing"

func TestMain(t *testing.T) {
	// Test passes if main doesn't panic
	main()
}
""")

        elif language == "javascript":
            config_content = """# Feature PRD Runner Configuration

# Project language
language: javascript
verify_profile: javascript

# Verification commands
test_command: npm test
lint_command: npm run lint
format_command: npm run format:check

# Ensure dependencies
ensure_deps: install

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true
"""

            # Create package.json
            package_json = {
                "name": "auth-example",
                "version": "0.1.0",
                "description": "Example authentication system",
                "scripts": {
                    "test": "jest",
                    "lint": "eslint .",
                    "format:check": "prettier --check ."
                }
            }
            import json
            (output_dir / "package.json").write_text(json.dumps(package_json, indent=2) + "\n")

            # Create sample JavaScript files
            (src_dir / "index.js").write_text("""function main() {
  console.log("Hello from Feature PRD Runner example!");
}

module.exports = { main };
main();
""")

            (tests_dir / "index.test.js").write_text("""const { main } = require("../src/index");

describe("main", () => {
  it("should run without error", () => {
    expect(() => main()).not.toThrow();
  });
});
""")

        elif language == "nextjs":
            config_content = """# Feature PRD Runner Configuration

# Project language
language: nextjs
verify_profile: nextjs

# Verification commands
test_command: npm test
lint_command: npx next lint
typecheck_command: npx next build
format_command: npx prettier --check .

# Ensure dependencies
ensure_deps: install

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true
"""

            # Create package.json
            package_json = {
                "name": "auth-example",
                "version": "0.1.0",
                "private": True,
                "description": "Example Next.js authentication system",
                "scripts": {
                    "dev": "next dev",
                    "build": "next build",
                    "start": "next start",
                    "test": "jest",
                    "lint": "next lint",
                    "format:check": "prettier --check ."
                },
                "dependencies": {
                    "next": "^14.0.0",
                    "react": "^18.2.0",
                    "react-dom": "^18.2.0"
                },
                "devDependencies": {
                    "typescript": "^5.0.0",
                    "@types/react": "^18.2.0",
                    "@types/node": "^20.0.0",
                    "jest": "^29.0.0",
                    "@testing-library/react": "^14.0.0",
                    "@testing-library/jest-dom": "^6.0.0",
                    "prettier": "^3.0.0"
                }
            }
            import json
            (output_dir / "package.json").write_text(json.dumps(package_json, indent=2) + "\n")

            # Create next.config.js
            (output_dir / "next.config.js").write_text("""/** @type {import('next').NextConfig} */
const nextConfig = {};

module.exports = nextConfig;
""")

            # Create tsconfig.json
            (output_dir / "tsconfig.json").write_text(json.dumps({
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "strict": True,
                    "noEmit": True,
                    "esModuleInterop": True,
                    "module": "esnext",
                    "moduleResolution": "bundler",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "jsx": "preserve",
                    "incremental": True,
                    "plugins": [{"name": "next"}],
                    "paths": {"@/*": ["./*"]}
                },
                "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
                "exclude": ["node_modules"]
            }, indent=2) + "\n")

            # Create app directory (App Router)
            app_dir = output_dir / "app"
            app_dir.mkdir()

            (app_dir / "layout.tsx").write_text("""export const metadata = {
  title: "Auth Example",
  description: "Example Next.js authentication app",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
""")

            (app_dir / "page.tsx").write_text("""export default function Home() {
  return (
    <main>
      <h1>Hello from Feature PRD Runner example!</h1>
    </main>
  );
}
""")

            # Create API route
            api_dir = app_dir / "api" / "hello"
            api_dir.mkdir(parents=True)

            (api_dir / "route.ts").write_text("""import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({ message: "Hello from Feature PRD Runner!" });
}
""")

            # Create tests
            test_dir = output_dir / "__tests__"
            test_dir.mkdir()

            (test_dir / "page.test.tsx").write_text("""import { render, screen } from "@testing-library/react";
import Home from "../app/page";

describe("Home", () => {
  it("renders the heading", () => {
    render(<Home />);
    expect(
      screen.getByText("Hello from Feature PRD Runner example!")
    ).toBeInTheDocument();
  });
});
""")

        elif language == "rust":
            config_content = """# Feature PRD Runner Configuration

# Project language
language: rust
verify_profile: rust

# Verification commands
test_command: cargo test
lint_command: cargo clippy -- -D warnings
format_command: cargo fmt -- --check

# Worker configuration
codex_command: codex exec -

# Git configuration
new_branch: feature/user-authentication

# Retry configuration
max_task_attempts: 10
max_review_attempts: 3
stop_on_blocking_issues: true
"""

            # Create Cargo.toml
            (output_dir / "Cargo.toml").write_text("""[package]
name = "auth-example"
version = "0.1.0"
edition = "2021"

[dependencies]
""")

            # Create sample Rust files
            (src_dir / "main.rs").write_text("""fn main() {
    println!("Hello from Feature PRD Runner example!");
}
""")

            (src_dir / "lib.rs").write_text("""pub fn greet() -> &'static str {
    "Hello from Feature PRD Runner example!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet(), "Hello from Feature PRD Runner example!");
    }
}
""")

        # Write config
        config_path = prd_runner_dir / "config.yaml"
        config_path.write_text(config_content)

        # Create README
        readme_content = f"""# Example Feature PRD Runner Project

This is an example project created by `feature-prd-runner example`.

## Project Structure

- `docs/` - Product Requirement Documents (PRDs)
- `src/` - Source code
- `tests/` - Test files
- `.prd_runner/` - Runner configuration and state

## Getting Started

1. Review the example PRD: `docs/authentication.md`

2. Initialize git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. Run the Feature PRD Runner:
   ```bash
   feature-prd-runner run --prd-file docs/authentication.md
   ```

4. Monitor progress:
   ```bash
   feature-prd-runner status
   feature-prd-runner list --tasks
   ```

## Configuration

The configuration is in `.prd_runner/config.yaml`. Key settings:

- `verify_profile`: Set to `{language}` for this project
- `test_command`: Command to run tests
- `lint_command`: Command to run linter
- `max_task_attempts`: Maximum retry attempts for each phase

## Learn More

- Documentation: https://github.com/anthropics/feature-prd-runner
- Edit `.prd_runner/config.yaml` to customize behavior
- Create new PRDs in `docs/` directory
"""
        readme_path = output_dir / "README.md"
        readme_path.write_text(readme_content)

        # Create .gitignore
        gitignore_content = """# Feature PRD Runner
.prd_runner/run_state.yaml
.prd_runner/task_queue.yaml
.prd_runner/phase_plan.yaml
.prd_runner/runs/
.prd_runner/artifacts/
.prd_runner/.lock

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/
dist/
build/

# TypeScript/JavaScript
node_modules/
dist/
*.js
*.d.ts
!jest.config.js

# Go
*.exe
*.test
*.out

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
        gitignore_path = output_dir / ".gitignore"
        gitignore_path.write_text(gitignore_content)

        sys.stdout.write(f"\n{'='*60}\n")
        sys.stdout.write(f"Example {language} project created successfully!\n")
        sys.stdout.write(f"{'='*60}\n\n")
        sys.stdout.write(f"Location: {output_dir}\n\n")
        sys.stdout.write("Files created:\n")
        sys.stdout.write(f"  - {prd_path.relative_to(output_dir)}  (Example PRD)\n")
        sys.stdout.write(f"  - {config_path.relative_to(output_dir)}  (Configuration)\n")
        sys.stdout.write(f"  - README.md  (Getting started guide)\n")
        sys.stdout.write(f"  - .gitignore  (Git ignore rules)\n")
        sys.stdout.write(f"  - src/  (Source files)\n")
        sys.stdout.write(f"  - tests/  (Test files)\n\n")
        sys.stdout.write("Next steps:\n")
        sys.stdout.write(f"  1. cd {output_dir}\n")
        sys.stdout.write("  2. git init\n")
        sys.stdout.write("  3. git add .\n")
        sys.stdout.write("  4. git commit -m 'Initial commit'\n")
        sys.stdout.write("  5. feature-prd-runner run --prd-file docs/authentication.md\n\n")

        return 0

    except Exception as e:
        sys.stdout.write(f"Error creating example project: {e}\n")
        # Clean up on error
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        return 1


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
            branch = phase.get("branch") or next_task.get("branch") or _default_run_branch(prd_path)

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
        if argv[0] == "workers":
            args = _build_workers_parser().parse_args(argv[1:])
            raise SystemExit(_workers_command(args))
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
                    args.worker,
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
        if argv[0] == "view-changes":
            args = _build_view_changes_parser().parse_args(argv[1:])
            raise SystemExit(
                _view_changes_command(
                    args.project_dir,
                    args.task_id,
                    args.staged,
                    args.context,
                )
            )
        if argv[0] == "correct":
            args = _build_correct_parser().parse_args(argv[1:])
            raise SystemExit(
                _correct_command(
                    args.task_id,
                    args.project_dir,
                    args.run_id,
                    args.file,
                    args.issue,
                    args.fix,
                )
            )
        if argv[0] == "require":
            args = _build_require_parser().parse_args(argv[1:])
            raise SystemExit(
                _require_command(
                    args.requirement,
                    args.project_dir,
                    args.run_id,
                    args.task_id,
                    args.priority,
                )
            )
        if argv[0] == "breakpoint":
            args = _build_breakpoint_parser().parse_args(argv[1:])
            raise SystemExit(_breakpoint_command(args))
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
        if argv[0] == "metrics":
            args = _build_metrics_parser().parse_args(argv[1:])
            raise SystemExit(
                _metrics_command(
                    args.project_dir,
                    as_json=bool(args.json),
                    export_format=args.export,
                    output_path=args.output,
                )
            )
        if argv[0] == "plan-parallel":
            args = _build_plan_parallel_parser().parse_args(argv[1:])
            raise SystemExit(
                _plan_parallel_command(
                    args.project_dir,
                    args.prd_file,
                    bool(args.tree),
                )
            )
        if argv[0] == "server":
            args = _build_server_parser().parse_args(argv[1:])
            raise SystemExit(
                _server_command(
                    args.project_dir,
                    args.host,
                    args.port,
                    bool(args.reload),
                    enable_cors=not bool(args.no_cors),
                )
            )
        if argv[0] == "example":
            args = _build_example_parser().parse_args(argv[1:])
            raise SystemExit(
                _example_command(
                    args.output,
                    args.language,
                )
            )

    parser = _build_run_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    # Load config file for language/profile overrides
    config, config_err = load_runner_config(args.project_dir)
    if config_err:
        logger.warning("Config file error (using CLI defaults): {}", config_err)

    # Resolve language: CLI > config > auto-detect
    language = args.language
    if language == "auto":
        # Check config file for language setting
        config_language = get_language_config(config)
        if config_language and config_language != "auto":
            language = config_language
            logger.info("Using language '{}' from config file", language)
        else:
            # Auto-detect from project files
            language = detect_language(args.project_dir)
            if language == "unknown":
                language = "python"  # Fallback for backwards compatibility
            logger.info("Auto-detected project language: {}", language)

    # Resolve verify_profile: CLI > config > language-based
    verify_profile = args.verify_profile
    if verify_profile == "none":
        # Check config file for verify_profile
        config_profile = get_verify_profile_config(config)
        if config_profile:
            verify_profile = config_profile
            logger.info("Using verify profile '{}' from config file", verify_profile)
        elif language != "unknown":
            # Use language-based profile
            verify_profile = get_verify_profile_for_language(language)  # type: ignore[arg-type]
            if verify_profile != "none":
                logger.info("Using verify profile '{}' based on detected language", verify_profile)

    run_feature_prd(
        project_dir=args.project_dir,
        prd_path=args.prd_file,
        codex_command=args.codex_command,
        worker=args.worker,
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
        verify_profile=verify_profile,
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
        parallel=bool(args.parallel),
        max_workers=args.max_workers,
        language=language,
    )


if __name__ == "__main__":
    main()
