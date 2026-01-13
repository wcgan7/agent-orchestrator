#!/usr/bin/env python3
"""
Feature PRD Runner (FSM)
========================

Coordinator entrypoint for planning, implementing, verifying, reviewing,
and committing changes using a step-based FSM.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

from .constants import (
    DEFAULT_HEARTBEAT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_AUTO_RESUMES,
    DEFAULT_SHIFT_MINUTES,
    DEFAULT_STOP_ON_BLOCKING_ISSUES,
    STATE_DIR_NAME,
)
from .orchestrator import run_feature_prd
from .prompts import _build_phase_prompt, _build_plan_prompt, _build_review_prompt
from .io_utils import _load_data, _load_data_with_error
from .tasks import (
    _blocking_tasks,
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
from .utils import _hash_file
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


def _dry_run_command(project_dir: Path, prd_path: Path | None, *, as_json: bool = False) -> int:
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve() if prd_path else None
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / "run_state.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    phase_plan_path = state_dir / "phase_plan.yaml"
    runs_dir = state_dir / "runs"
    gitignore_path = project_dir / ".gitignore"

    result: dict[str, object] = {
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
        result["errors"] = list(result.get("errors") or []) + schema_issues

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
                result["warnings"] = list(result.get("warnings") or []) + [
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
                result["errors"] = list(result.get("errors") or []) + [
                    f"PRD mismatch: stored={stored_resolved} requested={prd_path}"
                ]
        if stored_prd_hash and current_hash and str(stored_prd_hash).strip() != current_hash:
            result["errors"] = list(result.get("errors") or []) + [
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
        return 2 if (result.get("errors") or []) else 0

    sys.stdout.write(f"Project: {project_dir}\n")
    sys.stdout.write(f"State:   {state_dir}\n")
    errs = result.get("errors") or []
    warns = result.get("warnings") or []
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
        test_command=args.test_command,
        custom_prompt=args.custom_prompt,
        stop_on_blocking_issues=args.stop_on_blocking_issues,
        resume_blocked=args.resume_blocked,
        simple_review=args.simple_review,
        reset_state=args.reset_state,
        require_clean=args.require_clean,
        commit_enabled=args.commit,
        push_enabled=args.push,
    )


if __name__ == "__main__":
    main()
