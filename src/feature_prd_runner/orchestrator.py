from __future__ import annotations

import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from .actions.run_commit import run_commit_action
from .actions.run_verify import run_verify_action
from .actions.run_worker import run_resume_prompt_action, run_worker_action
from .constants import (
    DEFAULT_HEARTBEAT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_AUTO_RESUMES,
    DEFAULT_SHIFT_MINUTES,
    DEFAULT_STOP_ON_BLOCKING_ISSUES,
    ERROR_TYPE_BLOCKING_ISSUES,
    ERROR_TYPE_DIRTY_WORKTREE,
    ERROR_TYPE_PLAN_MISSING,
    ERROR_TYPE_PRD_READ_FAILED,
    ERROR_TYPE_PRD_MISMATCH,
    ERROR_TYPE_STATE_CORRUPT,
    ERROR_TYPE_STATE_INVALID,
    ERROR_TYPE_STATE_RESET_FAILED,
    LOCK_FILE,
    MAX_ALLOWLIST_EXPANSION_ATTEMPTS,
    MAX_IMPL_PLAN_ATTEMPTS,
    MAX_MANUAL_RESUME_ATTEMPTS,
    MAX_NO_PROGRESS_ATTEMPTS,
    MAX_REVIEW_ATTEMPTS,
    MAX_TEST_FAIL_ATTEMPTS,
    TASK_STATUS_BLOCKED,
)
from .fsm import reduce_task
from .git_utils import (
    _ensure_branch,
    _ensure_gitignore,
    _git_has_changes,
    _git_is_repo,
    _gitignore_change_is_prd_runner_only,
)
from .io_utils import (
    FileLock,
    _append_event,
    _load_data,
    _load_data_with_error,
    _require_yaml,
    _save_data,
    _update_progress,
)
from .signals import build_allowed_files
from .logging_utils import pretty, summarize_event
from .models import (
    AllowlistViolation,
    NoIntroducedChanges,
    ProgressHumanBlockers,
    ResumePromptResult,
    TaskLifecycle,
    TaskState,
    TaskStep,
    VerificationResult,
    WorkerFailed,
    WorkerSucceeded,
)
from .state import _active_run_is_stale, _ensure_state_files, _finalize_run_state, _reset_state_dir
from .tasks import (
    _auto_resume_blocked_dependencies,
    _blocking_event_payload,
    _blocking_tasks,
    _build_plan_task,
    _build_tasks_from_phases,
    _find_task,
    _impl_plan_path,
    _maybe_auto_resume_blocked,
    _maybe_resume_blocked_last_intent,
    _normalize_phases,
    _normalize_tasks,
    _phase_for_task,
    _record_blocked_intent,
    _report_blocking_tasks,
    _save_plan,
    _save_queue,
    _select_next_task,
    _summarize_blocking_tasks,
    _sync_phase_status,
    _task_summary,
)
from .utils import _hash_file, _now_iso
from .git_utils import _git_changed_files
from .constants import IGNORED_REVIEW_PATH_PREFIXES
from .validation import validate_phase_plan_schema, validate_task_queue_schema


def _write_blocked_report(
    state_dir: Path,
    *,
    error_type: str,
    summary: str,
    details: dict[str, object] | None = None,
    next_steps: list[str] | None = None,
) -> None:
    payload: dict[str, object] = {
        "error_type": error_type,
        "summary": summary,
        "created_at": _now_iso(),
    }
    if details:
        payload["details"] = details
    if next_steps:
        payload["next_steps"] = list(next_steps)
    try:
        _save_data(state_dir / "runner_blocked.json", payload)
    except Exception:
        # Best-effort only; avoid masking the original failure
        pass


def _step_suffix(task_type: str, task_step: str) -> str:
    """Return a suffix for the run folder based on task type and step."""
    if task_type == "plan":
        return "-plan"
    step_str = task_step.value if isinstance(task_step, TaskStep) else str(task_step or "")
    if step_str == TaskStep.RESUME_PROMPT.value:
        return "-resume_prompt"
    elif step_str == TaskStep.PLAN_IMPL.value:
        return "-plan_impl"
    elif step_str == TaskStep.IMPLEMENT.value:
        return "-implement"
    elif step_str == TaskStep.VERIFY.value:
        return "-verify"
    elif step_str == TaskStep.REVIEW.value:
        return "-review"
    elif step_str == TaskStep.COMMIT.value:
        return "-commit"
    return ""


def run_feature_prd(
    project_dir: Path,
    prd_path: Path,
    codex_command: str = "codex exec -",
    max_iterations: Optional[int] = None,
    shift_minutes: int = DEFAULT_SHIFT_MINUTES,
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
    heartbeat_grace_seconds: int = DEFAULT_HEARTBEAT_GRACE_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_auto_resumes: int = DEFAULT_MAX_AUTO_RESUMES,
    test_command: Optional[str] = None,
    format_command: Optional[str] = None,
    lint_command: Optional[str] = None,
    typecheck_command: Optional[str] = None,
    verify_profile: str = "none",
    ensure_ruff: str = "off",
    custom_prompt: Optional[str] = None,
    stop_on_blocking_issues: bool = DEFAULT_STOP_ON_BLOCKING_ISSUES,
    resume_blocked: bool = True,
    simple_review: bool = False,
    reset_state: bool = False,
    require_clean: bool = True,
    commit_enabled: bool = True,
    push_enabled: bool = True,
) -> None:
    _require_yaml()
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve()

    if reset_state:
        # Ensure archived state (e.g., `.prd_runner.bak-...`) doesn't show up as untracked / get committed.
        if _git_is_repo(project_dir):
            _ensure_gitignore(project_dir, only_if_clean=True)
        try:
            _reset_state_dir(project_dir)
        except Exception as exc:
            state_dir = project_dir / ".prd_runner"
            state_dir.mkdir(parents=True, exist_ok=True)
            msg = "Failed to reset .prd_runner state directory"
            logger.error("{}: {}", msg, exc)
            _write_blocked_report(
                state_dir,
                error_type=ERROR_TYPE_STATE_RESET_FAILED,
                summary=msg,
                details={"error": f"{exc.__class__.__name__}: {exc}"},
                next_steps=[
                    "Close other runners, fix permissions, or move/delete .prd_runner manually.",
                    "Re-run with --reset-state.",
                ],
            )
            return

    logger.info("=" * 70)
    logger.info("FEATURE PRD RUNNER (FSM)")
    logger.info("=" * 70)
    logger.info("Project directory: {}", project_dir)
    logger.info("PRD file: {}", prd_path)
    logger.info("Codex command: {}", codex_command)
    logger.info("Shift length: {} minutes", shift_minutes)
    logger.info("Heartbeat: {}s (grace {}s)", heartbeat_seconds, heartbeat_grace_seconds)
    logger.info("Max attempts per task: {}", max_attempts)
    logger.info("Max auto-resumes: {}", max_auto_resumes)
    logger.info("Stop on blocking issues: {}", stop_on_blocking_issues)
    if test_command:
        logger.info("Test command: {}", test_command)
    if custom_prompt:
        logger.info("Custom prompt provided (will run as standalone step)")
    if any([format_command, lint_command, typecheck_command]) or verify_profile != "none":
        logger.info(
            "Verify config: profile={} ensure_ruff={} format={} lint={} typecheck={}",
            verify_profile,
            ensure_ruff,
            bool(format_command),
            bool(lint_command),
            bool(typecheck_command),
        )
    logger.info("Require clean worktree: {}", require_clean)
    logger.info("Commit enabled: {}", commit_enabled)
    logger.info("Push enabled: {}", push_enabled)

    if (require_clean or commit_enabled or push_enabled) and not _git_is_repo(project_dir):
        msg = "Project directory is not a git repository"
        logger.error(msg)
        _write_blocked_report(
            project_dir / ".prd_runner",
            error_type="not_git_repo",
            summary=msg,
            next_steps=["Run inside a git repo (git init), or disable commit/push flags."],
        )
        return

    paths = _ensure_state_files(project_dir, prd_path)

    lock_path = paths["state_dir"] / LOCK_FILE
    iteration = 0

    # Detect corrupted durable state before touching the repo (e.g., .gitignore).
    run_state_initial, run_state_err = _load_data_with_error(paths["run_state"], {})
    queue_initial, queue_err = _load_data_with_error(paths["task_queue"], {})
    plan_initial, plan_err = _load_data_with_error(paths["phase_plan"], {})
    state_errors = [e for e in [run_state_err, queue_err, plan_err] if e]
    if state_errors:
        msg = "Corrupted .prd_runner state detected; refusing to continue"
        logger.error("{}: {}", msg, "; ".join(state_errors))
        _write_blocked_report(
            paths["state_dir"],
            error_type=ERROR_TYPE_STATE_CORRUPT,
            summary=msg,
            details={"errors": state_errors},
            next_steps=[
                "Fix/restore the corrupted file(s), or re-run with --reset-state (archives existing .prd_runner)."
            ],
        )
        if not run_state_err:
            try:
                run_state_initial.update(
                    {
                        "status": "blocked",
                        "last_error": msg,
                        "updated_at": _now_iso(),
                    }
                )
                _save_data(paths["run_state"], run_state_initial)
            except Exception:
                pass
        return

    current_prd_hash = _hash_file(str(prd_path))
    if not current_prd_hash:
        msg = "Failed to read PRD file for hashing"
        logger.error("{}: {}", msg, prd_path)
        _write_blocked_report(
            paths["state_dir"],
            error_type=ERROR_TYPE_PRD_READ_FAILED,
            summary=msg,
            details={"prd_path": str(prd_path)},
            next_steps=["Verify the PRD path exists and is readable, then re-run."],
        )
        _finalize_run_state(paths, lock_path, status="blocked", last_error=msg)
        return

    # PRD mismatch detection: refuse to reuse state for a different PRD unless reset_state was requested.
    stored_prd = run_state_initial.get("prd_path")
    if stored_prd:
        try:
            stored_resolved = Path(str(stored_prd)).expanduser().resolve()
        except Exception:
            stored_resolved = None
        if stored_resolved and stored_resolved != prd_path:
            msg = "Existing .prd_runner state was created for a different PRD"
            logger.error("{} (stored={}, requested={})", msg, stored_resolved, prd_path)
            _write_blocked_report(
                paths["state_dir"],
                error_type=ERROR_TYPE_PRD_MISMATCH,
                summary=msg,
                details={
                    "stored_prd_path": str(stored_resolved),
                    "requested_prd_path": str(prd_path),
                },
                next_steps=["Re-run with --reset-state to start fresh for the new PRD."],
            )
            _finalize_run_state(paths, lock_path, status="blocked", last_error=msg)
            return

    stored_prd_hash = run_state_initial.get("prd_hash")
    if stored_prd_hash and str(stored_prd_hash).strip() and str(stored_prd_hash).strip() != current_prd_hash:
        msg = "Existing .prd_runner state was created for different PRD content"
        logger.error("{} (stored_hash={}, current_hash={})", msg, stored_prd_hash, current_prd_hash)
        _write_blocked_report(
            paths["state_dir"],
            error_type=ERROR_TYPE_PRD_MISMATCH,
            summary=msg,
            details={
                "prd_path": str(prd_path),
                "stored_prd_hash": str(stored_prd_hash),
                "current_prd_hash": current_prd_hash,
            },
            next_steps=["Re-run with --reset-state to start fresh for the updated PRD."],
        )
        _finalize_run_state(paths, lock_path, status="blocked", last_error=msg)
        return

    # Persist current PRD identity for future mismatch detection.
    try:
        with FileLock(lock_path):
            rs = _load_data(paths["run_state"], {})
            changed = False
            if rs.get("prd_path") != str(prd_path):
                rs["prd_path"] = str(prd_path)
                changed = True
            if rs.get("prd_hash") != current_prd_hash:
                rs["prd_hash"] = current_prd_hash
                changed = True
            if changed:
                rs["updated_at"] = _now_iso()
                _save_data(paths["run_state"], rs)
    except Exception:
        pass

    # Validate persisted schemas to avoid confusing downstream behavior.
    phase_schema_issues = validate_phase_plan_schema(plan_initial)
    phase_ids = {
        str(p.get("id")).strip()
        for p in (plan_initial.get("phases") or [])
        if isinstance(p, dict) and str(p.get("id") or "").strip()
    }
    queue_schema_issues = validate_task_queue_schema(queue_initial, phase_ids=phase_ids)
    schema_issues = phase_schema_issues + queue_schema_issues
    if schema_issues:
        msg = "Invalid .prd_runner state schema detected; refusing to continue"
        logger.error("{} (sample={})", msg, schema_issues[:5])
        _write_blocked_report(
            paths["state_dir"],
            error_type=ERROR_TYPE_STATE_INVALID,
            summary=msg,
            details={"issues": schema_issues},
            next_steps=[
                "Fix the listed schema issues in .prd_runner/{phase_plan.yaml,task_queue.yaml}, or re-run with --reset-state.",
            ],
        )
        try:
            with FileLock(lock_path):
                rs = _load_data(paths["run_state"], {})
                rs.update({"status": "blocked", "last_error": msg, "updated_at": _now_iso()})
                _save_data(paths["run_state"], rs)
        except Exception:
            pass
        return

    if require_clean:
        non_runner_changes = _git_changed_files(
            project_dir,
            include_untracked=True,
            ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
        )
        try:
            prd_rel = prd_path.relative_to(project_dir).as_posix()
        except Exception:
            prd_rel = None
        if prd_rel:
            non_runner_changes = [p for p in non_runner_changes if p != prd_rel]
        if ".gitignore" in non_runner_changes and _gitignore_change_is_prd_runner_only(project_dir):
            non_runner_changes = [p for p in non_runner_changes if p != ".gitignore"]
        if non_runner_changes:
            msg = "Git working tree has changes outside .prd_runner; refusing to run with --require-clean"
            logger.error("{} (sample={})", msg, non_runner_changes[:8])
            _write_blocked_report(
                paths["state_dir"],
                error_type=ERROR_TYPE_DIRTY_WORKTREE,
                summary=msg,
                details={"changed_files": non_runner_changes},
                next_steps=["Commit/stash/reset your changes, or rerun with --no-require-clean."],
            )
            _finalize_run_state(paths, lock_path, status="blocked", last_error=msg)
            return

    # Safe to update ignore rules now that we know state is usable.
    _ensure_gitignore(project_dir, only_if_clean=True)

    # Handle custom_prompt as a standalone step before the main loop
    if custom_prompt:
        custom_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        custom_run_id = f"{custom_run_id}-{uuid.uuid4().hex[:8]}-custom_prompt"
        custom_run_dir = paths["runs"] / custom_run_id
        custom_run_dir.mkdir(parents=True, exist_ok=True)
        custom_progress_path = custom_run_dir / "progress.json"

        logger.info("=" * 70)
        logger.info("RUNNING CUSTOM PROMPT (standalone)")
        logger.info("=" * 70)
        logger.info("Run ID: {}", custom_run_id)

        _update_progress(
            custom_progress_path,
            {
                "run_id": custom_run_id,
                "task_id": "custom_prompt",
                "phase": "custom_prompt",
                "human_blocking_issues": [],
                "human_next_steps": [],
            },
        )

        with FileLock(lock_path):
            run_state = _load_data(paths["run_state"], {})
            run_state.update(
                {
                    "status": "running",
                    "current_task_id": "custom_prompt",
                    "current_phase_id": None,
                    "run_id": custom_run_id,
                    "last_run_id": custom_run_id,
                    "updated_at": _now_iso(),
                    "coordinator_pid": os.getpid(),
                    "worker_pid": None,
                    "coordinator_started_at": _now_iso(),
                    "last_heartbeat": _now_iso(),
                }
            )
            _save_data(paths["run_state"], run_state)

        def _on_custom_worker_spawn(pid: int) -> None:
            try:
                with FileLock(lock_path):
                    rs = _load_data(paths["run_state"], {})
                    if rs.get("status") == "running" and rs.get("run_id") == custom_run_id:
                        rs["worker_pid"] = pid
                        rs["last_heartbeat"] = _now_iso()
                        rs["updated_at"] = _now_iso()
                        _save_data(paths["run_state"], rs)
            except Exception:
                pass

        custom_event = run_resume_prompt_action(
            user_prompt=custom_prompt,
            project_dir=project_dir,
            run_dir=custom_run_dir,
            run_id=custom_run_id,
            codex_command=codex_command,
            progress_path=custom_progress_path,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_grace_seconds=heartbeat_grace_seconds,
            shift_minutes=shift_minutes,
            on_spawn=_on_custom_worker_spawn,
        )

        _append_event(paths["events"], custom_event.to_dict())

        # Check if custom prompt succeeded
        if isinstance(custom_event, ResumePromptResult) and custom_event.succeeded:
            logger.info("Custom prompt completed successfully, continuing with implementation cycle")
            _finalize_run_state(paths, lock_path, status="idle", last_error=None)
            # Continue to the main loop - don't pass custom_prompt to tasks
        elif isinstance(custom_event, ProgressHumanBlockers):
            logger.error("Custom prompt blocked: {}", "; ".join(custom_event.issues))
            _finalize_run_state(
                paths,
                lock_path,
                status="blocked",
                last_error=f"Custom prompt blocked: {'; '.join(custom_event.issues)}",
            )
            return
        elif isinstance(custom_event, WorkerFailed):
            logger.error("Custom prompt failed: {}", custom_event.error_detail)
            _finalize_run_state(
                paths,
                lock_path,
                status="blocked",
                last_error=f"Custom prompt failed: {custom_event.error_detail}",
            )
            return
        else:
            # Unexpected event type - treat as failure
            logger.error("Custom prompt returned unexpected event: {}", type(custom_event).__name__)
            _finalize_run_state(
                paths,
                lock_path,
                status="blocked",
                last_error=f"Custom prompt unexpected result: {type(custom_event).__name__}",
            )
            return

    # user_prompt is no longer used since custom_prompt is handled separately
    user_prompt = None

    def _log_task_start(
        *,
        step_value: str,
        task_id: str,
        phase_id: Optional[str],
        branch: Optional[str],
        task: dict,
        plan_path: Optional[Path],
    ) -> None:
        logger.info(
            "Running step={} for task={} (phase={})",
            step_value,
            task_id,
            phase_id,
        )
        logger.debug(
            "Action context: branch={} prompt_mode={} plan_path={}",
            branch,
            task.get("prompt_mode"),
            str(plan_path) if plan_path else None,
        )

    def _log_event(event: object, run_dir: Path) -> None:
        if not event:
            return
        logger.info(
            "Event emitted: {} (run_id={})",
            event.__class__.__name__,
            getattr(event, "run_id", None),
        )

        # Content-level observability for WorkerSucceeded
        if isinstance(event, WorkerSucceeded):
            event_step = getattr(event, "step", None)

            # 1) On PLAN_IMPL success: log plan summary
            if event_step == TaskStep.PLAN_IMPL:
                if getattr(event, "plan_valid", False):
                    plan_path = Path(getattr(event, "impl_plan_path", "") or "")
                    if plan_path and plan_path.exists():
                        plan = _load_data(plan_path, {})
                        files = plan.get("files_to_change") or []
                        new_files = plan.get("new_files") or []
                        spec = plan.get("spec_summary") or []
                        logger.info(
                            "Plan accepted: path={} files_to_change={} new_files={}",
                            str(plan_path),
                            len(files),
                            len(new_files),
                        )
                        logger.info("Plan allowlist sample: {}", (files + new_files)[:8])
                        if spec:
                            logger.debug("Plan spec_summary sample:\n- {}", "\n- ".join(str(s) for s in spec[:3]))
                else:
                    # Plan invalid warning
                    logger.warning("Plan invalid: {}", getattr(event, "plan_issue", "unknown"))

            # 3) After IMPLEMENT finishes: log introduced changes
            if event_step == TaskStep.IMPLEMENT:
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = _load_data(manifest_path, {})
                    intro = manifest.get("introduced_changes") or []
                    logger.info("Introduced changes: {} files (sample={})", len(intro), intro[:8])
                    disallowed = manifest.get("disallowed_files") or []
                    if disallowed:
                        logger.warning("Disallowed changes: {}", disallowed)

        # 4) VERIFY failure with expansion request at INFO level
        if isinstance(event, VerificationResult):
            logger.info(
                "VERIFY result: passed={} needs_expansion={}",
                event.passed,
                event.needs_allowlist_expansion,
            )
            if event.needs_allowlist_expansion:
                logger.info("Verify requested allowlist expansion: {}", event.failing_paths)
            elif not event.passed:
                # Log verify manifest for debugging test failures
                verify_manifest_path = run_dir / "verify_manifest.json"
                if verify_manifest_path.exists():
                    vm = _load_data(verify_manifest_path, {})
                    failed_tests = vm.get("failed_test_files") or []
                    trace_files = vm.get("trace_files") or []
                    suspects = vm.get("suspect_source_files") or []
                    if failed_tests:
                        logger.info("Failed test files: {}", failed_tests[:5])
                    if trace_files:
                        logger.info("Trace files: {}", trace_files[:5])
                    if suspects:
                        logger.info("Suspect source files: {}", suspects[:5])
            logger.debug("Verify manifest: {}", run_dir / "verify_manifest.json")

        if isinstance(event, AllowlistViolation):
            logger.warning(
                "Allowlist violation detected: {}",
                event.disallowed_paths,
            )
        if isinstance(event, NoIntroducedChanges):
            logger.debug(
                "No introduced changes detected (repo_dirty={})",
                event.repo_dirty,
            )
        if isinstance(event, WorkerFailed):
            event_step = getattr(event, "step", None)
            logger.warning(
                "WorkerFailed: step={} type={} detail={}",
                event_step,
                event.error_type,
                (event.error_detail[:240] + "…") if len(event.error_detail) > 240 else event.error_detail,
            )
            logger.info("Artifacts: run_dir={}", str(run_dir))
            logger.info("Codex logs: stdout={} stderr={}", run_dir / "stdout.log", run_dir / "stderr.log")
            manifest = run_dir / "manifest.json"
            if manifest.exists():
                logger.info("Manifest: {}", manifest)
                # Also log introduced changes on failure for debugging
                if event_step == TaskStep.IMPLEMENT:
                    m = _load_data(manifest, {})
                    intro = m.get("introduced_changes") or []
                    if intro:
                        logger.info("Introduced changes before failure: {} files (sample={})", len(intro), intro[:8])

    while True:
        if max_iterations and iteration >= max_iterations:
            logger.info("Reached max iterations ({})", max_iterations)
            _finalize_run_state(paths, lock_path, status="idle", last_error="Reached max iterations")
            break

        blocked_tasks_snapshot = None
        manually_resumed = False
        next_task = None
        run_id = None
        phase = None
        phase_id = None
        task_id = None
        task_type = None
        task_step = None
        branch = None

        with FileLock(lock_path):
            run_state, run_state_err = _load_data_with_error(paths["run_state"], {})
            queue, queue_err = _load_data_with_error(paths["task_queue"], {})
            plan, plan_err = _load_data_with_error(paths["phase_plan"], {})

            state_errors = [e for e in [run_state_err, queue_err, plan_err] if e]
            if state_errors:
                msg = "Corrupted .prd_runner state detected; refusing to continue"
                logger.error("{}: {}", msg, "; ".join(state_errors))
                _write_blocked_report(
                    paths["state_dir"],
                    error_type=ERROR_TYPE_STATE_CORRUPT,
                    summary=msg,
                    details={"errors": state_errors},
                    next_steps=[
                        "Fix/restore the corrupted file(s), or re-run with --reset-state (archives existing .prd_runner)."
                    ],
                )
                if not run_state_err:
                    try:
                        run_state.update(
                            {
                                "status": "blocked",
                                "last_error": msg,
                                "updated_at": _now_iso(),
                            }
                        )
                        _save_data(paths["run_state"], run_state)
                    except Exception:
                        pass
                # Do not overwrite state files; exit cleanly.
                return

            tasks = _normalize_tasks(queue)
            phases = _normalize_phases(plan)
            queue["tasks"] = tasks
            plan["phases"] = phases

            phase_schema_issues = validate_phase_plan_schema(plan)
            phase_ids = {
                str(p.get("id")).strip()
                for p in (plan.get("phases") or [])
                if isinstance(p, dict) and str(p.get("id") or "").strip()
            }
            queue_schema_issues = validate_task_queue_schema(queue, phase_ids=phase_ids)
            schema_issues = phase_schema_issues + queue_schema_issues
            if schema_issues:
                msg = "Invalid .prd_runner state schema detected; refusing to continue"
                logger.error("{} (sample={})", msg, schema_issues[:5])
                _write_blocked_report(
                    paths["state_dir"],
                    error_type=ERROR_TYPE_STATE_INVALID,
                    summary=msg,
                    details={"issues": schema_issues},
                    next_steps=[
                        "Fix the listed schema issues in .prd_runner/{phase_plan.yaml,task_queue.yaml}, or re-run with --reset-state.",
                    ],
                )
                try:
                    run_state.update({"status": "blocked", "last_error": msg, "updated_at": _now_iso()})
                    _save_data(paths["run_state"], run_state)
                except Exception:
                    pass
                return

            if run_state.get("status") == "running":
                if not _active_run_is_stale(
                    run_state,
                    paths["runs"],
                    heartbeat_grace_seconds,
                    shift_minutes,
                ):
                    logger.info("Another run is already active. Exiting to avoid overlap.")
                    return
                run_state.update(
                    {
                        "status": "idle",
                        "current_task_id": None,
                        "current_phase_id": None,
                        "run_id": None,
                        "branch": None,
                        "last_error": "Previous run marked stale; resuming",
                        "updated_at": _now_iso(),
                        "coordinator_pid": None,
                        "worker_pid": None,
                        "coordinator_started_at": None,
                    }
                )
                _save_data(paths["run_state"], run_state)

            for task in tasks:
                if task.get("lifecycle") == TaskLifecycle.RUNNING.value:
                    task["lifecycle"] = TaskLifecycle.READY.value
                    task["status"] = task.get("step") or task.get("status")

            if not tasks:
                tasks = [_build_plan_task()]
                queue["tasks"] = tasks
                queue["updated_at"] = _now_iso()
                _save_data(paths["task_queue"], queue)

            tasks, resumed = _maybe_auto_resume_blocked(queue, tasks, max_auto_resumes)
            if resumed:
                _save_data(paths["task_queue"], queue)
                logger.info("Auto-resumed blocked tasks after auto-resumable failure")

            if resume_blocked:
                tasks, manually_resumed = _maybe_resume_blocked_last_intent(
                    queue,
                    tasks,
                    MAX_MANUAL_RESUME_ATTEMPTS,
                )
                if manually_resumed:
                    run_state.update(
                        {
                            "status": "idle",
                            "current_task_id": None,
                            "current_phase_id": None,
                            "run_id": None,
                            "branch": None,
                            "last_error": None,
                            "updated_at": _now_iso(),
                            "coordinator_pid": None,
                            "worker_pid": None,
                            "coordinator_started_at": None,
                        }
                    )
                    _save_data(paths["run_state"], run_state)
                    _save_data(paths["task_queue"], queue)
                    logger.info("Resumed most recent blocked task to replay last step")

            if stop_on_blocking_issues and not manually_resumed:
                blocked_tasks = _blocking_tasks(tasks)
                if blocked_tasks:
                    blocked_tasks_snapshot = [dict(task) for task in blocked_tasks]
                    run_state.update(
                        {
                            "status": "blocked",
                            "current_task_id": None,
                            "current_phase_id": None,
                            "branch": None,
                            "last_error": _summarize_blocking_tasks(blocked_tasks),
                            "updated_at": _now_iso(),
                        }
                    )
                    _save_data(paths["run_state"], run_state)
                    _save_queue(paths["task_queue"], queue, tasks)
                    _append_event(paths["events"], _blocking_event_payload(blocked_tasks))

            if not blocked_tasks_snapshot:
                next_task = _select_next_task(tasks)
                if not next_task:
                    if _auto_resume_blocked_dependencies(queue, tasks, max_auto_resumes):
                        _save_data(paths["task_queue"], queue)
                        logger.info("Auto-resumed blocked dependency tasks to resolve deadlock")
                        continue
                    run_state.update(
                        {
                            "status": "idle",
                            "current_task_id": None,
                            "current_phase_id": None,
                            "run_id": None,
                            "branch": None,
                            "updated_at": _now_iso(),
                            "coordinator_pid": None,
                            "worker_pid": None,
                            "coordinator_started_at": None,
                        }
                    )
                    _save_data(paths["run_state"], run_state)
                    _save_data(paths["task_queue"], queue)
                    summary = _task_summary(tasks)
                    logger.info(
                        "No runnable tasks. Queue summary: {} ready, {} running, {} done, {} waiting_human",
                        summary[TaskLifecycle.READY.value],
                        summary[TaskLifecycle.RUNNING.value],
                        summary[TaskLifecycle.DONE.value],
                        summary[TaskLifecycle.WAITING_HUMAN.value],
                    )
                    break

                logger.info(
                    "Selected task={} phase={} step={} lifecycle={} prompt_mode={}",
                    next_task.get("id"),
                    next_task.get("phase_id"),
                    next_task.get("step"),
                    next_task.get("lifecycle"),
                    next_task.get("prompt_mode"),
                )

                task_id = str(next_task.get("id"))
                task_type = next_task.get("type", "implement")
                phase_id = next_task.get("phase_id")
                task_step = next_task.get("step") or TaskStep.PLAN_IMPL.value    
                
                logger.info(
                    "\n===== TASK START =====\n"
                    "task: {}\n"
                    "phase: {}\n"
                    "step: {}\n"
                    "branch: {}\n"
                    "======================",
                    task_id,
                    phase_id,
                    task_step,
                    branch,
                )

                next_task["step"] = task_step
                next_task["lifecycle"] = TaskLifecycle.RUNNING.value
                next_task["status"] = task_step

                if task_type != "plan":
                    phase = _phase_for_task(phases, next_task)
                    if not phase:
                        next_task["lifecycle"] = TaskLifecycle.WAITING_HUMAN.value
                        next_task["block_reason"] = "PLAN_MISSING"
                        next_task["last_error"] = "Phase not found for task"
                        next_task["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                        next_task["status"] = TASK_STATUS_BLOCKED
                        _record_blocked_intent(
                            next_task,
                            task_status=task_step,
                            task_type=task_type,
                            phase_id=phase_id or next_task.get("id"),
                            branch=None,
                            test_command=next_task.get("test_command"),
                            run_id=None,
                            step=task_step,
                            lifecycle=next_task["lifecycle"],
                            prompt_mode=next_task.get("prompt_mode"),
                        )
                        _save_queue(paths["task_queue"], queue, tasks)
                        run_state.update(
                            {
                                "status": "blocked",
                                "current_task_id": None,
                                "current_phase_id": None,
                                "run_id": None,
                                "branch": None,
                                "last_error": "Phase not found for task",
                                "updated_at": _now_iso(),
                                "coordinator_pid": None,
                                "worker_pid": None,
                                "coordinator_started_at": None,
                            }
                        )
                        _save_data(paths["run_state"], run_state)
                        blocked_tasks_snapshot = [dict(next_task)]

                    if phase and not blocked_tasks_snapshot:
                        branch = phase.get("branch") or next_task.get("branch") or f"feature/{phase_id or task_id}"
                        next_task["branch"] = branch
                        try:
                            _ensure_branch(project_dir, branch)
                            run_state["branch"] = branch
                            run_state["updated_at"] = _now_iso()
                            _save_data(paths["run_state"], run_state)
                        except subprocess.CalledProcessError as exc:
                            msg = f"Failed to checkout branch {branch}: {exc}"
                            next_task["lifecycle"] = TaskLifecycle.WAITING_HUMAN.value
                            next_task["block_reason"] = "GIT_CHECKOUT_FAILED"
                            next_task["last_error"] = msg
                            next_task["last_error_type"] = "git_checkout_failed"
                            next_task["status"] = TASK_STATUS_BLOCKED
                            _record_blocked_intent(
                                next_task,
                                task_status=task_step,
                                task_type=task_type,
                                phase_id=phase_id or next_task.get("id"),
                                branch=branch,
                                test_command=next_task.get("test_command"),
                                run_id=None,
                                step=task_step,
                                lifecycle=next_task["lifecycle"],
                                prompt_mode=next_task.get("prompt_mode"),
                            )
                            _save_queue(paths["task_queue"], queue, tasks)
                            run_state.update(
                                {
                                    "status": "blocked",
                                    "current_task_id": None,
                                    "current_phase_id": None,
                                    "run_id": None,
                                    "branch": None,
                                    "last_error": msg,
                                    "updated_at": _now_iso(),
                                    "coordinator_pid": None,
                                    "worker_pid": None,
                                    "coordinator_started_at": None,
                                }
                            )
                            _save_data(paths["run_state"], run_state)
                            blocked_tasks_snapshot = [dict(next_task)]

                if not blocked_tasks_snapshot and task_type != "plan" and _git_has_changes(project_dir):
                    dirty_note = (
                        "Workspace has uncommitted changes; continue from them and do not reset."
                    )
                    context = next_task.get("context", []) or []
                    if dirty_note not in context:
                        context.append(dirty_note)
                    next_task["context"] = context

                if not blocked_tasks_snapshot:
                    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
                    # Append step suffix for easier debugging
                    step_suffix = _step_suffix(task_type, task_step)
                    run_id = f"{run_id}{step_suffix}"

                    run_state.update(
                        {
                            "status": "running",
                            "current_task_id": task_id,
                            "current_phase_id": next_task.get("phase_id"),
                            "run_id": run_id,
                            "last_run_id": run_id,
                            "updated_at": _now_iso(),
                            "coordinator_pid": os.getpid(),
                            "worker_pid": None,
                            "coordinator_started_at": _now_iso(),
                            "last_heartbeat": _now_iso(),
                        }
                    )
                    _save_data(paths["run_state"], run_state)
                    _save_data(paths["task_queue"], queue)

        if blocked_tasks_snapshot:
            _report_blocking_tasks(blocked_tasks_snapshot, paths, stopping=stop_on_blocking_issues)
            _finalize_run_state(paths, lock_path, status="blocked")
            return

        iteration += 1
        run_dir = paths["runs"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_path = run_dir / "progress.json"
        logger.debug("Run context: run_id={} run_dir={}", run_id, str(run_dir))

        progress_phase = task_type if task_type == "plan" else str(phase_id or task_id)
        _update_progress(
            progress_path,
            {
                "run_id": run_id,
                "task_id": task_id,
                "phase": progress_phase,
                "step": task_step,
                "human_blocking_issues": [],
                "human_next_steps": [],
            },
        )

        _append_event(
            paths["events"],
            {
                "event_type": "task_start",
                "task_id": task_id,
                "task_type": task_type,
                "step": task_step,
            },
        )

        def _on_worker_spawn(pid: int) -> None:
            try:
                with FileLock(lock_path):
                    rs = _load_data(paths["run_state"], {})
                    if rs.get("status") == "running" and rs.get("run_id") == run_id:
                        rs["worker_pid"] = pid
                        rs["last_heartbeat"] = _now_iso()
                        rs["updated_at"] = _now_iso()
                        _save_data(paths["run_state"], rs)
            except Exception:
                pass

        step_enum = TaskStep.PLAN_IMPL
        if isinstance(task_step, TaskStep):
            step_enum = task_step
        elif isinstance(task_step, str):
            try:
                step_enum = TaskStep(task_step)
            except ValueError:
                step_enum = TaskStep.PLAN_IMPL
        event = None
        if task_type == "plan":
            _log_task_start(
                step_value=TaskStep.PLAN_IMPL.value,
                task_id=task_id,
                phase_id=phase_id,
                branch=branch,
                task=next_task,
                plan_path=None,
            )
            event = run_worker_action(
                step=TaskStep.PLAN_IMPL,
                task=next_task,
                phase=None,
                prd_path=prd_path,
                project_dir=project_dir,
                artifacts_dir=paths["artifacts"],
                phase_plan_path=paths["phase_plan"],
                task_queue_path=paths["task_queue"],
                run_dir=run_dir,
                run_id=run_id,
                codex_command=codex_command,
                user_prompt=user_prompt,
                progress_path=progress_path,
                events_path=paths["events"],
                heartbeat_seconds=heartbeat_seconds,
                heartbeat_grace_seconds=heartbeat_grace_seconds,
                shift_minutes=shift_minutes,
                test_command=test_command,
                on_spawn=_on_worker_spawn,
                simple_review=simple_review,
            )

        elif step_enum in {TaskStep.PLAN_IMPL, TaskStep.IMPLEMENT, TaskStep.REVIEW}:
            plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
            _log_task_start(
                step_value=step_enum.value,
                task_id=task_id,
                phase_id=phase_id,
                branch=branch,
                task=next_task,
                plan_path=plan_path,
            )
            # 2) On IMPLEMENT start: log the allowlist being enforced
            if step_enum == TaskStep.IMPLEMENT and plan_path.exists():
                plan_for_allowlist = _load_data(plan_path, {})
                allowlist = build_allowed_files(plan_for_allowlist)
                logger.info("Implement allowlist: {} files (sample={})", len(allowlist), allowlist[:8])
            event = run_worker_action(
                step=step_enum,
                task=next_task,
                phase=phase,
                prd_path=prd_path,
                project_dir=project_dir,
                artifacts_dir=paths["artifacts"],
                phase_plan_path=paths["phase_plan"],
                task_queue_path=paths["task_queue"],
                run_dir=run_dir,
                run_id=run_id,
                codex_command=codex_command,
                user_prompt=user_prompt,
                progress_path=progress_path,
                events_path=paths["events"],
                heartbeat_seconds=heartbeat_seconds,
                heartbeat_grace_seconds=heartbeat_grace_seconds,
                shift_minutes=shift_minutes,
                test_command=test_command,
                on_spawn=_on_worker_spawn,
                simple_review=simple_review,
            )

        elif step_enum == TaskStep.VERIFY:
            plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
            plan_data = _load_data(plan_path, {})
            _log_task_start(
                step_value=step_enum.value,
                task_id=task_id,
                phase_id=phase_id,
                branch=branch,
                task=next_task,
                plan_path=plan_path,
            )
            event = run_verify_action(
                project_dir=project_dir,
                artifacts_dir=paths["artifacts"],
                run_dir=run_dir,
                phase=phase,
                task=next_task,
                run_id=run_id,
                plan_data=plan_data,
                default_test_command=test_command,
                default_format_command=format_command,
                default_lint_command=lint_command,
                default_typecheck_command=typecheck_command,
                verify_profile=verify_profile,
                ensure_ruff=ensure_ruff,
                timeout_seconds=shift_minutes * 60,
            )

        elif step_enum == TaskStep.COMMIT:
            commit_message = f"{phase.get('id')}: {phase.get('name') or 'phase'}" if phase else task_id
            _log_task_start(
                step_value=step_enum.value,
                task_id=task_id,
                phase_id=phase_id,
                branch=branch,
                task=next_task,
                plan_path=None,
            )
            event = run_commit_action(
                project_dir=project_dir,
                branch=branch or next_task.get("branch") or "",
                commit_message=commit_message,
                run_id=run_id,
                commit_enabled=commit_enabled,
                push_enabled=push_enabled,
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
            
        _log_event(event, run_dir)
        summary = summarize_event(event, run_dir=run_dir)
        logger.debug("Event summary:\n{}", pretty(summary))

        if user_prompt:
            user_prompt = None

        if event:
            _append_event(paths["events"], event.to_dict())

        final_status = "idle"
        final_error = None
        blocked_snapshot = None
        plan_created = False

        with FileLock(lock_path):
            queue = _load_data(paths["task_queue"], {})
            tasks = _normalize_tasks(queue)
            plan = _load_data(paths["phase_plan"], {})
            phases = _normalize_phases(plan)

            target = _find_task(tasks, task_id)
            if not target:
                final_error = "Task not found after run"
            else:
                if task_type == "plan":
                    plan_state = TaskState.from_dict(target)
                    plan_state.last_run_id = run_id
                    if isinstance(event, ProgressHumanBlockers):
                        plan_state.human_blocking_issues = list(event.issues)
                        plan_state.human_next_steps = list(event.next_steps)
                        issue_summary = "; ".join(event.issues).strip() or "Human intervention required"
                        plan_state.last_error = issue_summary
                        plan_state.last_error_type = ERROR_TYPE_BLOCKING_ISSUES
                        plan_state.block_reason = "HUMAN_REQUIRED"
                        plan_state.lifecycle = TaskLifecycle.WAITING_HUMAN
                    elif isinstance(event, WorkerFailed):
                        plan_state.worker_attempts += 1
                        plan_state.last_error = event.error_detail
                        plan_state.last_error_type = event.error_type
                        if plan_state.worker_attempts >= max_attempts:
                            plan_state.block_reason = "WORKER_FAILED"
                            plan_state.lifecycle = TaskLifecycle.WAITING_HUMAN
                        else:
                            plan_state.lifecycle = TaskLifecycle.READY
                    elif isinstance(event, WorkerSucceeded):
                        plan_data = _load_data(paths["phase_plan"], {})
                        plan = plan_data
                        phases = _normalize_phases(plan_data)
                        if not phases:
                            plan_state.lifecycle = TaskLifecycle.WAITING_HUMAN
                            plan_state.block_reason = "PLAN_MISSING"
                            plan_state.last_error = "Phase plan not generated"
                            plan_state.last_error_type = ERROR_TYPE_PLAN_MISSING
                        else:
                            plan_state.lifecycle = TaskLifecycle.DONE
                            plan_state.last_error = None
                            plan_state.last_error_type = None
                            plan_state.block_reason = None
                            plan_task_dict = plan_state.to_dict()
                            tasks = [plan_task_dict] + _build_tasks_from_phases(phases)
                            plan_created = True
                            final_error = None
                    if plan_created:
                        queue["tasks"] = tasks
                    else:
                        updated = plan_state.to_dict()
                        target.clear()
                        target.update(updated)
                        if target.get("lifecycle") == TaskLifecycle.WAITING_HUMAN.value:
                            _record_blocked_intent(
                                target,
                                task_status=target.get("status", task_step),
                                task_type=task_type,
                                phase_id=phase_id or target.get("id"),
                                branch=branch or target.get("branch"),
                                test_command=target.get("test_command"),
                                run_id=run_id,
                                step=target.get("step"),
                                lifecycle=target.get("lifecycle"),
                                prompt_mode=target.get("prompt_mode"),
                            )
                            blocked_snapshot = [dict(target)]
                        final_error = target.get("last_error")
                else:
                    task_state = TaskState.from_dict(target)
                    previous_step = task_state.step
                    caps = {
                        "worker_attempts": max_attempts,
                        "plan_attempts": MAX_IMPL_PLAN_ATTEMPTS,
                        "no_progress_attempts": MAX_NO_PROGRESS_ATTEMPTS,
                        "test_fail_attempts": MAX_TEST_FAIL_ATTEMPTS,
                        "review_gen_attempts": MAX_REVIEW_ATTEMPTS,
                        "review_fix_attempts": MAX_REVIEW_ATTEMPTS,
                        "allowlist_expansion_attempts": MAX_ALLOWLIST_EXPANSION_ATTEMPTS,
                    }
                    task_state = reduce_task(task_state, event, caps=caps)
                    logger.info(
                        "FSM transition: {} -> {} (lifecycle={})",
                        previous_step,
                        task_state.step,
                        task_state.lifecycle,
                    )
                    logger.info(
                        "FSM decision:\n"
                        "  from_step: {}\n"
                        "  to_step: {}\n"
                        "  lifecycle: {}\n"
                        "  reason: {}\n"
                        "  error_type: {}",
                        previous_step,
                        task_state.step,
                        task_state.lifecycle,
                        task_state.block_reason or "none",
                        task_state.last_error_type or "none",
                    )

                    logger.debug(
                        "FSM counters:\n"
                        "  worker_attempts: {}\n"
                        "  plan_attempts: {}\n"
                        "  no_progress_attempts: {}\n"
                        "  test_fail_attempts: {}\n"
                        "  review_gen_attempts: {}\n"
                        "  review_fix_attempts: {}\n"
                        "  allowlist_expansion_attempts: {}",
                        task_state.worker_attempts,
                        task_state.plan_attempts,
                        task_state.no_progress_attempts,
                        task_state.test_fail_attempts,
                        task_state.review_gen_attempts,
                        task_state.review_fix_attempts,
                        task_state.allowlist_expansion_attempts,
                    )
                    logger.debug(
                        "FSM details: block_reason={} last_error_type={} prompt_mode={} last_error={}",
                        task_state.block_reason,
                        task_state.last_error_type,
                        task_state.prompt_mode,
                        task_state.last_error,
                    )


                    if not task_state.step:
                        logger.error(
                            "Task {} missing step after reduce_task",
                            task_id,
                        )

                    updated = task_state.to_dict()
                    target.clear()
                    target.update(updated)
                    if target.get("lifecycle") == TaskLifecycle.WAITING_HUMAN.value:
                        _record_blocked_intent(
                            target,
                            task_status=target.get("status", task_step),
                            task_type=task_type,
                            phase_id=phase_id or target.get("id"),
                            branch=branch or target.get("branch"),
                            test_command=target.get("test_command"),
                            run_id=run_id,
                            step=target.get("step"),
                            lifecycle=target.get("lifecycle"),
                            prompt_mode=target.get("prompt_mode"),
                        )
                        blocked_snapshot = [dict(target)]
                    if phase:
                        phase_entry = _phase_for_task(phases, target)
                        if phase_entry:
                            _sync_phase_status(phase_entry, target)
                    final_error = target.get("last_error")
                    if task_state.lifecycle == TaskLifecycle.WAITING_HUMAN:
                        logger.error(
                            "Task {} blocked: reason={} error={}",
                            task_id,
                            task_state.block_reason,
                            task_state.last_error,
                        )

            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)

        if blocked_snapshot:
            _append_event(paths["events"], _blocking_event_payload(blocked_snapshot))

        if blocked_snapshot and stop_on_blocking_issues:
            _report_blocking_tasks(blocked_snapshot, paths, stopping=True)
            _finalize_run_state(paths, lock_path, status="blocked", last_error=final_error)
            return

        _finalize_run_state(paths, lock_path, status=final_status, last_error=final_error)

        if blocked_snapshot and not stop_on_blocking_issues:
            _report_blocking_tasks(blocked_snapshot, paths, stopping=False)

        if plan_created:
            logger.info("Run {} complete. Phase plan created.", run_id)
            continue

        time.sleep(1)

    logger.info("Done!")
