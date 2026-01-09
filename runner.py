#!/usr/bin/env python3
"""
Feature PRD Runner (Goal-Oriented)
==================================

Standalone helper module for long-running feature development driven by a PRD.
Uses Codex CLI as a worker and keeps durable state in local files.

Refactored to remove rigid step-by-step enforcement in favor of a 
goal-oriented loop (Implement -> Test -> Review).

Usage:
  python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from .constants import (
        DEFAULT_HEARTBEAT_GRACE_SECONDS,
        DEFAULT_HEARTBEAT_SECONDS,
        DEFAULT_MAX_ATTEMPTS,
        DEFAULT_MAX_AUTO_RESUMES,
        DEFAULT_SHIFT_MINUTES,
        DEFAULT_STOP_ON_BLOCKING_ISSUES,
        ERROR_TYPE_BLOCKING_ISSUES,
        ERROR_TYPE_CODEX_EXIT,
        ERROR_TYPE_DISALLOWED_FILES,
        ERROR_TYPE_HEARTBEAT_TIMEOUT,
        ERROR_TYPE_PLAN_MISSING,
        ERROR_TYPE_SHIFT_TIMEOUT,
        IGNORED_REVIEW_PATH_PREFIXES,
        LOCK_FILE,
        MAX_IMPL_PLAN_ATTEMPTS,
        MAX_MANUAL_RESUME_ATTEMPTS,
        MAX_NO_PROGRESS_ATTEMPTS,
        MAX_REVIEW_ATTEMPTS,
        MAX_TEST_FAIL_ATTEMPTS,
        REVIEW_BLOCKING_SEVERITIES,
        TASK_RUN_CODEX_STATUSES,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_DOING,
        TASK_STATUS_DONE,
        TASK_STATUS_IMPLEMENTING,
        TASK_STATUS_PLAN_IMPL,
        TASK_STATUS_REVIEW,
        TASK_STATUS_TODO,
    )
    from .git_utils import (
        _diff_file_sets,
        _ensure_branch,
        _ensure_gitignore,
        _git_changed_files,
        _git_commit_and_push,
        _git_diff_stat,
        _git_diff_text,
        _git_has_changes,
        _git_status_porcelain,
        _is_prd_runner_artifact,
        _snapshot_repo_changes,
        _validate_changes_for_mode,
    )
    from .io_utils import (
        FileLock,
        _append_event,
        _load_data,
        _read_log_tail,
        _read_text_for_prompt,
        _render_json_for_prompt,
        _require_yaml,
        _save_data,
        _update_progress,
    )
    from .prompts import (
        _build_impl_plan_prompt,
        _build_phase_prompt,
        _build_plan_prompt,
        _build_review_prompt,
        _extract_prd_markers,
    )
    from .state import _active_run_is_stale, _ensure_state_files, _finalize_run_state
    from .tasks import (
        _auto_resume_blocked_dependencies,
        _blocking_event_payload,
        _blocking_tasks,
        _build_plan_task,
        _build_tasks_from_phases,
        _find_task,
        _impl_plan_path,
        _increment_task_counter,
        _maybe_auto_resume_blocked,
        _maybe_resume_blocked_last_intent,
        _normalize_phases,
        _normalize_tasks,
        _phase_for_task,
        _read_progress_human_blockers,
        _record_blocked_intent,
        _record_task_run,
        _report_blocking_tasks,
        _resolve_test_command,
        _review_output_path,
        _save_plan,
        _save_queue,
        _select_next_task,
        _summarize_blocking_tasks,
        _sync_phase_status,
        _task_summary,
        _tests_log_path,
    )
    from .utils import _hash_json_data, _now_iso
    from .validation import _extract_review_blocker_files, _validate_impl_plan_data, _validate_review_data
    from .worker import _capture_test_result_snapshot, _run_codex_worker, _run_command
except ImportError:  # pragma: no cover
    from constants import (
        DEFAULT_HEARTBEAT_GRACE_SECONDS,
        DEFAULT_HEARTBEAT_SECONDS,
        DEFAULT_MAX_ATTEMPTS,
        DEFAULT_MAX_AUTO_RESUMES,
        DEFAULT_SHIFT_MINUTES,
        DEFAULT_STOP_ON_BLOCKING_ISSUES,
        ERROR_TYPE_BLOCKING_ISSUES,
        ERROR_TYPE_CODEX_EXIT,
        ERROR_TYPE_DISALLOWED_FILES,
        ERROR_TYPE_HEARTBEAT_TIMEOUT,
        ERROR_TYPE_PLAN_MISSING,
        ERROR_TYPE_SHIFT_TIMEOUT,
        IGNORED_REVIEW_PATH_PREFIXES,
        LOCK_FILE,
        MAX_IMPL_PLAN_ATTEMPTS,
        MAX_MANUAL_RESUME_ATTEMPTS,
        MAX_NO_PROGRESS_ATTEMPTS,
        MAX_REVIEW_ATTEMPTS,
        MAX_TEST_FAIL_ATTEMPTS,
        REVIEW_BLOCKING_SEVERITIES,
        TASK_RUN_CODEX_STATUSES,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_DOING,
        TASK_STATUS_DONE,
        TASK_STATUS_IMPLEMENTING,
        TASK_STATUS_PLAN_IMPL,
        TASK_STATUS_REVIEW,
        TASK_STATUS_TODO,
    )
    from git_utils import (
        _diff_file_sets,
        _ensure_branch,
        _ensure_gitignore,
        _git_changed_files,
        _git_commit_and_push,
        _git_diff_stat,
        _git_diff_text,
        _git_has_changes,
        _git_status_porcelain,
        _is_prd_runner_artifact,
        _snapshot_repo_changes,
        _validate_changes_for_mode,
    )
    from io_utils import (
        FileLock,
        _append_event,
        _load_data,
        _read_log_tail,
        _read_text_for_prompt,
        _render_json_for_prompt,
        _require_yaml,
        _save_data,
        _update_progress,
    )
    from prompts import (
        _build_impl_plan_prompt,
        _build_phase_prompt,
        _build_plan_prompt,
        _build_review_prompt,
        _extract_prd_markers,
    )
    from state import _active_run_is_stale, _ensure_state_files, _finalize_run_state
    from tasks import (
        _auto_resume_blocked_dependencies,
        _blocking_event_payload,
        _blocking_tasks,
        _build_plan_task,
        _build_tasks_from_phases,
        _find_task,
        _impl_plan_path,
        _increment_task_counter,
        _maybe_auto_resume_blocked,
        _maybe_resume_blocked_last_intent,
        _normalize_phases,
        _normalize_tasks,
        _phase_for_task,
        _read_progress_human_blockers,
        _record_blocked_intent,
        _record_task_run,
        _report_blocking_tasks,
        _resolve_test_command,
        _review_output_path,
        _save_plan,
        _save_queue,
        _select_next_task,
        _summarize_blocking_tasks,
        _sync_phase_status,
        _task_summary,
        _tests_log_path,
    )
    from utils import _hash_json_data, _now_iso
    from validation import _extract_review_blocker_files, _validate_impl_plan_data, _validate_review_data
    from worker import _capture_test_result_snapshot, _run_codex_worker, _run_command


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
    resume_prompt: Optional[str] = None,
    stop_on_blocking_issues: bool = DEFAULT_STOP_ON_BLOCKING_ISSUES,
    resume_blocked: bool = True,
) -> None:
    _require_yaml()
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve()
    _ensure_gitignore(project_dir, only_if_clean=True)
    paths = _ensure_state_files(project_dir, prd_path)

    lock_path = paths["state_dir"] / LOCK_FILE
    iteration = 0

    print("\n" + "=" * 70)
    print("  FEATURE PRD RUNNER (Goal-Oriented)")
    print("=" * 70)
    print(f"\nProject directory: {project_dir}")
    print(f"PRD file: {prd_path}")
    print(f"Codex command: {codex_command}")
    print(f"Shift length: {shift_minutes} minutes")
    print(f"Heartbeat: {heartbeat_seconds}s (grace {heartbeat_grace_seconds}s)")
    print(f"Max attempts per task: {max_attempts}")
    print(f"Max auto-resumes: {max_auto_resumes}")
    print(f"Stop on blocking issues: {stop_on_blocking_issues}")
    if test_command:
        print(f"Test command: {test_command}")
    print()

    user_prompt = resume_prompt

    while True:
        if max_iterations and iteration >= max_iterations:
            print(f"\nReached max iterations ({max_iterations})")
            _finalize_run_state(paths, lock_path, status="idle", last_error="Reached max iterations")
            break

        with FileLock(lock_path):
            run_state = _load_data(paths["run_state"], {})
            queue = _load_data(paths["task_queue"], {})
            plan = _load_data(paths["phase_plan"], {})

            tasks = _normalize_tasks(queue)
            phases = _normalize_phases(plan)
            queue["tasks"] = tasks
            plan["phases"] = phases

            if run_state.get("status") == "running":
                if not _active_run_is_stale(
                    run_state,
                    paths["runs"],
                    heartbeat_grace_seconds,
                    shift_minutes,
                ):
                    print("\nAnother run is already active. Exiting to avoid overlap.")
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

            if not tasks:
                tasks = [_build_plan_task()]
                queue["tasks"] = tasks
                queue["updated_at"] = _now_iso()
                _save_data(paths["task_queue"], queue)

            tasks, resumed = _maybe_auto_resume_blocked(queue, tasks, max_auto_resumes)
            if resumed:
                _save_data(paths["task_queue"], queue)
                print("Auto-resumed blocked tasks after auto-resumable failure")

            manually_resumed = False
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
                    print("Resumed most recent blocked task to replay last step")

            blocked_tasks_snapshot = None
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

            next_task = None
            if not blocked_tasks_snapshot:
                next_task = _select_next_task(tasks)
                if not next_task:
                    if _auto_resume_blocked_dependencies(queue, tasks, max_auto_resumes):
                        _save_data(paths["task_queue"], queue)
                        print("Auto-resumed blocked dependency tasks to resolve deadlock")
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
                    print(
                        "\nNo runnable tasks. Queue summary: "
                        f"{summary['todo']} todo, {summary['doing']} doing, "
                        f"{summary['done']} done, {summary['blocked']} blocked"
                    )
                    break

                task_id = str(next_task.get("id"))
                task_type = next_task.get("type", "implement")
                phase_id = next_task.get("phase_id")
                task_status = next_task.get("status", TASK_STATUS_TODO)
                
                # Auto-transition: TODO -> PLAN_IMPL or DOING
                if task_status == TASK_STATUS_TODO:
                    if task_type == "plan":
                        next_task["status"] = TASK_STATUS_DOING
                    else:
                        next_task["status"] = TASK_STATUS_PLAN_IMPL
                    task_status = next_task["status"]
                
                if task_type != "plan":
                    phase_entry = _phase_for_task(phases, next_task)
                    plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                    prd_text, prd_truncated = _read_text_for_prompt(prd_path)
                    prd_markers = _extract_prd_markers(prd_text)
                    phase_test_command = None
                    if phase_entry:
                        phase_test_command = (
                            phase_entry.get("test_command")
                            or next_task.get("test_command")
                            or test_command
                        )
                    plan_data = _load_data(plan_path, {})
                    plan_valid, plan_issue = _validate_impl_plan_data(
                        plan_data,
                        phase_entry or {"id": phase_id or task_id, "acceptance_criteria": []},
                        prd_markers=prd_markers,
                        prd_truncated=prd_truncated,
                        prd_has_content=bool(prd_text.strip()),
                        expected_test_command=phase_test_command,
                    )

                    # --- LOGIC UPDATE: Remove Rigid Step Logic ---
                    # If we are in PLAN_IMPL and have a valid plan (technical approach), proceed to IMPLEMENTING
                    if task_status == TASK_STATUS_PLAN_IMPL and plan_valid:
                        next_task["status"] = TASK_STATUS_IMPLEMENTING
                        next_task["impl_plan_path"] = str(plan_path)
                        plan_hash = _hash_json_data(plan_data)
                        next_task["impl_plan_hash"] = plan_hash
                        next_task["no_progress_attempts"] = 0
                        task_status = next_task["status"]
                    
                    # --- RESTART FIX: Trust REVIEW status ---
                    # Only check for "evidence" if we are NOT in REVIEW.
                    if task_status == TASK_STATUS_IMPLEMENTING:
                        if not plan_valid:
                            next_task["status"] = TASK_STATUS_PLAN_IMPL
                            next_task["last_error"] = f"Implementation plan invalid: {plan_issue}"
                            next_task["last_error_type"] = "impl_plan_invalid"
                            task_status = next_task["status"]
                        # We trust the runner loop to handle "no evidence" via no_progress_attempts
                        # rather than preemptively resetting status here.

                if task_type != "plan" and _git_has_changes(project_dir):
                    dirty_note = (
                        "Workspace has uncommitted changes; continue from them and do not reset."
                    )
                    context = next_task.get("context", []) or []
                    if dirty_note not in context:
                        context.append(dirty_note)
                    next_task["context"] = context
                
                if task_type != "plan":
                    phase_entry = _phase_for_task(phases, next_task)
                    if phase_entry:
                        _sync_phase_status(phase_entry, next_task["status"])
                        _save_plan(paths["phase_plan"], plan, phases)

                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"

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
            _report_blocking_tasks(
                blocked_tasks_snapshot,
                paths,
                stopping=stop_on_blocking_issues,
            )
            _finalize_run_state(paths, lock_path, status="blocked")
            return

        iteration += 1
        run_dir = paths["runs"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_path = run_dir / "progress.json"

        progress_phase = task_type if task_type == "plan" else str(phase_id or task_id)
        _update_progress(
            progress_path,
            {
                "run_id": run_id,
                "task_id": task_id,
                "phase": progress_phase,
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
            },
        )

        prompt_hash = ""
        run_result: Optional[dict[str, Any]] = None
        phase = _phase_for_task(phases, next_task)
        branch = None

        if task_type != "plan":
            if not phase:
                print(f"\nPhase not found for task {task_id}. Blocking task.")
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        target["status"] = TASK_STATUS_BLOCKED
                        target["last_error"] = "Phase not found for task"
                        target["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                        intent_test_command = _resolve_test_command(None, target, test_command)
                        _record_blocked_intent(
                            target,
                            task_status=task_status,
                            task_type=task_type,
                            phase_id=phase_id or target.get("phase_id") or target.get("id"),
                            branch=None,
                            test_command=intent_test_command,
                            run_id=run_id,
                        )
                        _save_queue(paths["task_queue"], queue, tasks)
                _finalize_run_state(paths, lock_path, status="blocked", last_error="Phase not found for task")
                continue

            branch = phase.get("branch") or f"feature/{phase_id or task_id}"
            
            try:
                _ensure_branch(project_dir, branch)
                with FileLock(lock_path):
                    run_state = _load_data(paths["run_state"], {})
                    run_state["branch"] = branch
                    run_state["updated_at"] = _now_iso()
                    _save_data(paths["run_state"], run_state)
            except subprocess.CalledProcessError as exc:
                msg = f"Failed to checkout branch {branch}: {exc}"
                print(f"\n{msg}")
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        target["status"] = TASK_STATUS_BLOCKED
                        target["last_error"] = msg
                        target["last_error_type"] = "git_checkout_failed"
                        intent_test_command = _resolve_test_command(phase, target, test_command)
                        _record_blocked_intent(
                            target,
                            task_status=task_status,
                            task_type=task_type,
                            phase_id=phase_id or target.get("phase_id") or target.get("id"),
                            branch=branch,
                            test_command=intent_test_command,
                            run_id=run_id,
                        )
                        _save_queue(paths["task_queue"], queue, tasks)
                _finalize_run_state(paths, lock_path, status="blocked", last_error=msg)
                continue

        planning_impl = task_type != "plan" and task_status == TASK_STATUS_PLAN_IMPL
        run_codex = task_type == "plan" or task_status in TASK_RUN_CODEX_STATUSES
        
        # --- LOGIC UPDATE: No Step Slicing ---
        # We always allowed allowed_files based on 'files_to_change' in the plan if available
        allowed_files: list[str] = []
        allowed_files_set: Optional[set[str]] = None

        if run_codex:
            pre_run_changed = _snapshot_repo_changes(project_dir)
            try:
                if task_type == "plan":
                    prompt = _build_plan_prompt(
                        prd_path=prd_path,
                        phase_plan_path=paths["phase_plan"],
                        task_queue_path=paths["task_queue"],
                        events_path=paths["events"],
                        progress_path=progress_path,
                        run_id=run_id,
                        user_prompt=user_prompt,
                        heartbeat_seconds=heartbeat_seconds,
                    )
                elif planning_impl:
                    plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                    prd_text, prd_truncated = _read_text_for_prompt(prd_path)
                    prd_markers = _extract_prd_markers(prd_text)
                    phase_test_command = None
                    if phase:
                        phase_test_command = (
                            phase.get("test_command")
                            or next_task.get("test_command")
                            or test_command
                        )
                    prompt = _build_impl_plan_prompt(
                        phase=phase or {"id": phase_id or task_id, "acceptance_criteria": []},
                        prd_path=prd_path,
                        prd_text=prd_text,
                        prd_truncated=prd_truncated,
                        prd_markers=prd_markers,
                        impl_plan_path=plan_path,
                        user_prompt=user_prompt,
                        progress_path=progress_path,
                        run_id=run_id,
                        test_command=phase_test_command,
                        heartbeat_seconds=heartbeat_seconds,
                    )
                elif task_status == TASK_STATUS_REVIEW:
                    # REVIEW logic
                    if not phase:
                        raise ValueError("Phase not found for task")
                    plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                    plan_data = _load_data(plan_path, {})
                    
                    prd_text, prd_truncated = _read_text_for_prompt(prd_path)
                    prd_markers = _extract_prd_markers(prd_text)
                    plan_text, plan_truncated = _render_json_for_prompt(plan_data)
                    
                    diff_text, diff_truncated = _git_diff_text(project_dir)
                    diff_stat, diff_stat_truncated = _git_diff_stat(project_dir)
                    status_text, status_truncated = _git_status_porcelain(project_dir)
                    
                    review_path = _review_output_path(paths["artifacts"], str(phase.get("id") or phase_id or task_id))
                    changed_files = _git_changed_files(
                        project_dir,
                        include_untracked=True,
                        ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
                    )
                    
                    tests_snapshot = next_task.get("last_tests") if isinstance(next_task, dict) else None
                    
                    prompt = _build_review_prompt(
                        phase=phase,
                        review_path=review_path,
                        prd_path=prd_path,
                        prd_text=prd_text,
                        prd_truncated=prd_truncated,
                        prd_markers=prd_markers,
                        user_prompt=user_prompt,
                        progress_path=progress_path,
                        run_id=run_id,
                        changed_files=changed_files,
                        diff_text=diff_text,
                        diff_truncated=diff_truncated,
                        diff_stat=diff_stat,
                        diff_stat_truncated=diff_stat_truncated,
                        status_text=status_text,
                        status_truncated=status_truncated,
                        impl_plan_text=plan_text,
                        impl_plan_truncated=plan_truncated,
                        heartbeat_seconds=heartbeat_seconds,
                        tests_snapshot=tests_snapshot,
                    )
                else:
                    # IMPLEMENTATION logic
                    if not phase:
                        raise ValueError("Phase not found for task")
                    plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                    plan_data = _load_data(plan_path, {})
                    
                    # --- UPDATE: Use full technical approach text ---
                    tech_approach = ""
                    if "technical_approach" in plan_data:
                        raw = plan_data["technical_approach"]
                        if isinstance(raw, list):
                            tech_approach = "\n".join(str(x) for x in raw)
                        else:
                            tech_approach = str(raw)
                    elif "steps" in plan_data:
                         # Fallback if old plan exists
                         tech_approach = json.dumps(plan_data["steps"], indent=2)

                    # Allowed files for entire phase
                    allowed_files = plan_data.get("files_to_change", [])
                    if "new_files" in plan_data:
                        allowed_files.extend(plan_data.get("new_files", []))
                    allowed_files.append("README.md")
                    allowed_files_set = {path for path in allowed_files if path}
                    
                    prompt = _build_phase_prompt(
                        prd_path=prd_path,
                        phase=phase,
                        task=next_task,
                        events_path=paths["events"],
                        progress_path=progress_path,
                        run_id=run_id,
                        user_prompt=user_prompt,
                        impl_plan_path=plan_path,
                        allowed_files=allowed_files,
                        no_progress_attempts=int(next_task.get("no_progress_attempts", 0)),
                        technical_approach_text=tech_approach,
                        heartbeat_seconds=heartbeat_seconds,
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

                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                run_result = _run_codex_worker(
                    command=codex_command,
                    prompt=prompt,
                    project_dir=project_dir,
                    run_dir=run_dir,
                    timeout_seconds=shift_minutes * 60,
                    heartbeat_seconds=heartbeat_seconds,
                    heartbeat_grace_seconds=heartbeat_grace_seconds,
                    progress_path=progress_path,
                    expected_run_id=run_id,
                    on_spawn=_on_worker_spawn,
                )

            except Exception as exc:
                run_result = {
                    "command": codex_command,
                    "prompt_path": str(run_dir / "prompt.txt"),
                    "stdout_path": str(run_dir / "stdout.log"),
                    "stderr_path": str(run_dir / "stderr.log"),
                    "start_time": _now_iso(),
                    "end_time": _now_iso(),
                    "runtime_seconds": 0,
                    "exit_code": 1,
                    "timed_out": False,
                    "no_heartbeat": False,
                    "last_heartbeat": None,
                }
                with open(run_dir / "stderr.log", "a") as handle:
                    handle.write(f"Coordinator error: {exc}\n")

            if user_prompt:
                user_prompt = None

            stdout_tail = _read_log_tail(Path(run_result["stdout_path"]))
            stderr_tail = _read_log_tail(Path(run_result["stderr_path"]))

            human_blocking, human_next_steps = _read_progress_human_blockers(
                progress_path,
                expected_run_id=run_id,
            )
            human_blocking_detected = bool(human_blocking)

            failure = run_result["exit_code"] != 0 or run_result["no_heartbeat"]
            error_detail = None
            error_type = None
            if run_result["no_heartbeat"]:
                error_detail = "No heartbeat received within grace period"
                error_type = ERROR_TYPE_HEARTBEAT_TIMEOUT
            elif run_result["timed_out"]:
                error_detail = "Shift timed out"
                error_type = ERROR_TYPE_SHIFT_TIMEOUT
            elif run_result["exit_code"] != 0:
                error_detail = f"Codex CLI exited with code {run_result['exit_code']}"
                error_type = ERROR_TYPE_CODEX_EXIT
                if stderr_tail.strip():
                    error_detail = f"{error_detail}. stderr: {stderr_tail.strip()}"
                elif stdout_tail.strip():
                    error_detail = f"{error_detail}. stdout: {stdout_tail.strip()}"
                    
            post_run_changed = _snapshot_repo_changes(project_dir)
            introduced, removed = _diff_file_sets(pre_run_changed, post_run_changed)

            # Decide mode
            if task_type == "plan":
                mode = "plan"
            elif planning_impl:
                mode = "plan_impl"
            elif task_status == TASK_STATUS_REVIEW:
                mode = "review"
            else:
                mode = "implement"

            ok, msg, disallowed = _validate_changes_for_mode(
                project_dir=project_dir,
                mode=mode,
                introduced_changes=introduced,
                allowed_files=allowed_files if mode == "implement" else None,
            )
            disallowed_files: list[str] = []
            if not ok and not failure:
                failure = True
                error_detail = msg
                error_type = ERROR_TYPE_DISALLOWED_FILES
                disallowed_files = disallowed

            # Use post_run_changed as the definitive changed_files snapshot
            changed_files_after_run = post_run_changed
            progress_files = changed_files_after_run  # if you still need this later

            manifest = {
                "run_id": run_id,
                "task_id": task_id,
                "mode": mode,
                "start_time": run_result["start_time"],
                "end_time": run_result["end_time"],
                "exit_code": run_result["exit_code"],
                "changed_files": changed_files_after_run,
                "disallowed_files": disallowed_files,
                "pre_run_snapshot": pre_run_changed,
                "post_run_snapshot": post_run_changed,
                "introduced_changes": introduced,
                "removed_changes": removed,
            }
            _save_data(run_dir / "manifest.json", manifest)
            _finalize_run_state(paths, lock_path, status="idle", last_error=error_detail)

            # --- 1. HANDLE BLOCKING ISSUES ---
            if human_blocking_detected:
                issue_summary = "; ".join(human_blocking).strip()[:400]
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        _record_task_run(target, run_id, changed_files_after_run)
                        target["blocking_issues"] = human_blocking
                        target["blocking_next_steps"] = human_next_steps
                        target["last_error"] = f"Human intervention required: {issue_summary}"
                        target["last_error_type"] = ERROR_TYPE_BLOCKING_ISSUES
                        target["status"] = TASK_STATUS_BLOCKED
                        intent_test_command = _resolve_test_command(phase, target, test_command)
                        _record_blocked_intent(
                            target,
                            task_status=task_status,
                            task_type=task_type,
                            phase_id=phase_id or target.get("phase_id") or target.get("id"),
                            branch=branch,
                            test_command=intent_test_command,
                            run_id=run_id,
                        )
                        if phase:
                            _sync_phase_status(phase, target["status"])
                        _save_queue(paths["task_queue"], queue, tasks)
                        
                        blocked_tasks_snapshot = [dict(target)]
                        
                _append_event(paths["events"], _blocking_event_payload(blocked_tasks_snapshot))
                _report_blocking_tasks(blocked_tasks_snapshot, paths, stopping=stop_on_blocking_issues)
                if stop_on_blocking_issues:
                    _finalize_run_state(
                        paths,
                        lock_path,
                        status="blocked",
                        last_error=f"Blocking issues reported: {issue_summary}",
                    )
                    return
                _finalize_run_state(paths, lock_path, status="idle", last_error=f"Blocking issues reported: {issue_summary}")
                continue

            # --- 2. HANDLE FAILURE ---
            if failure:
                print(f"\nRun {run_id} failed: {error_detail}")
                final_status = "idle"
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        _record_task_run(target, run_id, changed_files_after_run)
                        attempts = _increment_task_counter(target, "attempts")
                        target["last_error"] = error_detail or "Run failed"
                        target["last_error_type"] = error_type
                        intent_test_command = _resolve_test_command(phase, target, test_command)
                        if error_type == ERROR_TYPE_DISALLOWED_FILES:
                            target["status"] = TASK_STATUS_BLOCKED
                            context = target.get("context", []) or []
                            context.append(error_detail or "Changes outside allowed files")
                            target["context"] = context
                            _record_blocked_intent(
                                target,
                                task_status=task_status,
                                task_type=task_type,
                                phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                branch=branch,
                                test_command=intent_test_command,
                                run_id=run_id,
                            )
                        elif attempts >= max_attempts:
                            target["status"] = TASK_STATUS_BLOCKED
                            _record_blocked_intent(
                                target,
                                task_status=task_status,
                                task_type=task_type,
                                phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                branch=branch,
                                test_command=intent_test_command,
                                run_id=run_id,
                            )
                        _save_queue(paths["task_queue"], queue, tasks)
                        if target and target.get("status") == TASK_STATUS_BLOCKED:
                            final_status = "blocked"
                _finalize_run_state(paths, lock_path, status=final_status, last_error=error_detail)
                continue

            # --- 3. HANDLE PLAN TASK ---
            if task_type == "plan":
                plan = _load_data(paths["phase_plan"], {})
                phases = _normalize_phases(plan)
                queue = _load_data(paths["task_queue"], {})
                tasks = _normalize_tasks(queue)
                
                # Check if we generated valid phases
                if not phases:
                    with FileLock(lock_path):
                        plan_task = _find_task(tasks, "plan-001")
                        if plan_task:
                            _record_task_run(plan_task, run_id, None)
                            plan_task["status"] = TASK_STATUS_BLOCKED
                            plan_task["last_error"] = "Phase plan not generated"
                            plan_task["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                            _record_blocked_intent(
                                plan_task,
                                task_status=task_status,
                                task_type=task_type,
                                phase_id=plan_task.get("phase_id") or plan_task.get("id"),
                                branch=None,
                                test_command=None,
                                run_id=run_id,
                            )
                        _save_queue(paths["task_queue"], queue, tasks)
                    print(f"\nRun {run_id} complete, but phase plan was not generated.")
                    _finalize_run_state(paths, lock_path, status="blocked", last_error="Phase plan not generated")
                    continue

                # Success - Update Tasks
                plan_task = _find_task(tasks, "plan-001")
                if plan_task:
                    _record_task_run(plan_task, run_id, None)
                    plan_task["status"] = TASK_STATUS_DONE
                
                tasks = [t for t in tasks if t["id"] == "plan-001"] + _build_tasks_from_phases(phases)
                with FileLock(lock_path):
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)

                _finalize_run_state(paths, lock_path, status="idle")
                print(f"\nRun {run_id} complete. Phase plan created.")
                continue

            # --- 4. HANDLE IMPL PLAN TASK ---
            if planning_impl:
                # We validated plan_valid BEFORE the run, but we check if file exists now
                plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                plan_data = _load_data(plan_path, {})
                if not plan_data:
                    # Failed to write plan file
                    print(f"\nRun {run_id} complete, but implementation plan missing.")
                    final_status = "idle"
                    with FileLock(lock_path):
                        queue = _load_data(paths["task_queue"], {})
                        tasks = _normalize_tasks(queue)
                        target = _find_task(tasks, task_id)
                        if target:
                            _record_task_run(target, run_id, None)
                            attempts = _increment_task_counter(target, "plan_attempts")
                            target["status"] = TASK_STATUS_PLAN_IMPL
                            target["last_error"] = "Implementation plan missing"
                            target["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                            if attempts >= MAX_IMPL_PLAN_ATTEMPTS:
                                target["status"] = TASK_STATUS_BLOCKED
                                final_status = "blocked"
                                intent_test_command = _resolve_test_command(phase, target, test_command)
                                _record_blocked_intent(
                                    target,
                                    task_status=TASK_STATUS_PLAN_IMPL,
                                    task_type=task_type,
                                    phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                    branch=branch,
                                    test_command=intent_test_command,
                                    run_id=run_id,
                                )
                        _save_queue(paths["task_queue"], queue, tasks)
                    _finalize_run_state(paths, lock_path, status=final_status, last_error="Implementation plan missing")
                    continue
                
                # Double check validation on the produced file
                prd_text, _ = _read_text_for_prompt(prd_path)
                prd_markers = _extract_prd_markers(prd_text)
                plan_valid, plan_issue = _validate_impl_plan_data(
                    plan_data,
                    phase or {"id": phase_id or task_id, "acceptance_criteria": []},
                    prd_markers=prd_markers,
                    prd_has_content=bool(prd_text.strip()),
                )
                
                final_status = "idle"
                final_error = None
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        _record_task_run(target, run_id, None)
                        if plan_valid:
                            print(f"\nRun {run_id} complete. Implementation plan created.")
                            # Status update handled in next loop iteration via validation
                            target["plan_attempts"] = 0
                        else:
                            print(f"\nImplementation plan invalid: {plan_issue}")
                            target["last_error"] = f"Invalid plan: {plan_issue}"
                            target["last_error_type"] = "impl_plan_invalid"
                            target["context"] = target.get("context", []) + [f"Fix plan: {plan_issue}"]
                            attempts = _increment_task_counter(target, "plan_attempts")
                            if attempts >= MAX_IMPL_PLAN_ATTEMPTS:
                                target["status"] = TASK_STATUS_BLOCKED
                                intent_test_command = _resolve_test_command(phase, target, test_command)
                                _record_blocked_intent(
                                    target,
                                    task_status=TASK_STATUS_PLAN_IMPL,
                                    task_type=task_type,
                                    phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                    branch=branch,
                                    test_command=intent_test_command,
                                    run_id=run_id,
                                )
                        _save_queue(paths["task_queue"], queue, tasks)
                        final_status = "blocked" if target.get("status") == TASK_STATUS_BLOCKED else "idle"
                        final_error = target.get("last_error")

                _finalize_run_state(paths, lock_path, status=final_status, last_error=final_error)
                continue

            # --- 5. HANDLING SUCCESS/FAIL LOOP (IMPLEMENTATION) ---
            if task_type != "plan" and not planning_impl and task_status != TASK_STATUS_REVIEW:
                # 1. Run Tests (if any)
                phase_test_command = (
                     phase.get("test_command") 
                     or next_task.get("test_command") 
                     or test_command
                )
                tests_passed = True
                tests_snapshot = None
                if phase_test_command:
                    test_log_path = _tests_log_path(paths["artifacts"], str(phase.get("id") or phase_id or task_id))
                    test_result = _run_command(phase_test_command, project_dir, test_log_path)
                    tests_snapshot = _capture_test_result_snapshot(
                        command=phase_test_command,
                        exit_code=test_result["exit_code"],
                        log_path=test_log_path,
                    )
                    if test_result["exit_code"] != 0:
                        tests_passed = False
                        print(f"\nTests failed for phase {phase.get('id')}. Log: {test_log_path}")

                # 2. Check Progress
                repo_changes = [p for p in changed_files_after_run if not _is_prd_runner_artifact(p)]
                has_changes = bool(repo_changes)
                
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    
                    if target:
                        _record_task_run(target, run_id, changed_files_after_run)
                        if tests_passed and has_changes:
                            # Success! Move to Review
                            target["status"] = TASK_STATUS_REVIEW
                            target["last_error"] = None
                            target["last_error_type"] = None
                            target["no_progress_attempts"] = 0
                            target["test_fail_attempts"] = 0
                            # keep target["last_tests"] so review can cite it
                            target.pop("test_failure", None)
                        else:
                            # Failure Loop
                            target["status"] = TASK_STATUS_IMPLEMENTING
                            if not tests_passed:
                                attempts = _increment_task_counter(target, "test_fail_attempts")
                                target["last_error"] = f"Tests failed (attempt {attempts}/{MAX_TEST_FAIL_ATTEMPTS})"
                                target["last_error_type"] = "tests_failed"
                                target["last_tests"] = tests_snapshot  # may be None if no test_command
                                log_tail = _read_log_tail(test_log_path, max_chars=4000)
                                target["test_failure"] = {
                                    "command": phase_test_command,
                                    "log_path": str(test_log_path),
                                    "log_tail": log_tail,
                                    "attempt": attempts,
                                    "max_attempts": MAX_TEST_FAIL_ATTEMPTS,
                                    "run_id": run_id,
                                }

                                ctx = target.get("context", []) or []
                                # keep 1 short context line for humans scanning history
                                ctx.append(f"Tests failed. See logs: {test_log_path}")
                                target["context"] = ctx

                                if attempts >= MAX_TEST_FAIL_ATTEMPTS:
                                    target["status"] = TASK_STATUS_BLOCKED
                                    target["last_error_type"] = "test_fail_attempts_exhausted"
                                    _record_blocked_intent(
                                        target,
                                        task_status=TASK_STATUS_IMPLEMENTING,
                                        task_type=task_type,
                                        phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                        branch=branch,
                                        test_command=phase_test_command,
                                        run_id=run_id,
                                    )
                            elif not has_changes:
                                attempts = _increment_task_counter(target, "no_progress_attempts")
                                target["last_error"] = "No changes detected"
                                target["test_fail_attempts"] = 0
                                if attempts >= MAX_NO_PROGRESS_ATTEMPTS:
                                     target["status"] = TASK_STATUS_BLOCKED
                                     _record_blocked_intent(
                                         target,
                                         task_status=TASK_STATUS_IMPLEMENTING,
                                         task_type=task_type,
                                         phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                         branch=branch,
                                         test_command=phase_test_command,
                                         run_id=run_id,
                                     )
                            
                        if phase:
                            _sync_phase_status(phase, target["status"])
                        _save_queue(paths["task_queue"], queue, tasks)
                continue

            # --- 6. HANDLING REVIEW COMPLETION ---
            if task_status == TASK_STATUS_REVIEW:
                review_path = _review_output_path(paths["artifacts"], str(phase.get("id") or phase_id or task_id))
                review_data = _load_data(review_path, {})
                
                # --- NEW: Validate Review Data ---
                prd_text, _ = _read_text_for_prompt(prd_path)
                prd_markers = _extract_prd_markers(prd_text)
                changed_files = _git_changed_files(project_dir, include_untracked=True, ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES)
                
                valid_review, review_issue = _validate_review_data(
                    review_data, 
                    phase, 
                    changed_files, 
                    prd_markers=prd_markers,
                    prd_has_content=bool(prd_text.strip())
                )

                if not valid_review:
                    print(f"\nReview output invalid: {review_issue}")
                    with FileLock(lock_path):
                        queue = _load_data(paths["task_queue"], {})
                        tasks = _normalize_tasks(queue)
                        target = _find_task(tasks, task_id)
                        if target:
                            _record_task_run(target, run_id, changed_files)
                            # Keep status as REVIEW to retry generation, but increment attempts
                            target["last_error"] = f"Review invalid: {review_issue}"
                            target["context"] = target.get("context", []) + [f"Review JSON invalid: {review_issue}"]
                            attempts = _increment_task_counter(target, "review_attempts")
                            if attempts >= MAX_REVIEW_ATTEMPTS:
                                target["status"] = TASK_STATUS_BLOCKED
                                target["last_error_type"] = "review_attempts_exhausted"
                                intent_test_command = _resolve_test_command(phase, target, test_command)
                                _record_blocked_intent(
                                    target,
                                    task_status=TASK_STATUS_REVIEW,
                                    task_type=task_type,
                                    phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                    branch=branch,
                                    test_command=intent_test_command,
                                    run_id=run_id,
                                )
                            _save_queue(paths["task_queue"], queue, tasks)
                    continue

                # Check for blocking issues in review
                issues = review_data.get("issues") or []
                blocking_issues = [
                    it for it in issues
                    if isinstance(it, dict)
                    and str(it.get("severity", "")).strip().lower() in REVIEW_BLOCKING_SEVERITIES
                ]
                
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    
                    if target:
                        if blocking_issues:
                            target["status"] = TASK_STATUS_IMPLEMENTING
                            target["last_error"] = "Review blockers found"
                            summaries = [str(it.get("severity","")).upper() + ": " + str(it.get("summary","")) for it in blocking_issues]
                            target["review_blockers"] = summaries
                            target["review_blocker_files"] = _extract_review_blocker_files(review_data)
                            target["context"] = target.get("context", []) + [f"Review blockers: {summaries}"]
                            attempts = _increment_task_counter(target, "review_attempts")
                            if attempts >= MAX_REVIEW_ATTEMPTS:
                                target["status"] = TASK_STATUS_BLOCKED
                                target["last_error_type"] = "review_attempts_exhausted"
                                intent_test_command = _resolve_test_command(phase, target, test_command)
                                _record_blocked_intent(
                                    target,
                                    task_status=TASK_STATUS_REVIEW,
                                    task_type=task_type,
                                    phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                    branch=branch,
                                    test_command=intent_test_command,
                                    run_id=run_id,
                                )
                                print(
                                    f"\nReview blockers persist for phase {phase.get('id')}. Blocking task."
                                )
                            else:
                                print(f"\nReview blockers found for phase {phase.get('id')}. Re-queueing task.")
                        else:
                            # Success! Commit and Done.
                            target["status"] = TASK_STATUS_DONE
                            target["last_error"] = None
                            target["review_attempts"] = 0
                            
                            # Commit
                            if _git_has_changes(project_dir):
                                commit_message = f"{phase.get('id')}: {phase.get('name') or 'phase'}"
                                try:
                                    _git_commit_and_push(project_dir, branch, commit_message)
                                    _append_event(paths["events"], {
                                        "event_type": "phase_committed",
                                        "phase_id": phase.get("id"),
                                        "commit_message": commit_message
                                    })
                                    print(f"\nPhase {phase.get('id')} complete. Committed and Pushed.")
                                except subprocess.CalledProcessError as exc:
                                    target["status"] = TASK_STATUS_BLOCKED
                                    target["last_error"] = f"Git push failed: {exc}"
                                    intent_test_command = _resolve_test_command(phase, target, test_command)
                                    _record_blocked_intent(
                                        target,
                                        task_status=TASK_STATUS_REVIEW,
                                        task_type=task_type,
                                        phase_id=phase_id or target.get("phase_id") or target.get("id"),
                                        branch=branch,
                                        test_command=intent_test_command,
                                        run_id=run_id,
                                    )
                                    print(f"\nPush failed for phase {phase.get('id')}: {exc}")
                            else:
                                print(f"\nPhase {phase.get('id')} complete (No changes to commit).")

                        _record_task_run(target, run_id, changed_files)
                        if phase:
                            _sync_phase_status(phase, target["status"])
                        _save_queue(paths["task_queue"], queue, tasks)
                        _save_plan(paths["phase_plan"], plan, phases)

        else:
            _finalize_run_state(
                paths,
                lock_path,
                status="blocked",
                last_error=f"Internal error: task {task_id} in status {task_status} not runnable by codex.",
            )
            continue
        time.sleep(1)

    print("\nDone!")


def parse_args() -> argparse.Namespace:
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
        "--resume-prompt",
        type=str,
        default=None,
        help="Special instructions to inject on resume (applies to next agent run only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        resume_prompt=args.resume_prompt,
        stop_on_blocking_issues=args.stop_on_blocking_issues,
        resume_blocked=args.resume_blocked,
    )


if __name__ == "__main__":
    main()
