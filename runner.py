#!/usr/bin/env python3
"""
Feature PRD Runner
==================

Standalone helper module for long-running feature development driven by a PRD.
Uses Codex CLI as a worker and keeps durable state in local files so runs can
resume safely across restarts or interruptions.

Usage:
  python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md

Optional:
  --codex-command "codex exec -"      # default
  --test-command "npm test"           # global fallback test command
  --max-iterations 10
  --no-stop-on-blocking-issues
  --resume-prompt "Focus on error handling first"
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


STATE_DIR_NAME = ".prd_runner"
RUN_STATE_FILE = "run_state.yaml"
TASK_QUEUE_FILE = "task_queue.yaml"
PHASE_PLAN_FILE = "phase_plan.yaml"
ARTIFACTS_DIR = "artifacts"
RUNS_DIR = "runs"
LOCK_FILE = ".lock"

DEFAULT_SHIFT_MINUTES = 45
DEFAULT_HEARTBEAT_SECONDS = 120
DEFAULT_HEARTBEAT_GRACE_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_AUTO_RESUMES = 3
DEFAULT_STOP_ON_BLOCKING_ISSUES = True

TRANSIENT_ERROR_MARKERS = (
    "No heartbeat",
    "Shift timed out",
)

TASK_STATUS_TODO = "todo"
TASK_STATUS_DOING = "doing"
TASK_STATUS_PLAN_IMPL = "plan_impl"
TASK_STATUS_IMPLEMENTING = "implementing"
TASK_STATUS_TESTING = "testing"
TASK_STATUS_REVIEW = "review"
TASK_STATUS_DONE = "done"
TASK_STATUS_BLOCKED = "blocked"

TASK_IN_PROGRESS_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
    TASK_STATUS_TESTING,
    TASK_STATUS_REVIEW,
}

TASK_RUN_CODEX_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
}

ERROR_TYPE_HEARTBEAT_TIMEOUT = "heartbeat_timeout"
ERROR_TYPE_SHIFT_TIMEOUT = "shift_timeout"
ERROR_TYPE_CODEX_EXIT = "codex_exit"
ERROR_TYPE_PLAN_MISSING = "plan_missing"
ERROR_TYPE_BLOCKING_ISSUES = "blocking_issues"
AUTO_RESUME_ERROR_TYPES = {
    ERROR_TYPE_HEARTBEAT_TIMEOUT,
    ERROR_TYPE_SHIFT_TIMEOUT,
}

BLOCKING_RESOLUTION_STEPS = {
    ERROR_TYPE_CODEX_EXIT: [
        "Verify Codex CLI is installed, authenticated, and reachable.",
        "Inspect the latest run logs for stderr output.",
    ],
    ERROR_TYPE_PLAN_MISSING: [
        "Open the PRD and regenerate the phase plan.",
        "Ensure phase_plan.yaml and task_queue.yaml are updated.",
    ],
    ERROR_TYPE_HEARTBEAT_TIMEOUT: [
        "Check Codex CLI connectivity and long-running command settings.",
        "Re-run the runner after the worker is healthy.",
    ],
    ERROR_TYPE_SHIFT_TIMEOUT: [
        "Increase shift minutes or split the phase into smaller steps.",
        "Re-run the runner after adjusting the plan.",
    ],
    "phase_missing": [
        "Update phase_plan.yaml or task_queue.yaml to include the missing phase.",
        "Re-run the runner.",
    ],
    "dirty_branch_mismatch": [
        "Commit or stash local changes on the current branch.",
        "Switch to the expected phase branch and re-run the runner.",
    ],
    "branch_checkout_failed": [
        "Ensure the phase branch exists and git can create or checkout it.",
        "Fix git errors, then re-run the runner.",
    ],
    "impl_plan_invalid": [
        "Inspect the implementation plan JSON and fix schema errors.",
        "Confirm PRD references and test command values are correct.",
    ],
    "impl_plan_attempts_exhausted": [
        "Review the implementation plan and fix schema or content issues.",
        "Regenerate the plan if needed.",
    ],
    "plan_deviation_missing": [
        "Update the implementation plan with plan_deviations explaining changes.",
        "Re-run the runner.",
    ],
    "no_progress_exhausted": [
        "Ensure the current step edits the allowed files.",
        "Update plan steps or allowed files to match actual edits.",
    ],
    "no_changes_exhausted": [
        "Make the required code changes or adjust the phase scope.",
        "Update README/docs if this phase is docs-only.",
    ],
    "review_attempts_exhausted": [
        "Open the review JSON and address all blocking issues.",
        "Re-run the runner once fixes are in place.",
    ],
    "review_invalid": [
        "Ensure the review JSON schema is valid and complete.",
        "Re-run the review step.",
    ],
    "review_blockers": [
        "Address the review blockers listed in the review output.",
        "Re-run the runner to continue the phase.",
    ],
    "git_push_failed": [
        "Check git remote/authentication and resolve conflicts.",
        "Push the branch manually, then re-run the runner.",
    ],
}

REVIEW_MIN_EVIDENCE_ITEMS = 2
REVIEW_MIN_SPEC_TRACE_ITEMS = 3
REVIEW_MIN_ARCH_SUMMARY_ITEMS = 3
REVIEW_MAX_ARCH_SUMMARY_ITEMS = 8
REVIEW_MET_VALUES = {"yes", "no", "partial"}
REVIEW_ARCHITECTURE_CHECKS = [
    "right abstractions introduced",
    "responsibilities split cleanly",
    "failure modes handled and observable",
    "state consistent or idempotent",
    "matches project conventions",
]
MAX_REVIEW_ATTEMPTS = 3
MAX_NO_CHANGE_ATTEMPTS = 3
MAX_IMPL_PLAN_ATTEMPTS = 3
MIN_IMPL_PLAN_STEPS = 5
MAX_NO_PROGRESS_ATTEMPTS = 2
MAX_PLAN_STEPS_PER_RUN = 1
IGNORED_REVIEW_PATH_PREFIXES = [
    ".prd_runner/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "venv/",
    "dist/",
    "build/",
    ".eggs/",
    "htmlcov/",
    "*.egg-info",
    "*.egg-info/*",
    ".coverage",
    "*.pyc",
    "*.pyo",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FileLock:
    """Best-effort cross-platform file lock."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.handle: Optional[Any] = None

    def __enter__(self) -> "FileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.lock_path, "w")
        try:
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.handle:
            return
        try:
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()
        self.handle = None


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _load_data(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        with open(path, "r") as handle:
            if path.suffix in {".yaml", ".yml"}:
                if yaml:
                    data = yaml.safe_load(handle)
                else:
                    data = json.load(handle)
            else:
                data = json.load(handle)
        return data if isinstance(data, dict) else default
    except (OSError, json.JSONDecodeError):
        return default
    except Exception as exc:
        if yaml and isinstance(exc, yaml.YAMLError):
            return default
        raise


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    if not yaml:
        _atomic_write_json(path, data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _atomic_write_data(path: Path, data: dict[str, Any]) -> None:
    if path.suffix in {".yaml", ".yml"}:
        _atomic_write_yaml(path, data)
        return
    _atomic_write_json(path, data)


def _save_data(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_data(path, data)


def _append_event(events_path: Path, event: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", _now_iso())
    with open(events_path, "a") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _update_progress(progress_path: Path, updates: dict[str, Any]) -> None:
    current = _load_data(progress_path, {})
    current.update(updates)
    current["timestamp"] = _now_iso()
    current["heartbeat"] = _now_iso()
    _save_data(progress_path, current)


def _ignore_file_has_entry(path: Path, ignore_entry: str) -> bool:
    if not path.exists():
        return False
    try:
        contents = path.read_text()
    except OSError:
        return False
    lines = {
        line.strip()
        for line in contents.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return STATE_DIR_NAME in lines or ignore_entry in lines


def _append_ignore_entry(path: Path, ignore_entry: str) -> None:
    contents = ""
    if path.exists():
        contents = path.read_text()
    if contents and not contents.endswith("\n"):
        contents += "\n"
    contents += ignore_entry + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _ensure_gitignore(project_dir: Path, only_if_clean: bool = False) -> None:
    ignore_entry = f"{STATE_DIR_NAME}/"
    gitignore_path = project_dir / ".gitignore"
    exclude_path = project_dir / ".git" / "info" / "exclude"
    if _ignore_file_has_entry(gitignore_path, ignore_entry) or _ignore_file_has_entry(
        exclude_path, ignore_entry
    ):
        return
    if exclude_path.parent.exists():
        try:
            _append_ignore_entry(exclude_path, ignore_entry)
            return
        except OSError as exc:
            print(f"Warning: unable to update .git/info/exclude: {exc}")
    if only_if_clean and _git_has_changes(project_dir):
        return
    try:
        _append_ignore_entry(gitignore_path, ignore_entry)
    except OSError as exc:
        print(f"Warning: unable to update .gitignore: {exc}")


def _ensure_state_files(project_dir: Path, prd_path: Path) -> dict[str, Path]:
    state_dir = project_dir / STATE_DIR_NAME
    run_state_path = state_dir / RUN_STATE_FILE
    task_queue_path = state_dir / TASK_QUEUE_FILE
    phase_plan_path = state_dir / PHASE_PLAN_FILE
    artifacts_dir = state_dir / ARTIFACTS_DIR
    runs_dir = state_dir / RUNS_DIR

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not run_state_path.exists():
        _save_data(
            run_state_path,
            {
                "status": "idle",
                "current_phase_id": None,
                "current_task_id": None,
                "run_id": None,
                "branch": None,
                "last_heartbeat": None,
                "last_error": None,
                "updated_at": _now_iso(),
                "prd_path": str(prd_path),
            },
        )

    if not task_queue_path.exists():
        _save_data(
            task_queue_path,
            {
                "updated_at": _now_iso(),
                "tasks": [],
                "task_template": {
                    "id": "phase-001",
                    "type": "implement",
                    "status": "todo",
                    "priority": 1,
                    "deps": [],
                    "description": "Implement phase requirements",
                    "acceptance_criteria": ["List clear success criteria"],
                    "test_command": "optional test command override",
                },
            },
        )

    if not phase_plan_path.exists():
        _save_data(
            phase_plan_path,
            {
                "updated_at": _now_iso(),
                "phases": [],
            },
        )

    events_path = artifacts_dir / "events.ndjson"

    if not events_path.exists():
        events_path.touch()

    return {
        "state_dir": state_dir,
        "run_state": run_state_path,
        "task_queue": task_queue_path,
        "phase_plan": phase_plan_path,
        "artifacts": artifacts_dir,
        "runs": runs_dir,
        "events": events_path,
    }


def _build_plan_task() -> dict[str, Any]:
    return {
        "id": "plan-001",
        "type": "plan",
        "status": TASK_STATUS_TODO,
        "priority": 0,
        "deps": [],
        "description": "Review PRD and repository, then create phases and tasks",
        "acceptance_criteria": [
            "phase_plan.yaml updated with phases",
            "task_queue.yaml contains one implement task per phase",
        ],
    }


def _build_tasks_from_phases(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, phase in enumerate(phases, start=1):
        phase_id = phase.get("id") or f"phase-{index}"
        description = phase.get("description") or phase.get("name") or f"Implement {phase_id}"
        tasks.append(
            {
                "id": phase_id,
                "type": "implement",
                "phase_id": phase_id,
                "status": TASK_STATUS_TODO,
                "priority": index,
                "deps": phase.get("deps", []) or [],
                "description": description,
                "acceptance_criteria": phase.get("acceptance_criteria", []) or [],
                "test_command": phase.get("test_command"),
                "branch": phase.get("branch"),
            }
        )
    return tasks


def _tasks_match_phases(tasks: list[dict[str, Any]], phases: list[dict[str, Any]]) -> bool:
    phase_ids = {phase.get("id") for phase in phases if phase.get("id")}
    if not phase_ids:
        return True
    implement_tasks = [task for task in tasks if task.get("type") == "implement"]
    task_phase_ids = {
        (task.get("phase_id") or task.get("id"))
        for task in implement_tasks
        if task.get("phase_id") or task.get("id")
    }
    return phase_ids.issubset(task_phase_ids)


def _normalize_tasks(queue: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = queue.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        status = task.get("status", TASK_STATUS_TODO)
        if status in {TASK_STATUS_DOING, "in_progress"}:
            if task.get("type") == "plan":
                status = TASK_STATUS_DOING
            else:
                status = TASK_STATUS_IMPLEMENTING
        task["status"] = status
        task.setdefault("priority", 0)
        deps = task.get("deps", [])
        if isinstance(deps, list):
            task["deps"] = deps
        elif deps:
            task["deps"] = [str(deps)]
        else:
            task["deps"] = []
        task.setdefault("attempts", 0)
        task.setdefault("auto_resume_attempts", 0)
        task.setdefault("last_error", None)
        task.setdefault("last_error_type", None)
        task.setdefault("review_attempts", 0)
        task.setdefault("no_change_attempts", 0)
        task.setdefault("no_progress_attempts", 0)
        task.setdefault("plan_attempts", 0)
        task.setdefault("impl_plan_path", None)
        task.setdefault("impl_plan_hash", None)
        task.setdefault("impl_plan_run_id", None)
        task.setdefault("next_plan_step_index", 0)
        task.setdefault("resume_plan_step_index", None)
        task.setdefault("review_blockers", [])
        task.setdefault("review_blocker_files", [])
        task["blocking_issues"] = _coerce_string_list(task.get("blocking_issues"))
        task["blocking_next_steps"] = _coerce_string_list(task.get("blocking_next_steps"))
        task.setdefault("last_changed_files", [])
        task.setdefault("context", [])
        acceptance = task.get("acceptance_criteria", [])
        if isinstance(acceptance, list):
            task["acceptance_criteria"] = acceptance
        elif acceptance:
            task["acceptance_criteria"] = [str(acceptance)]
        else:
            task["acceptance_criteria"] = []
        normalized.append(task)
    return normalized


def _normalize_phases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    phases = plan.get("phases", [])
    if not isinstance(phases, list):
        return []
    normalized: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase.setdefault("status", TASK_STATUS_TODO)
        phase.setdefault("acceptance_criteria", [])
        phase.setdefault("branch", None)
        phase.setdefault("test_command", None)
        normalized.append(phase)
    return normalized


def _task_summary(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"todo": 0, "doing": 0, "done": 0, "blocked": 0}
    for task in tasks:
        status = task.get("status", TASK_STATUS_TODO)
        if status in TASK_IN_PROGRESS_STATUSES:
            counts["doing"] += 1
        elif status in counts:
            counts[status] += 1
    return counts


def _deps_satisfied(task: dict[str, Any], tasks_by_id: dict[str, dict[str, Any]]) -> bool:
    deps = task.get("deps", []) or []
    for dep_id in deps:
        dep = tasks_by_id.get(dep_id)
        if not dep or dep.get("status") != "done":
            return False
    return True


def _select_next_task(tasks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    tasks_by_id = {task.get("id"): task for task in tasks if task.get("id")}
    sorted_tasks = sorted(
        enumerate(tasks),
        key=lambda item: (item[1].get("priority", 0), item[0]),
    )

    for _, task in sorted_tasks:
        if task.get("status") in TASK_IN_PROGRESS_STATUSES:
            return task

    for _, task in sorted_tasks:
        if task.get("status") == "todo" and _deps_satisfied(task, tasks_by_id):
            return task

    return None


def _is_auto_resumable_error(error: Optional[str], error_type: Optional[str] = None) -> bool:
    if error_type in AUTO_RESUME_ERROR_TYPES:
        return True
    if not error:
        return False
    return any(marker in error for marker in TRANSIENT_ERROR_MARKERS)


def _maybe_auto_resume_blocked(
    queue: dict[str, Any],
    tasks: list[dict[str, Any]],
    max_auto_resumes: int,
) -> tuple[list[dict[str, Any]], bool]:
    changed = False
    for task in tasks:
        if task.get("status") != "blocked":
            continue
        last_error = task.get("last_error")
        last_error_type = task.get("last_error_type")
        if not _is_auto_resumable_error(last_error, last_error_type):
            continue
        attempts = int(task.get("auto_resume_attempts", 0))
        if attempts >= max_auto_resumes:
            continue

        task["status"] = TASK_STATUS_TODO
        task["attempts"] = 0
        task["last_error"] = None
        task["last_error_type"] = None
        task["blocking_issues"] = []
        task["blocking_next_steps"] = []
        task["auto_resume_attempts"] = attempts + 1
        task["last_updated_at"] = _now_iso()
        changed = True

    if changed:
        queue["tasks"] = tasks
        queue["updated_at"] = _now_iso()
        queue["auto_resumed_at"] = _now_iso()
    return tasks, changed


def _blocked_dependency_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_id = {task.get("id"): task for task in tasks if task.get("id")}
    blocked: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for dep_id in task.get("deps", []) or []:
            dep = tasks_by_id.get(dep_id)
            if dep and dep.get("status") == TASK_STATUS_BLOCKED:
                blocked[str(dep.get("id"))] = dep
    return list(blocked.values())


def _auto_resume_blocked_dependencies(
    queue: dict[str, Any],
    tasks: list[dict[str, Any]],
    max_auto_resumes: int,
) -> bool:
    blocked_deps = _blocked_dependency_tasks(tasks)
    if not blocked_deps:
        return False
    changed = False
    for task in blocked_deps:
        last_error = task.get("last_error")
        last_error_type = task.get("last_error_type")
        if not _is_auto_resumable_error(last_error, last_error_type):
            continue
        attempts = int(task.get("auto_resume_attempts", 0))
        if attempts >= max_auto_resumes:
            continue
        task["status"] = TASK_STATUS_TODO
        task["attempts"] = 0
        task["last_error"] = None
        task["last_error_type"] = None
        task["blocking_issues"] = []
        task["blocking_next_steps"] = []
        task["auto_resume_attempts"] = attempts + 1
        task["last_updated_at"] = _now_iso()
        changed = True

    if changed:
        queue["tasks"] = tasks
        queue["updated_at"] = _now_iso()
        queue["auto_resumed_at"] = _now_iso()
    return changed


def _read_log_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(errors="replace")
    except OSError:
        return ""
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _validate_string_list(
    value: Any,
    field_name: str,
    min_items: int = 1,
    max_items: Optional[int] = None,
) -> tuple[bool, str]:
    if not isinstance(value, list) or len(value) < min_items:
        return False, f"{field_name} must be a list with at least {min_items} items"
    if max_items is not None and len(value) > max_items:
        return False, f"{field_name} must have no more than {max_items} items"
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return False, f"{field_name} must contain non-empty strings"
    return True, ""


def _increment_task_counter(task: dict[str, Any], field: str) -> int:
    attempts = int(task.get(field, 0)) + 1
    task[field] = attempts
    return attempts


def _sanitize_phase_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "phase"


def _impl_plan_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"impl_plan_{safe_id}.json"


def _hash_json_data(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _blocking_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("status") == TASK_STATUS_BLOCKED]


def _summarize_blocking_tasks(blocked_tasks: list[dict[str, Any]]) -> str:
    if not blocked_tasks:
        return "Blocking issues require human intervention."
    first = blocked_tasks[0]
    error = first.get("last_error") or "Blocking issue reported"
    if len(blocked_tasks) == 1:
        return f"Task {first.get('id')} blocked: {error}"
    return f"{len(blocked_tasks)} tasks blocked. First: {first.get('id')}: {error}"


def _blocking_event_payload(blocked_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_type": "human_intervention_required",
        "tasks": [
            {
                "id": task.get("id"),
                "phase_id": task.get("phase_id"),
                "last_error": task.get("last_error"),
                "last_error_type": task.get("last_error_type"),
                "blocking_issues": _coerce_string_list(task.get("blocking_issues")),
                "blocking_next_steps": _coerce_string_list(task.get("blocking_next_steps")),
            }
            for task in blocked_tasks
        ],
    }


def _resolve_steps_for_block(task: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    next_steps = _coerce_string_list(task.get("blocking_next_steps"))
    if next_steps:
        return next_steps
    error_type = task.get("last_error_type")
    steps = list(BLOCKING_RESOLUTION_STEPS.get(error_type, []))
    phase_id = task.get("phase_id") or task.get("id") or "phase"
    if error_type in {
        "impl_plan_invalid",
        "impl_plan_attempts_exhausted",
        "plan_deviation_missing",
    }:
        impl_plan = task.get("impl_plan_path") or _impl_plan_path(
            paths["artifacts"], str(phase_id)
        )
        steps.insert(0, f"Inspect implementation plan: {impl_plan}.")
    if error_type in {"review_attempts_exhausted", "review_invalid", "review_blockers"}:
        review_path = paths["artifacts"] / f"review_{phase_id}.json"
        steps.insert(0, f"Inspect review output: {review_path}.")
    last_run_id = task.get("last_run_id")
    if last_run_id:
        run_dir = paths["runs"] / str(last_run_id)
        steps.insert(0, f"Review run logs: {run_dir}.")
    if not steps:
        steps = [
            "Inspect the task's last_error and recent logs.",
            "Resolve the issue in the repo or configuration.",
            "Re-run the runner once the blocker is cleared.",
        ]
    return steps


def _report_blocking_tasks(
    blocked_tasks: list[dict[str, Any]],
    paths: dict[str, Path],
    stopping: bool = True,
) -> None:
    if not blocked_tasks:
        return
    status_note = "Stopping runner." if stopping else "Continuing runner."
    print(f"\nBlocking issues detected; human intervention required. {status_note}")
    for task in blocked_tasks:
        task_id = task.get("id") or "(unknown)"
        error_type = task.get("last_error_type") or "unknown"
        last_error = task.get("last_error") or "Blocking issue reported"
        print(f"\nTask {task_id} blocked ({error_type}): {last_error}")
        issues = _coerce_string_list(task.get("blocking_issues"))
        if issues:
            print("Reported blocking issues:")
            for issue in issues:
                print(f"- {issue}")
        steps = _resolve_steps_for_block(task, paths)
        if steps:
            print("Proposed resolve steps:")
            for step in steps:
                print(f"- {step}")


def _render_json_for_prompt(data: dict[str, Any], max_chars: int = 20000) -> tuple[str, bool]:
    text = json.dumps(data, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _format_plan_steps_for_prompt(
    steps: list[dict[str, Any]],
    start_index: int,
    total_steps: int,
) -> str:
    if not steps:
        return "(no steps available)"
    lines: list[str] = []
    for offset, step in enumerate(steps):
        step_number = start_index + offset + 1
        step_id = str(step.get("id") or f"S{step_number}").strip()
        title = str(step.get("title", "")).strip()
        details = str(step.get("details", "")).strip()
        files = step.get("files")
        done_when = step.get("done_when")
        lines.append(f"Step {step_number} of {total_steps}")
        lines.append(f"- id: {step_id}")
        if title:
            lines.append(f"- title: {title}")
        if details:
            lines.append(f"- details: {details}")
        if isinstance(files, list) and files:
            lines.append("- files:")
            for path in files:
                path_value = str(path).strip()
                if path_value:
                    lines.append(f"  - {path_value}")
        if isinstance(done_when, list) and done_when:
            lines.append("- done_when:")
            for item in done_when:
                item_value = str(item).strip()
                if item_value:
                    lines.append(f"  - {item_value}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _allowed_files_for_steps(
    steps: list[dict[str, Any]],
    plan_data: dict[str, Any],
    plan_path: Optional[Path] = None,
) -> list[str]:
    allowed: list[str] = []
    seen: set[str] = set()

    def _add(paths: Any) -> None:
        if not isinstance(paths, list):
            return
        for path in paths:
            path_value = str(path).strip()
            if not path_value or path_value in seen:
                continue
            allowed.append(path_value)
            seen.add(path_value)

    for step in steps:
        if isinstance(step, dict):
            _add(step.get("files"))

    if not allowed:
        _add(plan_data.get("files_to_change"))
        _add(plan_data.get("new_files"))

    if plan_path:
        plan_value = str(plan_path)
        if plan_value not in seen:
            allowed.append(plan_value)
            seen.add(plan_value)

    if "README.md" not in seen:
        allowed.append("README.md")

    return allowed


def _extract_review_blocker_files(review_data: dict[str, Any]) -> list[str]:
    if not isinstance(review_data, dict):
        return []
    collected: list[str] = []
    seen: set[str] = set()

    def _add(paths: Any) -> None:
        if not isinstance(paths, list):
            return
        for path in paths:
            path_value = str(path).strip()
            if not path_value or path_value in seen:
                continue
            collected.append(path_value)
            seen.add(path_value)

    _add(review_data.get("files_reviewed"))
    _add(review_data.get("changed_files"))

    for key, field in [
        ("spec_traceability", "files"),
        ("architecture_checklist", "files"),
        ("acceptance_criteria_checklist", "files"),
        ("logic_risks", "evidence_files"),
    ]:
        items = review_data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                _add(item.get(field))

    return collected


def _build_review_fix_steps(blocking: Any, files: Any) -> list[dict[str, Any]]:
    blockers: list[str] = []
    if isinstance(blocking, list):
        for item in blocking:
            item_value = str(item).strip()
            if item_value:
                blockers.append(item_value)

    file_list: list[str] = []
    if isinstance(files, list):
        for path in files:
            path_value = str(path).strip()
            if path_value:
                file_list.append(path_value)

    details_lines: list[str] = []
    if blockers:
        details_lines.append("Resolve the review blockers:")
        for item in blockers:
            details_lines.append(f"- {item}")
    else:
        details_lines.append("Resolve the latest review blockers recorded in task context.")

    if file_list:
        details_lines.append("Focus on these files:")
        for path in file_list:
            details_lines.append(f"- {path}")

    details = "\n".join(details_lines).strip()

    return [
        {
            "id": "REVIEW-FIX",
            "title": "Resolve review blockers",
            "details": details,
            "files": file_list,
            "done_when": [
                "All listed review blockers are resolved in code",
                "Ready to re-run review",
            ],
        }
    ]


def _is_docs_only_phase(phase: dict[str, Any]) -> bool:
    keywords = ["doc", "docs", "readme", "documentation"]
    haystack = " ".join(
        str(value or "")
        for value in [
            phase.get("name"),
            phase.get("description"),
            " ".join(phase.get("acceptance_criteria") or []),
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _plan_deviations_present(plan_data: dict[str, Any]) -> bool:
    deviations = plan_data.get("plan_deviations")
    if not isinstance(deviations, list):
        return False
    return any(isinstance(item, str) and item.strip() for item in deviations)


def _validate_impl_plan_data(
    plan_data: dict[str, Any],
    phase: dict[str, Any],
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
    expected_test_command: Optional[str] = None,
) -> tuple[bool, str]:
    if not isinstance(plan_data, dict):
        return False, "Implementation plan is not a JSON object"

    phase_id = phase.get("id")
    if not plan_data.get("phase_id"):
        return False, "phase_id missing"
    if phase_id and str(plan_data.get("phase_id")) != str(phase_id):
        return False, "phase_id does not match current phase"

    if not prd_has_content:
        return False, "PRD content missing for plan"

    spec_summary = plan_data.get("spec_summary")
    valid, error = _validate_string_list(spec_summary, "spec_summary")
    if not valid:
        return False, error

    prd_markers = prd_markers or []
    markers_normalized = [_normalize_text(marker) for marker in prd_markers if marker]
    if markers_normalized:
        summary_text = " ".join(_normalize_text(item) for item in spec_summary)
        marker_hits = sum(1 for marker in markers_normalized if marker in summary_text)
        if marker_hits == 0:
            return False, "spec_summary must reference PRD section headers or IDs"
        if prd_truncated:
            required_hits = min(2, len(markers_normalized))
            if marker_hits < required_hits:
                return False, "spec_summary must cite multiple PRD sections when truncated"

    steps = plan_data.get("steps")
    if not isinstance(steps, list) or len(steps) < MIN_IMPL_PLAN_STEPS:
        return False, f"steps must include at least {MIN_IMPL_PLAN_STEPS} items"
    step_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"steps[{index}] must be an object"
        missing_fields = []
        step_id = str(step.get("id", "")).strip()
        title = str(step.get("title", "")).strip()
        details = str(step.get("details", "")).strip()
        files = step.get("files")
        done_when = step.get("done_when")
        if not step_id:
            missing_fields.append("id")
        if not title:
            missing_fields.append("title")
        if not details:
            missing_fields.append("details")
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            missing_fields.append("files")
        if not isinstance(done_when, list) or not done_when or any(
            not isinstance(item, str) or not item.strip() for item in done_when
        ):
            missing_fields.append("done_when")
        if missing_fields:
            return False, f"steps[{index}] missing or invalid: {', '.join(missing_fields)}"
        if step_id in step_ids:
            return False, f"steps[{index}] has duplicate id"
        step_ids.add(step_id)

    files_to_change = plan_data.get("files_to_change")
    if not isinstance(files_to_change, list):
        return False, "files_to_change must be a list"
    if not files_to_change and not _is_docs_only_phase(phase):
        return False, "files_to_change must be non-empty for non-docs phases"
    if any(not isinstance(path, str) or not path.strip() for path in files_to_change):
        return False, "files_to_change must contain non-empty strings"

    new_files = plan_data.get("new_files")
    if not isinstance(new_files, list):
        return False, "new_files must be a list"
    if any(not isinstance(path, str) or not path.strip() for path in new_files):
        return False, "new_files must contain non-empty strings"

    design_notes = plan_data.get("design_notes")
    if not isinstance(design_notes, dict):
        return False, "design_notes must be an object"
    valid, error = _validate_string_list(
        design_notes.get("architecture"),
        "design_notes.architecture",
        min_items=REVIEW_MIN_ARCH_SUMMARY_ITEMS,
        max_items=REVIEW_MAX_ARCH_SUMMARY_ITEMS,
    )
    if not valid:
        return False, error
    for field in ["data_flow", "invariants", "edge_cases"]:
        valid, error = _validate_string_list(
            design_notes.get(field),
            f"design_notes.{field}",
        )
        if not valid:
            return False, error

    test_plan = plan_data.get("test_plan")
    if not isinstance(test_plan, dict):
        return False, "test_plan must be an object"
    commands = test_plan.get("commands")
    if expected_test_command and (
        not isinstance(commands, list) or not commands or any(
            not isinstance(cmd, str) or not cmd.strip() for cmd in commands
        )
    ):
        return False, "test_plan.commands must include at least one command"
    for field in ["new_tests", "manual_checks"]:
        value = test_plan.get(field)
        if not isinstance(value, list):
            return False, f"test_plan.{field} must be a list"

    migration = plan_data.get("migration_or_rollout")
    if not isinstance(migration, list):
        return False, "migration_or_rollout must be a list"
    open_questions = plan_data.get("open_questions")
    if not isinstance(open_questions, list):
        return False, "open_questions must be a list"
    assumptions = plan_data.get("assumptions")
    if not isinstance(assumptions, list):
        return False, "assumptions must be a list"

    acceptance = phase.get("acceptance_criteria") or []
    acceptance_mapping = plan_data.get("acceptance_mapping")
    if not isinstance(acceptance_mapping, list) or not acceptance_mapping:
        return False, "acceptance_mapping must be a non-empty list"
    mapping_criteria: set[str] = set()
    for index, item in enumerate(acceptance_mapping):
        if not isinstance(item, dict):
            return False, f"acceptance_mapping[{index}] must be an object"
        missing_fields = []
        criterion_raw = str(item.get("criterion", "")).strip()
        criterion = _normalize_text(criterion_raw)
        plan_steps = item.get("plan_steps")
        evidence_strategy = str(item.get("evidence_strategy", "")).strip()
        if not criterion:
            missing_fields.append("criterion")
        if not isinstance(plan_steps, list) or not plan_steps:
            missing_fields.append("plan_steps")
        if not evidence_strategy:
            missing_fields.append("evidence_strategy")
        if missing_fields:
            return False, f"acceptance_mapping[{index}] missing or invalid: {', '.join(missing_fields)}"
        if any(step_id not in step_ids for step_id in plan_steps):
            return False, f"acceptance_mapping[{index}] references unknown step ids"
        mapping_criteria.add(criterion)

    if isinstance(acceptance, list) and acceptance:
        normalized_acceptance = {
            _normalize_text(str(criterion))
            for criterion in acceptance
            if str(criterion).strip()
        }
        missing = [criterion for criterion in normalized_acceptance if criterion not in mapping_criteria]
        if missing:
            return False, "acceptance_mapping missing criteria entries"
    else:
        if "(none provided)" not in mapping_criteria:
            return False, 'acceptance_mapping must include "(none provided)"'

    return True, ""


def _validate_review_data(
    review_data: dict[str, Any],
    phase: dict[str, Any],
    changed_files: Optional[list[str]] = None,
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
    plan_step_ids: Optional[list[str]] = None,
) -> tuple[bool, str]:
    if not isinstance(review_data, dict):
        return False, "Review output is not a JSON object"

    phase_id = phase.get("id")
    review_phase_id = review_data.get("phase_id")
    if not review_phase_id:
        return False, "phase_id missing"
    if phase_id and str(review_phase_id) != str(phase_id):
        return False, "phase_id does not match current phase"

    blocking = review_data.get("blocking_issues")
    if not isinstance(blocking, list):
        return False, "blocking_issues must be a list"
    if not prd_has_content and not blocking:
        return False, "blocking_issues must include PRD access failure"

    files_reviewed = review_data.get("files_reviewed")
    if not isinstance(files_reviewed, list) or not files_reviewed:
        return False, "files_reviewed must be a non-empty list"
    if any(not isinstance(item, str) or not item.strip() for item in files_reviewed):
        return False, "files_reviewed must contain non-empty strings"
    files_reviewed_set = {item.strip() for item in files_reviewed if item.strip()}

    expected_files = [path.strip() for path in (changed_files or []) if str(path).strip()]
    expected_set = {path for path in expected_files if path}
    review_changed_files = review_data.get("changed_files")
    if expected_files or review_changed_files is not None:
        if not isinstance(review_changed_files, list):
            return False, "changed_files must be a list"
        review_changed_set = {
            str(path).strip() for path in review_changed_files if str(path).strip()
        }
        if expected_set and not review_changed_set.intersection(expected_set):
            return False, "changed_files in review do not overlap with actual changes"
        if not expected_set.issubset(files_reviewed_set):
            return False, "not all changed_files were reviewed"

    spec_summary = review_data.get("spec_summary")
    valid, error = _validate_string_list(spec_summary, "spec_summary")
    if not valid:
        return False, error

    prd_markers = prd_markers or []
    markers_normalized = [_normalize_text(marker) for marker in prd_markers if marker]
    if markers_normalized:
        summary_text = " ".join(_normalize_text(item) for item in spec_summary)
        marker_hits = sum(1 for marker in markers_normalized if marker in summary_text)
        if marker_hits == 0:
            return False, "spec_summary must reference PRD section headers or IDs"
        if prd_truncated:
            required_hits = min(2, len(markers_normalized))
            if marker_hits < required_hits:
                return False, "spec_summary must cite multiple PRD sections when truncated"

    evidence = review_data.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < REVIEW_MIN_EVIDENCE_ITEMS:
        return False, f"evidence must include at least {REVIEW_MIN_EVIDENCE_ITEMS} items"
    if any(not isinstance(item, str) or not item.strip() for item in evidence):
        return False, "evidence must contain non-empty strings"

    design_assessment = review_data.get("design_assessment")
    if not isinstance(design_assessment, dict):
        return False, "design_assessment must be an object"

    valid, error = _validate_string_list(
        design_assessment.get("architecture_summary"),
        "design_assessment.architecture_summary",
        min_items=REVIEW_MIN_ARCH_SUMMARY_ITEMS,
        max_items=REVIEW_MAX_ARCH_SUMMARY_ITEMS,
    )
    if not valid:
        return False, error

    for field in [
        "key_components",
        "data_flow",
        "invariants",
        "edge_cases",
        "security_privacy",
        "performance_scalability",
    ]:
        valid, error = _validate_string_list(
            design_assessment.get(field),
            f"design_assessment.{field}",
        )
        if not valid:
            return False, error

    architecture_checklist = review_data.get("architecture_checklist")
    if not isinstance(architecture_checklist, list) or not architecture_checklist:
        return False, "architecture_checklist must be a non-empty list"
    expected_checks = {_normalize_text(item) for item in REVIEW_ARCHITECTURE_CHECKS}
    observed_checks: set[str] = set()
    for index, item in enumerate(architecture_checklist):
        if not isinstance(item, dict):
            return False, f"architecture_checklist[{index}] must be an object"
        missing_fields = []
        check_value = _normalize_text(str(item.get("check", "")))
        met = str(item.get("met", "")).strip().lower()
        item_evidence = str(item.get("evidence", "")).strip()
        files = item.get("files")
        if not check_value:
            missing_fields.append("check")
        if met not in REVIEW_MET_VALUES:
            missing_fields.append("met")
        if not item_evidence:
            missing_fields.append("evidence")
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            missing_fields.append("files")
        if missing_fields:
            return False, f"architecture_checklist[{index}] missing or invalid: {', '.join(missing_fields)}"
        observed_checks.add(check_value)
    if not expected_checks.issubset(observed_checks):
        return False, "architecture_checklist missing required checks"

    spec_traceability = review_data.get("spec_traceability")
    if not isinstance(spec_traceability, list) or not spec_traceability:
        return False, "spec_traceability must be a non-empty list"
    spec_trace_min = 1
    acceptance = phase.get("acceptance_criteria") or []
    if (changed_files and len(changed_files) > 1) or (
        isinstance(acceptance, list) and len(acceptance) > 1
    ):
        spec_trace_min = REVIEW_MIN_SPEC_TRACE_ITEMS
    if markers_normalized and len(markers_normalized) < spec_trace_min:
        spec_trace_min = len(markers_normalized)
    if len(spec_traceability) < spec_trace_min:
        return False, f"spec_traceability must include at least {spec_trace_min} items"
    trace_has_marker = False
    trace_has_step = False
    normalized_steps = {_normalize_text(step_id) for step_id in (plan_step_ids or []) if step_id}
    for index, item in enumerate(spec_traceability):
        if not isinstance(item, dict):
            return False, f"spec_traceability[{index}] must be an object"
        missing_fields = []
        requirement_raw = str(item.get("requirement", "")).strip()
        requirement_norm = _normalize_text(requirement_raw)
        item_evidence = str(item.get("implementation_evidence", "")).strip()
        evidence_norm = _normalize_text(item_evidence)
        files = item.get("files")
        if not requirement_raw:
            missing_fields.append("requirement")
        if not item_evidence:
            missing_fields.append("implementation_evidence")
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            missing_fields.append("files")
        if missing_fields:
            return False, f"spec_traceability[{index}] missing or invalid: {', '.join(missing_fields)}"
        if markers_normalized and any(marker in requirement_norm for marker in markers_normalized):
            trace_has_marker = True
        if normalized_steps and any(
            step_id in requirement_norm or step_id in evidence_norm
            for step_id in normalized_steps
        ):
            trace_has_step = True
    if markers_normalized and not trace_has_marker:
        return False, "spec_traceability must reference PRD sections or IDs"
    if normalized_steps and not trace_has_step:
        return False, "spec_traceability must reference plan step IDs"

    logic_risks = review_data.get("logic_risks")
    if not isinstance(logic_risks, list) or not logic_risks:
        return False, "logic_risks must be a non-empty list"
    for index, item in enumerate(logic_risks):
        if not isinstance(item, dict):
            return False, f"logic_risks[{index}] must be an object"
        missing_fields = []
        risk = str(item.get("risk", "")).strip()
        impact = str(item.get("impact", "")).strip()
        mitigation = str(item.get("mitigation", "")).strip()
        files = item.get("evidence_files")
        if not risk:
            missing_fields.append("risk")
        if not impact:
            missing_fields.append("impact")
        if not mitigation:
            missing_fields.append("mitigation")
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            missing_fields.append("evidence_files")
        if missing_fields:
            return False, f"logic_risks[{index}] missing or invalid: {', '.join(missing_fields)}"

    checklist = review_data.get("acceptance_criteria_checklist")
    if not isinstance(checklist, list) or not checklist:
        return False, "acceptance_criteria_checklist must be a non-empty list"

    checklist_criteria = []
    for index, item in enumerate(checklist):
        if not isinstance(item, dict):
            return False, f"checklist[{index}] must be an object"
        missing_fields = []
        criterion_raw = str(item.get("criterion", "")).strip()
        criterion = _normalize_text(criterion_raw)
        met = str(item.get("met", "")).strip().lower()
        item_evidence = str(item.get("evidence", "")).strip()
        files = item.get("files")
        if not criterion:
            missing_fields.append("criterion")
        if met not in REVIEW_MET_VALUES:
            missing_fields.append("met")
        if not item_evidence:
            missing_fields.append("evidence")
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            missing_fields.append("files")
        if missing_fields:
            return False, f"checklist[{index}] missing or invalid: {', '.join(missing_fields)}"
        checklist_criteria.append(criterion)

    if isinstance(acceptance, list) and acceptance:
        normalized_acceptance = {
            _normalize_text(str(criterion))
            for criterion in acceptance
            if str(criterion).strip()
        }
        checklist_set = {criterion for criterion in checklist_criteria if criterion}
        missing = [criterion for criterion in normalized_acceptance if criterion not in checklist_set]
        if missing:
            return False, "acceptance_criteria_checklist missing criteria entries"
    else:
        if "(none provided)" not in checklist_criteria:
            return False, 'acceptance_criteria_checklist must include "(none provided)"'

    return True, ""


def _stream_pipe(pipe: Any, file_path: Path, label: str, to_stderr: bool) -> None:
    prefix = f"[codex {label}] "
    with open(file_path, "w") as handle:
        for line in iter(pipe.readline, ""):
            handle.write(line)
            handle.flush()
            if to_stderr:
                sys.stderr.write(prefix + line)
                sys.stderr.flush()
            else:
                sys.stdout.write(prefix + line)
                sys.stdout.flush()
    try:
        pipe.close()
    except Exception:
        pass


def _build_plan_prompt(
    prd_path: Path,
    phase_plan_path: Path,
    task_queue_path: Path,
    events_path: Path,
    progress_path: Path,
    run_id: str,
    user_prompt: Optional[str],
) -> str:
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    return f"""You are a Codex CLI worker. Your task is to plan phases for a new feature.

Inputs:
- PRD: {prd_path}
- Repository: current working directory
{user_block}

Output files (write/update):
- Phase plan: {phase_plan_path}
  Schema:
  {{
    "updated_at": "ISO-8601",
    "phases": [
      {{
        "id": "phase-1",
        "name": "Short phase name",
        "status": "todo",
        "description": "What this phase delivers",
        "acceptance_criteria": ["list of acceptance checks"],
        "branch": "feature/phase-1-short-name",
        "test_command": "optional command for this phase"
      }}
    ]
  }}
- Task queue: {task_queue_path}
  Include one task per phase with:
  id, type="implement", phase_id, status, priority, deps, description,
  acceptance_criteria, test_command, branch.

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, blocking_issues, heartbeat.

Review the PRD plan against the existing codebase. Split work into phases that
are independently reviewable and testable. If the PRD already includes phases,
align with them but adjust as needed based on the repository state.
"""


def _build_phase_prompt(
    prd_path: Path,
    phase: dict[str, Any],
    task: dict[str, Any],
    events_path: Path,
    progress_path: Path,
    run_id: str,
    user_prompt: Optional[str],
    impl_plan_path: Optional[Path] = None,
    plan_steps_text: str = "",
    allowed_files: Optional[list[str]] = None,
    no_progress_attempts: int = 0,
) -> str:
    phase_name = phase.get("name") or phase.get("id")
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = (
        "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none provided)"
    )
    context_items = task.get("context", []) or []
    context_block = "\n".join(f"- {item}" for item in context_items) if context_items else "- (none)"

    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    plan_path_display = impl_plan_path or "(missing)"
    allowed_files = allowed_files or []
    allowed_block = "\n".join(f"- {path}" for path in allowed_files) if allowed_files else "- (none)"
    no_progress_block = ""
    if no_progress_attempts > 0:
        no_progress_block = (
            "\nNOTE: Previous run made no code changes. You must edit the allowed files "
            "and complete the listed step(s) in this run.\n"
        )
    return f"""You are a Codex CLI worker. Implement the phase described below.

PRD: {prd_path}
Phase: {phase_name}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}

Additional context from previous runs:
{context_block}
{user_block}

Rules:
- Work only on this phase scope.
- Do not commit or push; the coordinator will handle git.
- If tests fail, fix them (the coordinator will also run tests).
- Keep the project's README.md updated with changes in this phase (features, setup, usage).
- Follow the plan steps below. If you must deviate, update the plan file with a
  non-empty "plan_deviations" list explaining why.
- Do only the step(s) listed below. Do not continue to later steps.
- Begin editing code immediately.
- Do not scan the repository.
- Read/edit only the allowed files.

Implementation plan file: {plan_path_display}
Plan step(s) for this run:
{plan_steps_text or "(missing)"}

Allowed files to read/edit:
{allowed_block}
{no_progress_block}

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, blocking_issues, heartbeat.
"""


def _read_text_for_prompt(path: Path, max_chars: int = 20000) -> tuple[str, bool]:
    try:
        content = path.read_text(errors="replace")
    except OSError:
        return "", False
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _extract_prd_markers(prd_text: str, max_items: int = 20) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    id_pattern = re.compile(r"\b[A-Z]{2,10}-\d+\b")
    req_pattern = re.compile(r"\bREQ(?:UIREMENT)?\s*\d+\b", re.IGNORECASE)

    for line in prd_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip()
            normalized = _normalize_text(header)
            if header and normalized not in seen:
                markers.append(header)
                seen.add(normalized)
                if len(markers) >= max_items:
                    return markers
        for match in id_pattern.findall(line):
            normalized = _normalize_text(match)
            if normalized not in seen:
                markers.append(match)
                seen.add(normalized)
                if len(markers) >= max_items:
                    return markers
        for match in req_pattern.findall(line):
            normalized = _normalize_text(match)
            if normalized not in seen:
                markers.append(match)
                seen.add(normalized)
                if len(markers) >= max_items:
                    return markers

    return markers


def _build_impl_plan_prompt(
    phase: dict[str, Any],
    prd_path: Path,
    prd_text: str,
    prd_truncated: bool,
    prd_markers: Optional[list[str]],
    impl_plan_path: Path,
    user_prompt: Optional[str],
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    test_command: Optional[str] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    prd_notice = ""
    if prd_truncated:
        prd_notice = (
            "\nNOTE: PRD content truncated. Open the PRD file to read the full spec.\n"
        )
    prd_markers = prd_markers or []
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"
    progress_block = ""
    if progress_path and run_id:
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, blocking_issues, heartbeat.\n"
        )
    test_block = test_command or "(none specified)"
    return f"""You are a Codex CLI worker. Produce an implementation plan for the phase below.

PRD: {prd_path}
PRD content (read first):
{prd_text}
{prd_notice}
PRD sections/IDs (cite in spec_summary):
{markers_block}

Phase: {phase.get("name") or phase.get("id")}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}
Global/phase test command: {test_block}
{user_block}
{progress_block}

Output file (write JSON): {impl_plan_path}

Plan schema (strict):
{{
  "phase_id": "{phase.get('id')}",
  "spec_summary": ["PRD requirements relevant to this phase (with section/IDs)"],
  "acceptance_mapping": [
    {{
      "criterion": "exact acceptance criterion",
      "plan_steps": ["S1", "S3"],
      "evidence_strategy": "how we'll prove it works"
    }}
  ],
  "files_to_change": ["path/a.py", "path/b.ts"],
  "new_files": ["path/new_file.py"],
  "steps": [
    {{
      "id": "S1",
      "title": "Short action",
      "details": "precise instructions",
      "files": ["path/a.py"],
      "risks": ["..."],
      "done_when": ["observable checks"]
    }}
  ],
  "design_notes": {{
    "architecture": ["3-8 bullets"],
    "data_flow": ["..."],
    "invariants": ["..."],
    "edge_cases": ["..."]
  }},
  "test_plan": {{
    "commands": ["npm test", "pytest -q"],
    "new_tests": ["tests/test_x.py::test_y"],
    "manual_checks": ["..."]
  }},
  "migration_or_rollout": ["if applicable, otherwise '(none)'"],
  "open_questions": ["if any"],
  "assumptions": ["..."],
  "plan_deviations": []
}}

Rules:
- Cite PRD section headers/IDs in spec_summary where available.
- Include at least {MIN_IMPL_PLAN_STEPS} steps with unique IDs.
- Steps must be concrete, reference real files, and include done_when checks.
- If you cannot infer exact files, include open_questions and propose a safe default plan.
- Ensure acceptance_mapping covers every acceptance criterion with step IDs.
- If acceptance criteria are empty, include one mapping with criterion "(none provided)".
- files_to_change must be non-empty unless the phase is docs-only.
- Set plan_deviations to an empty list in the initial plan.
"""


def _build_review_prompt(
    phase: dict[str, Any],
    review_path: Path,
    prd_path: Path,
    prd_text: str,
    prd_truncated: bool,
    prd_markers: Optional[list[str]],
    user_prompt: Optional[str],
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    changed_files: Optional[list[str]] = None,
    diff_text: str = "",
    diff_truncated: bool = False,
    diff_stat: str = "",
    diff_stat_truncated: bool = False,
    status_text: str = "",
    status_truncated: bool = False,
    impl_plan_text: str = "",
    impl_plan_truncated: bool = False,
    plan_step_ids: Optional[list[str]] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    progress_block = ""
    if progress_path and run_id:
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, blocking_issues, heartbeat.\n"
        )
    prd_notice = ""
    if prd_truncated:
        prd_notice = (
            "\nNOTE: PRD content truncated. Open the PRD file to read the full spec.\n"
        )
    prd_markers = prd_markers or []
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"
    changed_files = changed_files or []
    changed_block = "\n".join(f"- {path}" for path in changed_files) if changed_files else "- (none)"
    diff_notice = ""
    if diff_truncated:
        diff_notice = "\nNOTE: Diff truncated. Open git diff for full context.\n"
    diff_block = diff_text if diff_text else "(no diff output; open listed files directly)"
    diff_stat_notice = ""
    if diff_stat_truncated:
        diff_stat_notice = "\nNOTE: Diffstat truncated.\n"
    diff_stat_block = diff_stat if diff_stat else "(no diffstat output)"
    status_notice = ""
    if status_truncated:
        status_notice = "\nNOTE: Status truncated.\n"
    status_block = status_text if status_text else "(clean)"
    plan_notice = ""
    if impl_plan_truncated:
        plan_notice = "\nNOTE: Implementation plan truncated. Open the plan file for full details.\n"
    plan_step_ids = plan_step_ids or []
    plan_steps_block = ", ".join(plan_step_ids) if plan_step_ids else "(none)"
    return f"""Perform a code review for the phase below and write JSON to {review_path}.

Phase: {phase.get("name") or phase.get("id")}
PRD: {prd_path}
PRD content (read first):
{prd_text}
{prd_notice}
PRD sections/IDs (for reference in spec_summary):
{markers_block}
Acceptance criteria:
{acceptance_block}
{user_block}
{progress_block}

Git status (from coordinator):
{status_block}
{status_notice}

Changed files (from coordinator):
{changed_block}

Diffstat (from coordinator):
{diff_stat_block}
{diff_stat_notice}

Diff (from coordinator):
{diff_block}
{diff_notice}

Implementation plan (from coordinator):
{impl_plan_text or "(missing)"}
{plan_notice}
Plan step IDs: {plan_steps_block}

Review output schema:
{{
  "phase_id": "{phase.get('id')}",
  "spec_summary": ["bullets restating PRD requirements relevant to this phase"],
  "spec_traceability": [
    {{
      "requirement": "PRD section/ID and requirement text",
      "implementation_evidence": "specific evidence from code or diff",
      "files": ["path/to/file.ext"]
    }}
  ],
  "design_assessment": {{
    "architecture_summary": ["3-8 bullets describing solution structure"],
    "key_components": ["components/modules touched and their roles"],
    "data_flow": ["input -> transform -> output flow bullets"],
    "invariants": ["properties that must hold true"],
    "edge_cases": ["non-trivial cases and handling"],
    "security_privacy": ["concerns or (none)"],
    "performance_scalability": ["concerns or (none)"]
  }},
  "architecture_checklist": [
    {{
      "check": "right abstractions introduced",
      "met": "yes|no|partial",
      "evidence": "specific evidence from code or diff",
      "files": ["path/to/file.ext"]
    }}
  ],
  "acceptance_criteria_checklist": [
    {{
      "criterion": "exact acceptance criterion text",
      "met": "yes|no|partial",
      "evidence": "specific evidence from code or diff",
      "files": ["path/to/file.ext"]
    }}
  ],
  "summary": "Short summary",
  "blocking_issues": ["list of blockers"],
  "non_blocking": ["nice-to-have improvements"],
  "changed_files": ["exact list from coordinator"],
  "files_reviewed": ["list of paths"],
  "evidence": ["at least two concrete observations with file/diff references"],
  "logic_risks": [
    {{
      "risk": "risk statement or (none found)",
      "impact": "impact if risk occurs",
      "mitigation": "mitigation or follow-up",
      "evidence_files": ["path/to/file.ext"]
    }}
  ],
  "recommendations": ["actionable fixes"]
}}

Review instructions:
- Read the PRD content and restate the relevant requirements in spec_summary.
- Reference PRD section headers/IDs in spec_summary where available.
- Base your review on the provided changed files, diffstat, and diff (do not invent files).
- Copy the coordinator-provided changed_files list exactly into changed_files.
- Populate spec_traceability for PRD requirements, not just acceptance criteria.
- Complete design_assessment and architecture_checklist from the code and diff.
- If you cannot describe architecture/data flow/invariants from code, add a blocking_issue.
- Verify implementation aligns with the plan; cite plan step IDs in spec_traceability where relevant.
- For each acceptance criterion, fill acceptance_criteria_checklist with met + evidence.
- If acceptance criteria are empty, include one checklist item with criterion "(none provided)".
- Provide at least {REVIEW_MIN_EVIDENCE_ITEMS} concrete evidence items tied to files/diff.
- If README updates are missing or spec requirements are unmet, add blocking_issues.
- Do not treat passing tests as sufficient evidence of spec compliance.
- If you cannot access the PRD text or diff, add a blocking_issue explaining why.

Architecture checklist (use these exact check labels):
- right abstractions introduced
- responsibilities split cleanly
- failure modes handled and observable
- state consistent or idempotent
- matches project conventions
"""


def _run_codex_worker(
    command: str,
    prompt: str,
    project_dir: Path,
    run_dir: Path,
    timeout_seconds: int,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
    progress_path: Path,
    expected_run_id: Optional[str] = None,
) -> dict[str, Any]:
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    start_wall = datetime.now(timezone.utc)
    try:
        formatted_command = command.format(
            prompt_file=str(prompt_path),
            project_dir=str(project_dir),
            run_dir=str(run_dir),
            prompt=prompt,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in codex command: {exc}") from exc

    command_parts = shlex.split(formatted_command)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    start_time = time.monotonic()
    start_iso = _now_iso()
    timed_out = False
    no_heartbeat = False
    last_heartbeat = None

    process = subprocess.Popen(
        command_parts,
        cwd=project_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, stdout_path, "stdout", False),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, stderr_path, "stderr", True),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    if "{prompt_file}" not in command and "{prompt}" not in command:
        if process.stdin:
            process.stdin.write(prompt)
            process.stdin.flush()
            process.stdin.close()

    poll_interval = max(5, min(heartbeat_seconds // 2, 30))

    while True:
        if process.poll() is not None:
            break

        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break

        heartbeat = _heartbeat_from_progress(progress_path, expected_run_id)
        now = datetime.now(timezone.utc)
        if heartbeat and heartbeat >= start_wall:
            last_heartbeat = heartbeat
            age = (now - heartbeat).total_seconds()
        else:
            age = (now - start_wall).total_seconds()

        if age > heartbeat_grace_seconds:
            no_heartbeat = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break

        time.sleep(poll_interval)

    exit_code = process.poll()
    if exit_code is None:
        exit_code = -1

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    end_iso = _now_iso()
    runtime_seconds = int(time.monotonic() - start_time)

    return {
        "command": formatted_command,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "no_heartbeat": no_heartbeat,
        "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
    }


def _heartbeat_from_progress(
    progress_path: Path,
    expected_run_id: Optional[str] = None,
) -> Optional[datetime]:
    if not progress_path.exists():
        return None
    progress = _load_data(progress_path, {})
    if expected_run_id:
        run_id = progress.get("run_id")
        if run_id and run_id != expected_run_id:
            return None
    heartbeat = _parse_iso(progress.get("heartbeat")) or _parse_iso(progress.get("timestamp"))
    if heartbeat:
        return heartbeat
    try:
        mtime = progress_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        return None


def _run_command(command: str, project_dir: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as handle:
        result = subprocess.run(
            command,
            cwd=project_dir,
            shell=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {"command": command, "exit_code": result.returncode, "log_path": str(log_path)}


def _git_current_branch(project_dir: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_branch_exists(project_dir: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_branch(project_dir: Path, branch: str) -> None:
    current = _git_current_branch(project_dir)
    if current == branch:
        return
    if _git_branch_exists(project_dir, branch):
        subprocess.run(["git", "checkout", branch], cwd=project_dir, check=True)
    else:
        subprocess.run(["git", "checkout", "-b", branch], cwd=project_dir, check=True)


def _git_has_changes(project_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _path_is_ignored(path: str, ignore_patterns: Optional[list[str]] = None) -> bool:
    if not ignore_patterns:
        return False
    for pattern in ignore_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        if any(char in pattern for char in "*?["):
            if fnmatch.fnmatch(path, pattern):
                return True
            continue
        normalized = pattern.rstrip("/")
        if pattern.endswith("/"):
            prefix = f"{normalized}/"
            if path == normalized or path.startswith(prefix):
                return True
        else:
            if path == normalized:
                return True
    return False


def _git_changed_files(
    project_dir: Path,
    include_untracked: bool = True,
    ignore_prefixes: Optional[list[str]] = None,
) -> list[str]:
    changed: set[str] = set()
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--staged"],
    ]
    if include_untracked:
        commands.append(["git", "ls-files", "--others", "--exclude-standard"])
    ignore_prefixes = ignore_prefixes or []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                if _path_is_ignored(line, ignore_prefixes):
                    continue
                changed.add(line)
    return sorted(changed)


def _git_diff_text(project_dir: Path, max_chars: int = 20000) -> tuple[str, bool]:
    sections: list[str] = []
    commands = [
        ("UNSTAGED DIFF", ["git", "diff"]),
        ("STAGED DIFF", ["git", "diff", "--staged"]),
    ]
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        content = result.stdout.strip()
        if content:
            sections.append(f"{label}:\n{content}")
    diff_text = "\n\n".join(sections).strip()
    if not diff_text:
        return "", False
    if len(diff_text) <= max_chars:
        return diff_text, False
    return diff_text[:max_chars], True


def _git_diff_stat(project_dir: Path, max_chars: int = 4000) -> tuple[str, bool]:
    sections: list[str] = []
    commands = [
        ("UNSTAGED DIFFSTAT", ["git", "diff", "--stat"]),
        ("STAGED DIFFSTAT", ["git", "diff", "--stat", "--staged"]),
    ]
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        content = result.stdout.strip()
        if content:
            sections.append(f"{label}:\n{content}")
    diff_stat = "\n\n".join(sections).strip()
    if not diff_stat:
        return "", False
    if len(diff_stat) <= max_chars:
        return diff_stat, False
    return diff_stat[:max_chars], True


def _git_status_porcelain(project_dir: Path, max_chars: int = 2000) -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "", False
    content = result.stdout.strip()
    if not content:
        return "", False
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _git_is_ignored(project_dir: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _git_tracked_paths(project_dir: Path, path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", path],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_commit_and_push(project_dir: Path, branch: str, message: str) -> None:
    if _git_tracked_paths(project_dir, ".prd_runner"):
        raise RuntimeError(".prd_runner is tracked; remove it from git history before committing")
    if not _git_is_ignored(project_dir, ".prd_runner"):
        _ensure_gitignore(project_dir)
        if not _git_is_ignored(project_dir, ".prd_runner"):
            raise RuntimeError(".prd_runner is not ignored; add it to .gitignore before committing")
    subprocess.run(
        ["git", "add", "-A", "--", ".", ":(exclude).prd_runner", ":(exclude).prd_runner/**"],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", message], cwd=project_dir, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=project_dir, check=True)


def _has_implementation_evidence(task: dict[str, Any], runs_dir: Path, project_dir: Path) -> bool:
    tracked_changes = _git_changed_files(
        project_dir,
        include_untracked=False,
        ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
    )
    if tracked_changes:
        return True
    recorded_files = task.get("last_changed_files")
    if isinstance(recorded_files, list) and any(str(path).strip() for path in recorded_files):
        return True
    run_id = task.get("last_run_id")
    if not run_id:
        return False
    manifest_path = runs_dir / str(run_id) / "manifest.json"
    manifest = _load_data(manifest_path, {})
    manifest_files = manifest.get("changed_files")
    if isinstance(manifest_files, list) and any(str(path).strip() for path in manifest_files):
        return True
    return False


def _active_run_is_stale(
    run_state: dict[str, Any],
    runs_dir: Path,
    heartbeat_grace_seconds: int,
    shift_minutes: int,
) -> bool:
    if run_state.get("status") != "running":
        return False
    run_id = run_state.get("run_id")
    if not run_id:
        return True
    progress_path = runs_dir / str(run_id) / "progress.json"
    heartbeat = _heartbeat_from_progress(progress_path, expected_run_id=str(run_id))
    now = datetime.now(timezone.utc)
    if heartbeat:
        age = (now - heartbeat).total_seconds()
        return age > heartbeat_grace_seconds
    updated_at = _parse_iso(run_state.get("updated_at"))
    if updated_at:
        age = (now - updated_at).total_seconds()
        return age > max(heartbeat_grace_seconds, shift_minutes * 60)
    return True


def _read_progress_blocking_issues(
    progress_path: Path,
    expected_run_id: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    if not progress_path.exists():
        return [], []
    progress = _load_data(progress_path, {})
    if expected_run_id:
        run_id = progress.get("run_id")
        if run_id and str(run_id) != str(expected_run_id):
            return [], []
    issues = _coerce_string_list(progress.get("blocking_issues"))
    next_steps = _coerce_string_list(progress.get("next_steps"))
    return issues, next_steps


def _phase_for_task(phases: list[dict[str, Any]], task: dict[str, Any]) -> Optional[dict[str, Any]]:
    phase_id = task.get("phase_id") or task.get("id")
    for phase in phases:
        if phase.get("id") == phase_id:
            return phase
    return None


def _sync_phase_status(phase: dict[str, Any], task_status: str) -> None:
    if task_status in {TASK_STATUS_DOING, "in_progress"}:
        phase["status"] = TASK_STATUS_IMPLEMENTING
    else:
        phase["status"] = task_status


def _find_task(tasks: list[dict[str, Any]], task_id: str) -> Optional[dict[str, Any]]:
    for task in tasks:
        if str(task.get("id")) == task_id:
            return task
    return None


def _save_queue(path: Path, queue: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    queue["tasks"] = tasks
    queue["updated_at"] = _now_iso()
    _save_data(path, queue)


def _save_plan(path: Path, plan: dict[str, Any], phases: list[dict[str, Any]]) -> None:
    plan["phases"] = phases
    plan["updated_at"] = _now_iso()
    _save_data(path, plan)


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
) -> None:
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve()
    _ensure_gitignore(project_dir, only_if_clean=True)
    paths = _ensure_state_files(project_dir, prd_path)

    lock_path = paths["state_dir"] / LOCK_FILE
    iteration = 0

    print("\n" + "=" * 70)
    print("  FEATURE PRD RUNNER (Codex CLI)")
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
                        "last_error": "Previous run marked stale; resuming",
                        "updated_at": _now_iso(),
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

            blocked_tasks_snapshot = None
            if stop_on_blocking_issues:
                blocked_tasks = _blocking_tasks(tasks)
                if blocked_tasks:
                    blocked_tasks_snapshot = [dict(task) for task in blocked_tasks]
                    run_state.update(
                        {
                            "status": "blocked",
                            "current_task_id": None,
                            "current_phase_id": None,
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
                            "updated_at": _now_iso(),
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
                    steps = plan_data.get("steps") if isinstance(plan_data, dict) else []
                    step_count = len(steps) if isinstance(steps, list) else 0
                    step_index_raw = int(next_task.get("next_plan_step_index", 0))
                    review_fix_mode = step_index_raw < 0
                    step_index = max(0, step_index_raw)
                    if task_status == TASK_STATUS_PLAN_IMPL and plan_valid:
                        next_task["status"] = TASK_STATUS_IMPLEMENTING
                        next_task["impl_plan_path"] = str(plan_path)
                        plan_hash = _hash_json_data(plan_data)
                        existing_hash = next_task.get("impl_plan_hash")
                        next_task["impl_plan_hash"] = plan_hash
                        if not existing_hash or existing_hash != plan_hash:
                            next_task["next_plan_step_index"] = 0
                        next_task["no_progress_attempts"] = 0
                        task_status = next_task["status"]
                    if task_status in {
                        TASK_STATUS_IMPLEMENTING,
                        TASK_STATUS_TESTING,
                        TASK_STATUS_REVIEW,
                    }:
                        if not plan_valid:
                            next_task["status"] = TASK_STATUS_PLAN_IMPL
                            next_task["last_error"] = (
                                f"Implementation plan invalid: {plan_issue}"
                            )
                            next_task["last_error_type"] = "impl_plan_invalid"
                            task_status = next_task["status"]
                        elif not review_fix_mode and step_count and step_index >= step_count:
                            next_task["status"] = (
                                TASK_STATUS_TESTING
                                if phase_test_command
                                else TASK_STATUS_REVIEW
                            )
                            next_task["next_plan_step_index"] = step_count
                            task_status = next_task["status"]
                        elif not _has_implementation_evidence(
                            next_task, paths["runs"], project_dir
                        ):
                            next_task["status"] = TASK_STATUS_IMPLEMENTING
                            next_task["last_error"] = (
                                "No implementation evidence; rerunning implementation"
                            )
                            next_task["last_error_type"] = "missing_implementation"
                            task_status = next_task["status"]
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
                        "updated_at": _now_iso(),
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
            return

        iteration += 1
        run_dir = paths["runs"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_path = run_dir / "progress.json"

        _update_progress(
            progress_path,
            {
                "run_id": run_id,
                "task_id": task_id,
                "phase": task_type,
                "blocking_issues": [],
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
                        target["last_error_type"] = "phase_missing"
                        _save_queue(paths["task_queue"], queue, tasks)
                continue

            branch = phase.get("branch") or f"feature/{phase_id or task_id}"
            current_branch = _git_current_branch(project_dir)
            if _git_has_changes(project_dir) and current_branch and current_branch != branch:
                print(
                    f"\nDirty workspace on branch {current_branch}; expected {branch}. Blocking task."
                )
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        target["status"] = TASK_STATUS_BLOCKED
                        target["last_error"] = (
                            f"Dirty workspace on branch {current_branch}; expected {branch}"
                        )
                        target["last_error_type"] = "dirty_branch_mismatch"
                        _save_queue(paths["task_queue"], queue, tasks)
                continue
            try:
                _ensure_branch(project_dir, branch)
                _append_event(
                    paths["events"],
                    {
                        "event_type": "branch_checkout",
                        "phase_id": phase.get("id"),
                        "branch": branch,
                    },
                )
                with FileLock(lock_path):
                    run_state = _load_data(paths["run_state"], {})
                    run_state["branch"] = branch
                    run_state["updated_at"] = _now_iso()
                    _save_data(paths["run_state"], run_state)
            except subprocess.CalledProcessError as exc:
                print(f"\nFailed to checkout branch {branch}: {exc}")
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        target["status"] = TASK_STATUS_BLOCKED
                        target["last_error"] = f"Branch checkout failed: {exc}"
                        target["last_error_type"] = "branch_checkout_failed"
                        _save_queue(paths["task_queue"], queue, tasks)
                continue

        planning_impl = task_type != "plan" and task_status == TASK_STATUS_PLAN_IMPL
        run_codex = task_type == "plan" or task_status in TASK_RUN_CODEX_STATUSES
        selected_steps: list[dict[str, Any]] = []
        plan_steps_total = 0
        plan_step_index = 0
        review_fix_mode = False
        advance_to_tests = True
        allowed_files: list[str] = []
        allowed_files_set: Optional[set[str]] = None
        updated_status: Optional[str] = None

        if run_codex:
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
                    )
                else:
                    if not phase:
                        raise ValueError("Phase not found for task")
                    plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                    plan_data = _load_data(plan_path, {})
                    steps_all = plan_data.get("steps") if isinstance(plan_data, dict) else []
                    if not isinstance(steps_all, list):
                        steps_all = []
                    plan_steps_total = len(steps_all)
                    plan_step_index_raw = int(next_task.get("next_plan_step_index", 0))
                    review_fix_mode = plan_step_index_raw < 0
                    plan_step_index = max(0, plan_step_index_raw)
                    plan_steps_total_for_prompt = plan_steps_total
                    plan_step_index_for_prompt = plan_step_index
                    if review_fix_mode:
                        selected_steps = _build_review_fix_steps(
                            next_task.get("review_blockers"),
                            next_task.get("review_blocker_files"),
                        )
                        plan_steps_total_for_prompt = max(1, len(selected_steps))
                        plan_step_index_for_prompt = 0
                    elif plan_steps_total and plan_step_index < plan_steps_total:
                        selected_steps = steps_all[
                            plan_step_index : plan_step_index + MAX_PLAN_STEPS_PER_RUN
                        ]
                    plan_steps_text = _format_plan_steps_for_prompt(
                        selected_steps,
                        plan_step_index_for_prompt,
                        plan_steps_total_for_prompt,
                    )
                    allowed_files = _allowed_files_for_steps(selected_steps, plan_data, plan_path)
                    allowed_files_set = {path for path in allowed_files if path}
                    if not allowed_files_set:
                        allowed_files_set = None
                    prompt = _build_phase_prompt(
                        prd_path=prd_path,
                        phase=phase,
                        task=next_task,
                        events_path=paths["events"],
                        progress_path=progress_path,
                        run_id=run_id,
                        user_prompt=user_prompt,
                        impl_plan_path=plan_path,
                        plan_steps_text=plan_steps_text,
                        allowed_files=allowed_files,
                        no_progress_attempts=int(next_task.get("no_progress_attempts", 0)),
                    )

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

            progress_blocking, progress_next_steps = _read_progress_blocking_issues(
                progress_path,
                expected_run_id=run_id,
            )
            progress_blocking_detected = bool(progress_blocking)

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

            next_status = None
            if task_type != "plan":
                phase_test_command = (
                    phase.get("test_command")
                    or next_task.get("test_command")
                    or test_command
                )
                next_status = TASK_STATUS_TESTING if phase_test_command else TASK_STATUS_REVIEW

            changed_files_after_run = None
            progress_files = None
            if task_type != "plan" and not planning_impl and not failure:
                changed_files_after_run = _git_changed_files(
                    project_dir,
                    include_untracked=False,
                    ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
                )
                progress_files = changed_files_after_run
                if allowed_files_set and changed_files_after_run:
                    progress_files = sorted(
                        set(changed_files_after_run).intersection(allowed_files_set)
                    )

            manifest = {
                "run_id": run_id,
                "task_id": task_id,
                "start_time": run_result["start_time"],
                "end_time": run_result["end_time"],
                "exit_code": run_result["exit_code"],
                "timed_out": run_result["timed_out"],
                "no_heartbeat": run_result["no_heartbeat"],
                "runtime_seconds": run_result["runtime_seconds"],
                "command": run_result["command"],
                "prompt_hash": prompt_hash,
                "prompt_path": run_result["prompt_path"],
                "stdout_path": run_result["stdout_path"],
                "stderr_path": run_result["stderr_path"],
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            if changed_files_after_run is not None:
                manifest["changed_files"] = changed_files_after_run
            _save_data(run_dir / "manifest.json", manifest)

            plan_valid = None
            plan_issue = None
            plan_hash = None
            plan_path = None
            if planning_impl and not failure:
                plan_path = _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
                plan_data = _load_data(plan_path, {})
                prd_text, prd_truncated = _read_text_for_prompt(prd_path)
                prd_markers = _extract_prd_markers(prd_text)
                plan_valid, plan_issue = _validate_impl_plan_data(
                    plan_data,
                    phase or {"id": phase_id or task_id, "acceptance_criteria": []},
                    prd_markers=prd_markers,
                    prd_truncated=prd_truncated,
                    prd_has_content=bool(prd_text.strip()),
                    expected_test_command=phase_test_command,
                )
                if plan_valid:
                    plan_hash = _hash_json_data(plan_data)

            blocked_task_snapshot = None
            with FileLock(lock_path):
                run_state = _load_data(paths["run_state"], {})
                queue = _load_data(paths["task_queue"], {})
                plan = _load_data(paths["phase_plan"], {})
                tasks = _normalize_tasks(queue)
                phases = _normalize_phases(plan)
                target = _find_task(tasks, task_id)
                if target:
                    target["last_run_id"] = run_id
                    target["last_updated_at"] = _now_iso()
                    if progress_blocking_detected:
                        issue_summary = "; ".join(progress_blocking).strip()
                        if issue_summary:
                            issue_summary = issue_summary[:400]
                            target["last_error"] = (
                                f"Blocking issues reported: {issue_summary}"
                            )
                        else:
                            target["last_error"] = "Blocking issues reported by worker"
                        target["last_error_type"] = ERROR_TYPE_BLOCKING_ISSUES
                        target["blocking_issues"] = progress_blocking
                        target["blocking_next_steps"] = progress_next_steps
                        target["status"] = TASK_STATUS_BLOCKED
                        blocked_task_snapshot = dict(target)
                    elif failure:
                        target["attempts"] = int(target.get("attempts", 0)) + 1
                        target["last_error"] = error_detail or "Run failed"
                        target["last_error_type"] = error_type
                        if target["attempts"] >= max_attempts:
                            target["status"] = TASK_STATUS_BLOCKED
                        else:
                            target["status"] = (
                                TASK_STATUS_PLAN_IMPL if planning_impl else TASK_STATUS_TODO
                            )
                    else:
                        target["last_error"] = None
                        target["last_error_type"] = None
                        if task_type == "plan":
                            target["status"] = TASK_STATUS_DONE
                        elif planning_impl:
                            if plan_valid:
                                target["status"] = TASK_STATUS_IMPLEMENTING
                                target["impl_plan_path"] = str(plan_path)
                                existing_hash = target.get("impl_plan_hash")
                                target["impl_plan_hash"] = plan_hash
                                target["impl_plan_run_id"] = run_id
                                target["plan_attempts"] = 0
                                if not existing_hash or existing_hash != plan_hash:
                                    target["next_plan_step_index"] = 0
                                target["no_progress_attempts"] = 0
                            else:
                                target["status"] = TASK_STATUS_PLAN_IMPL
                                target["last_error"] = (
                                    f"Implementation plan invalid: {plan_issue}"
                                )
                                target["last_error_type"] = "impl_plan_invalid"
                                attempts = _increment_task_counter(target, "plan_attempts")
                                if attempts >= MAX_IMPL_PLAN_ATTEMPTS:
                                    target["status"] = TASK_STATUS_BLOCKED
                                    target["last_error"] = (
                                        f"Plan invalid after {attempts} attempts; blocking task."
                                    )
                                    target["last_error_type"] = "impl_plan_attempts_exhausted"
                        else:
                            if changed_files_after_run is None:
                                target["status"] = next_status or TASK_STATUS_REVIEW
                            elif not progress_files:
                                attempts = _increment_task_counter(target, "no_progress_attempts")
                                target["status"] = TASK_STATUS_IMPLEMENTING
                                target["last_error"] = "No changes detected in allowed files"
                                target["last_error_type"] = "no_progress"
                                context_note = "No code edits detected in allowed files; complete the assigned plan step."
                                if changed_files_after_run:
                                    context_note = (
                                        "Changes were outside allowed files; edit only the files for this step."
                                    )
                                target["context"] = target.get("context", []) + [context_note]
                                if attempts >= MAX_NO_PROGRESS_ATTEMPTS:
                                    target["status"] = TASK_STATUS_BLOCKED
                                    target["last_error"] = (
                                        f"No progress after {attempts} attempts; blocking task."
                                    )
                                    target["last_error_type"] = "no_progress_exhausted"
                            else:
                                target["last_changed_files"] = progress_files
                                target["last_error"] = None
                                target["last_error_type"] = None
                                target["no_progress_attempts"] = 0
                                if review_fix_mode:
                                    target["review_blockers"] = []
                                    target["review_blocker_files"] = []
                                    resume_index = target.get("resume_plan_step_index")
                                    if resume_index is None:
                                        resume_index = target.get("next_plan_step_index", 0)
                                    try:
                                        resume_index = int(resume_index)
                                    except (TypeError, ValueError):
                                        resume_index = 0
                                    resume_index = max(0, resume_index)
                                    if plan_steps_total:
                                        resume_index = min(resume_index, plan_steps_total)
                                    target["next_plan_step_index"] = resume_index
                                    target["resume_plan_step_index"] = None
                                    target["status"] = next_status or TASK_STATUS_REVIEW
                                else:
                                    next_index = plan_step_index + len(selected_steps)
                                    if plan_steps_total:
                                        next_index = min(next_index, plan_steps_total)
                                        target["next_plan_step_index"] = next_index
                                        if next_index >= plan_steps_total:
                                            target["status"] = next_status or TASK_STATUS_REVIEW
                                        else:
                                            target["status"] = TASK_STATUS_IMPLEMENTING
                                    else:
                                        target["next_plan_step_index"] = next_index
                                        target["status"] = next_status or TASK_STATUS_REVIEW
                    if task_type != "plan":
                        phase_entry = _phase_for_task(
                            phases, {"phase_id": phase_id or task_id, "id": task_id}
                        )
                        if phase_entry:
                            _sync_phase_status(phase_entry, target["status"])
                    updated_status = target.get("status") if target else None
                elif task_type != "plan":
                    raise ValueError(f"Task {task_id} not found in queue")

                run_state_status = (
                    "blocked" if progress_blocking_detected and stop_on_blocking_issues else "idle"
                )
                run_state.update(
                    {
                        "status": run_state_status,
                        "current_task_id": None,
                        "current_phase_id": phase_id,
                        "last_error": error_detail or (target.get("last_error") if target else None),
                        "last_heartbeat": run_result.get("last_heartbeat"),
                        "updated_at": _now_iso(),
                    }
                )

                _save_data(paths["run_state"], run_state)
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)

            if progress_blocking_detected:
                if not blocked_task_snapshot:
                    blocked_task_snapshot = {
                        "id": task_id,
                        "phase_id": phase_id,
                        "status": TASK_STATUS_BLOCKED,
                        "last_error": "Blocking issues reported by worker",
                        "last_error_type": ERROR_TYPE_BLOCKING_ISSUES,
                        "blocking_issues": progress_blocking,
                        "blocking_next_steps": progress_next_steps,
                    }
                _append_event(
                    paths["events"], _blocking_event_payload([blocked_task_snapshot])
                )
                _report_blocking_tasks(
                    [blocked_task_snapshot],
                    paths,
                    stopping=stop_on_blocking_issues,
                )
                if stop_on_blocking_issues:
                    return
                continue

            if failure:
                print(f"\nRun {run_id} failed: {error_detail}")
                print(f"See logs: {run_dir / 'stderr.log'}")
                continue

            if planning_impl:
                if plan_valid:
                    print(f"\nRun {run_id} complete. Implementation plan created.")
                else:
                    print(
                        f"\nRun {run_id} complete, but implementation plan invalid: {plan_issue}"
                    )
                continue

            if task_type == "plan":
                plan = _load_data(paths["phase_plan"], {})
                phases = _normalize_phases(plan)
                queue = _load_data(paths["task_queue"], {})
                tasks = _normalize_tasks(queue)
                tasks_empty = not tasks or (len(tasks) == 1 and tasks[0].get("id") == "plan-001")
                tasks_match = _tasks_match_phases(tasks, phases)
                if phases and (tasks_empty or not tasks_match):
                    plan_task = _find_task(tasks, "plan-001") or _build_plan_task()
                    plan_task["status"] = TASK_STATUS_DONE
                    tasks = [plan_task] + _build_tasks_from_phases(phases)
                    queue["tasks"] = tasks
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)
                    print(f"\nRun {run_id} complete. Phase plan created.")
                    continue

                if not phases or not tasks:
                    if not tasks:
                        tasks = [_build_plan_task()]
                    plan_task = _find_task(tasks, "plan-001")
                    if plan_task:
                        plan_task["status"] = TASK_STATUS_BLOCKED
                        plan_task["last_error"] = "Phase plan not generated"
                        plan_task["last_error_type"] = ERROR_TYPE_PLAN_MISSING
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)
                    print(f"\nRun {run_id} complete, but phase plan was not generated.")
                else:
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)
                    print(f"\nRun {run_id} complete. Phase plan created.")
                continue
            if updated_status:
                advance_to_tests = updated_status in {
                    TASK_STATUS_TESTING,
                    TASK_STATUS_REVIEW,
                    TASK_STATUS_DONE,
                }
            if task_type != "plan" and not planning_impl and not advance_to_tests and not review_fix_mode:
                if plan_steps_total:
                    print(
                        f"\nCompleted plan step {plan_step_index + 1}/{plan_steps_total}. "
                        "Continuing to next step."
                    )
                continue
        else:
            with FileLock(lock_path):
                run_state = _load_data(paths["run_state"], {})
                run_state.update(
                    {
                        "status": "idle",
                        "current_task_id": None,
                        "current_phase_id": phase_id,
                        "last_error": None,
                        "last_heartbeat": None,
                        "updated_at": _now_iso(),
                    }
                )
                _save_data(paths["run_state"], run_state)

        plan = _load_data(paths["phase_plan"], {})
        phases = _normalize_phases(plan)
        phase = _phase_for_task(phases, {"phase_id": phase_id or task_id, "id": task_id})
        if not phase:
            print(f"\nRun {run_id} complete, but phase not found.")
            continue
        branch = phase.get("branch") or f"feature/{phase.get('id')}"

        queue = _load_data(paths["task_queue"], {})
        tasks = _normalize_tasks(queue)
        target = _find_task(tasks, task_id)
        if not target:
            print(f"\nTask {task_id} missing after run; skipping phase work.")
            continue

        plan_path = Path(
            target.get("impl_plan_path") or _impl_plan_path(paths["artifacts"], str(phase_id or task_id))
        )
        prd_text, prd_truncated = _read_text_for_prompt(prd_path)
        prd_markers = _extract_prd_markers(prd_text)
        plan_data = _load_data(plan_path, {})
        phase_test_command = phase.get("test_command") or target.get("test_command") or test_command
        plan_valid, plan_issue = _validate_impl_plan_data(
            plan_data,
            phase,
            prd_markers=prd_markers,
            prd_truncated=prd_truncated,
            prd_has_content=bool(prd_text.strip()),
            expected_test_command=phase_test_command,
        )
        if not plan_valid:
            target["status"] = TASK_STATUS_PLAN_IMPL
            target["last_error"] = f"Implementation plan invalid: {plan_issue}"
            target["last_error_type"] = "impl_plan_invalid"
            attempts = _increment_task_counter(target, "plan_attempts")
            target["context"] = target.get("context", []) + [
                f"Implementation plan invalid: {plan_issue}"
            ]
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)
            if attempts >= MAX_IMPL_PLAN_ATTEMPTS:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = (
                    f"Plan invalid after {attempts} attempts; blocking task."
                )
                target["last_error_type"] = "impl_plan_attempts_exhausted"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(
                    f"\nImplementation plan invalid for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                )
                continue
            print(f"\nImplementation plan invalid for phase {phase.get('id')}. Re-queueing task.")
            continue

        plan_hash_current = _hash_json_data(plan_data)
        target["impl_plan_path"] = str(plan_path)
        if target.get("impl_plan_hash") and target["impl_plan_hash"] != plan_hash_current:
            if not _plan_deviations_present(plan_data):
                target["status"] = TASK_STATUS_IMPLEMENTING
                target["last_error"] = "Plan changed without plan_deviations"
                target["last_error_type"] = "plan_deviation_missing"
                attempts = _increment_task_counter(target, "plan_attempts")
                target["context"] = target.get("context", []) + [
                    "Plan changed; add plan_deviations explaining why."
                ]
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                if attempts >= MAX_IMPL_PLAN_ATTEMPTS:
                    target["status"] = TASK_STATUS_BLOCKED
                    target["last_error"] = (
                        f"Plan deviation missing after {attempts} attempts; blocking task."
                    )
                    target["last_error_type"] = "impl_plan_attempts_exhausted"
                    _sync_phase_status(phase, target["status"])
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)
                    print(
                        f"\nPlan deviations missing for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                    )
                    continue
                print(
                    f"\nPlan changed without deviations for phase {phase.get('id')}. Re-queueing implementation."
                )
                continue
        target["impl_plan_hash"] = plan_hash_current
        target["plan_attempts"] = 0

        test_log = None
        if phase_test_command:
            test_log_path = paths["artifacts"] / f"tests_{phase.get('id')}.log"
            test_result = _run_command(phase_test_command, project_dir, test_log_path)
            test_log = test_result["log_path"]
            if test_result["exit_code"] != 0:
                target["status"] = TASK_STATUS_IMPLEMENTING
                target["last_error"] = f"Tests failed. See {test_log}"
                target["last_error_type"] = "tests_failed"
                if target.get("resume_plan_step_index") is None:
                    target["resume_plan_step_index"] = target.get(
                        "next_plan_step_index", 0
                    )
                target["next_plan_step_index"] = -1
                target["review_blockers"] = [f"Tests failed. See log: {test_log}"]
                target["review_blocker_files"] = target.get(
                    "last_changed_files", []
                )
                _sync_phase_status(phase, target["status"])
                _append_event(
                    paths["events"],
                    {
                        "event_type": "tests_failed",
                        "phase_id": phase.get("id"),
                        "log_path": test_log,
                    },
                )
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(f"\nTests failed for phase {phase.get('id')}. Re-queueing task.")
                continue
            if target.get("status") != TASK_STATUS_REVIEW:
                target["status"] = TASK_STATUS_REVIEW
                target["last_error"] = None
                target["last_error_type"] = None
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
        elif target.get("status") == TASK_STATUS_TESTING:
            target["status"] = TASK_STATUS_REVIEW
            target["last_error"] = None
            target["last_error_type"] = None
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)

        prd_text, prd_truncated = _read_text_for_prompt(prd_path)
        prd_markers = _extract_prd_markers(prd_text)
        plan_text, plan_truncated = _render_json_for_prompt(plan_data)
        plan_step_ids = [
            str(step.get("id")).strip()
            for step in plan_data.get("steps", [])
            if isinstance(step, dict) and str(step.get("id")).strip()
        ]
        changed_files = _git_changed_files(
            project_dir,
            include_untracked=True,
            ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
        )
        if not changed_files:
            target["status"] = TASK_STATUS_IMPLEMENTING
            target["last_error"] = "No changed files to review"
            target["last_error_type"] = "review_no_changes"
            attempts = _increment_task_counter(target, "no_change_attempts")
            target["context"] = target.get("context", []) + [
                "No changed files detected; implement changes before review."
            ]
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)
            if attempts >= MAX_NO_CHANGE_ATTEMPTS:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = (
                    f"No changes detected after {attempts} attempts; blocking task."
                )
                target["last_error_type"] = "no_changes_exhausted"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(
                    f"\nNo changes for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                )
                continue
            print(f"\nNo changed files for phase {phase.get('id')}. Re-queueing task.")
            continue
        target["no_change_attempts"] = 0

        diff_text, diff_truncated = _git_diff_text(project_dir)
        diff_stat, diff_stat_truncated = _git_diff_stat(project_dir)
        status_text, status_truncated = _git_status_porcelain(project_dir)

        review_path = paths["artifacts"] / f"review_{phase.get('id')}.json"
        review_run_id = f"{run_id}-review"
        review_run_dir = paths["runs"] / review_run_id
        review_run_dir.mkdir(parents=True, exist_ok=True)
        review_progress_path = review_run_dir / "progress.json"
        _update_progress(
            review_progress_path,
            {
                "run_id": review_run_id,
                "task_id": task_id,
                "phase": "review",
                "blocking_issues": [],
            },
        )
        review_prompt = _build_review_prompt(
            phase=phase,
            review_path=review_path,
            prd_path=prd_path,
            prd_text=prd_text,
            prd_truncated=prd_truncated,
            prd_markers=prd_markers,
            user_prompt=user_prompt,
            progress_path=review_progress_path,
            run_id=review_run_id,
            changed_files=changed_files,
            diff_text=diff_text,
            diff_truncated=diff_truncated,
            diff_stat=diff_stat,
            diff_stat_truncated=diff_stat_truncated,
            status_text=status_text,
            status_truncated=status_truncated,
            impl_plan_text=plan_text,
            impl_plan_truncated=plan_truncated,
            plan_step_ids=plan_step_ids,
        )

        review_result = _run_codex_worker(
            command=codex_command,
            prompt=review_prompt,
            project_dir=project_dir,
            run_dir=review_run_dir,
            timeout_seconds=shift_minutes * 60,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_grace_seconds=heartbeat_grace_seconds,
            progress_path=review_progress_path,
            expected_run_id=review_run_id,
        )
        if user_prompt:
            user_prompt = None
        review_stdout = _read_log_tail(Path(review_result["stdout_path"]))
        review_stderr = _read_log_tail(Path(review_result["stderr_path"]))

        review_blocking, review_next_steps = _read_progress_blocking_issues(
            review_progress_path,
            expected_run_id=review_run_id,
        )
        if review_blocking:
            blocked_task_snapshot = None
            with FileLock(lock_path):
                queue = _load_data(paths["task_queue"], {})
                tasks = _normalize_tasks(queue)
                target = _find_task(tasks, task_id)
                if target:
                    issue_summary = "; ".join(review_blocking).strip()
                    if issue_summary:
                        issue_summary = issue_summary[:400]
                        target["last_error"] = (
                            f"Blocking issues reported: {issue_summary}"
                        )
                    else:
                        target["last_error"] = "Blocking issues reported by worker"
                    target["last_error_type"] = ERROR_TYPE_BLOCKING_ISSUES
                    target["blocking_issues"] = review_blocking
                    target["blocking_next_steps"] = review_next_steps
                    target["status"] = TASK_STATUS_BLOCKED
                    target["last_updated_at"] = _now_iso()
                    blocked_task_snapshot = dict(target)
                    _sync_phase_status(phase, target["status"])
                    _save_queue(paths["task_queue"], queue, tasks)
                    _save_plan(paths["phase_plan"], plan, phases)
                run_state = _load_data(paths["run_state"], {})
                run_state_status = "blocked" if stop_on_blocking_issues else "idle"
                run_state.update(
                    {
                        "status": run_state_status,
                        "current_task_id": None,
                        "current_phase_id": phase.get("id"),
                        "last_error": _summarize_blocking_tasks(
                            [blocked_task_snapshot] if blocked_task_snapshot else []
                        ),
                        "updated_at": _now_iso(),
                    }
                )
                _save_data(paths["run_state"], run_state)

            if not blocked_task_snapshot:
                blocked_task_snapshot = {
                    "id": task_id,
                    "phase_id": phase.get("id"),
                    "status": TASK_STATUS_BLOCKED,
                    "last_error": "Blocking issues reported by worker",
                    "last_error_type": ERROR_TYPE_BLOCKING_ISSUES,
                    "blocking_issues": review_blocking,
                    "blocking_next_steps": review_next_steps,
                }
            _append_event(
                paths["events"], _blocking_event_payload([blocked_task_snapshot])
            )
            _report_blocking_tasks(
                [blocked_task_snapshot],
                paths,
                stopping=stop_on_blocking_issues,
            )
            if stop_on_blocking_issues:
                return
            continue

        if review_result["exit_code"] != 0:
            target["status"] = TASK_STATUS_REVIEW
            target["last_error"] = f"Review failed: {review_stderr.strip() or review_stdout.strip()}"
            target["last_error_type"] = "review_failed"
            attempts = _increment_task_counter(target, "review_attempts")
            target["context"] = target.get("context", []) + [
                "Review failed; rerun review and fix issues."
            ]
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)
            if attempts >= MAX_REVIEW_ATTEMPTS:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = (
                    f"Review failed after {attempts} attempts; blocking task."
                )
                target["last_error_type"] = "review_attempts_exhausted"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(
                    f"\nReview failed for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                )
                continue
            print(f"\nReview step failed for phase {phase.get('id')}. Re-queueing task.")
            continue

        review_data = _load_data(review_path, {})
        valid_review, review_issue = _validate_review_data(
            review_data,
            phase,
            changed_files,
            prd_markers=prd_markers,
            prd_truncated=prd_truncated,
            prd_has_content=bool(prd_text.strip()),
            plan_step_ids=plan_step_ids,
        )
        if not valid_review:
            target["status"] = TASK_STATUS_REVIEW
            target["last_error"] = f"Review output invalid: {review_issue}"
            target["last_error_type"] = "review_invalid"
            attempts = _increment_task_counter(target, "review_attempts")
            target["context"] = target.get("context", []) + [
                f"Review output invalid: {review_issue}"
            ]
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)
            if attempts >= MAX_REVIEW_ATTEMPTS:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = (
                    f"Review invalid after {attempts} attempts; blocking task."
                )
                target["last_error_type"] = "review_attempts_exhausted"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(
                    f"\nReview invalid for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                )
                continue
            print(f"\nReview output missing or invalid for phase {phase.get('id')}. Re-queueing task.")
            continue

        blocking = review_data.get("blocking_issues") or []
        if blocking:
            target["status"] = TASK_STATUS_IMPLEMENTING
            target["last_error"] = "Review blockers found"
            target["last_error_type"] = "review_blockers"
            target["review_blockers"] = blocking
            target["review_blocker_files"] = _extract_review_blocker_files(review_data)
            if target.get("resume_plan_step_index") is None:
                target["resume_plan_step_index"] = target.get("next_plan_step_index", 0)
            target["next_plan_step_index"] = -1
            attempts = _increment_task_counter(target, "review_attempts")
            target["context"] = target.get("context", []) + [
                f"Review blockers: {blocking}"
            ]
            _append_event(
                paths["events"],
                {
                    "event_type": "review_blockers",
                    "phase_id": phase.get("id"),
                    "blocking": blocking,
                },
            )
            _sync_phase_status(phase, target["status"])
            _save_queue(paths["task_queue"], queue, tasks)
            _save_plan(paths["phase_plan"], plan, phases)
            if attempts >= MAX_REVIEW_ATTEMPTS:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = (
                    f"Review blockers persisted after {attempts} attempts; blocking task."
                )
                target["last_error_type"] = "review_attempts_exhausted"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(
                    f"\nReview blockers for phase {phase.get('id')} after {attempts} attempts. Blocking task."
                )
                continue
            print(f"\nReview blockers found for phase {phase.get('id')}. Re-queueing task.")
            continue

        post_review_files = _git_changed_files(
            project_dir,
            include_untracked=False,
            ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
        )
        reviewed_files = review_data.get("changed_files") or []
        if set(post_review_files) != set(reviewed_files):
            print(
                "Warning: File state changed during review, but proceeding because no blockers found."
            )

        target["status"] = TASK_STATUS_DONE
        target["last_error"] = None
        target["last_error_type"] = None
        target["review_attempts"] = 0
        target["review_blockers"] = []
        target["review_blocker_files"] = []
        target["resume_plan_step_index"] = None
        _sync_phase_status(phase, target["status"])
        _save_plan(paths["phase_plan"], plan, phases)
        _save_queue(paths["task_queue"], queue, tasks)

        if _git_has_changes(project_dir):
            commit_message = f"{phase.get('id')}: {phase.get('name') or 'phase'}"
            try:
                _git_commit_and_push(project_dir, branch, commit_message)
                _append_event(
                    paths["events"],
                    {
                        "event_type": "phase_committed",
                        "phase_id": phase.get("id"),
                        "branch": branch,
                        "commit_message": commit_message,
                    },
                )
            except subprocess.CalledProcessError as exc:
                target["status"] = TASK_STATUS_BLOCKED
                target["last_error"] = f"Git push failed: {exc}"
                target["last_error_type"] = "git_push_failed"
                _sync_phase_status(phase, target["status"])
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(f"\nPush failed for phase {phase.get('id')}: {exc}")
                continue
        else:
            _append_event(
                paths["events"],
                {
                    "event_type": "phase_no_changes",
                    "phase_id": phase.get("id"),
                },
            )

        summary = _task_summary(tasks)
        print(
            f"\nPhase {phase.get('id')} complete. "
            f"Queue: {summary['todo']} todo, {summary['doing']} doing, "
            f"{summary['done']} done, {summary['blocked']} blocked"
        )

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
    )


if __name__ == "__main__":
    main()
