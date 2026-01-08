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
from typing import Any, Optional, Callable

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
DEFAULT_MAX_ATTEMPTS = 5  # Increased default since we loop on tests
DEFAULT_MAX_AUTO_RESUMES = 3
DEFAULT_STOP_ON_BLOCKING_ISSUES = True
WINDOWS_LOCK_BYTES = 4096

TRANSIENT_ERROR_MARKERS = (
    "No heartbeat",
    "Shift timed out",
)

TASK_STATUS_TODO = "todo"
TASK_STATUS_DOING = "doing"
TASK_STATUS_PLAN_IMPL = "plan_impl"
TASK_STATUS_IMPLEMENTING = "implementing"
TASK_STATUS_TESTING = "testing"  # Kept for compatibility, mapped to implementing logic
TASK_STATUS_REVIEW = "review"
TASK_STATUS_DONE = "done"
TASK_STATUS_BLOCKED = "blocked"

TASK_IN_PROGRESS_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
    TASK_STATUS_REVIEW,
}

TASK_RUN_CODEX_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
    TASK_STATUS_REVIEW, 
}

ERROR_TYPE_HEARTBEAT_TIMEOUT = "heartbeat_timeout"
ERROR_TYPE_SHIFT_TIMEOUT = "shift_timeout"
ERROR_TYPE_CODEX_EXIT = "codex_exit"
ERROR_TYPE_PLAN_MISSING = "plan_missing"
ERROR_TYPE_BLOCKING_ISSUES = "blocking_issues"
ERROR_TYPE_DISALLOWED_FILES = "disallowed_files"
AUTO_RESUME_ERROR_TYPES = {
    ERROR_TYPE_HEARTBEAT_TIMEOUT,
    ERROR_TYPE_SHIFT_TIMEOUT,
}

# Resolution steps help the user when the runner stops
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
    "review_attempts_exhausted": [
        "Open the review JSON and address all blocking issues.",
        "Re-run the runner once fixes are in place.",
    ],
    ERROR_TYPE_DISALLOWED_FILES: [
        "Revert or move the out-of-scope changes.",
        "Update the implementation plan to include needed files before re-running.",
    ],
    "git_push_failed": [
        "Check git remote/authentication and resolve conflicts.",
        "Push the branch manually, then re-run the runner.",
    ],
}

REVIEW_MIN_EVIDENCE_ITEMS = 2
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
MAX_NO_PROGRESS_ATTEMPTS = 3  # Allowed "no-op" runs before blocking
MAX_MANUAL_RESUME_ATTEMPTS = 10

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
        if not isinstance(value, str):
            value = str(value)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        # If a naive timestamp slips in, assume UTC to avoid crashes.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None



def _require_yaml() -> None:
    if not yaml:
        raise RuntimeError("PyYAML is required to read/write .yaml files. Install pyyaml.")


class FileLock:
    """Best-effort cross-platform file lock."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.handle: Optional[Any] = None
        self.lock_bytes = WINDOWS_LOCK_BYTES

    def __enter__(self) -> "FileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.lock_path, "w")
        try:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except ImportError:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                self.handle.truncate(self.lock_bytes)
                self.handle.flush()
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, self.lock_bytes)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.handle:
            return
        try:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        except ImportError:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, self.lock_bytes)
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
                _require_yaml()
                data = yaml.safe_load(handle)
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
    _require_yaml()
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


def _save_data(path: Path, data: dict[str, Any]) -> None:
    if path.suffix in {".yaml", ".yml"}:
        _atomic_write_yaml(path, data)
    else:
        _atomic_write_json(path, data)


def _append_event(events_path: Path, event: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", _now_iso())
    line = json.dumps(payload) + "\n"
    with open(events_path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


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
        line.strip().rstrip("/")
        for line in contents.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    normalized_entry = ignore_entry.strip().rstrip("/")
    return normalized_entry in lines or STATE_DIR_NAME in lines


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
    if _ignore_file_has_entry(gitignore_path, ignore_entry):
        return
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
                "last_run_id": None,
                "branch": None,
                "last_heartbeat": None,
                "last_error": None,
                "updated_at": _now_iso(),
                "prd_path": str(prd_path),
                "coordinator_pid": None,
                "worker_pid": None,
                "coordinator_started_at": None,
            },
        )

    if not task_queue_path.exists():
        _save_data(task_queue_path, {"updated_at": _now_iso(), "tasks": []})

    if not phase_plan_path.exists():
        _save_data(phase_plan_path, {"updated_at": _now_iso(), "phases": []})

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

        # Schema compatibility: title <-> description
        if not task.get("description") and task.get("title"):
            task["description"] = str(task.get("title"))
        if not task.get("title") and task.get("description"):
            task["title"] = str(task.get("description"))

        status = task.get("status") or TASK_STATUS_TODO
        if not isinstance(status, str):
            status = str(status)
        status = status.strip() or TASK_STATUS_TODO

        if status in {TASK_STATUS_DOING, "in_progress"}:
            status = TASK_STATUS_DOING if task.get("type") == "plan" else TASK_STATUS_IMPLEMENTING
        if status == TASK_STATUS_TESTING and task.get("type") != "plan":
            status = TASK_STATUS_IMPLEMENTING
        task["status"] = status

        task["priority"] = _coerce_int(task.get("priority"), 0)

        deps = task.get("deps", [])
        if isinstance(deps, list):
            task["deps"] = [str(dep).strip() for dep in deps if str(dep).strip()]
        elif deps:
            task["deps"] = [str(deps).strip()]
        else:
            task["deps"] = []

        # Coerce numeric counters safely
        for field in [
            "attempts",
            "auto_resume_attempts",
            "review_attempts",
            "no_change_attempts",
            "no_progress_attempts",
            "plan_attempts",
            "manual_resume_attempts",
        ]:
            task[field] = _coerce_int(task.get(field), 0)

        task.setdefault("last_error", None)
        task.setdefault("last_error_type", None)
        task.setdefault("impl_plan_path", None)
        task.setdefault("impl_plan_hash", None)
        task.setdefault("review_blockers", [])
        task.setdefault("review_blocker_files", [])
        task.setdefault("blocked_intent", None)
        task.setdefault("blocked_at", None)
        task.setdefault("last_run_id", None)

        # Lists
        task["blocking_issues"] = _coerce_string_list(task.get("blocking_issues"))
        task["blocking_next_steps"] = _coerce_string_list(task.get("blocking_next_steps"))

        lcf = task.get("last_changed_files", [])
        if not isinstance(lcf, list):
            lcf = _coerce_string_list(lcf)
        task["last_changed_files"] = [str(p).strip() for p in lcf if str(p).strip()]

        ctx = task.get("context", [])
        if not isinstance(ctx, list):
            ctx = _coerce_string_list(ctx)
        task["context"] = [str(item).strip() for item in ctx if str(item).strip()]

        acceptance = task.get("acceptance_criteria", [])
        if isinstance(acceptance, list):
            task["acceptance_criteria"] = [str(x).strip() for x in acceptance if str(x).strip()]
        elif acceptance:
            task["acceptance_criteria"] = [str(acceptance).strip()]
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

        # Schema compatibility: title -> name, summary -> description
        if not phase.get("name") and phase.get("title"):
            phase["name"] = str(phase.get("title"))
        if not phase.get("description") and phase.get("summary"):
            phase["description"] = str(phase.get("summary"))

        phase.setdefault("status", TASK_STATUS_TODO)
        phase.setdefault("branch", None)
        phase.setdefault("test_command", None)

        acceptance = phase.get("acceptance_criteria", [])
        if isinstance(acceptance, list):
            phase["acceptance_criteria"] = [str(x).strip() for x in acceptance if str(x).strip()]
        elif acceptance:
            phase["acceptance_criteria"] = [str(acceptance).strip()]
        else:
            phase["acceptance_criteria"] = []

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


def _is_placeholder_text(value: str) -> bool:
    normalized = _normalize_text(value)
    normalized = normalized.strip().strip("()[]{}").strip(".,;:")
    return normalized in {"none", "n/a", "na", "nil", "null"}


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _pid_is_running(pid_value: Any) -> bool:
    pid = _coerce_int(pid_value, 0)
    if pid <= 0:
        return False

    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            # Process exists but we may not have permission to signal it.
            return True
        except OSError:
            return False
        else:
            return True

    # Windows: best-effort OpenProcess check.
    try:
        import ctypes
        import ctypes.wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            ctypes.wintypes.DWORD(pid),
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        # If we can't determine, treat as NOT running to avoid deadlocks.
        # (Worker PID check + heartbeat still provide safety.)
        return False


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
    attempts = _coerce_int(task.get(field), 0) + 1
    task[field] = attempts
    return attempts


def _record_task_run(
    task: dict[str, Any],
    run_id: str,
    changed_files: Optional[list[str]],
) -> None:
    task["last_run_id"] = run_id
    if changed_files is not None:
        task["last_changed_files"] = list(changed_files)


def _resolve_test_command(
    phase: Optional[dict[str, Any]],
    task: dict[str, Any],
    default_test_command: Optional[str],
) -> Optional[str]:
    if phase and phase.get("test_command"):
        return phase.get("test_command")
    task_command = task.get("test_command")
    if task_command:
        return task_command
    return default_test_command


def _record_blocked_intent(
    task: dict[str, Any],
    *,
    task_status: str,
    task_type: str,
    phase_id: Optional[str],
    branch: Optional[str],
    test_command: Optional[str],
    run_id: Optional[str],
) -> None:
    task["blocked_intent"] = {
        "task_status": task_status,
        "task_type": task_type,
        "phase_id": phase_id,
        "branch": branch,
        "test_command": test_command,
        "run_id": run_id,
    }
    task["blocked_at"] = _now_iso()


def _maybe_resume_blocked_last_intent(
    queue: dict[str, Any],
    tasks: list[dict[str, Any]],
    max_manual_resumes: int,
) -> tuple[list[dict[str, Any]], bool]:
    blocked = [task for task in tasks if task.get("status") == TASK_STATUS_BLOCKED]
    if not blocked:
        return tasks, False
    candidates = [
        task
        for task in blocked
        if int(task.get("manual_resume_attempts", 0)) < max_manual_resumes
    ]
    if not candidates:
        print(
            "Blocked tasks found, but manual resume attempts exhausted; skipping auto-resume."
        )
        return tasks, False

    def sort_key(task: dict[str, Any]) -> str:
        return str(
            task.get("blocked_at")
            or task.get("last_updated_at")
            or task.get("last_run_id")
            or ""
        )

    target = sorted(candidates, key=sort_key)[-1]
    intent = target.get("blocked_intent") or {}
    prev_status = intent.get("task_status") or TASK_STATUS_TODO
    if prev_status in {
        TASK_STATUS_PLAN_IMPL,
        TASK_STATUS_IMPLEMENTING,
        TASK_STATUS_REVIEW,
        TASK_STATUS_DOING,
    }:
        restore_status = prev_status
    else:
        restore_status = TASK_STATUS_TODO

    target["status"] = restore_status
    target["last_error"] = None
    target["last_error_type"] = None
    target["blocking_issues"] = []
    target["blocking_next_steps"] = []
    target["last_updated_at"] = _now_iso()
    target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1

    context = target.get("context", []) or []
    note = "Human intervention noted. Replaying last blocked step."
    if note not in context:
        context.append(note)
    target["context"] = context

    queue["tasks"] = tasks
    queue["updated_at"] = _now_iso()
    queue["manual_resumed_at"] = _now_iso()
    return tasks, True


def _sanitize_phase_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "phase"


def _impl_plan_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"impl_plan_{safe_id}.json"


def _review_output_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"review_{safe_id}.json"


def _tests_log_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"tests_{safe_id}.log"


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


def _render_json_for_prompt(data: dict[str, Any], max_chars: int = 20000) -> tuple[str, bool]:
    text = json.dumps(data, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


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

    # REMOVED: Rigid step count validation.
    # Added: Check for technical approach or steps.
    steps = plan_data.get("steps")
    tech_approach = plan_data.get("technical_approach")
    
    if not steps and not tech_approach:
        return False, "Must provide either 'technical_approach' (text) or 'steps' (list)"

    files_to_change = plan_data.get("files_to_change")
    if not isinstance(files_to_change, list):
        return False, "files_to_change must be a list"
    if not files_to_change and not _is_docs_only_phase(phase):
        return False, "files_to_change must be non-empty for non-docs phases"
    if any(not isinstance(path, str) or not path.strip() for path in files_to_change):
        return False, "files_to_change must contain non-empty strings"

    return True, ""


def _validate_review_data(
    review_data: dict[str, Any],
    phase: dict[str, Any],
    changed_files: Optional[list[str]] = None,
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
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
    
    # We are lenient here if expected_set is empty, assuming changes might be untracked or previously committed
    if expected_set and not expected_set.issubset(files_reviewed_set):
        # We can make this a warning or just assume the reviewer knows best
        pass

    evidence = review_data.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < REVIEW_MIN_EVIDENCE_ITEMS:
        return False, f"evidence must include at least {REVIEW_MIN_EVIDENCE_ITEMS} items"

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
    heartbeat_seconds: Optional[int] = None,
) -> str:
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
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
{heartbeat_block}"""


def _build_phase_prompt(
    prd_path: Path,
    phase: dict[str, Any],
    task: dict[str, Any],
    events_path: Path,
    progress_path: Path,
    run_id: str,
    user_prompt: Optional[str],
    impl_plan_path: Optional[Path] = None,
    allowed_files: Optional[list[str]] = None,
    no_progress_attempts: int = 0,
    technical_approach_text: str = "",
    heartbeat_seconds: Optional[int] = None,
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
            "\nNOTE: Previous run made no code changes. You MUST edit files "
            "to implement the requirements.\n"
        )
    
    approach_block = ""
    if technical_approach_text:
        approach_block = f"\nTechnical Approach (from Plan):\n{technical_approach_text}\n"
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"

    return f"""You are a Codex CLI worker. Implement the COMPLETE phase described below.

PRD: {prd_path}
Phase: {phase_name}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}

Additional context from previous runs:
{context_block}
{user_block}

Implementation plan file: {plan_path_display}
{approach_block}

Rules:
- Work on the ENTIRE phase scope. Implement all necessary changes.
- Do not commit or push; the coordinator will handle git.
- If tests fail, fix them (the coordinator will run tests after you finish).
- Keep the project's README.md updated with changes in this phase.
- Allowed files to read/edit:
{allowed_block}
{no_progress_block}

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, blocking_issues, heartbeat.
{heartbeat_block}"""


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
    heartbeat_seconds: Optional[int] = None,
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
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, blocking_issues, heartbeat.\n"
            f"{heartbeat_block}"
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
  "files_to_change": ["path/a.py", "path/b.ts"],
  "new_files": ["path/new_file.py"],
  "technical_approach": [
      "1. High-level step description",
      "2. Another step description",
      "3. Integration notes"
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
- Focus on a coherent technical approach, not just a list of rigid steps.
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
    heartbeat_seconds: Optional[int] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, blocking_issues, heartbeat.\n"
            f"{heartbeat_block}"
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
{diff_text}
{diff_notice}

Implementation plan (from coordinator):
{impl_plan_text or "(missing)"}
{plan_notice}

Review output schema:
{{
  "phase_id": "{phase.get('id')}",
  "spec_summary": ["bullets restating PRD requirements relevant to this phase"],
  "design_assessment": {{
    "architecture_summary": ["3-8 bullets describing solution structure"],
    "key_components": ["components/modules touched and their roles"]
  }},
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
  "changed_files": ["exact list from coordinator"],
  "files_reviewed": ["list of paths"],
  "evidence": ["at least two concrete observations with file/diff references"],
  "recommendations": ["actionable fixes"]
}}

Review instructions:
- Verify implementation aligns with the plan.
- If acceptance criteria are empty, include one checklist item with criterion "(none provided)".
- Provide at least {REVIEW_MIN_EVIDENCE_ITEMS} concrete evidence items tied to files/diff.
- If tests passed (coordinator verified), assume logic works but check for code quality and specs.
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
    on_spawn: Optional[Callable[[int], None]] = None,
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
    uses_prompt_placeholder = "{prompt_file}" in command or "{prompt}" in command
    expects_stdin = "-" in command_parts
    if not uses_prompt_placeholder and not expects_stdin:
        raise ValueError(
            "Codex command must include {prompt_file}, {prompt}, or '-' to accept stdin input."
        )

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

    if on_spawn:
        try:
            on_spawn(process.pid)
        except Exception:
            pass

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

    if not uses_prompt_placeholder and expects_stdin:
        if process.stdin:
            try:
                process.stdin.write(prompt)
                process.stdin.flush()
                process.stdin.close()
            except BrokenPipeError:
                pass

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


def _path_is_allowed(project_dir: Path, path: str, allowed_patterns: list[str]) -> bool:
    for pattern in allowed_patterns or []:
        pattern = str(pattern).strip()
        if not pattern:
            continue

        # Glob patterns
        if any(ch in pattern for ch in "*?["):
            if fnmatch.fnmatch(path, pattern):
                return True
            continue

        normalized = pattern.rstrip("/")

        # Treat directories as prefixes (even if the pattern omitted the trailing slash)
        candidate = project_dir / normalized
        if pattern.endswith("/") or (candidate.exists() and candidate.is_dir()):
            prefix = f"{normalized}/"
            if path == normalized or path.startswith(prefix):
                return True

        # Exact file match
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


def _gitignore_change_is_prd_runner_only(project_dir: Path) -> bool:
    commands = [
        ["git", "diff", "--", ".gitignore"],
        ["git", "diff", "--staged", "--", ".gitignore"],
    ]
    changes: list[str] = []
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
            if line.startswith(("+++ ", "--- ", "@@ ")):
                continue
            if line.startswith("+") or line.startswith("-"):
                changes.append(line)
    if not changes:
        return False
    if any(line.startswith("-") for line in changes):
        return False
    allowed = {".prd_runner", ".prd_runner/"}
    additions = [line[1:].strip() for line in changes if line.startswith("+")]
    if not additions:
        return False
    return all(entry in allowed for entry in additions)


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
        ["git", "add", "-A", "--", "."],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", message], cwd=project_dir, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=project_dir, check=True)


_LAST_ERROR_UNSET = object()

def _finalize_run_state(
    paths: dict[str, Path],
    lock_path: Path,
    *,
    status: str = "idle",
    last_error: Any = _LAST_ERROR_UNSET,
) -> None:
    """Mark coordinator run as finished; clear running markers."""
    with FileLock(lock_path):
        run_state = _load_data(paths["run_state"], {})
        run_state.update(
            {
                "status": status,  # "idle" or "blocked"
                "current_task_id": None,
                "current_phase_id": None,
                "run_id": None,
                "branch": None,
                "worker_pid": None,
                "coordinator_pid": None,
                "coordinator_started_at": None,
                "updated_at": _now_iso(),
                "last_heartbeat": _now_iso(),
            }
        )

        if last_error is not _LAST_ERROR_UNSET:
            # Explicitly set (including None)
            run_state["last_error"] = last_error
        else:
            # If we're going idle and no explicit error was provided, clear old errors.
            if status == "idle":
                run_state["last_error"] = None
            # If blocked and not provided, preserve existing run_state["last_error"].

        _save_data(paths["run_state"], run_state)


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

    # If we know the worker PID and it's alive, treat as active.
    worker_pid = _coerce_int(run_state.get("worker_pid"), 0)
    if worker_pid and _pid_is_running(worker_pid):
        return False

    # If the coordinator PID is recorded and dead, treat as stale immediately.
    coordinator_pid = _coerce_int(run_state.get("coordinator_pid"), 0)
    if coordinator_pid and not _pid_is_running(coordinator_pid):
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
    issues = [item for item in issues if not _is_placeholder_text(item)]
    next_steps = [item for item in next_steps if not _is_placeholder_text(item)]
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

            changed_files_after_run = None
            progress_files = None
            disallowed_files: list[str] = []
            if task_type != "plan" and not planning_impl and not failure:
                changed_files_after_run = _git_changed_files(
                    project_dir,
                    include_untracked=True,
                    ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
                )
                progress_files = changed_files_after_run
                if allowed_files and changed_files_after_run:
                    progress_files = [
                        p for p in changed_files_after_run
                        if _path_is_allowed(project_dir, p, allowed_files)
                    ]
                    disallowed_files = [
                        p for p in changed_files_after_run
                        if not _path_is_allowed(project_dir, p, allowed_files)
                    ]
                    if ".gitignore" in disallowed_files and _gitignore_change_is_prd_runner_only(
                        project_dir
                    ):
                        disallowed_files = [path for path in disallowed_files if path != ".gitignore"]
            if disallowed_files and not failure:
                failure = True
                error_detail = (
                    "Changes outside allowed files: "
                    + ", ".join(disallowed_files)[:400]
                )
                error_type = ERROR_TYPE_DISALLOWED_FILES

            manifest = {
                "run_id": run_id,
                "task_id": task_id,
                # ... standard manifest fields ...
                "start_time": run_result["start_time"],
                "end_time": run_result["end_time"],
                "exit_code": run_result["exit_code"],
                "changed_files": changed_files_after_run or [],
                "disallowed_files": disallowed_files,
            }
            _save_data(run_dir / "manifest.json", manifest)
            _finalize_run_state(paths, lock_path, status="idle", last_error=error_detail)

            # --- 1. HANDLE BLOCKING ISSUES ---
            if progress_blocking_detected:
                issue_summary = "; ".join(progress_blocking).strip()[:400]
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    if target:
                        _record_task_run(target, run_id, changed_files_after_run)
                        target["last_error"] = f"Blocking issues reported: {issue_summary}"
                        target["last_error_type"] = ERROR_TYPE_BLOCKING_ISSUES
                        target["blocking_issues"] = progress_blocking
                        target["blocking_next_steps"] = progress_next_steps
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
                if phase_test_command:
                    test_log_path = _tests_log_path(paths["artifacts"], str(phase.get("id") or phase_id or task_id))
                    test_result = _run_command(phase_test_command, project_dir, test_log_path)
                    if test_result["exit_code"] != 0:
                        tests_passed = False
                        print(f"\nTests failed for phase {phase.get('id')}. Log: {test_log_path}")

                # 2. Check Progress
                has_changes = bool(progress_files)
                
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
                            target["no_progress_attempts"] = 0
                        else:
                            # Failure Loop
                            target["status"] = TASK_STATUS_IMPLEMENTING
                            if not tests_passed:
                                target["last_error"] = "Tests failed"
                                target["context"] = target.get("context", []) + [f"Tests failed. See logs."]
                            elif not has_changes:
                                attempts = _increment_task_counter(target, "no_progress_attempts")
                                target["last_error"] = "No changes detected"
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
                review_blocking = review_data.get("blocking_issues") or []
                
                with FileLock(lock_path):
                    queue = _load_data(paths["task_queue"], {})
                    tasks = _normalize_tasks(queue)
                    target = _find_task(tasks, task_id)
                    
                    if target:
                        if review_blocking:
                            target["status"] = TASK_STATUS_IMPLEMENTING
                            target["last_error"] = "Review blockers found"
                            target["review_blockers"] = review_blocking
                            target["review_blocker_files"] = _extract_review_blocker_files(review_data)
                            target["context"] = target.get("context", []) + [f"Review blockers: {review_blocking}"]
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

# ... (main/parse_args remain same) ...
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
