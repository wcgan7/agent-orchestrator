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
  --resume-prompt "Focus on error handling first"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

TRANSIENT_ERROR_MARKERS = (
    "No heartbeat",
    "Shift timed out",
)


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
            if yaml:
                data = yaml.safe_load(handle)
            else:
                data = json.load(handle)
        return data if isinstance(data, dict) else default
    except (OSError, json.JSONDecodeError):
        return default


def _save_data(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_json(path, data)


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
    progress_path = artifacts_dir / "progress.json"

    if not events_path.exists():
        events_path.touch()

    if not progress_path.exists():
        _save_data(
            progress_path,
            {
                "run_id": None,
                "task_id": None,
                "phase": "idle",
                "actions": [],
                "files_changed": [],
                "claims": [],
                "next_steps": [],
                "blocking_issues": [],
                "heartbeat": _now_iso(),
            },
        )

    return {
        "state_dir": state_dir,
        "run_state": run_state_path,
        "task_queue": task_queue_path,
        "phase_plan": phase_plan_path,
        "artifacts": artifacts_dir,
        "runs": runs_dir,
        "events": events_path,
        "progress": progress_path,
    }


def _build_plan_task() -> dict[str, Any]:
    return {
        "id": "plan-001",
        "type": "plan",
        "status": "todo",
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
                "status": "todo",
                "priority": index,
                "deps": phase.get("deps", []) or [],
                "description": description,
                "acceptance_criteria": phase.get("acceptance_criteria", []) or [],
                "test_command": phase.get("test_command"),
                "branch": phase.get("branch"),
            }
        )
    return tasks


def _normalize_tasks(queue: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = queue.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task.setdefault("status", "todo")
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
        phase.setdefault("status", "todo")
        phase.setdefault("acceptance_criteria", [])
        phase.setdefault("branch", None)
        phase.setdefault("test_command", None)
        normalized.append(phase)
    return normalized


def _task_summary(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"todo": 0, "doing": 0, "done": 0, "blocked": 0}
    for task in tasks:
        status = task.get("status", "todo")
        if status == "in_progress":
            status = "doing"
        if status in counts:
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
        if task.get("status") in {"doing", "in_progress"}:
            return task

    for _, task in sorted_tasks:
        if task.get("status") == "todo" and _deps_satisfied(task, tasks_by_id):
            return task

    return None


def _is_transient_error(error: Optional[str]) -> bool:
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
        if not _is_transient_error(last_error):
            continue
        attempts = int(task.get("auto_resume_attempts", 0))
        if attempts >= max_auto_resumes:
            continue

        task["status"] = "todo"
        task["attempts"] = 0
        task["last_error"] = None
        task["auto_resume_attempts"] = attempts + 1
        task["last_updated_at"] = _now_iso()
        changed = True

    if changed:
        queue["tasks"] = tasks
        queue["updated_at"] = _now_iso()
        queue["auto_resumed_at"] = _now_iso()
    return tasks, changed


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
) -> str:
    phase_name = phase.get("name") or phase.get("id")
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = (
        "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none provided)"
    )
    context_items = task.get("context", []) or []
    context_block = "\n".join(f"- {item}" for item in context_items) if context_items else "- (none)"

    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
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

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, blocking_issues, heartbeat.
"""


def _build_review_prompt(
    phase: dict[str, Any],
    review_path: Path,
    prd_path: Path,
    user_prompt: Optional[str],
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    return f"""Perform a code review for the phase below and write JSON to {review_path}.

Phase: {phase.get("name") or phase.get("id")}
PRD: {prd_path}
Acceptance criteria:
{acceptance_block}
{user_block}

Review output schema:
{{
  "phase_id": "{phase.get('id')}",
  "summary": "Short summary",
  "blocking_issues": ["list of blockers"],
  "non_blocking": ["nice-to-have improvements"],
  "files_reviewed": ["list of paths"],
  "recommendations": ["actionable fixes"]
}}

Review instructions:
- Compare changes against PRD requirements and phase acceptance criteria.
- Call out gaps, regressions, or missing README updates.
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
) -> dict[str, Any]:
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    start_wall = datetime.now(timezone.utc)
    try:
        formatted_command = command.format(
            prompt_file=str(prompt_path),
            project_dir=str(project_dir),
            run_dir=str(run_dir),
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

        heartbeat = _heartbeat_from_progress(progress_path)
        if heartbeat:
            if heartbeat >= start_wall:
                last_heartbeat = heartbeat
                age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
                if age > heartbeat_grace_seconds:
                    no_heartbeat = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
        else:
            age = (datetime.now(timezone.utc) - start_wall).total_seconds()
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


def _heartbeat_from_progress(progress_path: Path) -> Optional[datetime]:
    if not progress_path.exists():
        return None
    progress = _load_data(progress_path, {})
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


def _git_commit_and_push(project_dir: Path, branch: str, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=project_dir, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=project_dir, check=True)


def _phase_for_task(phases: list[dict[str, Any]], task: dict[str, Any]) -> Optional[dict[str, Any]]:
    phase_id = task.get("phase_id") or task.get("id")
    for phase in phases:
        if phase.get("id") == phase_id:
            return phase
    return None


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
) -> None:
    project_dir = project_dir.resolve()
    prd_path = prd_path.resolve()
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

            if not tasks:
                tasks = [_build_plan_task()]
                queue["tasks"] = tasks
                queue["updated_at"] = _now_iso()
                _save_data(paths["task_queue"], queue)

            tasks, resumed = _maybe_auto_resume_blocked(queue, tasks, max_auto_resumes)
            if resumed:
                _save_data(paths["task_queue"], queue)
                print("Auto-resumed blocked tasks after transient failure")

            next_task = _select_next_task(tasks)
            if not next_task:
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
            if next_task.get("status") == "todo":
                next_task["status"] = "doing"

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

        iteration += 1
        run_dir = paths["runs"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        _update_progress(
            paths["progress"],
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
                        target["status"] = "blocked"
                        target["last_error"] = "Phase not found for task"
                        _save_queue(paths["task_queue"], queue, tasks)
                continue

            branch = phase.get("branch") or f"feature/{phase_id or task_id}"
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
                        target["status"] = "blocked"
                        target["last_error"] = f"Branch checkout failed: {exc}"
                        _save_queue(paths["task_queue"], queue, tasks)
                continue

        try:
            if task_type == "plan":
                prompt = _build_plan_prompt(
                    prd_path=prd_path,
                    phase_plan_path=paths["phase_plan"],
                    task_queue_path=paths["task_queue"],
                    events_path=paths["events"],
                    progress_path=paths["progress"],
                    run_id=run_id,
                    user_prompt=user_prompt,
                )
            else:
                if not phase:
                    raise ValueError("Phase not found for task")
                prompt = _build_phase_prompt(
                    prd_path=prd_path,
                    phase=phase,
                    task=next_task,
                    events_path=paths["events"],
                    progress_path=paths["progress"],
                    run_id=run_id,
                    user_prompt=user_prompt,
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
                progress_path=paths["progress"],
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
        _save_data(run_dir / "manifest.json", manifest)

        failure = run_result["exit_code"] != 0 or run_result["no_heartbeat"]
        error_detail = None
        if run_result["no_heartbeat"]:
            error_detail = "No heartbeat received within grace period"
        elif run_result["timed_out"]:
            error_detail = "Shift timed out"
        elif run_result["exit_code"] != 0:
            error_detail = f"Codex CLI exited with code {run_result['exit_code']}"
            if stderr_tail.strip():
                error_detail = f"{error_detail}. stderr: {stderr_tail.strip()}"
            elif stdout_tail.strip():
                error_detail = f"{error_detail}. stdout: {stdout_tail.strip()}"

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
                if failure:
                    target["attempts"] = int(target.get("attempts", 0)) + 1
                    target["last_error"] = error_detail or "Run failed"
                    if target["attempts"] >= max_attempts:
                        target["status"] = "blocked"
                    else:
                        target["status"] = "todo"
                else:
                    target["status"] = "done" if task_type == "plan" else "doing"
            elif task_type != "plan":
                raise ValueError(f"Task {task_id} not found in queue")

            if not failure and task_type != "plan":
                phase_entry = _phase_for_task(phases, {"phase_id": phase_id or task_id, "id": task_id})
                if phase_entry:
                    phase_entry["status"] = "doing"

            run_state.update(
                {
                    "status": "idle",
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

        if failure:
            print(f"\nRun {run_id} failed: {error_detail}")
            print(f"See logs: {run_dir / 'stderr.log'}")
            continue

        if task_type == "plan":
            plan = _load_data(paths["phase_plan"], {})
            phases = _normalize_phases(plan)
            queue = _load_data(paths["task_queue"], {})
            tasks = _normalize_tasks(queue)
            tasks_empty = not tasks or (len(tasks) == 1 and tasks[0].get("id") == "plan-001")
            if phases and tasks_empty:
                tasks = _build_tasks_from_phases(phases)
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
                    plan_task["status"] = "blocked"
                    plan_task["last_error"] = "Phase plan not generated"
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(f"\nRun {run_id} complete, but phase plan was not generated.")
            else:
                _save_queue(paths["task_queue"], queue, tasks)
                _save_plan(paths["phase_plan"], plan, phases)
                print(f"\nRun {run_id} complete. Phase plan created.")
            continue

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

        phase_test_command = phase.get("test_command") or target.get("test_command") or test_command
        test_log = None
        if phase_test_command:
            test_log_path = paths["artifacts"] / f"tests_{phase.get('id')}.log"
            test_result = _run_command(phase_test_command, project_dir, test_log_path)
            test_log = test_result["log_path"]
            if test_result["exit_code"] != 0:
                target["status"] = "todo"
                target["last_error"] = f"Tests failed. See {test_log}"
                context = target.get("context", [])
                context.append(f"Tests failed: {test_log}")
                target["context"] = context
                _append_event(
                    paths["events"],
                    {
                        "event_type": "tests_failed",
                        "phase_id": phase.get("id"),
                        "log_path": test_log,
                    },
                )
                _save_queue(paths["task_queue"], queue, tasks)
                print(f"\nTests failed for phase {phase.get('id')}. Re-queueing task.")
                continue

        review_path = paths["artifacts"] / f"review_{phase.get('id')}.json"
        review_prompt = _build_review_prompt(
            phase=phase,
            review_path=review_path,
            prd_path=prd_path,
            user_prompt=user_prompt,
        )
        review_run_dir = paths["runs"] / f"{run_id}-review"
        review_run_dir.mkdir(parents=True, exist_ok=True)

        review_result = _run_codex_worker(
            command=codex_command,
            prompt=review_prompt,
            project_dir=project_dir,
            run_dir=review_run_dir,
            timeout_seconds=shift_minutes * 60,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_grace_seconds=heartbeat_grace_seconds,
            progress_path=paths["progress"],
        )
        review_stdout = _read_log_tail(Path(review_result["stdout_path"]))
        review_stderr = _read_log_tail(Path(review_result["stderr_path"]))

        if review_result["exit_code"] != 0:
            target["status"] = "todo"
            target["last_error"] = f"Review failed: {review_stderr.strip() or review_stdout.strip()}"
            target["context"] = target.get("context", []) + [
                "Review failed; rerun review and fix issues."
            ]
            _save_queue(paths["task_queue"], queue, tasks)
            print(f"\nReview step failed for phase {phase.get('id')}. Re-queueing task.")
            continue

        review_data = _load_data(review_path, {})
        if not review_data or "blocking_issues" not in review_data:
            target["status"] = "todo"
            target["last_error"] = f"Review output missing or invalid: {review_path}"
            target["context"] = target.get("context", []) + [
                f"Review output missing or invalid: {review_path}"
            ]
            _save_queue(paths["task_queue"], queue, tasks)
            print(f"\nReview output missing or invalid for phase {phase.get('id')}. Re-queueing task.")
            continue
        blocking = review_data.get("blocking_issues") or []
        if isinstance(blocking, list) and blocking:
            target["status"] = "todo"
            target["last_error"] = "Review blockers found"
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
            _save_queue(paths["task_queue"], queue, tasks)
            print(f"\nReview blockers found for phase {phase.get('id')}. Re-queueing task.")
            continue

        phase["status"] = "done"
        target["status"] = "done"
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
                target["status"] = "blocked"
                target["last_error"] = f"Git push failed: {exc}"
                _save_queue(paths["task_queue"], queue, tasks)
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
    )


if __name__ == "__main__":
    main()
