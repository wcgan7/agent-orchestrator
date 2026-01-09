from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from .constants import (
        AUTO_RESUME_ERROR_TYPES,
        ERROR_TYPE_HEARTBEAT_TIMEOUT,
        ERROR_TYPE_SHIFT_TIMEOUT,
        IGNORED_REVIEW_PATH_PREFIXES,
        TASK_IN_PROGRESS_STATUSES,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_DOING,
        TASK_STATUS_IMPLEMENTING,
        TASK_STATUS_PLAN_IMPL,
        TASK_STATUS_REVIEW,
        TASK_STATUS_TESTING,
        TASK_STATUS_TODO,
        TRANSIENT_ERROR_MARKERS,
    )
    from .git_utils import _git_changed_files
    from .io_utils import _load_data, _save_data
    from .utils import (
        _coerce_int,
        _coerce_string_list,
        _is_placeholder_text,
        _now_iso,
        _sanitize_phase_id,
    )
except ImportError:  # pragma: no cover
    from constants import (
        AUTO_RESUME_ERROR_TYPES,
        ERROR_TYPE_HEARTBEAT_TIMEOUT,
        ERROR_TYPE_SHIFT_TIMEOUT,
        IGNORED_REVIEW_PATH_PREFIXES,
        TASK_IN_PROGRESS_STATUSES,
        TASK_STATUS_BLOCKED,
        TASK_STATUS_DOING,
        TASK_STATUS_IMPLEMENTING,
        TASK_STATUS_PLAN_IMPL,
        TASK_STATUS_REVIEW,
        TASK_STATUS_TESTING,
        TASK_STATUS_TODO,
        TRANSIENT_ERROR_MARKERS,
    )
    from git_utils import _git_changed_files
    from io_utils import _load_data, _save_data
    from utils import (
        _coerce_int,
        _coerce_string_list,
        _is_placeholder_text,
        _now_iso,
        _sanitize_phase_id,
    )


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
            "test_fail_attempts",
        ]:
            task[field] = _coerce_int(task.get(field), 0)

        task.setdefault("last_tests", None)
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


def _impl_plan_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"impl_plan_{safe_id}.json"


def _review_output_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"review_{safe_id}.json"


def _tests_log_path(artifacts_dir: Path, phase_id: str) -> Path:
    safe_id = _sanitize_phase_id(phase_id)
    return artifacts_dir / f"tests_{safe_id}.log"


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


def _read_progress_human_blockers(
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

    issues = _coerce_string_list(progress.get("human_blocking_issues"))
    next_steps = _coerce_string_list(progress.get("human_next_steps"))

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
