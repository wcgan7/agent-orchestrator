from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .constants import ARTIFACTS_DIR, LOCK_FILE, PHASE_PLAN_FILE, RUNS_DIR, RUN_STATE_FILE, STATE_DIR_NAME, TASK_QUEUE_FILE
    from .io_utils import FileLock, _load_data, _save_data
    from .utils import _coerce_int, _now_iso, _parse_iso, _pid_is_running
except ImportError:  # pragma: no cover
    from constants import ARTIFACTS_DIR, LOCK_FILE, PHASE_PLAN_FILE, RUNS_DIR, RUN_STATE_FILE, STATE_DIR_NAME, TASK_QUEUE_FILE
    from io_utils import FileLock, _load_data, _save_data
    from utils import _coerce_int, _now_iso, _parse_iso, _pid_is_running

try:
    from .io_utils import _heartbeat_from_progress
except ImportError:  # pragma: no cover
    from io_utils import _heartbeat_from_progress


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
