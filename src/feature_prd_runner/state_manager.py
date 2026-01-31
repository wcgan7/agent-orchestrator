"""Thread-safe state management for task queue and phase plan.

This module provides a StateTransaction context manager that encapsulates
the lock-load-modify-save pattern used throughout the codebase, reducing
code duplication and making state operations more consistent.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from .io_utils import FileLock, _load_data
from .tasks import (
    _find_task,
    _normalize_phases,
    _normalize_tasks,
    _phase_for_task,
    _save_plan,
    _save_queue,
)


@dataclass
class StateSnapshot:
    """Snapshot of task queue and optionally phase plan state.

    This class holds the loaded state and provides helper methods for
    common operations. Modifications to the contained data structures
    will be saved when the transaction completes.
    """

    queue: dict[str, Any]
    tasks: list[dict[str, Any]]
    plan: Optional[dict[str, Any]] = None
    phases: Optional[list[dict[str, Any]]] = None

    # Internal paths for saving
    _task_queue_path: Optional[Path] = field(default=None, repr=False)
    _phase_plan_path: Optional[Path] = field(default=None, repr=False)
    _save_plan_on_exit: bool = field(default=False, repr=False)

    def find_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Find a task by ID.

        Args:
            task_id: The task identifier to search for.

        Returns:
            The task dict if found, None otherwise.
        """
        return _find_task(self.tasks, task_id)

    def phase_for_task(self, task: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Get the phase associated with a task.

        Args:
            task: The task dict.

        Returns:
            The phase dict if found and phases were loaded, None otherwise.
        """
        if self.phases is None:
            return None
        return _phase_for_task(self.phases, task)


@contextmanager
def state_transaction(
    lock_path: Path,
    task_queue_path: Path,
    phase_plan_path: Optional[Path] = None,
    *,
    load_plan: bool = False,
    save_plan: bool = False,
) -> Generator[StateSnapshot, None, None]:
    """Context manager for thread-safe state transactions.

    Acquires an exclusive lock, loads state from disk, yields a snapshot
    for modification, and saves state on successful exit.

    This encapsulates the common pattern:
        with FileLock(lock_path):
            queue = _load_data(task_queue_path, {})
            tasks = _normalize_tasks(queue)
            plan = _load_data(phase_plan_path, {})
            phases = _normalize_phases(plan)
            # ... modify state ...
            _save_queue(task_queue_path, queue, tasks)

    Args:
        lock_path: Path to the lock file.
        task_queue_path: Path to task_queue.yaml.
        phase_plan_path: Path to phase_plan.yaml (required if load_plan=True).
        load_plan: Whether to load phase plan. Set to True when you need
            to access phase data or call phase_for_task(). Default False
            for efficiency when only task operations are needed.
        save_plan: Whether to save phase plan on exit. Default False.
            Set to True when you modify phase data.

    Yields:
        StateSnapshot containing loaded state. Modify the queue/tasks/plan/phases
        in place; changes will be saved on successful context exit.

    Raises:
        ValueError: If load_plan=True but phase_plan_path is None.

    Example:
        with state_transaction(
            lock_path,
            task_queue_path,
            phase_plan_path,
            load_plan=True,
        ) as state:
            task = state.find_task("task-001")
            if task:
                task["status"] = "running"
            # State is automatically saved on exit
    """
    if load_plan and phase_plan_path is None:
        raise ValueError("phase_plan_path is required when load_plan=True")

    with FileLock(lock_path):
        # Load task queue (always required)
        queue = _load_data(task_queue_path, {})
        tasks = _normalize_tasks(queue)

        # Load phase plan (optional, for efficiency)
        plan: Optional[dict[str, Any]] = None
        phases: Optional[list[dict[str, Any]]] = None
        if load_plan and phase_plan_path:
            plan = _load_data(phase_plan_path, {})
            phases = _normalize_phases(plan)

        snapshot = StateSnapshot(
            queue=queue,
            tasks=tasks,
            plan=plan,
            phases=phases,
            _task_queue_path=task_queue_path,
            _phase_plan_path=phase_plan_path,
            _save_plan_on_exit=save_plan,
        )

        yield snapshot

        # Save state on successful exit
        _save_queue(task_queue_path, snapshot.queue, snapshot.tasks)

        if save_plan and phase_plan_path and snapshot.plan is not None:
            _save_plan(phase_plan_path, snapshot.plan, snapshot.phases or [])


def load_task_state_quick(
    lock_path: Path,
    task_queue_path: Path,
    task_id: str,
) -> Optional[dict[str, Any]]:
    """Quickly load a single task's state (read-only snapshot).

    This is a convenience function for when you only need to read
    a task's current state without modifying it.

    Args:
        lock_path: Path to the lock file.
        task_queue_path: Path to task_queue.yaml.
        task_id: The task identifier to find.

    Returns:
        A copy of the task dict if found, None otherwise.
    """
    with FileLock(lock_path):
        queue = _load_data(task_queue_path, {})
        tasks = _normalize_tasks(queue)
        task = _find_task(tasks, task_id)
        if task:
            # Return a copy to prevent accidental modification
            return dict(task)
        return None
