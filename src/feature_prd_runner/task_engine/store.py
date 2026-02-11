"""File-based task store with thread-safe locking.

Stores tasks in a single YAML file (``tasks_v2.yaml``) inside the project's
``.prd_runner/`` directory.  All reads and writes go through :func:`transaction`
which acquires an exclusive file lock to guarantee thread-safety.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from ..io_utils import FileLock
from .model import Task

try:
    import yaml
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    import yaml  # type: ignore[no-redef]
    Dumper = getattr(yaml, "Dumper")  # type: ignore[assignment,misc]
    Loader = getattr(yaml, "Loader")  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORE_FILENAME = "tasks_v2.yaml"
LOCK_FILENAME = "tasks_v2.lock"
LOCK_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> list[dict[str, Any]]:
    """Load the raw task list from *path*, returning ``[]`` if missing."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    data = yaml.load(text, Loader=Loader)
    if not isinstance(data, dict) or "tasks" not in data:
        return []
    tasks = data["tasks"]
    return list(tasks) if isinstance(tasks, list) else []


def _save_raw(path: Path, tasks: list[dict[str, Any]]) -> None:
    """Atomically write *tasks* to *path* (write-tmp-then-rename)."""
    payload = {"version": 2, "tasks": tasks}
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        import os
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, Dumper=Dumper, default_flow_style=False, sort_keys=False)
        shutil.move(tmp, str(path))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------

class TaskStore:
    """Thread-safe, file-backed store for :class:`Task` objects.

    Parameters
    ----------
    state_dir:
        Path to the ``.prd_runner/`` directory for the project.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._store_path = state_dir / STORE_FILENAME
        self._lock_path = state_dir / LOCK_FILENAME
        self._lock = FileLock(self._lock_path)

    # -- internal helpers ---------------------------------------------------

    def _load(self) -> list[Task]:
        raw = _load_raw(self._store_path)
        return [Task.from_dict(d) for d in raw]

    def _save(self, tasks: list[Task]) -> None:
        _save_raw(self._store_path, [t.to_dict() for t in tasks])

    # -- public API ---------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[_TaskTx]:
        """Acquire the lock, load tasks, yield a transaction, and save on exit.

        Usage::

            with store.transaction() as tx:
                task = tx.get("task-abc123")
                task.status = TaskStatus.IN_PROGRESS
                # automatically saved on exit
        """
        with self._lock:
            tasks = self._load()
            tx = _TaskTx(tasks)
            yield tx
            if tx.dirty:
                self._save(tx.tasks)

    def read_snapshot(self) -> list[Task]:
        """Return a read-only snapshot (no lock held after return)."""
        with self._lock:
            return self._load()

    def get_one(self, task_id: str) -> Optional[Task]:
        """Convenience: fetch a single task without holding the lock."""
        with self._lock:
            for t in self._load():
                if t.id == task_id:
                    return t
        return None


class _TaskTx:
    """In-memory transaction over a list of tasks.

    Mutations are collected and flushed back to disk when the ``transaction``
    context-manager exits.
    """

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks
        self.dirty = False
        self._index: dict[str, int] = {t.id: i for i, t in enumerate(tasks)}

    # -- lookups ------------------------------------------------------------

    def get(self, task_id: str) -> Optional[Task]:
        idx = self._index.get(task_id)
        return self.tasks[idx] if idx is not None else None

    def list_all(self) -> list[Task]:
        return list(self.tasks)

    def find(
        self,
        *,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        label: Optional[str] = None,
        search: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> list[Task]:
        out: list[Task] = []
        for t in self.tasks:
            if status and t.status.value != status:
                continue
            if task_type and t.task_type.value != task_type:
                continue
            if priority and t.priority.value != priority:
                continue
            if assignee and t.assignee != assignee:
                continue
            if label and label not in t.labels:
                continue
            if parent_id is not None and t.parent_id != parent_id:
                continue
            if search:
                q = search.lower()
                if q not in t.title.lower() and q not in t.description.lower() and q not in t.id.lower():
                    continue
            out.append(t)
        return out

    # -- mutations ----------------------------------------------------------

    def add(self, task: Task) -> Task:
        if task.id in self._index:
            raise ValueError(f"Task {task.id} already exists")
        self._index[task.id] = len(self.tasks)
        self.tasks.append(task)
        self.dirty = True
        return task

    def add_many(self, tasks: list[Task]) -> list[Task]:
        for t in tasks:
            self.add(t)
        return tasks

    def update(self, task_id: str, changes: dict[str, Any]) -> Optional[Task]:
        task = self.get(task_id)
        if task is None:
            return None
        for key, value in changes.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.touch()
        self.dirty = True
        return task

    def remove(self, task_id: str) -> bool:
        """Soft-delete: set status to CANCELLED."""
        task = self.get(task_id)
        if task is None:
            return False
        task.transition(task.status.__class__("cancelled"))
        self.dirty = True
        return True

    def hard_remove(self, task_id: str) -> bool:
        """Physically remove a task from the store."""
        idx = self._index.pop(task_id, None)
        if idx is None:
            return False
        self.tasks.pop(idx)
        # rebuild index
        self._index = {t.id: i for i, t in enumerate(self.tasks)}
        self.dirty = True
        return True

    def reorder(self, task_ids: list[str]) -> None:
        """Reorder tasks to match *task_ids* ordering (unmentioned tasks keep position)."""
        id_set = set(task_ids)
        ordered = []
        remaining = []
        for t in self.tasks:
            if t.id in id_set:
                remaining.append(t)
            else:
                ordered.append(t)
        task_map = {t.id: t for t in remaining}
        for tid in task_ids:
            if tid in task_map:
                ordered.append(task_map[tid])
        self.tasks = ordered
        self._index = {t.id: i for i, t in enumerate(self.tasks)}
        self.dirty = True
