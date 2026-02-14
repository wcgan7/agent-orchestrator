from __future__ import annotations

import json
import os
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

from ...io_utils import FileLock
from ..domain.models import AgentRecord, QuickActionRun, ReviewCycle, RunRecord, Task, now_iso
from .interfaces import (
    AgentRepository,
    EventRepository,
    QuickActionRepository,
    ReviewRepository,
    RunRepository,
    TaskRepository,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for runtime file repositories")


T = TypeVar("T")


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 99)


class _YamlCollectionRepo(Generic[T]):
    def __init__(
        self,
        path: Path,
        lock_path: Path,
        key: str,
        loader: Callable[[dict[str, Any]], T],
        dumper: Callable[[T], dict[str, Any]],
    ) -> None:
        self._path = path
        self._lock = FileLock(lock_path)
        self._thread_lock = threading.RLock()
        self._key = key
        self._loader = loader
        self._dumper = dumper

    def _load(self) -> list[T]:
        _require_yaml()
        if not self._path.exists():
            return []
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return []
        items = raw.get(self._key, [])
        if not isinstance(items, list):
            return []
        out: list[T] = []
        for item in items:
            if isinstance(item, dict):
                out.append(self._loader(item))
        return out

    def _save(self, items: list[T]) -> None:
        _require_yaml()
        payload = {"version": 3, self._key: [self._dumper(item) for item in items]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._path)


class FileTaskRepository(TaskRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._repo = _YamlCollectionRepo[Task](
            path,
            lock_path,
            "tasks",
            loader=Task.from_dict,
            dumper=lambda t: t.to_dict(),
        )

    def list(self) -> list[Task]:
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, task_id: str) -> Optional[Task]:
        for task in self.list():
            if task.id == task_id:
                return task
        return None

    def upsert(self, task: Task) -> Task:
        with self._repo._thread_lock:
            with self._repo._lock:
                tasks = self._repo._load()
                found = False
                for idx, existing in enumerate(tasks):
                    if existing.id == task.id:
                        task.updated_at = now_iso()
                        tasks[idx] = task
                        found = True
                        break
                if not found:
                    task.created_at = task.created_at or now_iso()
                    task.updated_at = now_iso()
                    tasks.append(task)
                self._repo._save(tasks)
        return task

    def delete(self, task_id: str) -> bool:
        with self._repo._thread_lock:
            with self._repo._lock:
                tasks = self._repo._load()
                keep = [t for t in tasks if t.id != task_id]
                if len(keep) == len(tasks):
                    return False
                self._repo._save(keep)
        return True

    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        with self._repo._thread_lock:
            with self._repo._lock:
                tasks = self._repo._load()
                in_progress = [t for t in tasks if t.status == "in_progress"]
                if len(in_progress) >= max_in_progress:
                    return None
                terminal = {"done", "cancelled"}
                by_id = {t.id: t for t in tasks}

                def _is_runnable(task: Task) -> bool:
                    if task.status != "ready":
                        return False
                    if task.pending_gate:
                        return False
                    for dep_id in task.blocked_by:
                        dep = by_id.get(dep_id)
                        if dep is None or dep.status not in terminal:
                            return False
                    return True

                runnable = [t for t in tasks if _is_runnable(t)]
                runnable.sort(key=lambda t: (_priority_rank(t.priority), t.retry_count, t.created_at))
                if not runnable:
                    return None
                selected = runnable[0]
                for idx, task in enumerate(tasks):
                    if task.id == selected.id:
                        selected.status = "in_progress"
                        selected.updated_at = now_iso()
                        tasks[idx] = selected
                        self._repo._save(tasks)
                        return selected
        return None


class FileRunRepository(RunRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._repo = _YamlCollectionRepo[RunRecord](
            path,
            lock_path,
            "runs",
            loader=RunRecord.from_dict,
            dumper=lambda r: r.to_dict(),
        )

    def list(self) -> list[RunRecord]:
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def upsert(self, run: RunRecord) -> RunRecord:
        with self._repo._thread_lock:
            with self._repo._lock:
                runs = self._repo._load()
                for idx, existing in enumerate(runs):
                    if existing.id == run.id:
                        runs[idx] = run
                        self._repo._save(runs)
                        return run
                runs.append(run)
                self._repo._save(runs)
        return run


class FileReviewRepository(ReviewRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._repo = _YamlCollectionRepo[ReviewCycle](
            path,
            lock_path,
            "review_cycles",
            loader=ReviewCycle.from_dict,
            dumper=lambda c: c.to_dict(),
        )

    def list(self) -> list[ReviewCycle]:
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def for_task(self, task_id: str) -> list[ReviewCycle]:
        return [cycle for cycle in self.list() if cycle.task_id == task_id]

    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        with self._repo._thread_lock:
            with self._repo._lock:
                cycles = self._repo._load()
                cycles.append(cycle)
                self._repo._save(cycles)
        return cycle


class FileAgentRepository(AgentRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._repo = _YamlCollectionRepo[AgentRecord](
            path,
            lock_path,
            "agents",
            loader=AgentRecord.from_dict,
            dumper=lambda a: a.to_dict(),
        )

    def list(self) -> list[AgentRecord]:
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        for agent in self.list():
            if agent.id == agent_id:
                return agent
        return None

    def upsert(self, agent: AgentRecord) -> AgentRecord:
        with self._repo._thread_lock:
            with self._repo._lock:
                agents = self._repo._load()
                for idx, existing in enumerate(agents):
                    if existing.id == agent.id:
                        agents[idx] = agent
                        self._repo._save(agents)
                        return agent
                agents.append(agent)
                self._repo._save(agents)
        return agent

    def delete(self, agent_id: str) -> bool:
        with self._repo._thread_lock:
            with self._repo._lock:
                agents = self._repo._load()
                filtered = [agent for agent in agents if agent.id != agent_id]
                if len(filtered) == len(agents):
                    return False
                self._repo._save(filtered)
                return True


class FileQuickActionRepository(QuickActionRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._repo = _YamlCollectionRepo[QuickActionRun](
            path,
            lock_path,
            "quick_actions",
            loader=QuickActionRun.from_dict,
            dumper=lambda q: q.to_dict(),
        )

    def list(self) -> list[QuickActionRun]:
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, quick_action_id: str) -> Optional[QuickActionRun]:
        for run in self.list():
            if run.id == quick_action_id:
                return run
        return None

    def upsert(self, quick_action: QuickActionRun) -> QuickActionRun:
        with self._repo._thread_lock:
            with self._repo._lock:
                runs = self._repo._load()
                for idx, existing in enumerate(runs):
                    if existing.id == quick_action.id:
                        # Preserve promotion linkage across async status updates.
                        if existing.promoted_task_id and not quick_action.promoted_task_id:
                            quick_action.promoted_task_id = existing.promoted_task_id
                        runs[idx] = quick_action
                        self._repo._save(runs)
                        return quick_action
                runs.append(quick_action)
                self._repo._save(runs)
        return quick_action


class FileEventRepository(EventRepository):
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._path = path
        self._lock = FileLock(lock_path)
        self._thread_lock = threading.RLock()

    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        event = {
            "id": f"evt-{uuid.uuid4().hex[:10]}",
            "ts": now_iso(),
            "channel": channel,
            "type": event_type,
            "entity_id": entity_id,
            "payload": payload,
            "project_id": project_id,
        }
        with self._thread_lock:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return event

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0 or not self._path.exists():
            return []
        with self._thread_lock:
            with self._lock:
                with self._path.open("r", encoding="utf-8") as handle:
                    selected = list(deque(handle, maxlen=limit))
        events: list[dict[str, Any]] = []
        for line in selected:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events


class FileConfigRepository:
    def __init__(self, path: Path, lock_path: Path) -> None:
        self._path = path
        self._lock = FileLock(lock_path)
        self._thread_lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        _require_yaml()
        with self._thread_lock:
            with self._lock:
                if not self._path.exists():
                    return {}
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
                return raw if isinstance(raw, dict) else {}

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        _require_yaml()
        with self._thread_lock:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
                with tmp_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self._path)
        return config
