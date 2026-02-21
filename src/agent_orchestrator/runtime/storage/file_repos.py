"""File-backed repository implementations for runtime state."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Generic, List, Optional, TypeVar

from ...io_utils import FileLock
from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, ReviewCycle, RunRecord, Task, TerminalSession, now_iso
from .interfaces import (
    AgentRepository,
    EventRepository,
    PlanRefineJobRepository,
    PlanRevisionRepository,
    TerminalSessionRepository,
    ReviewRepository,
    RunRepository,
    TaskRepository,
)

try:
    import yaml  # type: ignore[import-untyped]
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
        """Initialize the YamlCollectionRepo.

        Args:
            path (Path): YAML file path containing this repository collection.
            lock_path (Path): Lock file path used for cross-process synchronization.
            key (str): Top-level YAML key that stores serialized collection items.
            loader (Callable[[dict[str, Any]], T]): Callable converting raw dictionaries
                into domain models.
            dumper (Callable[[T], dict[str, Any]]): Callable converting domain models
                into dictionaries for persistence.
        """
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
    """YAML-backed task repository with coarse file/process locking."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileTaskRepository.

        Args:
            path (Path): YAML file path for task records.
            lock_path (Path): Lock file path used while mutating task data.
        """
        self._repo = _YamlCollectionRepo[Task](
            path,
            lock_path,
            "tasks",
            loader=Task.from_dict,
            dumper=lambda t: t.to_dict(),
        )

    def list(self) -> list[Task]:
        """Load all persisted tasks.

        Returns:
            list[Task]: All persisted task records.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, task_id: str) -> Optional[Task]:
        """Fetch a single task by identifier.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            Optional[Task]: Requested value when available; otherwise `None`.
        """
        for task in self.list():
            if task.id == task_id:
                return task
        return None

    def upsert(self, task: Task) -> Task:
        """Insert or update a task and refresh timestamps.

        Args:
            task (Task): Task model to insert or replace by id.

        Returns:
            Task: Persisted task record after timestamps are refreshed.
        """
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
        """Delete a task by id.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                tasks = self._repo._load()
                keep = [t for t in tasks if t.id != task_id]
                if len(keep) == len(tasks):
                    return False
                self._repo._save(keep)
        return True

    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        """Select and atomically mark the next runnable queued task.

        Args:
            max_in_progress (int): Maximum number of tasks allowed in `in_progress`.

        Returns:
            Optional[Task]: Claimed runnable task, or `None` when none can be claimed.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                tasks = self._repo._load()
                in_progress = [t for t in tasks if t.status == "in_progress"]
                if len(in_progress) >= max_in_progress:
                    return None
                terminal = {"done", "cancelled"}
                by_id = {t.id: t for t in tasks}

                def _is_runnable(task: Task) -> bool:
                    if task.status != "queued":
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
    """YAML-backed repository for run execution records."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileRunRepository.

        Args:
            path (Path): YAML file path for run records.
            lock_path (Path): Lock file path used while mutating run data.
        """
        self._repo = _YamlCollectionRepo[RunRecord](
            path,
            lock_path,
            "runs",
            loader=RunRecord.from_dict,
            dumper=lambda r: r.to_dict(),
        )

    def list(self) -> list[RunRecord]:
        """Load all runs.

        Returns:
            list[RunRecord]: All persisted run records.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a single run by identifier.

        Args:
            run_id (str): Identifier for the target run.

        Returns:
            Optional[RunRecord]: Requested value when available; otherwise `None`.
        """
        for run in self.list():
            if run.id == run_id:
                return run
        return None

    def upsert(self, run: RunRecord) -> RunRecord:
        """Insert or update a run record.

        Args:
            run (RunRecord): Run model to insert or replace by id.

        Returns:
            RunRecord: Persisted run record after the write operation.
        """
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
    """YAML-backed repository for review cycle history."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileReviewRepository.

        Args:
            path (Path): YAML file path for review cycle records.
            lock_path (Path): Lock file path used while mutating review data.
        """
        self._repo = _YamlCollectionRepo[ReviewCycle](
            path,
            lock_path,
            "review_cycles",
            loader=ReviewCycle.from_dict,
            dumper=lambda c: c.to_dict(),
        )

    def list(self) -> list[ReviewCycle]:
        """Load all review cycles.

        Returns:
            list[ReviewCycle]: All persisted review cycles.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def for_task(self, task_id: str) -> List[ReviewCycle]:
        """List review cycles associated with one task id.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[ReviewCycle]: Review cycles associated with ``task_id``.
        """
        return [cycle for cycle in self.list() if cycle.task_id == task_id]

    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        """Append a new review cycle entry.

        Args:
            cycle (ReviewCycle): Review cycle record to append.

        Returns:
            ReviewCycle: Persisted review cycle record.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                cycles = self._repo._load()
                cycles.append(cycle)
                self._repo._save(cycles)
        return cycle


class FileAgentRepository(AgentRepository):
    """YAML-backed repository for active and historical agents."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileAgentRepository.

        Args:
            path (Path): YAML file path for agent records.
            lock_path (Path): Lock file path used while mutating agent data.
        """
        self._repo = _YamlCollectionRepo[AgentRecord](
            path,
            lock_path,
            "agents",
            loader=AgentRecord.from_dict,
            dumper=lambda a: a.to_dict(),
        )

    def list(self) -> list[AgentRecord]:
        """Load all agents.

        Returns:
            list[AgentRecord]: All persisted agent records.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Fetch a single agent by identifier.

        Args:
            agent_id (str): Identifier for the target agent.

        Returns:
            Optional[AgentRecord]: Requested value when available; otherwise `None`.
        """
        for agent in self.list():
            if agent.id == agent_id:
                return agent
        return None

    def upsert(self, agent: AgentRecord) -> AgentRecord:
        """Insert or update an agent record.

        Args:
            agent (AgentRecord): Agent record to insert or replace by id.

        Returns:
            AgentRecord: Persisted agent record after the write operation.
        """
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
        """Delete an agent record by id.

        Args:
            agent_id (str): Identifier for the target agent.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                agents = self._repo._load()
                filtered = [agent for agent in agents if agent.id != agent_id]
                if len(filtered) == len(agents):
                    return False
                self._repo._save(filtered)
                return True


class FileTerminalSessionRepository(TerminalSessionRepository):
    """YAML-backed repository for terminal session metadata."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileTerminalSessionRepository.

        Args:
            path (Path): YAML file path for terminal session records.
            lock_path (Path): Lock file path used while mutating session data.
        """
        self._repo = _YamlCollectionRepo[TerminalSession](
            path,
            lock_path,
            "terminal_sessions",
            loader=TerminalSession.from_dict,
            dumper=lambda q: q.to_dict(),
        )

    def list(self) -> list[TerminalSession]:
        """Load all terminal sessions.

        Returns:
            list[TerminalSession]: All persisted terminal session records.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                return self._repo._load()

    def get(self, session_id: str) -> Optional[TerminalSession]:
        """Fetch a single terminal session by identifier.

        Args:
            session_id (str): Identifier for the target session.

        Returns:
            Optional[TerminalSession]: Requested value when available; otherwise `None`.
        """
        for run in self.list():
            if run.id == session_id:
                return run
        return None

    def upsert(self, session: TerminalSession) -> TerminalSession:
        """Insert or update a terminal session record.

        Args:
            session (TerminalSession): Terminal session record to insert or replace by id.

        Returns:
            TerminalSession: Persisted terminal session record after the write operation.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                runs = self._repo._load()
                for idx, existing in enumerate(runs):
                    if existing.id == session.id:
                        runs[idx] = session
                        self._repo._save(runs)
                        return session
                runs.append(session)
                self._repo._save(runs)
        return session


class FileEventRepository(EventRepository):
    """JSONL-backed event stream repository."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileEventRepository.

        Args:
            path (Path): JSONL file path where event envelopes are appended.
            lock_path (Path): Lock file path used while writing or reading events.
        """
        self._path = path
        self._lock = FileLock(lock_path)
        self._thread_lock = threading.RLock()

    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        """Append one event envelope to the JSONL stream.

        Args:
            channel (str): Channel namespace for the event stream.
            event_type (str): Specific event type emitted in the channel.
            entity_id (str): Identifier for the related entity.
            payload (dict[str, Any]): JSON-serializable event payload body.
            project_id (str): Identifier for the related project.

        Returns:
            dict[str, Any]: Persisted event envelope including generated id and timestamp.
        """
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
        """Read the newest events up to ``limit``.

        Args:
            limit (int): Maximum number of newest events to return.

        Returns:
            list[dict[str, Any]]: Parsed event envelopes from the tail of the stream.
        """
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


class FilePlanRevisionRepository(PlanRevisionRepository):
    """YAML-backed repository for plan revisions."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FilePlanRevisionRepository.

        Args:
            path (Path): YAML file path for plan revision records.
            lock_path (Path): Lock file path used while mutating revision data.
        """
        self._repo = _YamlCollectionRepo[PlanRevision](
            path,
            lock_path,
            "plan_revisions",
            loader=PlanRevision.from_dict,
            dumper=lambda item: item.to_dict(),
        )

    def list(self) -> list[PlanRevision]:
        """Load and return revisions sorted by creation time.

        Returns:
            list[PlanRevision]: Persisted revisions sorted by creation timestamp.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                items = self._repo._load()
        return sorted(items, key=lambda item: item.created_at)

    def for_task(self, task_id: str) -> List[PlanRevision]:
        """List revisions that belong to a given task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[PlanRevision]: Plan revisions associated with ``task_id``.
        """
        return [item for item in self.list() if item.task_id == task_id]

    def get(self, revision_id: str) -> Optional[PlanRevision]:
        """Fetch a single revision by identifier.

        Args:
            revision_id (str): Identifier for the target revision.

        Returns:
            Optional[PlanRevision]: Requested value when available; otherwise `None`.
        """
        for item in self.list():
            if item.id == revision_id:
                return item
        return None

    def upsert(self, revision: PlanRevision) -> PlanRevision:
        """Insert or update a revision.

        Args:
            revision (PlanRevision): Plan revision record to insert or replace by id.

        Returns:
            PlanRevision: Persisted plan revision record after the write operation.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                items = self._repo._load()
                for idx, existing in enumerate(items):
                    if existing.id == revision.id:
                        items[idx] = revision
                        self._repo._save(items)
                        return revision
                items.append(revision)
                self._repo._save(items)
                return revision


class FilePlanRefineJobRepository(PlanRefineJobRepository):
    """YAML-backed repository for plan-refine job state."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FilePlanRefineJobRepository.

        Args:
            path (Path): YAML file path for plan-refine job records.
            lock_path (Path): Lock file path used while mutating job data.
        """
        self._repo = _YamlCollectionRepo[PlanRefineJob](
            path,
            lock_path,
            "plan_refine_jobs",
            loader=PlanRefineJob.from_dict,
            dumper=lambda item: item.to_dict(),
        )

    def list(self) -> list[PlanRefineJob]:
        """Load and return jobs newest-first.

        Returns:
            list[PlanRefineJob]: Persisted jobs sorted newest-first by creation time.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                items = self._repo._load()
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def for_task(self, task_id: str) -> List[PlanRefineJob]:
        """List plan-refine jobs that belong to a given task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[PlanRefineJob]: Plan-refine jobs associated with ``task_id``.
        """
        return [item for item in self.list() if item.task_id == task_id]

    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        """Fetch a single plan-refine job by identifier.

        Args:
            job_id (str): Identifier for the target job.

        Returns:
            Optional[PlanRefineJob]: Requested value when available; otherwise `None`.
        """
        for item in self.list():
            if item.id == job_id:
                return item
        return None

    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        """Insert or update a refine job.

        Args:
            job (PlanRefineJob): Plan-refine job record to insert or replace by id.

        Returns:
            PlanRefineJob: Persisted plan-refine job record after the write operation.
        """
        with self._repo._thread_lock:
            with self._repo._lock:
                items = self._repo._load()
                for idx, existing in enumerate(items):
                    if existing.id == job.id:
                        items[idx] = job
                        self._repo._save(items)
                        return job
                items.append(job)
                self._repo._save(items)
                return job


class FileConfigRepository:
    """YAML-backed repository for runtime configuration."""
    def __init__(self, path: Path, lock_path: Path) -> None:
        """Initialize the FileConfigRepository.

        Args:
            path (Path): YAML file path for runtime configuration.
            lock_path (Path): Lock file path used while reading or writing config.
        """
        self._path = path
        self._lock = FileLock(lock_path)
        self._thread_lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        """Load configuration from disk.

        Returns:
            dict[str, Any]: Configuration mapping from disk, or an empty mapping.
        """
        _require_yaml()
        with self._thread_lock:
            with self._lock:
                if not self._path.exists():
                    return {}
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
                return raw if isinstance(raw, dict) else {}

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        """Persist configuration to disk atomically.

        Args:
            config (dict[str, Any]): Configuration mapping to persist.

        Returns:
            dict[str, Any]: Saved configuration mapping.
        """
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
