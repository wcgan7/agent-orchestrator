"""SQLite-backed repository implementations for runtime state."""
# ruff: noqa: D102,D107

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, List, Optional

from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, ReviewCycle, RunRecord, Task, TerminalSession, now_iso
from .interfaces import (
    AgentRepository,
    EventRepository,
    PlanRefineJobRepository,
    PlanRevisionRepository,
    ReviewRepository,
    RunRepository,
    TaskRepository,
    TerminalSessionRepository,
)
from .sqlite_db import SQLiteDB


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def _json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 99)


class SqliteTaskRepository(TaskRepository):
    """SQLite-backed task repository with transactional runnable claims."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[Task]:
        rows = self._db.fetch_all("SELECT payload FROM tasks ORDER BY created_at ASC, id ASC")
        return [Task.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, task_id: str) -> Optional[Task]:
        row = self._db.fetch_one("SELECT payload FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            return None
        return Task.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, task: Task) -> Task:
        with self._db.transaction() as conn:
            existing = conn.execute("SELECT created_at FROM tasks WHERE id = ?", (task.id,)).fetchone()
            if existing is not None:
                if not str(task.created_at or "").strip():
                    task.created_at = str(existing["created_at"] or now_iso())
            else:
                task.created_at = task.created_at or now_iso()
            task.updated_at = now_iso()
            payload = task.to_dict()
            conn.execute(
                """
                INSERT INTO tasks(id, status, priority, retry_count, created_at, updated_at, pending_gate, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    priority = excluded.priority,
                    retry_count = excluded.retry_count,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    pending_gate = excluded.pending_gate,
                    payload = excluded.payload
                """,
                (
                    task.id,
                    task.status,
                    task.priority,
                    int(task.retry_count),
                    task.created_at,
                    task.updated_at,
                    task.pending_gate,
                    _json_dumps(payload),
                ),
            )
        return task

    def delete(self, task_id: str) -> bool:
        with self._db.transaction() as conn:
            row = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return bool(row.rowcount)

    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        with self._db.transaction() as conn:
            rows = conn.execute("SELECT payload FROM tasks").fetchall()
            tasks = [Task.from_dict(_json_loads(str(row["payload"]))) for row in rows]

            def _resume_requested(task: Task) -> bool:
                if task.status != "in_progress" or task.pending_gate:
                    return False
                if not isinstance(task.metadata, dict):
                    return False
                checkpoint = task.metadata.get("execution_checkpoint")
                if not isinstance(checkpoint, dict):
                    return False
                return bool(str(checkpoint.get("resume_requested_at") or "").strip())

            in_progress = [
                task
                for task in tasks
                if task.status == "in_progress" and not task.pending_gate and not _resume_requested(task)
            ]
            if len(in_progress) >= max_in_progress:
                return None

            terminal = {"done", "cancelled"}
            by_id = {task.id: task for task in tasks}

            def _is_runnable(task: Task) -> bool:
                if task.status not in {"queued", "in_progress"}:
                    return False
                if task.status == "queued" and task.pending_gate:
                    return False
                if task.status == "in_progress" and not _resume_requested(task):
                    return False
                for dep_id in task.blocked_by:
                    dep = by_id.get(dep_id)
                    if dep is None or dep.status not in terminal:
                        return False
                return True

            runnable = [task for task in tasks if _is_runnable(task)]
            runnable.sort(key=lambda task: (_priority_rank(task.priority), task.retry_count, task.created_at))
            if not runnable:
                return None

            selected = runnable[0]
            selected.status = "in_progress"
            if isinstance(selected.metadata, dict):
                checkpoint = selected.metadata.get("execution_checkpoint")
                if isinstance(checkpoint, dict):
                    checkpoint_copy = dict(checkpoint)
                    checkpoint_copy.pop("resume_requested_at", None)
                    selected.metadata["execution_checkpoint"] = checkpoint_copy
            selected.updated_at = now_iso()

            conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    updated_at = ?,
                    pending_gate = ?,
                    retry_count = ?,
                    priority = ?,
                    payload = ?
                WHERE id = ?
                """,
                (
                    selected.status,
                    selected.updated_at,
                    selected.pending_gate,
                    int(selected.retry_count),
                    selected.priority,
                    _json_dumps(selected.to_dict()),
                    selected.id,
                ),
            )
            return selected


class SqliteRunRepository(RunRepository):
    """SQLite-backed run record repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[RunRecord]:
        rows = self._db.fetch_all("SELECT payload FROM runs ORDER BY COALESCE(started_at, ''), id")
        return [RunRecord.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, run_id: str) -> Optional[RunRecord]:
        row = self._db.fetch_one("SELECT payload FROM runs WHERE id = ?", (run_id,))
        if row is None:
            return None
        return RunRecord.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, run: RunRecord) -> RunRecord:
        payload = run.to_dict()
        self._db.execute(
            """
            INSERT INTO runs(id, task_id, status, started_at, finished_at, payload)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                payload = excluded.payload
            """,
            (
                run.id,
                run.task_id,
                run.status,
                run.started_at,
                run.finished_at,
                _json_dumps(payload),
            ),
        )
        return run


class SqliteReviewRepository(ReviewRepository):
    """SQLite-backed review cycle repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[ReviewCycle]:
        rows = self._db.fetch_all("SELECT payload FROM reviews ORDER BY created_at ASC, id ASC")
        return [ReviewCycle.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def for_task(self, task_id: str) -> List[ReviewCycle]:
        rows = self._db.fetch_all(
            "SELECT payload FROM reviews WHERE task_id = ? ORDER BY created_at ASC, id ASC",
            (task_id,),
        )
        return [ReviewCycle.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        payload = cycle.to_dict()
        self._db.execute(
            """
            INSERT INTO reviews(id, task_id, created_at, payload)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                created_at = excluded.created_at,
                payload = excluded.payload
            """,
            (
                cycle.id,
                cycle.task_id,
                cycle.created_at,
                _json_dumps(payload),
            ),
        )
        return cycle


class SqliteAgentRepository(AgentRepository):
    """SQLite-backed agent metadata repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[AgentRecord]:
        rows = self._db.fetch_all("SELECT payload FROM agents ORDER BY last_seen_at ASC, id ASC")
        return [AgentRecord.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        row = self._db.fetch_one("SELECT payload FROM agents WHERE id = ?", (agent_id,))
        if row is None:
            return None
        return AgentRecord.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, agent: AgentRecord) -> AgentRecord:
        payload = agent.to_dict()
        self._db.execute(
            """
            INSERT INTO agents(id, status, last_seen_at, payload)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                last_seen_at = excluded.last_seen_at,
                payload = excluded.payload
            """,
            (
                agent.id,
                agent.status,
                agent.last_seen_at,
                _json_dumps(payload),
            ),
        )
        return agent

    def delete(self, agent_id: str) -> bool:
        with self._db.transaction() as conn:
            row = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            return bool(row.rowcount)


class SqliteTerminalSessionRepository(TerminalSessionRepository):
    """SQLite-backed terminal session repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[TerminalSession]:
        rows = self._db.fetch_all(
            "SELECT payload FROM terminal_sessions ORDER BY COALESCE(started_at, ''), id"
        )
        return [TerminalSession.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, session_id: str) -> Optional[TerminalSession]:
        row = self._db.fetch_one("SELECT payload FROM terminal_sessions WHERE id = ?", (session_id,))
        if row is None:
            return None
        return TerminalSession.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, session: TerminalSession) -> TerminalSession:
        payload = session.to_dict()
        self._db.execute(
            """
            INSERT INTO terminal_sessions(id, project_id, status, started_at, finished_at, payload)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                payload = excluded.payload
            """,
            (
                session.id,
                session.project_id,
                session.status,
                session.started_at,
                session.finished_at,
                _json_dumps(payload),
            ),
        )
        return session


class SqliteEventRepository(EventRepository):
    """SQLite-backed event repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def append(
        self,
        *,
        channel: str,
        event_type: str,
        entity_id: str,
        payload: dict[str, Any],
        project_id: str,
    ) -> dict[str, Any]:
        event_id = f"evt-{uuid.uuid4().hex[:10]}"
        event_ts = now_iso()
        event_payload = dict(payload)
        event = {
            "id": event_id,
            "ts": event_ts,
            "channel": channel,
            "type": event_type,
            "entity_id": entity_id,
            "payload": event_payload,
            "project_id": project_id,
        }
        self._db.execute(
            """
            INSERT INTO events(event_id, ts, channel, event_type, entity_id, project_id, payload)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_ts,
                channel,
                event_type,
                entity_id,
                project_id,
                _json_dumps(event_payload),
            ),
        )
        return event

    def list_recent(self, limit: int = 100) -> List[dict[str, Any]]:
        if limit <= 0:
            return []
        rows = self._db.fetch_all(
            """
            SELECT event_id, ts, channel, event_type, entity_id, project_id, payload
            FROM events
            ORDER BY seq DESC
            LIMIT ?
            """,
            (limit,),
        )
        newest_first = []
        for row in rows:
            newest_first.append(
                {
                    "id": str(row["event_id"]),
                    "ts": str(row["ts"]),
                    "channel": str(row["channel"]),
                    "type": str(row["event_type"]),
                    "entity_id": str(row["entity_id"]),
                    "project_id": str(row["project_id"]),
                    "payload": _json_loads(str(row["payload"])),
                }
            )
        return list(reversed(newest_first))


class SqlitePlanRevisionRepository(PlanRevisionRepository):
    """SQLite-backed plan revision repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[PlanRevision]:
        rows = self._db.fetch_all("SELECT payload FROM plan_revisions ORDER BY created_at ASC, id ASC")
        return [PlanRevision.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def for_task(self, task_id: str) -> List[PlanRevision]:
        rows = self._db.fetch_all(
            "SELECT payload FROM plan_revisions WHERE task_id = ? ORDER BY created_at ASC, id ASC",
            (task_id,),
        )
        return [PlanRevision.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, revision_id: str) -> Optional[PlanRevision]:
        row = self._db.fetch_one("SELECT payload FROM plan_revisions WHERE id = ?", (revision_id,))
        if row is None:
            return None
        return PlanRevision.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, revision: PlanRevision) -> PlanRevision:
        payload = revision.to_dict()
        self._db.execute(
            """
            INSERT INTO plan_revisions(id, task_id, created_at, status, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                created_at = excluded.created_at,
                status = excluded.status,
                payload = excluded.payload
            """,
            (
                revision.id,
                revision.task_id,
                revision.created_at,
                revision.status,
                _json_dumps(payload),
            ),
        )
        return revision


class SqlitePlanRefineJobRepository(PlanRefineJobRepository):
    """SQLite-backed plan-refine job repository."""

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def list(self) -> List[PlanRefineJob]:
        rows = self._db.fetch_all("SELECT payload FROM plan_refine_jobs ORDER BY created_at DESC, id DESC")
        return [PlanRefineJob.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def for_task(self, task_id: str) -> List[PlanRefineJob]:
        rows = self._db.fetch_all(
            "SELECT payload FROM plan_refine_jobs WHERE task_id = ? ORDER BY created_at DESC, id DESC",
            (task_id,),
        )
        return [PlanRefineJob.from_dict(_json_loads(str(row["payload"]))) for row in rows]

    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        row = self._db.fetch_one("SELECT payload FROM plan_refine_jobs WHERE id = ?", (job_id,))
        if row is None:
            return None
        return PlanRefineJob.from_dict(_json_loads(str(row["payload"])))

    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        payload = job.to_dict()
        self._db.execute(
            """
            INSERT INTO plan_refine_jobs(id, task_id, created_at, status, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                created_at = excluded.created_at,
                status = excluded.status,
                payload = excluded.payload
            """,
            (
                job.id,
                job.task_id,
                job.created_at,
                job.status,
                _json_dumps(payload),
            ),
        )
        return job


class SqliteConfigRepository:
    """SQLite-backed runtime settings repository."""

    _CONFIG_KEY = "config"

    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def load(self) -> dict[str, Any]:
        row = self._db.fetch_one("SELECT value FROM config_kv WHERE key = ?", (self._CONFIG_KEY,))
        if row is None:
            return {}
        try:
            parsed = json.loads(str(row["value"]))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        self._db.execute(
            """
            INSERT INTO config_kv(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (self._CONFIG_KEY, _json_dumps(config)),
        )
        return config

    def update(self, updater: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        with self._db.transaction() as conn:
            row = conn.execute("SELECT value FROM config_kv WHERE key = ?", (self._CONFIG_KEY,)).fetchone()
            if row is None:
                current: dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(str(row["value"]))
                except json.JSONDecodeError:
                    parsed = {}
                current = parsed if isinstance(parsed, dict) else {}
            updated = updater(dict(current))
            conn.execute(
                """
                INSERT INTO config_kv(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (self._CONFIG_KEY, _json_dumps(updated)),
            )
            return updated
