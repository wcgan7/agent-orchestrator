"""SQLite database manager for runtime state persistence."""
# ruff: noqa: D102,D107,D202

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

from ..domain.models import now_iso


class SQLiteDB:
    """Own the SQLite connection lifecycle, schema, and transactional helpers."""

    SCHEMA_VERSION = 4

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._active_connection: ContextVar[sqlite3.Connection | None] = ContextVar(
            f"sqlite_runtime_conn_{id(self)}",
            default=None,
        )
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _connection(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        active = self._active_connection.get()
        if active is not None:
            yield active
            return

        with self._thread_lock:
            conn = self._connect()
            try:
                if write:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
                if write:
                    conn.execute("COMMIT")
            except Exception:
                if write:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Open a write transaction that can span multiple repository operations."""

        active = self._active_connection.get()
        if active is not None:
            yield active
            return

        with self._thread_lock:
            conn = self._connect()
            token = self._active_connection.set(conn)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._active_connection.reset(token)
                conn.close()

    def fetch_one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        with self._connection(write=False) as conn:
            row = conn.execute(sql, params).fetchone()
            return row if isinstance(row, sqlite3.Row) else None

    def fetch_all(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._connection(write=False) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [row for row in rows if isinstance(row, sqlite3.Row)]

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        with self._connection(write=True) as conn:
            conn.execute(sql, params)

    def execute_many(self, sql: str, params_list: list[tuple[object, ...]]) -> None:
        if not params_list:
            return
        with self._connection(write=True) as conn:
            conn.executemany(sql, params_list)

    def load_orchestrator_state(self) -> dict[str, Any]:
        """Load persisted orchestrator heartbeat state from the dedicated table."""

        rows = self.fetch_all("SELECT key, value FROM orchestrator_state")
        state: dict[str, Any] = {}
        for row in rows:
            key = str(row["key"] or "").strip()
            if not key:
                continue
            raw_value = str(row["value"] or "")
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = raw_value
            state[key] = parsed
        return state

    def save_orchestrator_state(self, state: dict[str, Any]) -> None:
        """Replace orchestrator heartbeat state rows atomically."""

        with self.transaction() as conn:
            conn.execute("DELETE FROM orchestrator_state")
            for key, value in state.items():
                normalized_key = str(key or "").strip()
                if not normalized_key:
                    continue
                conn.execute(
                    "INSERT INTO orchestrator_state(key, value) VALUES(?, ?)",
                    (normalized_key, json.dumps(value, separators=(",", ":"), sort_keys=False)),
                )

    def load_execution_lease(self, task_id: str) -> dict[str, Any] | None:
        """Load one task execution lease payload from the dedicated lease table."""

        row = self.fetch_one("SELECT payload FROM execution_leases WHERE task_id = ?", (task_id,))
        if row is None:
            return None
        raw_payload = str(row["payload"] or "")
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def save_execution_lease(self, task_id: str, lease: dict[str, Any]) -> None:
        """Upsert one task execution lease into the dedicated lease table."""

        owner = str(lease.get("owner") or "orchestrator")
        heartbeat_at = str(lease.get("heartbeat_at") or "")
        expires_at = str(lease.get("expires_at") or "")
        payload = json.dumps(lease, separators=(",", ":"), sort_keys=False)
        self.execute(
            """
            INSERT INTO execution_leases(task_id, owner, heartbeat_at, expires_at, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                owner = excluded.owner,
                heartbeat_at = excluded.heartbeat_at,
                expires_at = excluded.expires_at,
                payload = excluded.payload
            """,
            (task_id, owner, heartbeat_at, expires_at, payload),
        )

    def delete_execution_lease(self, task_id: str) -> bool:
        """Delete one task execution lease by task id."""

        with self.transaction() as conn:
            row = conn.execute("DELETE FROM execution_leases WHERE task_id = ?", (task_id,))
            return bool(row.rowcount)

    def verify_integrity(self) -> None:
        """Fail fast when SQLite integrity check reports corruption."""

        with self._connection(write=False) as conn:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        messages = [str(row[0]) for row in rows if row and row[0] is not None]
        if messages == ["ok"]:
            return
        preview = "; ".join(messages[:3]) if messages else "unknown integrity error"
        raise RuntimeError(f"SQLite integrity check failed: {preview}")

    def _ensure_schema(self) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = conn.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
            current_version = int(row["version"] or 0) if row else 0

            self._apply_schema_v4(conn)

            if current_version < self.SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                    (self.SCHEMA_VERSION, now_iso()),
                )

    @staticmethod
    def _apply_schema_v4(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                pending_gate TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_updated ON tasks(status, updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_pending_gate ON tasks(pending_gate)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_task_id_created ON reviews(task_id, created_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                ts TEXT NOT NULL,
                channel TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts_seq ON events(ts, seq)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_project_channel ON events(project_id, channel)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terminal_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_terminal_sessions_project_started ON terminal_sessions(project_id, started_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_revisions (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_revisions_task_created ON plan_revisions(task_id, created_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_refine_jobs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_refine_jobs_task_created ON plan_refine_jobs(task_id, created_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config_kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orchestrator_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_leases (
                task_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
