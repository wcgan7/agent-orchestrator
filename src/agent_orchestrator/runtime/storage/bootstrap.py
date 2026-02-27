"""Bootstrap helpers for runtime state storage layout."""

from __future__ import annotations

from collections import Counter
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .file_repos import (
    FileAgentRepository,
    FileConfigRepository,
    FilePlanRefineJobRepository,
    FilePlanRevisionRepository,
    FileReviewRepository,
    FileRunRepository,
    FileTaskRepository,
    FileTerminalSessionRepository,
)
from .sqlite_db import SQLiteDB
from .sqlite_repos import SqliteConfigRepository


LEGACY_STATE_FILES = {
    "tasks": "tasks.yaml",
    "runs": "runs.yaml",
    "review_cycles": "review_cycles.yaml",
    "agents": "agents.yaml",
    "terminal_sessions": "terminal_sessions.yaml",
    "plan_revisions": "plan_revisions.yaml",
    "plan_refine_jobs": "plan_refine_jobs.yaml",
    "events": "events.jsonl",
    "config": "config.yaml",
}
LEGACY_LOCK_FILES = {
    "tasks.lock",
    "runs.lock",
    "review_cycles.lock",
    "agents.lock",
    "terminal_sessions.lock",
    "plan_revisions.lock",
    "plan_refine_jobs.lock",
    "events.lock",
    "config.lock",
}
SQLITE_DB_FILE = "runtime.db"
ARCHIVE_DIR_NAME = ".agent_orchestrator_archive"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _next_archive_path(root: Path, prefix: str) -> Path:
    stamp = _utc_stamp()
    candidate = root / f"{prefix}_{stamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{prefix}_{stamp}_{suffix}"
    return candidate


def _legacy_yaml_state_exists(state_root: Path) -> bool:
    return any((state_root / file_name).exists() for file_name in LEGACY_STATE_FILES.values())


def _archive_non_runtime_state(project_dir: Path, state_root: Path) -> None:
    archive_target = _next_archive_path(project_dir, ".agent_orchestrator_legacy")
    state_root.rename(archive_target)


def _create_migration_snapshot(project_dir: Path, state_root: Path) -> Path:
    archive_root = project_dir / ARCHIVE_DIR_NAME
    archive_root.mkdir(parents=True, exist_ok=True)
    snapshot = _next_archive_path(archive_root, "state_migration")
    shutil.copytree(state_root, snapshot)
    return snapshot


def _cleanup_sqlite_artifacts(state_root: Path) -> None:
    db_path = state_root / SQLITE_DB_FILE
    for path in (
        db_path,
        state_root / f"{SQLITE_DB_FILE}-wal",
        state_root / f"{SQLITE_DB_FILE}-shm",
    ):
        if path.exists():
            path.unlink()


def _cleanup_legacy_runtime_files(state_root: Path) -> None:
    for file_name in LEGACY_STATE_FILES.values():
        path = state_root / file_name
        if path.exists():
            path.unlink()
    for lock_name in LEGACY_LOCK_FILES:
        path = state_root / lock_name
        if path.exists():
            path.unlink()


def _load_legacy_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events


def _migrate_yaml_state_to_sqlite(project_dir: Path, state_root: Path) -> None:
    _create_migration_snapshot(project_dir, state_root)

    tasks = FileTaskRepository(state_root / "tasks.yaml", state_root / "tasks.lock").list()
    runs = FileRunRepository(state_root / "runs.yaml", state_root / "runs.lock").list()
    reviews = FileReviewRepository(state_root / "review_cycles.yaml", state_root / "review_cycles.lock").list()
    agents = FileAgentRepository(state_root / "agents.yaml", state_root / "agents.lock").list()
    terminal_sessions = FileTerminalSessionRepository(
        state_root / "terminal_sessions.yaml",
        state_root / "terminal_sessions.lock",
    ).list()
    plan_revisions = FilePlanRevisionRepository(
        state_root / "plan_revisions.yaml",
        state_root / "plan_revisions.lock",
    ).list()
    plan_refine_jobs = FilePlanRefineJobRepository(
        state_root / "plan_refine_jobs.yaml",
        state_root / "plan_refine_jobs.lock",
    ).list()
    config = FileConfigRepository(state_root / "config.yaml", state_root / "config.lock").load()
    events = _load_legacy_events(state_root / "events.jsonl")

    db = SQLiteDB(state_root / SQLITE_DB_FILE)

    with db.transaction() as conn:
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM agents")
        conn.execute("DELETE FROM terminal_sessions")
        conn.execute("DELETE FROM plan_revisions")
        conn.execute("DELETE FROM plan_refine_jobs")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM execution_leases")
        conn.execute("DELETE FROM orchestrator_state")
        conn.execute("DELETE FROM config_kv")

        for task in tasks:
            raw_lease = task.metadata.get("execution_lease") if isinstance(task.metadata, dict) else None
            if isinstance(task.metadata, dict):
                task.metadata.pop("execution_lease", None)
            payload = task.to_dict()
            conn.execute(
                """
                INSERT INTO tasks(id, status, priority, retry_count, created_at, updated_at, pending_gate, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.status,
                    task.priority,
                    int(task.retry_count),
                    task.created_at,
                    task.updated_at,
                    task.pending_gate,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )
            if isinstance(raw_lease, dict):
                owner = str(raw_lease.get("owner") or "orchestrator")
                heartbeat_at = str(raw_lease.get("heartbeat_at") or task.updated_at)
                expires_at = str(raw_lease.get("expires_at") or task.updated_at)
                conn.execute(
                    """
                    INSERT INTO execution_leases(task_id, owner, heartbeat_at, expires_at, payload)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        owner = excluded.owner,
                        heartbeat_at = excluded.heartbeat_at,
                        expires_at = excluded.expires_at,
                        payload = excluded.payload
                    """,
                    (
                        task.id,
                        owner,
                        heartbeat_at,
                        expires_at,
                        json.dumps(raw_lease, separators=(",", ":"), sort_keys=False),
                    ),
                )

        for run in runs:
            payload = run.to_dict()
            conn.execute(
                """
                INSERT INTO runs(id, task_id, status, started_at, finished_at, payload)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.task_id,
                    run.status,
                    run.started_at,
                    run.finished_at,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for cycle in reviews:
            payload = cycle.to_dict()
            conn.execute(
                """
                INSERT INTO reviews(id, task_id, created_at, payload)
                VALUES(?, ?, ?, ?)
                """,
                (
                    cycle.id,
                    cycle.task_id,
                    cycle.created_at,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for agent in agents:
            payload = agent.to_dict()
            conn.execute(
                """
                INSERT INTO agents(id, status, last_seen_at, payload)
                VALUES(?, ?, ?, ?)
                """,
                (
                    agent.id,
                    agent.status,
                    agent.last_seen_at,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for session in terminal_sessions:
            payload = session.to_dict()
            conn.execute(
                """
                INSERT INTO terminal_sessions(id, project_id, status, started_at, finished_at, payload)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.project_id,
                    session.status,
                    session.started_at,
                    session.finished_at,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for revision in plan_revisions:
            payload = revision.to_dict()
            conn.execute(
                """
                INSERT INTO plan_revisions(id, task_id, created_at, status, payload)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    revision.id,
                    revision.task_id,
                    revision.created_at,
                    revision.status,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for job in plan_refine_jobs:
            payload = job.to_dict()
            conn.execute(
                """
                INSERT INTO plan_refine_jobs(id, task_id, created_at, status, payload)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.task_id,
                    job.created_at,
                    job.status,
                    json.dumps(payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        for event in events:
            event_id = str(event.get("id") or "")
            ts = str(event.get("ts") or "")
            channel = str(event.get("channel") or "")
            event_type = str(event.get("type") or "")
            entity_id = str(event.get("entity_id") or "")
            project_id = str(event.get("project_id") or project_dir.name)
            event_payload_raw = event.get("payload")
            event_payload = event_payload_raw if isinstance(event_payload_raw, dict) else {}
            if not event_id or not ts or not channel or not event_type or not entity_id:
                continue
            conn.execute(
                """
                INSERT INTO events(event_id, ts, channel, event_type, entity_id, project_id, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts,
                    channel,
                    event_type,
                    entity_id,
                    project_id,
                    json.dumps(event_payload, separators=(",", ":"), sort_keys=False),
                ),
            )

        migrated_config = dict(config) if isinstance(config, dict) else {}
        migrated_config["schema_version"] = 4
        migrated_config["storage_backend"] = "sqlite"
        conn.execute(
            "INSERT INTO config_kv(key, value) VALUES(?, ?)",
            (
                "config",
                json.dumps(migrated_config, separators=(",", ":"), sort_keys=False),
            ),
        )

        parity_specs = [
            ("tasks", "tasks", "id", [task.id for task in tasks]),
            ("runs", "runs", "id", [run.id for run in runs]),
            ("reviews", "reviews", "id", [cycle.id for cycle in reviews]),
            ("agents", "agents", "id", [agent.id for agent in agents]),
            ("terminal_sessions", "terminal_sessions", "id", [session.id for session in terminal_sessions]),
            ("plan_revisions", "plan_revisions", "id", [revision.id for revision in plan_revisions]),
            ("plan_refine_jobs", "plan_refine_jobs", "id", [job.id for job in plan_refine_jobs]),
        ]
        for label, table, id_column, source_ids in parity_specs:
            rows = conn.execute(f"SELECT {id_column} AS id FROM {table}").fetchall()
            dest_ids = [str(row["id"]) for row in rows]
            if Counter(dest_ids) != Counter(source_ids):
                raise RuntimeError(f"YAML -> SQLite migration parity failed for {label}")

        source_event_ids = [str(event.get("id") or "") for event in events if str(event.get("id") or "")]
        event_rows = conn.execute("SELECT event_id FROM events").fetchall()
        dest_event_ids = [str(row["event_id"]) for row in event_rows]
        if Counter(dest_event_ids) != Counter(source_event_ids):
            raise RuntimeError("YAML -> SQLite migration parity failed for events")

    _cleanup_legacy_runtime_files(state_root)


def _ensure_gitignored(project_dir: Path) -> None:
    """Add runtime files to project `.gitignore` if not already present."""
    gitignore = project_dir / ".gitignore"
    entries = [".agent_orchestrator/", f"{ARCHIVE_DIR_NAME}/", ".workdoc.md"]
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        existing_stripped = {line.strip() for line in content.splitlines()}
        missing = [entry for entry in entries if entry not in existing_stripped and entry.rstrip("/") not in existing_stripped]
        if not missing:
            return
        if content and not content.endswith("\n"):
            content += "\n"
        if "# Agent Orchestrator runtime data" not in content:
            content += "\n# Agent Orchestrator runtime data\n"
        for entry in missing:
            content += f"{entry}\n"
        gitignore.write_text(content, encoding="utf-8")
    else:
        lines = "# Agent Orchestrator runtime data\n"
        for entry in entries:
            lines += f"{entry}\n"
        gitignore.write_text(lines, encoding="utf-8")


def ensure_state_root(project_dir: Path) -> Path:
    """Ensure runtime state root exists and is initialized with SQLite storage."""
    state_root = project_dir / ".agent_orchestrator"

    if state_root.exists() and not (state_root / SQLITE_DB_FILE).exists() and not _legacy_yaml_state_exists(state_root):
        _archive_non_runtime_state(project_dir, state_root)

    state_root.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(project_dir)
    (state_root / "workdocs").mkdir(parents=True, exist_ok=True)

    runtime_db = state_root / SQLITE_DB_FILE
    if not runtime_db.exists() and _legacy_yaml_state_exists(state_root):
        try:
            _migrate_yaml_state_to_sqlite(project_dir, state_root)
        except Exception as exc:  # pragma: no cover - defensive path exercised by dedicated tests
            _cleanup_sqlite_artifacts(state_root)
            raise RuntimeError("Failed to migrate runtime state from YAML to SQLite") from exc

    db = SQLiteDB(runtime_db)
    try:
        db.verify_integrity()
    except Exception as exc:  # pragma: no cover - defensive path exercised by dedicated tests
        raise RuntimeError("Runtime SQLite integrity check failed") from exc
    config_repo = SqliteConfigRepository(db)
    config = config_repo.load()
    config["schema_version"] = 4
    config["storage_backend"] = "sqlite"
    config.setdefault("pinned_projects", [])
    config.setdefault(
        "orchestrator",
        {
            "status": "running",
            "concurrency": 2,
            "max_review_attempts": 10,
            "max_verify_fix_attempts": 3,
            "gate_reminder_minutes": 30,
            "gate_stale_minutes": 0,
            "gate_max_wait_minutes": 0,
            "gate_timeout_action": "none",
            "reliability_mode": "strict",
            "reconcile_interval_seconds": 30,
            "lease_ttl_seconds": 120,
            "tick_stale_seconds": 15,
            "tick_failure_threshold": 5,
        },
    )
    config.setdefault(
        "defaults",
        {
            "quality_gate": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "dependency_policy": "prudent",
            "hitl_mode": "autopilot",
        },
    )
    project_cfg = config.get("project")
    if not isinstance(project_cfg, dict):
        project_cfg = {}
        config["project"] = project_cfg
    project_cfg.setdefault("commands", {})
    project_cfg.setdefault("prompt_overrides", {})
    project_cfg.setdefault("prompt_injections", {})
    config_repo.save(config)

    return state_root


def archive_state_root(project_dir: Path) -> Path | None:
    """Archive the current runtime state root under a timestamped folder."""
    state_root = project_dir / ".agent_orchestrator"
    if not state_root.exists():
        return None

    archive_root = project_dir / ARCHIVE_DIR_NAME
    archive_root.mkdir(parents=True, exist_ok=True)

    candidate = _next_archive_path(archive_root, "state")
    state_root.rename(candidate)
    return candidate
