"""Dependency container for runtime repositories."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import sqlite3

from .bootstrap import ensure_state_root
from .sqlite_db import SQLiteDB
from .sqlite_repos import (
    SqliteAgentRepository,
    SqliteConfigRepository,
    SqliteEventRepository,
    SqlitePlanRefineJobRepository,
    SqlitePlanRevisionRepository,
    SqliteReviewRepository,
    SqliteRunRepository,
    SqliteTaskRepository,
    SqliteTerminalSessionRepository,
)


class Container:
    """Wire SQLite-backed repositories and project-scoped runtime settings."""
    def __init__(self, project_dir: Path) -> None:
        """Initialize the Container.

        Args:
            project_dir (Path): Project dir for this call.
        """
        self.project_dir = project_dir.resolve()
        self.state_root = ensure_state_root(self.project_dir)
        self.db = SQLiteDB(self.state_root / "runtime.db")

        self.tasks = SqliteTaskRepository(self.db)
        self.runs = SqliteRunRepository(self.db)
        self.reviews = SqliteReviewRepository(self.db)
        self.agents = SqliteAgentRepository(self.db)
        self.terminal_sessions = SqliteTerminalSessionRepository(self.db)
        self.events = SqliteEventRepository(self.db)
        self.plan_revisions = SqlitePlanRevisionRepository(self.db)
        self.plan_refine_jobs = SqlitePlanRefineJobRepository(self.db)
        self.config = SqliteConfigRepository(self.db)

    @property
    def project_id(self) -> str:
        """Expose the stable project identifier derived from directory name.

        Returns:
            str: str result produced by this operation.
        """
        return self.project_dir.name

    def transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        """Open a container-wide transactional scope for multi-repo updates."""
        return self.db.transaction()
