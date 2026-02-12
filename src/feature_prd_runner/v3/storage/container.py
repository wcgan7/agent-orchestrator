from __future__ import annotations

from pathlib import Path

from .bootstrap import ensure_v3_state_root
from .file_repos import (
    FileAgentRepository,
    FileConfigRepository,
    FileEventRepository,
    FileQuickActionRepository,
    FileReviewRepository,
    FileRunRepository,
    FileTaskRepository,
)


class V3Container:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self.v3_root = ensure_v3_state_root(self.project_dir)

        self.tasks = FileTaskRepository(self.v3_root / "tasks.yaml", self.v3_root / "tasks.lock")
        self.runs = FileRunRepository(self.v3_root / "runs.yaml", self.v3_root / "runs.lock")
        self.reviews = FileReviewRepository(self.v3_root / "review_cycles.yaml", self.v3_root / "review_cycles.lock")
        self.agents = FileAgentRepository(self.v3_root / "agents.yaml", self.v3_root / "agents.lock")
        self.quick_actions = FileQuickActionRepository(self.v3_root / "quick_actions.yaml", self.v3_root / "quick_actions.lock")
        self.events = FileEventRepository(self.v3_root / "events.jsonl", self.v3_root / "events.lock")
        self.config = FileConfigRepository(self.v3_root / "config.yaml", self.v3_root / "config.lock")

    @property
    def project_id(self) -> str:
        return self.project_dir.name
