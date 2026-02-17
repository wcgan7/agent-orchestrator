from __future__ import annotations

from pathlib import Path

from .bootstrap import ensure_state_root
from .file_repos import (
    FileAgentRepository,
    FileConfigRepository,
    FileEventRepository,
    FilePlanRefineJobRepository,
    FilePlanRevisionRepository,
    FileQuickActionRepository,
    FileReviewRepository,
    FileRunRepository,
    FileTaskRepository,
)


class Container:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self.state_root = ensure_state_root(self.project_dir)

        self.tasks = FileTaskRepository(self.state_root / "tasks.yaml", self.state_root / "tasks.lock")
        self.runs = FileRunRepository(self.state_root / "runs.yaml", self.state_root / "runs.lock")
        self.reviews = FileReviewRepository(self.state_root / "review_cycles.yaml", self.state_root / "review_cycles.lock")
        self.agents = FileAgentRepository(self.state_root / "agents.yaml", self.state_root / "agents.lock")
        self.quick_actions = FileQuickActionRepository(self.state_root / "quick_actions.yaml", self.state_root / "quick_actions.lock")
        self.events = FileEventRepository(self.state_root / "events.jsonl", self.state_root / "events.lock")
        self.plan_revisions = FilePlanRevisionRepository(self.state_root / "plan_revisions.yaml", self.state_root / "plan_revisions.lock")
        self.plan_refine_jobs = FilePlanRefineJobRepository(self.state_root / "plan_refine_jobs.yaml", self.state_root / "plan_refine_jobs.lock")
        self.config = FileConfigRepository(self.state_root / "config.yaml", self.state_root / "config.lock")

    @property
    def project_id(self) -> str:
        return self.project_dir.name
