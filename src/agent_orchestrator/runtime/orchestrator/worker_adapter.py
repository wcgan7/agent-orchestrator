"""Worker adapter protocol and default deterministic adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..domain.models import RunRecord, Task


@dataclass
class StepResult:
    """Represents StepResult."""
    status: str = "ok"
    summary: str | None = None
    findings: list[dict[str, Any]] | None = None
    generated_tasks: list[dict[str, Any]] | None = None
    dependency_edges: list[dict[str, str]] | None = None
    human_blocking_issues: list[dict[str, str]] | None = None


class WorkerAdapter(Protocol):
    """Represents WorkerAdapter."""
    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Return run step."""
        ...

    def run_step_ephemeral(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Like run_step but does not persist the task to storage.

        Use for synthetic/throwaway tasks (e.g. pipeline classification,
        dependency analysis) where the task object is only a prompt carrier.
        """
        ...


class DefaultWorkerAdapter:
    """Default adapter used in local-first mode.

    Behavior is deterministic for tests by honoring `task.metadata['scripted_steps']`
    and `task.metadata['scripted_findings']`.
    """

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Return run step."""
        scripted_steps = task.metadata.get("scripted_steps") if isinstance(task.metadata, dict) else None
        if isinstance(scripted_steps, dict):
            key = f"{step}:{attempt}"
            raw = scripted_steps.get(key) or scripted_steps.get(step)
            if isinstance(raw, dict):
                return StepResult(
                    status=str(raw.get("status") or "ok"),
                    summary=raw.get("summary"),
                    findings=list(raw.get("findings") or []) if isinstance(raw.get("findings"), list) else None,
                    generated_tasks=list(raw.get("generated_tasks") or []) if isinstance(raw.get("generated_tasks"), list) else None,
                    dependency_edges=list(raw.get("dependency_edges") or []) if isinstance(raw.get("dependency_edges"), list) else None,
                    human_blocking_issues=(
                        list(raw.get("human_blocking_issues") or [])
                        if isinstance(raw.get("human_blocking_issues"), list)
                        else None
                    ),
                )

        if step == "review":
            scripted_findings = task.metadata.get("scripted_findings") if isinstance(task.metadata, dict) else None
            if isinstance(scripted_findings, list) and attempt <= len(scripted_findings):
                item = scripted_findings[attempt - 1]
                findings = list(item) if isinstance(item, list) else []
                return StepResult(status="ok", findings=findings)

        if step == "generate_tasks":
            scripted = task.metadata.get("scripted_generated_tasks") if isinstance(task.metadata, dict) else None
            if isinstance(scripted, list):
                return StepResult(status="ok", generated_tasks=scripted)

        if step == "analyze_deps":
            scripted = task.metadata.get("scripted_dependency_edges") if isinstance(task.metadata, dict) else None
            if isinstance(scripted, list):
                return StepResult(status="ok", dependency_edges=scripted)
            return StepResult(status="ok", dependency_edges=[])

        # Write scripted files into the worktree so the "no changes" guard passes.
        if isinstance(task.metadata, dict):
            scripted_files = task.metadata.get("scripted_files")
            if isinstance(scripted_files, dict):
                wt = task.metadata.get("worktree_dir")
                base = Path(wt) if wt else None
                if base and base.is_dir():
                    for rel_path, content in scripted_files.items():
                        fp = base / rel_path
                        fp.parent.mkdir(parents=True, exist_ok=True)
                        fp.write_text(str(content), encoding="utf-8")

        return StepResult(status="ok")

    def run_step_ephemeral(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Return run step ephemeral."""
        return self.run_step(task=task, step=step, attempt=attempt)

    def generate_run_summary(self, *, task: Task, run: RunRecord, project_dir: Path) -> str:
        """Return generate run summary."""
        return ""
