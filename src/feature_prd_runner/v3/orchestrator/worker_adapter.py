from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..domain.models import Task


@dataclass
class StepResult:
    status: str = "ok"
    summary: str | None = None
    findings: list[dict[str, Any]] | None = None
    generated_tasks: list[dict[str, Any]] | None = None


class WorkerAdapter(Protocol):
    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        ...


class DefaultWorkerAdapter:
    """Default adapter used in local-first mode.

    Behavior is deterministic for tests by honoring `task.metadata['scripted_steps']`
    and `task.metadata['scripted_findings']`.
    """

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
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

        return StepResult(status="ok")
