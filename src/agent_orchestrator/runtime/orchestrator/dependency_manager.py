"""Automatic dependency analysis helpers for queued tasks."""

from __future__ import annotations

import logging
from typing import Callable

from ..domain.models import Task
from ..events.bus import EventBus
from ..storage.container import Container
from .worker_adapter import WorkerAdapter

logger = logging.getLogger(__name__)


def _has_cycle(adj: dict[str, list[str]], from_id: str, to_id: str) -> bool:
    """Check whether adding edge from_id->to_id introduces a cycle."""
    visited: set[str] = set()
    stack = [to_id]
    while stack:
        node = stack.pop()
        if node == from_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))
    return False


class DependencyManager:
    """Analyze and apply inferred task dependencies."""

    def __init__(
        self,
        container: Container,
        bus: EventBus,
        *,
        worker_adapter_getter: Callable[[], WorkerAdapter],
    ) -> None:
        """Initialize the manager with orchestrator dependencies."""
        self.container = container
        self.bus = bus
        self._worker_adapter_getter = worker_adapter_getter

    def maybe_analyze_dependencies(self) -> None:
        """Run automatic dependency analysis on unanalyzed queued tasks."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if not orchestrator_cfg.get("auto_deps", True):
            return

        all_tasks = self.container.tasks.list()
        candidates = [
            task for task in all_tasks
            if task.status == "queued"
            and not (isinstance(task.metadata, dict) and task.metadata.get("deps_analyzed"))
            and task.source != "prd_import"
        ]

        def _mark_analyzed(tasks: list[Task]) -> None:
            for task in tasks:
                if not isinstance(task.metadata, dict):
                    task.metadata = {}
                task.metadata["deps_analyzed"] = True
                self.container.tasks.upsert(task)

        if len(candidates) < 2:
            _mark_analyzed(candidates)
            return

        terminal = {"done", "cancelled"}
        existing = [
            task for task in all_tasks
            if isinstance(task.metadata, dict) and task.metadata.get("deps_analyzed")
            and task.status not in terminal
        ]

        candidate_data = [
            {
                "id": task.id,
                "title": task.title,
                "description": (task.description or "")[:200],
                "task_type": task.task_type,
                "labels": task.labels,
            }
            for task in candidates
        ]
        existing_data = [
            {"id": task.id, "title": task.title, "status": task.status}
            for task in existing
        ]

        synthetic = Task(
            title="Dependency analysis",
            description="Analyze task dependencies",
            task_type="research",
            source="system",
            metadata={
                "candidate_tasks": candidate_data,
                "existing_tasks": existing_data,
            },
        )

        try:
            worker_adapter = self._worker_adapter_getter()
            result = worker_adapter.run_step_ephemeral(task=synthetic, step="analyze_deps", attempt=1)
            if result.status == "ok" and result.dependency_edges:
                self.apply_dependency_edges(candidates, result.dependency_edges, all_tasks)
        except Exception:
            logger.exception("Dependency analysis failed; tasks will run without inferred deps")
        finally:
            _mark_analyzed(candidates)

    def apply_dependency_edges(
        self,
        candidates: list[Task],
        edges: list[dict[str, str]],
        all_tasks: list[Task],
    ) -> None:
        """Apply inferred dependency edges with cycle detection."""
        task_map: dict[str, Task] = {}
        for task in all_tasks:
            task_map[task.id] = task
        for task in candidates:
            task_map[task.id] = task

        adj: dict[str, list[str]] = {}
        for task in task_map.values():
            for dep_id in task.blocked_by:
                adj.setdefault(dep_id, []).append(task.id)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            reason = edge.get("reason", "")
            if not from_id or not to_id:
                continue
            if from_id not in task_map or to_id not in task_map:
                continue
            if from_id == to_id:
                continue

            if _has_cycle(adj, from_id, to_id):
                logger.warning("Skipping edge %s->%s: would create cycle", from_id, to_id)
                continue

            from_task = task_map[from_id]
            to_task = task_map[to_id]

            if from_id not in to_task.blocked_by:
                to_task.blocked_by.append(from_id)
            if to_id not in from_task.blocks:
                from_task.blocks.append(to_id)

            if not isinstance(to_task.metadata, dict):
                to_task.metadata = {}
            inferred = to_task.metadata.setdefault("inferred_deps", [])
            inferred.append({"from": from_id, "reason": reason})

            adj.setdefault(from_id, []).append(to_id)

            self.container.tasks.upsert(from_task)
            self.container.tasks.upsert(to_task)
            self.bus.emit(
                channel="tasks",
                event_type="task.dependency_inferred",
                entity_id=to_id,
                payload={"from": from_id, "to": to_id, "reason": reason},
            )
