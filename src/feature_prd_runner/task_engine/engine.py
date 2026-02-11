"""Task engine — high-level CRUD, dependency management, and board operations.

This is the primary entry-point for all task manipulation.  It wraps
:class:`TaskStore` with business logic (cycle detection, cascading status
updates, board grouping, migration).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from .model import (
    Task,
    TaskPriority,
    TaskSource,
    TaskStatus,
    TaskType,
    from_legacy_task,
)
from .store import TaskStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid status transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.BACKLOG: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.BACKLOG, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.IN_REVIEW, TaskStatus.BLOCKED, TaskStatus.READY, TaskStatus.CANCELLED, TaskStatus.DONE},
    TaskStatus.IN_REVIEW: {TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.BLOCKED: {TaskStatus.READY, TaskStatus.CANCELLED, TaskStatus.BACKLOG},
    TaskStatus.DONE: {TaskStatus.READY},  # allow reopening
    TaskStatus.CANCELLED: {TaskStatus.BACKLOG},  # allow restoring
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TaskEngine:
    """Manage the full lifecycle of tasks on the dynamic board.

    Parameters
    ----------
    state_dir:
        Path to the ``.prd_runner/`` directory.
    """

    def __init__(self, state_dir: Path) -> None:
        self.store = TaskStore(state_dir)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str = "",
        task_type: str = "feature",
        priority: str = "P2",
        labels: Optional[list[str]] = None,
        acceptance_criteria: Optional[list[str]] = None,
        context_files: Optional[list[str]] = None,
        parent_id: Optional[str] = None,
        effort: Optional[str] = None,
        pipeline_template: Optional[str] = None,
        source: str = "manual",
        created_by: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Task:
        """Create and persist a new task, returning it."""
        task = Task(
            title=title,
            description=description,
            task_type=TaskType(task_type) if task_type else TaskType.FEATURE,
            priority=TaskPriority(priority) if priority else TaskPriority.P2,
            labels=labels or [],
            acceptance_criteria=acceptance_criteria or [],
            context_files=context_files or [],
            parent_id=parent_id,
            effort=None,  # set below
            pipeline_template=pipeline_template,
            source=TaskSource(source) if source else TaskSource.MANUAL,
            created_by=created_by,
            metadata=metadata or {},
        )
        if effort:
            from .model import EffortEstimate
            try:
                task.effort = EffortEstimate(effort)
            except (ValueError, KeyError):
                pass

        with self.store.transaction() as tx:
            tx.add(task)
            # If parent_id given, update parent's children_ids
            if parent_id:
                parent = tx.get(parent_id)
                if parent and task.id not in parent.children_ids:
                    parent.children_ids.append(task.id)
                    parent.touch()
                    tx.dirty = True

        logger.info("Created task %s: %s", task.id, title)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.store.get_one(task_id)

    def list_tasks(
        self,
        *,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        label: Optional[str] = None,
        search: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> list[Task]:
        with self.store.transaction() as tx:
            return tx.find(
                status=status,
                task_type=task_type,
                priority=priority,
                assignee=assignee,
                label=label,
                search=search,
                parent_id=parent_id,
            )

    def update_task(self, task_id: str, changes: dict[str, Any]) -> Optional[Task]:
        """Apply partial updates to a task.  Returns the updated task or None."""
        # Coerce enum fields if given as strings
        enum_fields = {
            "task_type": TaskType,
            "priority": TaskPriority,
            "status": TaskStatus,
            "source": TaskSource,
        }
        for field_name, enum_cls in enum_fields.items():
            if field_name in changes and isinstance(changes[field_name], str):
                try:
                    changes[field_name] = enum_cls(changes[field_name])
                except (ValueError, KeyError):
                    pass

        with self.store.transaction() as tx:
            return tx.update(task_id, changes)

    def delete_task(self, task_id: str) -> bool:
        """Soft-delete a task (set status=cancelled).

        Also removes it from other tasks' blocked_by / blocks lists.
        """
        with self.store.transaction() as tx:
            task = tx.get(task_id)
            if task is None:
                return False
            task.transition(TaskStatus.CANCELLED)
            # Clean up dependency references
            for other in tx.list_all():
                if task_id in other.blocked_by:
                    other.remove_blocked_by(task_id)
                    tx.dirty = True
                if task_id in other.blocks:
                    other.remove_blocks(task_id)
                    tx.dirty = True
            tx.dirty = True
            return True

    def bulk_create(self, tasks: list[Task]) -> list[Task]:
        with self.store.transaction() as tx:
            return tx.add_many(tasks)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def transition_task(self, task_id: str, new_status: str) -> Optional[Task]:
        """Move a task to a new status, enforcing valid transitions."""
        target = TaskStatus(new_status)
        with self.store.transaction() as tx:
            task = tx.get(task_id)
            if task is None:
                return None
            # Basic transition validation
            valid = _VALID_TRANSITIONS.get(task.status, set())
            if target not in valid:
                raise ValueError(
                    f"Cannot transition {task.id} from {task.status.value} to {target.value}. "
                    f"Valid targets: {[s.value for s in valid]}"
                )
            task.transition(target)
            # If moving to DONE, unblock dependents
            if target == TaskStatus.DONE:
                self._unblock_dependents(tx, task_id)
            tx.dirty = True
            return task

    def assign_task(self, task_id: str, assignee: str, assignee_type: str = "agent") -> Optional[Task]:
        with self.store.transaction() as tx:
            task = tx.get(task_id)
            if task is None:
                return None
            task.assign(assignee, assignee_type)
            tx.dirty = True
            return task

    def unassign_task(self, task_id: str) -> Optional[Task]:
        with self.store.transaction() as tx:
            task = tx.get(task_id)
            if task is None:
                return None
            task.unassign()
            tx.dirty = True
            return task

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def add_dependency(self, task_id: str, depends_on_id: str) -> None:
        """Add ``depends_on_id`` as a blocker for ``task_id``.

        Raises :class:`ValueError` if the dependency would create a cycle.
        """
        if task_id == depends_on_id:
            raise ValueError("A task cannot depend on itself")

        with self.store.transaction() as tx:
            task = tx.get(task_id)
            dep = tx.get(depends_on_id)
            if task is None or dep is None:
                raise ValueError("One or both task IDs not found")

            # Cycle check: would adding depends_on_id → task_id create a cycle?
            if self._would_cycle(tx, task_id, depends_on_id):
                raise ValueError(
                    f"Adding dependency {task_id} → {depends_on_id} would create a cycle"
                )

            task.add_blocked_by(depends_on_id)
            dep.add_blocks(task_id)

            # If the blocker is not done, mark task as blocked
            if not dep.is_terminal and task.status == TaskStatus.READY:
                task.status = TaskStatus.BLOCKED
                task.touch()

            tx.dirty = True

    def remove_dependency(self, task_id: str, depends_on_id: str) -> None:
        with self.store.transaction() as tx:
            task = tx.get(task_id)
            dep = tx.get(depends_on_id)
            if task:
                task.remove_blocked_by(depends_on_id)
            if dep:
                dep.remove_blocks(task_id)

            # If no more blockers, unblock
            if task and not task.blocked_by and task.status == TaskStatus.BLOCKED:
                task.status = TaskStatus.READY
                task.touch()
            tx.dirty = True

    def get_dependency_graph(self, task_id: Optional[str] = None) -> dict[str, list[str]]:
        """Return adjacency list: ``{task_id: [blocked_by_ids]}``.

        If *task_id* is given, return only the subgraph reachable from it.
        """
        tasks = self.store.read_snapshot()
        graph: dict[str, list[str]] = {t.id: list(t.blocked_by) for t in tasks}
        if task_id is None:
            return graph
        # BFS from task_id
        visited: set[str] = set()
        queue: deque[str] = deque([task_id])
        sub: dict[str, list[str]] = {}
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            deps = graph.get(nid, [])
            sub[nid] = deps
            for d in deps:
                queue.append(d)
            # also follow forward edges
            for tid, dep_list in graph.items():
                if nid in dep_list and tid not in visited:
                    queue.append(tid)
        return sub

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks that are ready and have all dependencies satisfied."""
        tasks = self.store.read_snapshot()
        done_ids = {t.id for t in tasks if t.is_terminal}
        ready: list[Task] = []
        for t in tasks:
            if t.status not in (TaskStatus.READY, TaskStatus.BACKLOG):
                continue
            if t.blocked_by and not all(bid in done_ids for bid in t.blocked_by):
                continue
            ready.append(t)
        # Sort by priority then creation time
        ready.sort(key=lambda t: (t.priority.sort_key, t.created_at))
        return ready

    def get_execution_order(self) -> list[list[str]]:
        """Topological sort into batches of independent tasks (Kahn's algorithm)."""
        tasks = self.store.read_snapshot()
        task_map = {t.id: t for t in tasks if not t.is_terminal}
        in_degree: dict[str, int] = {tid: 0 for tid in task_map}
        adj: dict[str, list[str]] = defaultdict(list)

        for t in task_map.values():
            for dep_id in t.blocked_by:
                if dep_id in task_map:
                    adj[dep_id].append(t.id)
                    in_degree[t.id] = in_degree.get(t.id, 0) + 1

        batches: list[list[str]] = []
        queue = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda tid: task_map[tid].priority.sort_key,
        )

        while queue:
            batches.append(list(queue))
            next_queue: list[str] = []
            for tid in queue:
                for neighbor in adj.get(tid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            next_queue.sort(key=lambda tid: task_map[tid].priority.sort_key)
            queue = next_queue

        # Detect cycles: any remaining tasks with in_degree > 0
        remaining = [tid for tid, deg in in_degree.items() if deg > 0]
        if remaining:
            logger.warning("Dependency cycle detected among tasks: %s", remaining)

        return batches

    # ------------------------------------------------------------------
    # Board view
    # ------------------------------------------------------------------

    def get_board(self) -> dict[str, list[dict[str, Any]]]:
        """Return tasks grouped by status column for the Kanban board."""
        tasks = self.store.read_snapshot()
        columns: dict[str, list[dict[str, Any]]] = {
            "backlog": [],
            "ready": [],
            "in_progress": [],
            "in_review": [],
            "blocked": [],
            "done": [],
        }
        for t in tasks:
            col = t.status.value
            if col == "cancelled":
                continue  # hide cancelled from board
            bucket = columns.get(col, columns["backlog"])
            bucket.append(t.to_dict())

        # Sort each column by priority then created_at
        for col_tasks in columns.values():
            col_tasks.sort(key=lambda d: (
                TaskPriority(d.get("priority", "P2")).sort_key,
                d.get("created_at", ""),
            ))
        return columns

    def reorder_tasks(self, task_ids: list[str]) -> None:
        with self.store.transaction() as tx:
            tx.reorder(task_ids)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def migrate_legacy_tasks(self, state_dir: Path) -> list[Task]:
        """Import tasks from the legacy ``task_queue.yaml`` into the v2 store.

        Returns the newly created tasks.
        """
        from ..state_manager import state_transaction

        legacy_tasks: list[dict[str, Any]] = []
        lock_path = state_dir / ".lock"
        tq_path = state_dir / "task_queue.yaml"

        if not tq_path.exists():
            return []

        with state_transaction(
            lock_path=lock_path,
            task_queue_path=tq_path,
        ) as snap:
            legacy_tasks = list(snap.tasks)

        migrated: list[Task] = []
        with self.store.transaction() as tx:
            # Check if we already migrated (look for legacy_migration source)
            existing_legacy_ids = {
                t.legacy_task_id for t in tx.list_all() if t.legacy_task_id
            }
            for raw in legacy_tasks:
                lid = str(raw.get("id", ""))
                if lid in existing_legacy_ids:
                    continue  # already migrated
                task = from_legacy_task(raw)
                tx.add(task)
                migrated.append(task)

        if migrated:
            logger.info("Migrated %d legacy tasks to v2 store", len(migrated))
        return migrated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _would_cycle(tx: Any, task_id: str, new_dep_id: str) -> bool:
        """Return True if adding task_id→new_dep_id creates a cycle.

        We check: can we reach task_id starting from new_dep_id's blocked_by?
        If so, adding the edge would create a cycle.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([new_dep_id])
        while queue:
            current = queue.popleft()
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            node = tx.get(current)
            if node:
                for dep in node.blocked_by:
                    queue.append(dep)
        return False

    @staticmethod
    def _unblock_dependents(tx: Any, completed_task_id: str) -> None:
        """When a task completes, remove it from dependents' blocked_by lists.

        If a dependent has no remaining blockers, move it to READY.
        """
        for task in tx.list_all():
            if completed_task_id in task.blocked_by:
                task.remove_blocked_by(completed_task_id)
                if not task.blocked_by and task.status == TaskStatus.BLOCKED:
                    task.status = TaskStatus.READY
                    task.touch()
                tx.dirty = True
