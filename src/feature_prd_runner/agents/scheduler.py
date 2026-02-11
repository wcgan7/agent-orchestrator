"""Smart scheduler — assigns tasks to agents based on priority, affinity, and dependencies.

The scheduler integrates the task engine (which tasks are ready) with the agent
pool (which agents are available) to make optimal assignment decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .registry import AgentRegistry
from .pool import AgentPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduling decision
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    """A scheduling decision: assign task X to agent Y."""
    task_id: str
    agent_id: str
    agent_role: str
    reason: str


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Priority-based task scheduler with agent affinity.

    Algorithm:
    1. Fetch all ready tasks (deps satisfied, status=ready) from task engine
    2. Sort by priority (P0 first), then creation time
    3. For each task, find the best available agent:
       a. Prefer agents with matching role affinity for the task type
       b. Fall back to any idle agent
       c. If no idle agents, spawn a new one if pool has capacity
    4. Skip task pairs that would conflict on files
    5. Assign and return the list of assignments
    """

    def __init__(
        self,
        pool: AgentPool,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        self.pool = pool
        self.registry = registry or AgentRegistry()
        self._recent_assignments: list[Assignment] = []
        self._pending_tasks: list[dict[str, Any]] = []

    def schedule(self, ready_tasks: list[dict[str, Any]]) -> list[Assignment]:
        """Given a list of ready tasks (dicts with id, task_type, priority, etc.),
        produce a list of assignments.

        Tasks should already be filtered to only include those with satisfied
        dependencies and status=ready.

        Tasks that would conflict on files with already-assigned tasks are
        skipped and left in the pending queue for the next scheduling cycle.
        """
        if not ready_tasks:
            return []

        # Sort: P0 > P1 > P2 > P3, then by created_at ascending
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        sorted_tasks = sorted(
            ready_tasks,
            key=lambda t: (
                priority_order.get(t.get("priority", "P2"), 2),
                t.get("created_at", ""),
            ),
        )

        assignments: list[Assignment] = []
        # Track tasks assigned in this cycle for file-conflict detection
        assigned_in_cycle: list[dict[str, Any]] = []

        for task in sorted_tasks:
            task_id = task["id"]
            task_type = task.get("task_type", "feature")
            priority = task.get("priority", "P2")

            # Skip if already assigned to a running agent
            existing = self.pool.get_agent_for_task(task_id)
            if existing:
                continue

            # Skip if this task conflicts on files with any task assigned
            # in this cycle
            conflict_found = False
            for assigned_task in assigned_in_cycle:
                if self._detect_file_conflicts(task, assigned_task):
                    logger.debug(
                        "Skipping task %s — file conflict with task %s",
                        task_id, assigned_task["id"],
                    )
                    conflict_found = True
                    break
            if conflict_found:
                continue

            # Find best agent
            agent = self._find_best_agent(task_type)
            if agent is None:
                # No available agents — try to spawn
                agent = self._try_spawn(task_type)

            if agent is None:
                # No capacity — skip this task for now
                logger.debug(
                    "No agent available for task %s (type=%s, priority=%s)",
                    task_id, task_type, priority,
                )
                continue

            # Make assignment
            self.pool.assign_task(agent.id, task_id)
            assignment = Assignment(
                task_id=task_id,
                agent_id=agent.id,
                agent_role=agent.agent_type,
                reason=f"Priority {priority}, affinity match for {task_type}",
            )
            assignments.append(assignment)
            assigned_in_cycle.append(task)

        # Track for queue/history introspection
        self._recent_assignments = list(assignments)
        self._pending_tasks = [
            t for t in sorted_tasks
            if not self.pool.get_agent_for_task(t["id"])
            and t not in assigned_in_cycle
        ]

        return assignments

    def preempt_for_p0(self, p0_task: dict[str, Any]) -> Optional[Assignment]:
        """If a P0 task arrives and no agents are idle, preempt the lowest-priority
        running agent.

        Returns an assignment if preemption occurred, None otherwise.
        """
        task_id = p0_task["id"]
        task_type = p0_task.get("task_type", "feature")

        # First try normal assignment
        agent = self._find_best_agent(task_type) or self._try_spawn(task_type)
        if agent:
            self.pool.assign_task(agent.id, task_id)
            return Assignment(
                task_id=task_id,
                agent_id=agent.id,
                agent_role=agent.agent_type,
                reason="P0 priority assignment",
            )

        # No idle/spawn capacity — preempt lowest priority running agent
        running = self.pool.list_agents()
        running = [a for a in running if a.status.value == "running" and a.task_id]

        if not running:
            return None

        # We'd need the task info to determine priority — for now just pick the
        # agent that has been running the longest (least progress)
        running.sort(key=lambda a: a.started_at or "")
        victim = running[0]

        logger.info(
            "Preempting agent %s (task %s) for P0 task %s",
            victim.id, victim.task_id, task_id,
        )

        old_task_id = self.pool.unassign_task(victim.id)
        self.pool.assign_task(victim.id, task_id)

        return Assignment(
            task_id=task_id,
            agent_id=victim.id,
            agent_role=victim.agent_type,
            reason=f"P0 preemption (displaced task {old_task_id})",
        )

    # -- Callback methods -----------------------------------------------------

    def on_task_complete(self, task_id: str) -> list[Assignment]:
        """Called when a task finishes.  Unblocks dependent tasks and
        triggers a re-schedule with the current pending queue.

        The caller is responsible for updating the task engine status;
        this method only handles the scheduling side-effects.

        Returns any new assignments produced by the re-schedule.
        """
        # Free the agent that was running this task
        agent = self.pool.get_agent_for_task(task_id)
        if agent:
            self.pool.unassign_task(agent.id)
            logger.info("Task %s complete — agent %s is now idle", task_id, agent.id)

        # Re-schedule with whatever is still pending
        return self.schedule(list(self._pending_tasks))

    def on_agent_free(self, agent_id: str) -> Optional[Assignment]:
        """Called when an agent becomes free.  Finds the next pending task
        for this agent and assigns it.

        Returns the assignment if one was made, ``None`` otherwise.
        """
        agent = self.pool.get(agent_id)
        if agent is None or agent.status.value != "idle":
            return None

        # Walk the pending queue in priority order and pick the first task
        # that matches the agent's affinity (or any task as fallback).
        best: Optional[dict[str, Any]] = None
        for task in self._pending_tasks:
            preferred_role = self.registry.best_role_for_task_type(
                task.get("task_type", "feature"),
            )
            if preferred_role == agent.agent_type:
                best = task
                break

        if best is None and self._pending_tasks:
            best = self._pending_tasks[0]

        if best is None:
            return None

        task_id = best["id"]
        self.pool.assign_task(agent_id, task_id)
        self._pending_tasks = [t for t in self._pending_tasks if t["id"] != task_id]
        assignment = Assignment(
            task_id=task_id,
            agent_id=agent_id,
            agent_role=agent.agent_type,
            reason=f"Assigned to free agent {agent_id}",
        )
        self._recent_assignments.append(assignment)
        return assignment

    def rebalance(self) -> list[Assignment]:
        """Reassign pending tasks after priority changes.

        Re-sorts the pending queue and performs a fresh scheduling pass
        using currently idle agents.  Already-running assignments are not
        disturbed.
        """
        if not self._pending_tasks:
            return []

        return self.schedule(list(self._pending_tasks))

    # -- Queue introspection --------------------------------------------------

    def get_queue(self) -> dict[str, Any]:
        """Return a snapshot of the current scheduling state for the API."""
        return {
            "pending_tasks": list(self._pending_tasks),
            "recent_assignments": [
                {
                    "task_id": a.task_id,
                    "agent_id": a.agent_id,
                    "agent_role": a.agent_role,
                    "reason": a.reason,
                }
                for a in self._recent_assignments
            ],
        }

    # -- File conflict detection ----------------------------------------------

    @staticmethod
    def _detect_file_conflicts(
        task1: dict[str, Any],
        task2: dict[str, Any],
    ) -> bool:
        """Check whether two tasks edit overlapping files.

        Each task may carry a ``context_files`` key (a list of file paths).
        Returns ``True`` if the two sets intersect.
        """
        files1 = set(task1.get("context_files") or [])
        files2 = set(task2.get("context_files") or [])
        if not files1 or not files2:
            return False
        return bool(files1 & files2)

    # -- Internal helpers -----------------------------------------------------

    def _find_best_agent(self, task_type: str) -> Any:
        """Find the best idle agent for a task type, preferring role affinity."""
        # 1. Try to find an idle agent with matching affinity
        preferred_role = self.registry.best_role_for_task_type(task_type)
        if preferred_role:
            agent = self.pool.find_idle_agent(role=preferred_role)
            if agent:
                return agent

        # 2. Fall back to any idle agent
        return self.pool.find_idle_agent()

    def _try_spawn(self, task_type: str) -> Any:
        """Try to spawn a new agent if pool has capacity."""
        if self.pool.capacity <= 0:
            return None

        role = self.registry.best_role_for_task_type(task_type) or "implementer"
        try:
            return self.pool.spawn(role)
        except RuntimeError:
            return None
