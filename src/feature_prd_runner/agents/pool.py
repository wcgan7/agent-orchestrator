"""Agent pool manager — lifecycle management for concurrent agent instances.

The pool spawns, monitors, and manages multiple agent instances running in
parallel. It tracks health via heartbeats, handles failures with redistribution,
and enforces resource limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from .registry import (
    AgentInstance,
    AgentRegistry,
    AgentRole,
    AgentStatus,
    AgentType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_AGENTS = 5
HEARTBEAT_INTERVAL_SECONDS = 10
HEARTBEAT_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Agent Pool
# ---------------------------------------------------------------------------

class AgentPool:
    """Manages a pool of concurrent agent instances.

    Responsibilities:
    - Spawn agents from registered types
    - Monitor agent health via heartbeats
    - Reassign tasks when agents fail
    - Enforce per-agent resource limits
    - Provide real-time status for the UI
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        max_agents: int = DEFAULT_MAX_AGENTS,
        on_agent_event: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        max_agents_per_role: Optional[dict[str, int]] = None,
    ) -> None:
        self._registry = registry or AgentRegistry()
        self._max_agents = max_agents
        self._max_agents_per_role: dict[str, int] = max_agents_per_role or {}
        self._agents: dict[str, AgentInstance] = {}
        self._on_event = on_agent_event  # callback(agent_id, event_type, data)

    # -- Properties ----------------------------------------------------------

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status in (AgentStatus.RUNNING, AgentStatus.PAUSED))

    @property
    def idle_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == AgentStatus.IDLE)

    @property
    def capacity(self) -> int:
        return max(0, self._max_agents - self.active_count)

    # -- Agent lifecycle -----------------------------------------------------

    def spawn(self, role: str, task_id: Optional[str] = None, **overrides: Any) -> AgentInstance:
        """Create and register a new agent instance.

        Raises ``RuntimeError`` if the pool is at global capacity or the
        per-role limit for *role* has been reached.
        """
        if self.active_count >= self._max_agents:
            raise RuntimeError(
                f"Agent pool at capacity ({self._max_agents}). "
                "Terminate an agent or increase max_agents."
            )

        # Enforce per-role limit if configured
        if role in self._max_agents_per_role:
            role_limit = self._max_agents_per_role[role]
            current_role_count = sum(
                1 for a in self._agents.values()
                if a.agent_type == role
                and a.status in (AgentStatus.RUNNING, AgentStatus.PAUSED, AgentStatus.IDLE)
            )
            if current_role_count >= role_limit:
                raise RuntimeError(
                    f"Per-role limit reached for '{role}' ({role_limit}). "
                    "Terminate an existing agent of this role first."
                )

        agent = self._registry.create_instance(role, **overrides)
        agent.started_at = _now_iso()
        agent.last_heartbeat = agent.started_at

        if task_id:
            agent.task_id = task_id
            agent.status = AgentStatus.RUNNING
        else:
            agent.status = AgentStatus.IDLE

        self._agents[agent.id] = agent
        self._emit(agent.id, "spawned", {"role": role, "task_id": task_id})
        return agent

    def terminate(self, agent_id: str) -> None:
        """Terminate an agent and free its slot."""
        agent = self._get(agent_id)
        agent.status = AgentStatus.TERMINATED
        self._emit(agent_id, "terminated", {})

    def remove(self, agent_id: str) -> None:
        """Remove a terminated agent from the pool entirely."""
        self._agents.pop(agent_id, None)

    def pause(self, agent_id: str) -> None:
        agent = self._get(agent_id)
        if agent.status == AgentStatus.RUNNING:
            agent.status = AgentStatus.PAUSED
            self._emit(agent_id, "paused", {})

    def resume(self, agent_id: str) -> None:
        agent = self._get(agent_id)
        if agent.status == AgentStatus.PAUSED:
            agent.status = AgentStatus.RUNNING
            self._emit(agent_id, "resumed", {})

    # -- Task assignment -----------------------------------------------------

    def assign_task(self, agent_id: str, task_id: str) -> None:
        agent = self._get(agent_id)
        if agent.status not in (AgentStatus.IDLE, AgentStatus.PAUSED):
            raise RuntimeError(f"Agent {agent_id} is {agent.status.value}, cannot assign task")
        agent.task_id = task_id
        agent.status = AgentStatus.RUNNING
        agent.current_step = None
        agent.current_file = None
        self._emit(agent_id, "task_assigned", {"task_id": task_id})

    def unassign_task(self, agent_id: str) -> Optional[str]:
        """Remove task assignment. Returns the old task_id."""
        agent = self._get(agent_id)
        old_task = agent.task_id
        agent.task_id = None
        agent.current_step = None
        agent.current_file = None
        agent.status = AgentStatus.IDLE
        self._emit(agent_id, "task_unassigned", {"task_id": old_task})
        return old_task

    # -- Progress / heartbeat -----------------------------------------------

    def heartbeat(self, agent_id: str) -> None:
        agent = self._get(agent_id)
        agent.last_heartbeat = _now_iso()

    def update_progress(
        self,
        agent_id: str,
        *,
        current_step: Optional[str] = None,
        current_file: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        output_line: Optional[str] = None,
    ) -> None:
        agent = self._get(agent_id)
        if current_step is not None:
            agent.current_step = current_step
        if current_file is not None:
            agent.current_file = current_file
        if tokens_used is not None:
            agent.tokens_used = tokens_used
        if cost_usd is not None:
            agent.cost_usd = cost_usd
        if output_line is not None:
            agent.output_tail.append(output_line)
            # Keep tail bounded
            if len(agent.output_tail) > 200:
                agent.output_tail = agent.output_tail[-100:]
        agent.last_heartbeat = _now_iso()
        self._emit(agent_id, "progress", agent.to_dict())

    def mark_failed(self, agent_id: str, error: str) -> None:
        agent = self._get(agent_id)
        agent.status = AgentStatus.FAILED
        agent.output_tail.append(f"[ERROR] {error}")
        self._emit(agent_id, "failed", {"error": error, "task_id": agent.task_id})

    # -- Resource limit checks -----------------------------------------------

    def check_limits(self, agent_id: str) -> Optional[str]:
        """Check if agent has exceeded any resource limit. Returns reason or None."""
        agent = self._get(agent_id)
        atype = self._registry.get_type(agent.agent_type)

        if agent.tokens_used >= atype.limits.max_tokens:
            return f"Token limit exceeded ({agent.tokens_used}/{atype.limits.max_tokens})"
        if agent.cost_usd >= atype.limits.max_cost_usd:
            return f"Cost limit exceeded (${agent.cost_usd:.2f}/${atype.limits.max_cost_usd:.2f})"
        if agent.elapsed_seconds >= atype.limits.max_time_seconds:
            return f"Time limit exceeded ({agent.elapsed_seconds:.0f}s/{atype.limits.max_time_seconds}s)"

        return None

    # -- Health monitoring ---------------------------------------------------

    def get_stale_agents(self, timeout_seconds: float = HEARTBEAT_TIMEOUT_SECONDS) -> list[AgentInstance]:
        """Return agents that have not sent a heartbeat within the timeout."""
        stale = []
        now = time.time()
        for agent in self._agents.values():
            if agent.status != AgentStatus.RUNNING:
                continue
            if agent.last_heartbeat:
                import datetime
                hb_time = datetime.datetime.fromisoformat(agent.last_heartbeat).timestamp()
                if now - hb_time > timeout_seconds:
                    stale.append(agent)
        return stale

    # -- Queries -------------------------------------------------------------

    def get(self, agent_id: str) -> Optional[AgentInstance]:
        return self._agents.get(agent_id)

    def list_agents(self, status: Optional[AgentStatus] = None) -> list[AgentInstance]:
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def find_idle_agent(self, role: Optional[str] = None) -> Optional[AgentInstance]:
        """Find an idle agent, optionally filtered by role."""
        for agent in self._agents.values():
            if agent.status != AgentStatus.IDLE:
                continue
            if role and agent.agent_type != role:
                continue
            return agent
        return None

    def get_agent_for_task(self, task_id: str) -> Optional[AgentInstance]:
        """Find the agent currently assigned to a task."""
        for agent in self._agents.values():
            if agent.task_id == task_id:
                return agent
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_agents": self._max_agents,
            "active_count": self.active_count,
            "idle_count": self.idle_count,
            "capacity": self.capacity,
            "agents": [a.to_dict() for a in self._agents.values()],
        }

    # -- Reassignment --------------------------------------------------------

    def reassign(self, agent_id: str, new_task_id: str) -> Optional[str]:
        """Unassign the current task from *agent_id* and assign *new_task_id*.

        Returns the old task id (or ``None`` if the agent had no task).
        """
        agent = self._get(agent_id)
        old_task_id: Optional[str] = None
        if agent.task_id:
            old_task_id = self.unassign_task(agent_id)
        self.assign_task(agent_id, new_task_id)
        return old_task_id

    # -- Auto-restart / reap -------------------------------------------------

    def auto_restart(self, agent_id: str) -> bool:
        """Attempt to restart a failed agent if it has retries remaining.

        Returns ``True`` if the agent was restarted, ``False`` otherwise.
        """
        agent = self._get(agent_id)
        if agent.status != AgentStatus.FAILED:
            return False

        atype = self._registry.get_type(agent.agent_type)
        if agent.retries >= atype.limits.max_retries:
            logger.info(
                "Agent %s has exhausted retries (%d/%d) — not restarting",
                agent_id, agent.retries, atype.limits.max_retries,
            )
            return False

        agent.retries += 1
        agent.status = AgentStatus.IDLE
        agent.last_heartbeat = _now_iso()
        self._emit(agent_id, "auto_restarted", {"retries": agent.retries})
        logger.info("Auto-restarted agent %s (retry %d/%d)", agent_id, agent.retries, atype.limits.max_retries)
        return True

    def reap_dead_agents(self, timeout_seconds: float = HEARTBEAT_TIMEOUT_SECONDS) -> list[str]:
        """Find stale (heartbeat-timed-out) agents, mark them failed, and
        attempt ``auto_restart`` on each.

        Returns a list of agent ids that were reaped.
        """
        stale = self.get_stale_agents(timeout_seconds=timeout_seconds)
        reaped: list[str] = []
        for agent in stale:
            logger.warning("Reaping stale agent %s (task=%s)", agent.id, agent.task_id)
            self.mark_failed(agent.id, "Heartbeat timeout — agent presumed dead")
            reaped.append(agent.id)
            # Attempt auto-restart
            self.auto_restart(agent.id)
        return reaped

    # -- Internal ------------------------------------------------------------

    def _get(self, agent_id: str) -> AgentInstance:
        agent = self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return agent

    def _emit(self, agent_id: str, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(agent_id, event_type, data)
            except Exception:
                logger.exception("Error in agent event callback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
