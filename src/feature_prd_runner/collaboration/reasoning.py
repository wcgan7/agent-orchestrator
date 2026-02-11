"""Agent reasoning capture and retrieval.

Stores step-by-step reasoning for each agent working on a task, so the
ReasoningViewer can display why agents made certain decisions.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReasoningStep:
    """One step in an agent's reasoning chain."""
    step_name: str
    status: str = "pending"       # pending | running | completed | failed | skipped
    reasoning: str = ""           # why the agent decided to do this
    output: str = ""              # key output/artifacts
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at is None or self.completed_at is None:
            return None
        return round((self.completed_at - self.started_at) * 1000, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status,
            "reasoning": self.reasoning,
            "output": self.output,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentReasoning:
    """Complete reasoning trace for one agent on one task."""
    agent_id: str
    agent_role: str
    task_id: str
    pipeline_id: str = ""
    steps: list[ReasoningStep] = field(default_factory=list)
    current_step: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "task_id": self.task_id,
            "pipeline_id": self.pipeline_id,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
        }


class ReasoningStore:
    """In-memory store for agent reasoning, indexed by task_id."""

    def __init__(self) -> None:
        self._reasoning: dict[str, list[AgentReasoning]] = defaultdict(list)

    def get_or_create(self, task_id: str, agent_id: str, agent_role: str) -> AgentReasoning:
        """Get existing reasoning for an agent on a task, or create a new one."""
        for r in self._reasoning[task_id]:
            if r.agent_id == agent_id:
                return r
        entry = AgentReasoning(agent_id=agent_id, agent_role=agent_role, task_id=task_id)
        self._reasoning[task_id].append(entry)
        return entry

    def get_for_task(self, task_id: str) -> list[AgentReasoning]:
        """Get all reasoning entries for a task."""
        return self._reasoning.get(task_id, [])

    def start_step(
        self, task_id: str, agent_id: str, agent_role: str,
        step_name: str, reasoning: str = "",
    ) -> ReasoningStep:
        """Record that an agent is starting a pipeline step."""
        entry = self.get_or_create(task_id, agent_id, agent_role)
        step = ReasoningStep(
            step_name=step_name,
            status="running",
            reasoning=reasoning,
            started_at=time.time(),
        )
        entry.steps.append(step)
        entry.current_step = step_name
        return step

    def complete_step(
        self, task_id: str, agent_id: str,
        step_name: str, status: str = "completed", output: str = "",
    ) -> bool:
        """Record that an agent finished a pipeline step."""
        for entry in self._reasoning.get(task_id, []):
            if entry.agent_id == agent_id:
                for step in reversed(entry.steps):
                    if step.step_name == step_name and step.status == "running":
                        step.status = status
                        step.output = output
                        step.completed_at = time.time()
                        if entry.current_step == step_name:
                            entry.current_step = None
                        return True
        return False

    def clear_task(self, task_id: str) -> None:
        self._reasoning.pop(task_id, None)
