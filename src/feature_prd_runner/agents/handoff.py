"""Agent-to-agent handoff protocol and shared context bus.

Enables structured communication between agents working on related tasks.
For example, a reviewer agent can send actionable feedback directly to an
implementer agent, or a researcher can share findings with an implementer.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Handoff types
# ---------------------------------------------------------------------------

class HandoffType(str, Enum):
    REVIEW_FEEDBACK = "review_feedback"     # reviewer → implementer
    RESEARCH_CONTEXT = "research_context"   # researcher → implementer
    ARCHITECTURE_PLAN = "architecture_plan" # architect → implementer
    BUG_DIAGNOSIS = "bug_diagnosis"         # tester → implementer
    TEST_RESULTS = "test_results"           # tester → reviewer
    TASK_SPLIT = "task_split"              # architect → scheduler (creates sub-tasks)
    ESCALATION = "escalation"              # any agent → human (needs intervention)


@dataclass
class Handoff:
    """Structured handoff between agents."""
    id: str = field(default_factory=lambda: f"ho-{uuid.uuid4().hex[:8]}")
    type: HandoffType = HandoffType.REVIEW_FEEDBACK

    from_agent_id: str = ""
    to_agent_id: Optional[str] = None      # None = broadcast to any suitable agent
    task_id: str = ""

    # Content
    summary: str = ""
    details: str = ""
    action_items: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)     # relevant file paths
    code_snippets: dict[str, str] = field(default_factory=dict)  # file → snippet

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "task_id": self.task_id,
            "summary": self.summary,
            "details": self.details,
            "action_items": self.action_items,
            "files": self.files,
            "code_snippets": self.code_snippets,
            "created_at": self.created_at,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
        }


# ---------------------------------------------------------------------------
# Context Bus — shared memory between agents
# ---------------------------------------------------------------------------

@dataclass
class ContextEntry:
    """A piece of shared context available to all agents on a task."""
    key: str
    value: Any
    source_agent_id: str
    task_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl_seconds: Optional[int] = None  # None = permanent


class ContextBus:
    """Shared context bus for inter-agent communication.

    Agents can:
    - Post handoffs to specific agents or broadcast
    - Store and retrieve shared context by task
    - Subscribe to handoff notifications
    """

    def __init__(self) -> None:
        # Handoffs indexed by task_id
        self._handoffs: dict[str, list[Handoff]] = defaultdict(list)
        # Shared context indexed by (task_id, key)
        self._context: dict[str, dict[str, ContextEntry]] = defaultdict(dict)
        # Event subscribers
        self._handlers: list[Callable[[Handoff], None]] = []

    # -- Handoffs ------------------------------------------------------------

    def send_handoff(self, handoff: Handoff) -> None:
        """Send a handoff from one agent to another."""
        self._handoffs[handoff.task_id].append(handoff)
        for handler in self._handlers:
            try:
                handler(handoff)
            except Exception:
                pass

    def get_handoffs(
        self,
        task_id: str,
        *,
        to_agent_id: Optional[str] = None,
        handoff_type: Optional[HandoffType] = None,
        unacknowledged_only: bool = False,
    ) -> list[Handoff]:
        """Retrieve handoffs for a task with optional filters."""
        handoffs = self._handoffs.get(task_id, [])
        if to_agent_id:
            handoffs = [h for h in handoffs if h.to_agent_id in (to_agent_id, None)]
        if handoff_type:
            handoffs = [h for h in handoffs if h.type == handoff_type]
        if unacknowledged_only:
            handoffs = [h for h in handoffs if not h.acknowledged]
        return handoffs

    def acknowledge_handoff(self, handoff_id: str) -> bool:
        """Mark a handoff as acknowledged. Returns True if found."""
        for handoffs in self._handoffs.values():
            for h in handoffs:
                if h.id == handoff_id:
                    h.acknowledged = True
                    h.acknowledged_at = datetime.now(timezone.utc).isoformat()
                    return True
        return False

    def on_handoff(self, handler: Callable[[Handoff], None]) -> None:
        """Subscribe to handoff events."""
        self._handlers.append(handler)

    # -- Shared Context ------------------------------------------------------

    def set_context(self, task_id: str, key: str, value: Any, source_agent_id: str) -> None:
        """Store a shared context value for a task."""
        self._context[task_id][key] = ContextEntry(
            key=key,
            value=value,
            source_agent_id=source_agent_id,
            task_id=task_id,
        )

    def get_context(self, task_id: str, key: str) -> Any:
        """Retrieve a shared context value. Returns None if not found."""
        entry = self._context.get(task_id, {}).get(key)
        return entry.value if entry else None

    def get_all_context(self, task_id: str) -> dict[str, Any]:
        """Retrieve all shared context for a task."""
        return {k: v.value for k, v in self._context.get(task_id, {}).items()}

    def delete_context(self, task_id: str, key: str) -> None:
        if task_id in self._context:
            self._context[task_id].pop(key, None)

    def clear_task_context(self, task_id: str) -> None:
        """Remove all context and handoffs for a task."""
        self._handoffs.pop(task_id, None)
        self._context.pop(task_id, None)

    # -- Convenience factory methods -----------------------------------------

    @staticmethod
    def create_review_feedback(
        from_agent_id: str,
        to_agent_id: str,
        task_id: str,
        *,
        summary: str,
        action_items: list[str],
        files: Optional[list[str]] = None,
    ) -> Handoff:
        return Handoff(
            type=HandoffType.REVIEW_FEEDBACK,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            task_id=task_id,
            summary=summary,
            action_items=action_items,
            files=files or [],
        )

    @staticmethod
    def create_research_context(
        from_agent_id: str,
        task_id: str,
        *,
        summary: str,
        details: str,
        files: Optional[list[str]] = None,
        code_snippets: Optional[dict[str, str]] = None,
    ) -> Handoff:
        return Handoff(
            type=HandoffType.RESEARCH_CONTEXT,
            from_agent_id=from_agent_id,
            task_id=task_id,
            summary=summary,
            details=details,
            files=files or [],
            code_snippets=code_snippets or {},
        )

    @staticmethod
    def create_architecture_plan(
        from_agent_id: str,
        task_id: str,
        *,
        summary: str,
        details: str,
        action_items: list[str],
    ) -> Handoff:
        return Handoff(
            type=HandoffType.ARCHITECTURE_PLAN,
            from_agent_id=from_agent_id,
            task_id=task_id,
            summary=summary,
            details=details,
            action_items=action_items,
        )


# ---------------------------------------------------------------------------
# Escalation factory function
# ---------------------------------------------------------------------------

def create_escalation(
    from_agent_id: str,
    task_id: str,
    reason: str,
    context: Optional[dict[str, Any]] = None,
) -> Handoff:
    """Create an escalation handoff that routes to a human operator.

    Use this when an agent encounters a situation it cannot resolve
    autonomously — for example, ambiguous requirements, conflicting
    constraints, or repeated failures.
    """
    return Handoff(
        type=HandoffType.ESCALATION,
        from_agent_id=from_agent_id,
        to_agent_id=None,  # always targets a human, not another agent
        task_id=task_id,
        summary=reason,
        details=str(context) if context else "",
    )


# ---------------------------------------------------------------------------
# Auto-routing
# ---------------------------------------------------------------------------

# Maps each handoff type to the agent *role* that should receive it.
# None means the handoff cannot be auto-routed (goes to a human).
_AUTO_ROUTE_TABLE: dict[HandoffType, Optional[str]] = {
    HandoffType.REVIEW_FEEDBACK:  "implementer",
    HandoffType.RESEARCH_CONTEXT: "implementer",
    HandoffType.BUG_DIAGNOSIS:    "implementer",
    HandoffType.TEST_RESULTS:     "reviewer",
    HandoffType.ARCHITECTURE_PLAN: "implementer",
    HandoffType.TASK_SPLIT:       "architect",
    HandoffType.ESCALATION:       None,
}


def auto_route(
    handoff: Handoff,
    registry: Any,
) -> Optional[str]:
    """Select the best target agent role for a handoff.

    *registry* is expected to be an ``AgentRegistry``-like object but is
    typed as ``Any`` to avoid circular imports.

    Returns the agent **role** string (e.g. ``"implementer"``) that should
    handle the handoff, or ``None`` when the handoff requires human
    intervention (e.g. ``ESCALATION``).
    """
    target_role = _AUTO_ROUTE_TABLE.get(handoff.type)
    return target_role
