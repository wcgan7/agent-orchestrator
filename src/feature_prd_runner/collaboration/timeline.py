"""Activity timeline — server-side aggregation of all events for a task.

Collects events from feedback, review comments, agent reasoning, and task
state changes into a unified chronological stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .feedback import FeedbackStore
from .reasoning import ReasoningStore


@dataclass
class TimelineEvent:
    """One event in the activity timeline."""
    id: str
    type: str          # status_change | agent_output | feedback | comment | file_change | reasoning
    timestamp: str     # ISO format
    actor: str         # username or agent_id
    actor_type: str    # "human" | "agent" | "system"
    summary: str
    details: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "summary": self.summary,
            "details": self.details,
            "metadata": self.metadata,
        }


class StateChangeStore:
    """In-memory store for task state change events and commit events."""

    def __init__(self) -> None:
        # task_id -> list of events
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._counter = 0

    def record_state_change(
        self,
        task_id: str,
        old_status: str,
        new_status: str,
        actor: str = "system",
        actor_type: str = "system",
    ) -> TimelineEvent:
        """Record a task status transition."""
        self._counter += 1
        event = TimelineEvent(
            id=f"sc-{self._counter}",
            type="status_change",
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            actor_type=actor_type,
            summary=f"Status changed: {old_status} → {new_status}",
            details="",
            metadata={"old_status": old_status, "new_status": new_status},
        )
        self._events.setdefault(task_id, []).append(event.to_dict())
        return event

    def record_commit(
        self,
        task_id: str,
        commit_hash: str,
        message: str,
        actor: str = "agent",
        actor_type: str = "agent",
    ) -> TimelineEvent:
        """Record a git commit event."""
        self._counter += 1
        event = TimelineEvent(
            id=f"cm-{self._counter}",
            type="commit",
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            actor_type=actor_type,
            summary=f"Commit: {message[:80]}",
            details=message,
            metadata={"commit_hash": commit_hash},
        )
        self._events.setdefault(task_id, []).append(event.to_dict())
        return event

    def record_file_change(
        self,
        task_id: str,
        file_path: str,
        change_type: str,
        actor: str = "agent",
        actor_type: str = "agent",
    ) -> TimelineEvent:
        """Record a file modification event."""
        self._counter += 1
        event = TimelineEvent(
            id=f"fc-{self._counter}",
            type="file_change",
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            actor_type=actor_type,
            summary=f"File {change_type}: {file_path}",
            details="",
            metadata={"file_path": file_path, "change_type": change_type},
        )
        self._events.setdefault(task_id, []).append(event.to_dict())
        return event

    def get_events(self, task_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(task_id, []))


class TimelineAggregator:
    """Aggregates events from multiple sources into a unified timeline."""

    def __init__(
        self,
        feedback_store: FeedbackStore,
        reasoning_store: Optional[ReasoningStore] = None,
        state_change_store: Optional[StateChangeStore] = None,
    ) -> None:
        self.feedback_store = feedback_store
        self.reasoning_store = reasoning_store
        self.state_change_store = state_change_store

    def get_timeline(self, task_id: str, limit: int = 100) -> list[TimelineEvent]:
        """Aggregate all events for a task, sorted newest first."""
        events: list[TimelineEvent] = []

        # Feedback events
        for fb in self.feedback_store.get_feedback(task_id):
            events.append(TimelineEvent(
                id=fb.id,
                type="feedback",
                timestamp=fb.created_at,
                actor=fb.created_by or "user",
                actor_type="human",
                summary=f"{fb.feedback_type.value}: {fb.summary}",
                details=fb.details,
                metadata={"priority": fb.priority.value, "status": fb.status.value},
            ))

        # Review comment events
        for c in self.feedback_store.get_comments(task_id):
            events.append(TimelineEvent(
                id=c.id,
                type="comment",
                timestamp=c.created_at,
                actor=c.author or "user",
                actor_type=c.author_type,
                summary=f"Comment on {c.file_path}:{c.line_number}",
                details=c.body,
                metadata={
                    "file_path": c.file_path,
                    "line_number": c.line_number,
                    "resolved": c.resolved,
                },
            ))

        # Reasoning events (from agents)
        if self.reasoning_store:
            for reasoning in self.reasoning_store.get_for_task(task_id):
                for step in reasoning.steps:
                    if step.started_at:
                        ts = datetime.fromtimestamp(step.started_at, tz=timezone.utc).isoformat()
                        events.append(TimelineEvent(
                            id=f"{reasoning.agent_id}:{step.step_name}",
                            type="reasoning",
                            timestamp=ts,
                            actor=reasoning.agent_id,
                            actor_type="agent",
                            summary=f"{reasoning.agent_role} / {step.step_name}: {step.status}",
                            details=step.reasoning or step.output,
                            metadata={
                                "step_name": step.step_name,
                                "status": step.status,
                                "duration_ms": step.duration_ms,
                            },
                        ))

        # State change, commit, and file change events
        if self.state_change_store:
            for raw in self.state_change_store.get_events(task_id):
                events.append(TimelineEvent(
                    id=raw["id"],
                    type=raw["type"],
                    timestamp=raw["timestamp"],
                    actor=raw["actor"],
                    actor_type=raw["actor_type"],
                    summary=raw["summary"],
                    details=raw.get("details", ""),
                    metadata=raw.get("metadata"),
                ))

        # Sort newest first
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
