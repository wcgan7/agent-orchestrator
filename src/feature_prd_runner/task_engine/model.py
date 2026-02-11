"""Enhanced task model for the dynamic task engine.

This module defines the new task model that supports types, priorities, labels,
parent/child relationships, dependencies, and flexible pipeline assignment.
It is designed to coexist with the existing TaskState model in models.py —
legacy tasks are converted via ``from_legacy_task()``.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """The kind of work a task represents."""

    FEATURE = "feature"
    BUG = "bug"
    REFACTOR = "refactor"
    RESEARCH = "research"
    REVIEW = "review"
    TEST = "test"
    DOCS = "docs"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class TaskPriority(str, Enum):
    """Priority level — P0 is most urgent."""

    P0 = "P0"  # Critical / drop everything
    P1 = "P1"  # High
    P2 = "P2"  # Medium (default)
    P3 = "P3"  # Low / nice-to-have

    @property
    def sort_key(self) -> int:
        return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[self.value]


class TaskStatus(str, Enum):
    """Board-level status used for Kanban columns."""

    BACKLOG = "backlog"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class EffortEstimate(str, Enum):
    """T-shirt size effort estimate."""

    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"


class TaskSource(str, Enum):
    """How the task was created."""

    MANUAL = "manual"
    PRD_IMPORT = "prd_import"
    REPO_REVIEW = "repo_review"
    BUG_SCAN = "bug_scan"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_AUDIT = "performance_audit"
    ENHANCEMENT_BRAINSTORM = "enhancement_brainstorm"
    AGENT_DISCOVERED = "agent_discovered"
    LEGACY_MIGRATION = "legacy_migration"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    """Short human-friendly task ID: ``task-<8hex>``."""
    return f"task-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A first-class task that lives on the dynamic task board.

    This replaces the implicit task model previously derived from PRD phases.
    It is richer (types, priorities, labels, hierarchy) while remaining fully
    serializable to YAML / JSON for file-based persistence.
    """

    # Identity
    id: str = field(default_factory=_generate_id)
    title: str = ""
    description: str = ""

    # Classification
    task_type: TaskType = TaskType.FEATURE
    priority: TaskPriority = TaskPriority.P2
    status: TaskStatus = TaskStatus.BACKLOG
    effort: Optional[EffortEstimate] = None
    labels: list[str] = field(default_factory=list)

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Dependencies
    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    # Assignment
    assignee: Optional[str] = None  # agent-id or username
    assignee_type: Optional[str] = None  # "agent" or "human"

    # Work definition
    acceptance_criteria: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    related_tasks: list[str] = field(default_factory=list)
    pipeline_template: Optional[str] = None  # None = auto-detect from type

    # Execution tracking (populated at runtime)
    current_step: Optional[str] = None
    current_agent_id: Optional[str] = None
    run_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0

    # Provenance
    source: TaskSource = TaskSource.MANUAL
    created_by: Optional[str] = None  # username or system
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    completed_at: Optional[str] = None

    # Bridge to legacy system (non-None when migrated from old TaskState)
    legacy_phase_id: Optional[str] = None
    legacy_task_id: Optional[str] = None

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # JSON Schema
    # ------------------------------------------------------------------

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema (draft-07) describing the Task model.

        This is used for validation at API boundaries and for documentation.
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Task",
            "type": "object",
            "required": ["id", "title"],
            "properties": {
                "id": {"type": "string", "pattern": "^task-[a-f0-9]{8}$"},
                "title": {"type": "string", "minLength": 1, "maxLength": 500},
                "description": {"type": "string"},
                "task_type": {"type": "string", "enum": [e.value for e in TaskType]},
                "priority": {"type": "string", "enum": [e.value for e in TaskPriority]},
                "status": {"type": "string", "enum": [e.value for e in TaskStatus]},
                "effort": {"type": ["string", "null"], "enum": [None] + [e.value for e in EffortEstimate]},
                "labels": {"type": "array", "items": {"type": "string"}},
                "parent_id": {"type": ["string", "null"]},
                "children_ids": {"type": "array", "items": {"type": "string"}},
                "blocked_by": {"type": "array", "items": {"type": "string"}},
                "blocks": {"type": "array", "items": {"type": "string"}},
                "assignee": {"type": ["string", "null"]},
                "assignee_type": {"type": ["string", "null"], "enum": [None, "agent", "human"]},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                "context_files": {"type": "array", "items": {"type": "string"}},
                "related_tasks": {"type": "array", "items": {"type": "string"}},
                "pipeline_template": {"type": ["string", "null"]},
                "current_step": {"type": ["string", "null"]},
                "current_agent_id": {"type": ["string", "null"]},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "error": {"type": ["string", "null"]},
                "error_type": {"type": ["string", "null"]},
                "retry_count": {"type": "integer", "minimum": 0},
                "source": {"type": "string", "enum": [e.value for e in TaskSource]},
                "created_by": {"type": ["string", "null"]},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": ["string", "null"], "format": "date-time"},
                "legacy_phase_id": {"type": ["string", "null"]},
                "legacy_task_id": {"type": ["string", "null"]},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        }

    @classmethod
    def validate_dict(cls, data: dict[str, Any]) -> list[str]:
        """Lightweight validation of a task dict against the schema.

        Returns a list of error strings (empty = valid). This avoids a
        dependency on ``jsonschema`` by checking the most important
        constraints directly.
        """
        errors: list[str] = []
        if not isinstance(data, dict):
            return ["Expected a dict"]
        if not data.get("title"):
            errors.append("'title' is required and must be non-empty")
        task_type = data.get("task_type")
        if task_type is not None:
            valid_types = {e.value for e in TaskType}
            if task_type not in valid_types:
                errors.append(f"'task_type' must be one of {sorted(valid_types)}, got '{task_type}'")
        priority = data.get("priority")
        if priority is not None:
            valid_prios = {e.value for e in TaskPriority}
            if priority not in valid_prios:
                errors.append(f"'priority' must be one of {sorted(valid_prios)}, got '{priority}'")
        status = data.get("status")
        if status is not None:
            valid_statuses = {e.value for e in TaskStatus}
            if status not in valid_statuses:
                errors.append(f"'status' must be one of {sorted(valid_statuses)}, got '{status}'")
        source = data.get("source")
        if source is not None:
            valid_sources = {e.value for e in TaskSource}
            if source not in valid_sources:
                errors.append(f"'source' must be one of {sorted(valid_sources)}, got '{source}'")
        for list_field in ("labels", "children_ids", "blocked_by", "blocks",
                           "acceptance_criteria", "context_files", "related_tasks", "run_ids"):
            val = data.get(list_field)
            if val is not None and not isinstance(val, list):
                errors.append(f"'{list_field}' must be an array")
        retry = data.get("retry_count")
        if retry is not None:
            if not isinstance(retry, int) or retry < 0:
                errors.append("'retry_count' must be a non-negative integer")
        return errors

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for YAML/JSON persistence."""
        data: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, Enum):
                data[k] = v.value
            else:
                data[k] = v
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Deserialize from a plain dict, coercing enums gracefully."""
        d = dict(data)  # shallow copy

        def _enum(enum_cls: type[Enum], key: str, default: Enum) -> Enum:
            raw = d.pop(key, None)
            if raw is None:
                return default
            if isinstance(raw, enum_cls):
                return raw
            try:
                return enum_cls(str(raw))
            except (ValueError, KeyError):
                return default

        task_type = _enum(TaskType, "task_type", TaskType.FEATURE)
        priority = _enum(TaskPriority, "priority", TaskPriority.P2)
        status = _enum(TaskStatus, "status", TaskStatus.BACKLOG)
        source = _enum(TaskSource, "source", TaskSource.MANUAL)
        effort_raw = d.pop("effort", None)
        effort: Optional[EffortEstimate] = None
        if effort_raw is not None:
            try:
                effort = EffortEstimate(str(effort_raw))
            except (ValueError, KeyError):
                effort = None

        return cls(
            id=str(d.pop("id", _generate_id())),
            title=str(d.pop("title", "")),
            description=str(d.pop("description", "")),
            task_type=task_type,
            priority=priority,
            status=status,
            effort=effort,
            labels=list(d.pop("labels", []) or []),
            parent_id=d.pop("parent_id", None),
            children_ids=list(d.pop("children_ids", []) or []),
            blocked_by=list(d.pop("blocked_by", []) or []),
            blocks=list(d.pop("blocks", []) or []),
            assignee=d.pop("assignee", None),
            assignee_type=d.pop("assignee_type", None),
            acceptance_criteria=list(d.pop("acceptance_criteria", []) or []),
            context_files=list(d.pop("context_files", []) or []),
            related_tasks=list(d.pop("related_tasks", []) or []),
            pipeline_template=d.pop("pipeline_template", None),
            current_step=d.pop("current_step", None),
            current_agent_id=d.pop("current_agent_id", None),
            run_ids=list(d.pop("run_ids", []) or []),
            error=d.pop("error", None),
            error_type=d.pop("error_type", None),
            retry_count=int(d.pop("retry_count", 0) or 0),
            source=source,
            created_by=d.pop("created_by", None),
            created_at=str(d.pop("created_at", _now_iso())),
            updated_at=str(d.pop("updated_at", _now_iso())),
            completed_at=d.pop("completed_at", None),
            legacy_phase_id=d.pop("legacy_phase_id", None),
            legacy_task_id=d.pop("legacy_task_id", None),
            metadata=dict(d.pop("metadata", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Bump ``updated_at`` to now."""
        self.updated_at = _now_iso()

    def transition(self, new_status: TaskStatus) -> None:
        """Move to *new_status* with timestamp bookkeeping."""
        self.status = new_status
        if new_status == TaskStatus.DONE:
            self.completed_at = _now_iso()
        self.touch()

    def assign(self, assignee_id: str, assignee_type: str = "agent") -> None:
        self.assignee = assignee_id
        self.assignee_type = assignee_type
        self.touch()

    def unassign(self) -> None:
        self.assignee = None
        self.assignee_type = None
        self.current_agent_id = None
        self.touch()

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.CANCELLED)

    @property
    def is_actionable(self) -> bool:
        """True if the task can be picked up by an agent right now."""
        return self.status == TaskStatus.READY and not self.blocked_by

    # ------------------------------------------------------------------
    # Dependency helpers
    # ------------------------------------------------------------------

    def add_blocked_by(self, task_id: str) -> None:
        if task_id not in self.blocked_by:
            self.blocked_by.append(task_id)
            self.touch()

    def remove_blocked_by(self, task_id: str) -> None:
        if task_id in self.blocked_by:
            self.blocked_by.remove(task_id)
            self.touch()

    def add_blocks(self, task_id: str) -> None:
        if task_id not in self.blocks:
            self.blocks.append(task_id)
            self.touch()

    def remove_blocks(self, task_id: str) -> None:
        if task_id in self.blocks:
            self.blocks.remove(task_id)
            self.touch()


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

_LEGACY_LIFECYCLE_TO_STATUS = {
    "ready": TaskStatus.READY,
    "running": TaskStatus.IN_PROGRESS,
    "waiting_human": TaskStatus.BLOCKED,
    "done": TaskStatus.DONE,
    "cancelled": TaskStatus.CANCELLED,
}


def from_legacy_task(data: dict[str, Any]) -> Task:
    """Convert a legacy ``task_queue.yaml`` task dict into a new :class:`Task`.

    This preserves the original task/phase IDs in ``legacy_task_id`` and
    ``legacy_phase_id`` so the orchestrator can still find them.
    """
    legacy_id = str(data.get("id", ""))
    phase_id = data.get("phase_id")

    lifecycle = str(data.get("lifecycle", "ready"))
    status = _LEGACY_LIFECYCLE_TO_STATUS.get(lifecycle, TaskStatus.BACKLOG)

    # Map legacy integer priority to TaskPriority
    legacy_prio = int(data.get("priority", 0) or 0)
    if legacy_prio <= 0:
        prio = TaskPriority.P2
    elif legacy_prio == 1:
        prio = TaskPriority.P1
    else:
        prio = TaskPriority.P0

    return Task(
        id=_generate_id(),
        title=data.get("title") or f"Phase: {phase_id or legacy_id}",
        description=data.get("description", "") or "",
        task_type=TaskType.FEATURE,
        priority=prio,
        status=status,
        labels=["migrated"],
        acceptance_criteria=list(data.get("acceptance_criteria", []) or []),
        context_files=list(data.get("last_changed_files", []) or []),
        source=TaskSource.LEGACY_MIGRATION,
        created_by="system",
        error=data.get("last_error"),
        error_type=data.get("last_error_type"),
        legacy_phase_id=phase_id,
        legacy_task_id=legacy_id,
        metadata={
            "legacy_step": data.get("step"),
            "legacy_branch": data.get("branch"),
            "legacy_worker_attempts": data.get("worker_attempts", 0),
        },
    )
