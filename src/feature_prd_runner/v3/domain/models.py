from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional


TaskStatus = Literal[
    "backlog",
    "ready",
    "in_progress",
    "in_review",
    "done",
    "blocked",
    "cancelled",
]

Priority = Literal["P0", "P1", "P2", "P3"]
ApprovalMode = Literal["human_review", "auto_approve"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class ReviewFinding:
    id: str = field(default_factory=lambda: _id("finding"))
    task_id: str = ""
    severity: str = "medium"
    category: str = "quality"
    summary: str = ""
    file: Optional[str] = None
    line: Optional[int] = None
    suggested_fix: Optional[str] = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewFinding":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class ReviewCycle:
    id: str = field(default_factory=lambda: _id("rc"))
    task_id: str = ""
    attempt: int = 1
    findings: list[ReviewFinding] = field(default_factory=list)
    open_counts: dict[str, int] = field(default_factory=dict)
    decision: str = "changes_requested"
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["findings"] = [f.to_dict() for f in self.findings]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewCycle":
        findings = [ReviewFinding.from_dict(f) for f in list(data.get("findings", []) or []) if isinstance(f, dict)]
        return cls(
            id=str(data.get("id") or _id("rc")),
            task_id=str(data.get("task_id") or ""),
            attempt=int(data.get("attempt") or 1),
            findings=findings,
            open_counts=dict(data.get("open_counts") or {}),
            decision=str(data.get("decision") or "changes_requested"),
            created_at=str(data.get("created_at") or now_iso()),
        )


@dataclass
class Task:
    id: str = field(default_factory=lambda: _id("task"))
    title: str = ""
    description: str = ""
    task_type: str = "feature"
    priority: Priority = "P2"
    status: TaskStatus = "backlog"
    labels: list[str] = field(default_factory=list)

    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    pipeline_template: list[str] = field(default_factory=lambda: ["plan", "implement", "verify", "review"])
    current_step: Optional[str] = None
    current_agent_id: Optional[str] = None
    run_ids: list[str] = field(default_factory=list)
    retry_count: int = 0
    error: Optional[str] = None

    quality_gate: dict[str, int] = field(default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})
    approval_mode: ApprovalMode = "human_review"

    source: str = "manual"

    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        payload = {k: data.get(k) for k in cls.__dataclass_fields__}
        payload["id"] = str(data.get("id") or _id("task"))
        payload["title"] = str(data.get("title") or "")
        payload["priority"] = str(data.get("priority") or "P2")
        payload["status"] = str(data.get("status") or "backlog")
        payload["created_at"] = str(data.get("created_at") or now_iso())
        payload["updated_at"] = str(data.get("updated_at") or now_iso())
        payload["blocked_by"] = list(data.get("blocked_by") or [])
        payload["blocks"] = list(data.get("blocks") or [])
        payload["children_ids"] = list(data.get("children_ids") or [])
        payload["run_ids"] = list(data.get("run_ids") or [])
        payload["labels"] = list(data.get("labels") or [])
        payload["pipeline_template"] = list(data.get("pipeline_template") or ["plan", "implement", "verify", "review"])
        payload["quality_gate"] = dict(data.get("quality_gate") or {"critical": 0, "high": 0, "medium": 0, "low": 0})
        payload["metadata"] = dict(data.get("metadata") or {})
        return cls(**payload)


@dataclass
class RunRecord:
    id: str = field(default_factory=lambda: _id("run"))
    task_id: str = ""
    branch: Optional[str] = None
    status: str = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[str] = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            id=str(data.get("id") or _id("run")),
            task_id=str(data.get("task_id") or ""),
            branch=data.get("branch"),
            status=str(data.get("status") or "queued"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            summary=data.get("summary"),
            steps=list(data.get("steps") or []),
        )


@dataclass
class QuickActionRun:
    id: str = field(default_factory=lambda: _id("qrun"))
    prompt: str = ""
    status: str = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result_summary: Optional[str] = None
    promoted_task_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuickActionRun":
        return cls(
            id=str(data.get("id") or _id("qrun")),
            prompt=str(data.get("prompt") or ""),
            status=str(data.get("status") or "queued"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result_summary=data.get("result_summary"),
            promoted_task_id=data.get("promoted_task_id"),
        )


@dataclass
class AgentRecord:
    id: str = field(default_factory=lambda: _id("agent"))
    role: str = "general"
    status: str = "running"
    capacity: int = 1
    override_provider: Optional[str] = None
    last_seen_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRecord":
        return cls(
            id=str(data.get("id") or _id("agent")),
            role=str(data.get("role") or "general"),
            status=str(data.get("status") or "running"),
            capacity=int(data.get("capacity") or 1),
            override_provider=data.get("override_provider"),
            last_seen_at=str(data.get("last_seen_at") or now_iso()),
        )
