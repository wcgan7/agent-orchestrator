from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Optional


TaskStatus = Literal[
    "backlog",
    "queued",
    "in_progress",
    "in_review",
    "done",
    "blocked",
    "cancelled",
]

Priority = Literal["P0", "P1", "P2", "P3"]
ApprovalMode = Literal["human_review", "auto_approve"]
DependencyPolicy = Literal["permissive", "prudent", "strict"]
PlanRevisionSource = Literal["worker_plan", "worker_refine", "human_edit", "import"]
PlanRevisionStatus = Literal["draft", "committed"]
PlanRefineJobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def content_sha256(value: str) -> str:
    return sha256(str(value or "").encode("utf-8")).hexdigest()


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
    status: TaskStatus = "queued"
    labels: list[str] = field(default_factory=list)

    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    pipeline_template: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    current_agent_id: Optional[str] = None
    run_ids: list[str] = field(default_factory=list)
    retry_count: int = 0
    error: Optional[str] = None

    quality_gate: dict[str, int] = field(default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})
    approval_mode: ApprovalMode = "human_review"
    hitl_mode: str = "autopilot"
    dependency_policy: DependencyPolicy = "prudent"
    pending_gate: Optional[str] = None

    source: str = "manual"
    worker_model: Optional[str] = None

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
        raw_status = str(data.get("status") or "backlog")
        payload["status"] = "queued" if raw_status == "ready" else raw_status
        payload["created_at"] = str(data.get("created_at") or now_iso())
        payload["updated_at"] = str(data.get("updated_at") or now_iso())
        payload["blocked_by"] = list(data.get("blocked_by") or [])
        payload["blocks"] = list(data.get("blocks") or [])
        payload["children_ids"] = list(data.get("children_ids") or [])
        payload["run_ids"] = list(data.get("run_ids") or [])
        payload["labels"] = list(data.get("labels") or [])
        payload["pipeline_template"] = list(data.get("pipeline_template") or [])
        payload["quality_gate"] = dict(data.get("quality_gate") or {"critical": 0, "high": 0, "medium": 0, "low": 0})
        payload["metadata"] = dict(data.get("metadata") or {})
        payload["hitl_mode"] = str(data.get("hitl_mode") or "autopilot")
        raw_dep_policy = str(data.get("dependency_policy") or "prudent")
        payload["dependency_policy"] = raw_dep_policy if raw_dep_policy in ("permissive", "prudent", "strict") else "prudent"
        payload["pending_gate"] = data.get("pending_gate")
        if "hitl_mode" not in data:
            am = str(data.get("approval_mode") or "human_review")
            payload["hitl_mode"] = "autopilot" if am == "auto_approve" else "review_only"
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
    kind: Optional[str] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None

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
            kind=data.get("kind"),
            command=data.get("command"),
            exit_code=data.get("exit_code"),
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


@dataclass
class PlanRevision:
    id: str = field(default_factory=lambda: _id("pr"))
    task_id: str = ""
    created_at: str = field(default_factory=now_iso)
    source: PlanRevisionSource = "human_edit"
    parent_revision_id: Optional[str] = None
    step: Optional[str] = None
    feedback_note: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    content: str = ""
    content_hash: str = ""
    status: PlanRevisionStatus = "draft"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content_hash"] = self.content_hash or content_sha256(self.content)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRevision":
        source = str(data.get("source") or "human_edit")
        if source not in {"worker_plan", "worker_refine", "human_edit", "import"}:
            source = "human_edit"
        status = str(data.get("status") or "draft")
        if status not in {"draft", "committed"}:
            status = "draft"
        content = str(data.get("content") or "")
        return cls(
            id=str(data.get("id") or _id("pr")),
            task_id=str(data.get("task_id") or ""),
            created_at=str(data.get("created_at") or now_iso()),
            source=source,
            parent_revision_id=(str(data.get("parent_revision_id")).strip() if data.get("parent_revision_id") else None),
            step=(str(data.get("step")).strip() if data.get("step") else None),
            feedback_note=(str(data.get("feedback_note")).strip() if data.get("feedback_note") else None),
            provider=(str(data.get("provider")).strip() if data.get("provider") else None),
            model=(str(data.get("model")).strip() if data.get("model") else None),
            content=content,
            content_hash=str(data.get("content_hash") or content_sha256(content)),
            status=status,
        )


@dataclass
class PlanRefineJob:
    id: str = field(default_factory=lambda: _id("prj"))
    task_id: str = ""
    base_revision_id: str = ""
    status: PlanRefineJobStatus = "queued"
    created_at: str = field(default_factory=now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    feedback: str = ""
    instructions: Optional[str] = None
    priority: str = "normal"
    result_revision_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRefineJob":
        status = str(data.get("status") or "queued")
        if status not in {"queued", "running", "completed", "failed", "cancelled"}:
            status = "queued"
        priority = str(data.get("priority") or "normal").lower()
        if priority not in {"normal", "high"}:
            priority = "normal"
        return cls(
            id=str(data.get("id") or _id("prj")),
            task_id=str(data.get("task_id") or ""),
            base_revision_id=str(data.get("base_revision_id") or ""),
            status=status,
            created_at=str(data.get("created_at") or now_iso()),
            started_at=(str(data.get("started_at")) if data.get("started_at") else None),
            finished_at=(str(data.get("finished_at")) if data.get("finished_at") else None),
            feedback=str(data.get("feedback") or ""),
            instructions=(str(data.get("instructions")) if data.get("instructions") else None),
            priority=priority,
            result_revision_id=(str(data.get("result_revision_id")) if data.get("result_revision_id") else None),
            error=(str(data.get("error")) if data.get("error") else None),
        )
