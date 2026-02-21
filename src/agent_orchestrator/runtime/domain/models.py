"""Domain model dataclasses and normalization helpers."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Optional, cast


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
_VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}
_VALID_TASK_STATUSES = {"backlog", "queued", "in_progress", "in_review", "done", "blocked", "cancelled"}
_VALID_APPROVAL_MODES = {"human_review", "auto_approve"}
_VALID_DEPENDENCY_POLICIES = {"permissive", "prudent", "strict"}
_VALID_PLAN_REVISION_SOURCES = {"worker_plan", "worker_refine", "human_edit", "import"}
_VALID_PLAN_REVISION_STATUSES = {"draft", "committed"}
_VALID_PLAN_REFINE_JOB_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}


def now_iso() -> str:
    """Get the current UTC timestamp formatted as ISO-8601 text."""
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def content_sha256(value: str) -> str:
    """Hash text content to a deterministic SHA-256 hex digest."""
    return sha256(str(value or "").encode("utf-8")).hexdigest()


@dataclass
class ReviewFinding:
    """A single review issue captured for a task."""
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
        """Serialize the finding to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewFinding":
        """Deserialize a finding, normalizing optional numeric fields."""
        raw_line = data.get("line")
        line: int | None
        try:
            line = int(raw_line) if raw_line is not None else None
        except (TypeError, ValueError):
            line = None
        return cls(
            id=str(data.get("id") or _id("finding")),
            task_id=str(data.get("task_id") or ""),
            severity=str(data.get("severity") or "medium"),
            category=str(data.get("category") or "quality"),
            summary=str(data.get("summary") or ""),
            file=(str(data.get("file")) if data.get("file") is not None else None),
            line=line,
            suggested_fix=(str(data.get("suggested_fix")) if data.get("suggested_fix") is not None else None),
            status=str(data.get("status") or "open"),
        )


@dataclass
class ReviewCycle:
    """Review attempt snapshot for a task, including findings."""
    id: str = field(default_factory=lambda: _id("rc"))
    task_id: str = ""
    attempt: int = 1
    findings: list[ReviewFinding] = field(default_factory=list)
    open_counts: dict[str, int] = field(default_factory=dict)
    decision: str = "changes_requested"
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the review cycle, including nested findings."""
        data = asdict(self)
        data["findings"] = [f.to_dict() for f in self.findings]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewCycle":
        """Deserialize a review cycle from persisted storage."""
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
    """Primary unit of orchestrated work tracked by the runtime."""
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
    project_commands: Optional[dict[str, dict[str, str]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize a task to a dictionary payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Deserialize and normalize a task from persisted data."""
        priority = str(data.get("priority") or "P2")
        if priority not in _VALID_PRIORITIES:
            priority = "P2"
        raw_status = str(data.get("status") or "backlog")
        status = "queued" if raw_status == "ready" else raw_status
        if status not in _VALID_TASK_STATUSES:
            status = "queued"
        raw_pc = data.get("project_commands")
        project_commands = (
            {k: dict(v) for k, v in raw_pc.items() if isinstance(v, dict) and v}
            if isinstance(raw_pc, dict) and raw_pc
            else None
        )
        approval_mode = str(data.get("approval_mode") or "human_review")
        if approval_mode not in _VALID_APPROVAL_MODES:
            approval_mode = "human_review"
        hitl_mode = str(data.get("hitl_mode") or "autopilot")
        raw_dep_policy = str(data.get("dependency_policy") or "prudent")
        dependency_policy = raw_dep_policy if raw_dep_policy in _VALID_DEPENDENCY_POLICIES else "prudent"
        if "hitl_mode" not in data:
            hitl_mode = "autopilot" if approval_mode == "auto_approve" else "review_only"
        raw_retry_count = data.get("retry_count")
        try:
            retry_count = int(raw_retry_count) if raw_retry_count is not None else 0
        except (TypeError, ValueError):
            retry_count = 0
        return cls(
            id=str(data.get("id") or _id("task")),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            task_type=str(data.get("task_type") or "feature"),
            priority=cast(Priority, priority),
            status=cast(TaskStatus, status),
            labels=list(data.get("labels") or []),
            blocked_by=list(data.get("blocked_by") or []),
            blocks=list(data.get("blocks") or []),
            parent_id=(str(data.get("parent_id")) if data.get("parent_id") else None),
            children_ids=list(data.get("children_ids") or []),
            pipeline_template=list(data.get("pipeline_template") or []),
            current_step=(str(data.get("current_step")) if data.get("current_step") else None),
            current_agent_id=(str(data.get("current_agent_id")) if data.get("current_agent_id") else None),
            run_ids=list(data.get("run_ids") or []),
            retry_count=retry_count,
            error=(str(data.get("error")) if data.get("error") is not None else None),
            quality_gate=dict(data.get("quality_gate") or {"critical": 0, "high": 0, "medium": 0, "low": 0}),
            approval_mode=cast(ApprovalMode, approval_mode),
            hitl_mode=hitl_mode,
            dependency_policy=cast(DependencyPolicy, dependency_policy),
            pending_gate=(str(data.get("pending_gate")) if data.get("pending_gate") else None),
            source=str(data.get("source") or "manual"),
            worker_model=(str(data.get("worker_model")) if data.get("worker_model") else None),
            created_at=str(data.get("created_at") or now_iso()),
            updated_at=str(data.get("updated_at") or now_iso()),
            metadata=dict(data.get("metadata") or {}),
            project_commands=project_commands,
        )


@dataclass
class RunRecord:
    """Execution run metadata and per-step outputs for a task."""
    id: str = field(default_factory=lambda: _id("run"))
    task_id: str = ""
    branch: Optional[str] = None
    status: str = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[str] = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a run record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        """Deserialize a run record from persisted data."""
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
class TerminalSession:
    """Persisted terminal session metadata and log locations."""
    id: str = field(default_factory=lambda: _id("term"))
    project_id: str = ""
    status: str = "starting"
    shell: str = ""
    cwd: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    pid: Optional[int] = None
    cols: int = 120
    rows: int = 36
    audit_log_path: Optional[str] = None
    output_log_path: Optional[str] = None
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize a terminal session record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminalSession":
        """Deserialize a terminal session and coerce numeric fields."""
        raw_exit = data.get("exit_code")
        exit_code = int(raw_exit) if raw_exit is not None else None
        raw_pid = data.get("pid")
        pid = int(raw_pid) if raw_pid is not None else None
        try:
            cols = int(data.get("cols") or 120)
        except (TypeError, ValueError):
            cols = 120
        try:
            rows = int(data.get("rows") or 36)
        except (TypeError, ValueError):
            rows = 36
        return cls(
            id=str(data.get("id") or _id("term")),
            project_id=str(data.get("project_id") or ""),
            status=str(data.get("status") or "starting"),
            shell=str(data.get("shell") or ""),
            cwd=str(data.get("cwd") or ""),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            exit_code=exit_code,
            pid=pid,
            cols=cols,
            rows=rows,
            audit_log_path=data.get("audit_log_path"),
            output_log_path=data.get("output_log_path"),
            last_error=data.get("last_error"),
        )


@dataclass
class AgentRecord:
    """Persisted runtime state for a worker agent process."""
    id: str = field(default_factory=lambda: _id("agent"))
    role: str = "general"
    status: str = "running"
    capacity: int = 1
    override_provider: Optional[str] = None
    last_seen_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize an agent record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRecord":
        """Deserialize an agent record from persisted data."""
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
    """Versioned plan document snapshot associated with a task."""
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
        """Serialize a plan revision and ensure ``content_hash`` is set."""
        data = asdict(self)
        data["content_hash"] = self.content_hash or content_sha256(self.content)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRevision":
        """Deserialize and validate a plan revision payload."""
        source = str(data.get("source") or "human_edit")
        if source not in _VALID_PLAN_REVISION_SOURCES:
            source = "human_edit"
        status = str(data.get("status") or "draft")
        if status not in _VALID_PLAN_REVISION_STATUSES:
            status = "draft"
        content = str(data.get("content") or "")
        return cls(
            id=str(data.get("id") or _id("pr")),
            task_id=str(data.get("task_id") or ""),
            created_at=str(data.get("created_at") or now_iso()),
            source=cast(PlanRevisionSource, source),
            parent_revision_id=(str(data.get("parent_revision_id")).strip() if data.get("parent_revision_id") else None),
            step=(str(data.get("step")).strip() if data.get("step") else None),
            feedback_note=(str(data.get("feedback_note")).strip() if data.get("feedback_note") else None),
            provider=(str(data.get("provider")).strip() if data.get("provider") else None),
            model=(str(data.get("model")).strip() if data.get("model") else None),
            content=content,
            content_hash=str(data.get("content_hash") or content_sha256(content)),
            status=cast(PlanRevisionStatus, status),
        )


@dataclass
class PlanRefineJob:
    """Asynchronous worker job used to refine an existing plan revision."""
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
        """Serialize a refine job record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRefineJob":
        """Deserialize and normalize a refine job payload."""
        status = str(data.get("status") or "queued")
        if status not in _VALID_PLAN_REFINE_JOB_STATUSES:
            status = "queued"
        priority = str(data.get("priority") or "normal").lower()
        if priority not in {"normal", "high"}:
            priority = "normal"
        return cls(
            id=str(data.get("id") or _id("prj")),
            task_id=str(data.get("task_id") or ""),
            base_revision_id=str(data.get("base_revision_id") or ""),
            status=cast(PlanRefineJobStatus, status),
            created_at=str(data.get("created_at") or now_iso()),
            started_at=(str(data.get("started_at")) if data.get("started_at") else None),
            finished_at=(str(data.get("finished_at")) if data.get("finished_at") else None),
            feedback=str(data.get("feedback") or ""),
            instructions=(str(data.get("instructions")) if data.get("instructions") else None),
            priority=priority,
            result_revision_id=(str(data.get("result_revision_id")) if data.get("result_revision_id") else None),
            error=(str(data.get("error")) if data.get("error") else None),
        )
