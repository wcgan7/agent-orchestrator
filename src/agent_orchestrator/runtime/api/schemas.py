"""Pydantic request/response schemas for runtime API routes."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CreateTaskRequest(BaseModel):
    """Payload for creating a new task."""

    title: str
    description: str = ""
    task_type: str = "feature"
    priority: str = "P2"
    status: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    pipeline_template: Optional[list[str]] = None
    approval_mode: str = "human_review"
    hitl_mode: str = "autopilot"
    dependency_policy: str = ""
    source: str = "manual"
    worker_model: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_commands: Optional[dict[str, dict[str, str]]] = None
    classifier_pipeline_id: Optional[str] = None
    classifier_confidence: Optional[Literal["high", "low"]] = None
    classifier_reason: Optional[str] = None
    was_user_override: Optional[bool] = None


class PipelineClassificationRequest(BaseModel):
    """Payload used to classify a task into a pipeline."""

    title: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineClassificationResponse(BaseModel):
    """Normalized pipeline-classification result."""

    pipeline_id: str
    task_type: str
    confidence: Literal["high", "low"]
    reason: str
    allowed_pipelines: list[str]


class UpdateTaskRequest(BaseModel):
    """Patch payload for updating mutable task fields."""

    title: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    labels: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = None
    approval_mode: Optional[str] = None
    hitl_mode: Optional[str] = None
    dependency_policy: Optional[str] = None
    worker_model: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    project_commands: Optional[dict[str, dict[str, str]]] = None


class TransitionRequest(BaseModel):
    """Payload for transitioning a task status."""

    status: str


class AddDependencyRequest(BaseModel):
    """Payload for adding a task dependency edge."""

    depends_on: str


class PrdPreviewRequest(BaseModel):
    """Payload for PRD preview ingestion."""

    title: Optional[str] = None
    content: str
    default_priority: str = "P2"


class PrdCommitRequest(BaseModel):
    """Payload for committing a previously previewed PRD import job."""

    job_id: str


class StartTerminalSessionRequest(BaseModel):
    """Payload for starting an interactive terminal session."""

    cols: Optional[int] = Field(default=120, ge=2, le=500)
    rows: Optional[int] = Field(default=36, ge=2, le=300)
    shell: Optional[str] = None


class TerminalInputRequest(BaseModel):
    """Payload containing terminal input bytes as UTF-8 text."""

    data: str


class TerminalResizeRequest(BaseModel):
    """Payload for resizing a terminal PTY window."""

    cols: int = Field(ge=2, le=500)
    rows: int = Field(ge=2, le=300)


class StopTerminalSessionRequest(BaseModel):
    """Payload for stopping a terminal session."""

    signal: Literal["TERM", "KILL"] = "TERM"


class PlanRefineRequest(BaseModel):
    """Payload for queueing an asynchronous plan-refine job."""

    base_revision_id: Optional[str] = None
    feedback: str
    instructions: Optional[str] = None
    priority: Literal["normal", "high"] = "normal"


class CommitPlanRequest(BaseModel):
    """Payload for committing a specific plan revision."""

    revision_id: str


class CreatePlanRevisionRequest(BaseModel):
    """Payload for creating a manual plan revision."""

    content: str
    parent_revision_id: Optional[str] = None
    feedback_note: Optional[str] = None


class GenerateTasksRequest(BaseModel):
    """Payload for generating child tasks from plan content."""

    source: Optional[Literal["committed", "revision", "override", "latest"]] = None
    revision_id: Optional[str] = None
    plan_override: Optional[str] = None
    infer_deps: bool = True


class ApproveGateRequest(BaseModel):
    """Payload for approving a pending human gate."""

    gate: Optional[str] = None


class OrchestratorControlRequest(BaseModel):
    """Payload for orchestrator control actions."""

    action: str


class OrchestratorSettingsRequest(BaseModel):
    """Settings payload for orchestrator concurrency and policies."""

    concurrency: int = Field(2, ge=1, le=128)
    auto_deps: bool = True
    max_review_attempts: int = Field(10, ge=1, le=50)


class AgentRoutingSettingsRequest(BaseModel):
    """Settings payload for role-based agent routing."""

    default_role: str = "general"
    task_type_roles: dict[str, str] = Field(default_factory=dict)
    role_provider_overrides: dict[str, str] = Field(default_factory=dict)


class WorkerProviderSettingsRequest(BaseModel):
    """Settings payload for one worker provider definition."""

    type: str = "codex"
    command: Optional[str] = None
    reasoning_effort: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    num_ctx: Optional[int] = None


class WorkersSettingsRequest(BaseModel):
    """Settings payload for worker defaults, health checks, and routing."""

    default: str = "codex"
    default_model: Optional[str] = None
    heartbeat_seconds: Optional[int] = Field(None, ge=1, le=3600)
    heartbeat_grace_seconds: Optional[int] = Field(None, ge=1, le=7200)
    routing: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, WorkerProviderSettingsRequest] = Field(default_factory=dict)


class QualityGateSettingsRequest(BaseModel):
    """Thresholds for severity-based quality gates."""

    critical: int = Field(0, ge=0)
    high: int = Field(0, ge=0)
    medium: int = Field(0, ge=0)
    low: int = Field(0, ge=0)


class DefaultsSettingsRequest(BaseModel):
    """Project-wide default policies applied to new tasks."""

    quality_gate: QualityGateSettingsRequest = QualityGateSettingsRequest(
        critical=0,
        high=0,
        medium=0,
        low=0,
    )
    dependency_policy: str = "prudent"


class LanguageCommandsRequest(BaseModel):
    """Per-language command overrides used in worker prompts."""

    test: Optional[str] = None
    lint: Optional[str] = None
    typecheck: Optional[str] = None
    format: Optional[str] = None


class ProjectSettingsRequest(BaseModel):
    """Settings payload for project-specific command configuration."""

    commands: Optional[dict[str, LanguageCommandsRequest]] = None


class UpdateSettingsRequest(BaseModel):
    """Top-level PATCH payload for runtime settings."""

    orchestrator: Optional[OrchestratorSettingsRequest] = None
    agent_routing: Optional[AgentRoutingSettingsRequest] = None
    defaults: Optional[DefaultsSettingsRequest] = None
    workers: Optional[WorkersSettingsRequest] = None
    project: Optional[ProjectSettingsRequest] = None


class SpawnAgentRequest(BaseModel):
    """Payload for creating a new runtime agent record."""

    role: str = "general"
    capacity: int = 1
    override_provider: Optional[str] = None


class ReviewActionRequest(BaseModel):
    """Payload for human review approval/request-changes actions."""

    guidance: Optional[str] = None


class RetryTaskRequest(BaseModel):
    """Payload for retrying a task from a chosen step."""

    guidance: Optional[str] = None
    start_from_step: Optional[str] = None


class AddFeedbackRequest(BaseModel):
    """Payload for posting reviewer feedback on a task."""

    task_id: str
    feedback_type: str = "general"
    priority: str = "should"
    summary: str
    details: str = ""
    target_file: Optional[str] = None


class AddCommentRequest(BaseModel):
    """Payload for adding a threaded code comment."""

    task_id: str
    file_path: str
    line_number: int = 0
    body: str
    line_type: Optional[str] = None
    parent_id: Optional[str] = None
