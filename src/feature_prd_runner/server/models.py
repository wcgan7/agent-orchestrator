"""Pydantic models for API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PhaseInfo(BaseModel):
    """Phase information."""

    id: str
    name: str
    description: str
    status: str
    deps: list[str] = Field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0


class TaskInfo(BaseModel):
    """Task information."""

    id: str
    type: str
    phase_id: Optional[str] = None
    step: str
    lifecycle: str
    status: str
    branch: Optional[str] = None
    last_error: Optional[str] = None
    last_run_id: Optional[str] = None
    worker_attempts: int = 0


class RunMetrics(BaseModel):
    """Run metrics."""

    tokens_used: int = 0
    api_calls: int = 0
    estimated_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    phases_completed: int = 0
    phases_total: int = 0
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0


class RunInfo(BaseModel):
    """Run information."""

    run_id: str
    task_id: str
    phase: str
    step: str
    status: str
    started_at: str
    updated_at: str
    metrics: Optional[RunMetrics] = None


class RunDetail(BaseModel):
    """Detailed run information."""

    run_id: str
    task_id: str
    phase: str
    step: str
    status: str
    started_at: str
    updated_at: str
    current_task_id: Optional[str] = None
    current_phase_id: Optional[str] = None
    last_error: Optional[str] = None
    phases: list[PhaseInfo] = Field(default_factory=list)
    tasks: list[TaskInfo] = Field(default_factory=list)
    metrics: Optional[RunMetrics] = None


class ProjectStatus(BaseModel):
    """Project status overview."""

    project_dir: str
    status: str
    current_task_id: Optional[str] = None
    current_phase_id: Optional[str] = None
    run_id: Optional[str] = None
    last_error: Optional[str] = None
    phases_completed: int = 0
    phases_total: int = 0
    tasks_ready: int = 0
    tasks_running: int = 0
    tasks_done: int = 0
    tasks_blocked: int = 0


class LogEntry(BaseModel):
    """Log entry."""

    timestamp: str
    level: str
    message: str
    run_id: Optional[str] = None
    task_id: Optional[str] = None


class ProjectInfo(BaseModel):
    """Project information."""

    name: str
    path: str
    status: str  # active, idle, error
    last_run: Optional[str] = None
    phases_total: int = 0
    phases_completed: int = 0


class ControlAction(BaseModel):
    """Control action request."""

    action: str  # pause, resume, skip, retry, stop
    task_id: Optional[str] = None
    step: Optional[str] = None
    reason: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class ControlResponse(BaseModel):
    """Control action response."""

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None


class LoginRequest(BaseModel):
    """Login request."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    token_type: str = "bearer"
    username: str


class AuthStatus(BaseModel):
    """Authentication status."""

    enabled: bool
    authenticated: bool
    username: Optional[str] = None


class ApprovalGateInfo(BaseModel):
    """Approval gate information."""

    request_id: str
    gate_type: str
    message: str
    task_id: Optional[str] = None
    phase_id: Optional[str] = None
    created_at: str
    timeout: Optional[int] = None
    context: dict[str, Any] = Field(default_factory=dict)
    show_diff: bool = False
    show_plan: bool = False
    show_tests: bool = False
    show_review: bool = False


class ApprovalAction(BaseModel):
    """Approval action request."""

    request_id: str
    approved: bool
    feedback: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message."""

    type: str  # status_update, log_entry, phase_complete, error
    timestamp: str
    data: dict[str, Any]


class ChatMessage(BaseModel):
    """Chat message between human and worker."""

    id: str
    type: str  # guidance, clarification_request, clarification_response
    content: str
    timestamp: str
    from_human: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Request to send a message to the worker."""

    content: str
    type: str = "guidance"  # guidance, clarification_request
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileChange(BaseModel):
    """File change with diff."""

    file_path: str
    status: str  # added, modified, deleted
    additions: int
    deletions: int
    diff: str
    approved: Optional[bool] = None
    comments: list[str] = Field(default_factory=list)


class FileReviewRequest(BaseModel):
    """Request to approve/reject a file."""

    file_path: str
    approved: bool
    comment: Optional[str] = None


class BreakpointInfo(BaseModel):
    """Breakpoint information."""

    id: str
    trigger: str
    target: str
    task_id: Optional[str] = None
    condition: Optional[str] = None
    action: str = "pause"
    enabled: bool = True
    hit_count: int = 0
    created_at: Optional[str] = None


class BreakpointCreateRequest(BaseModel):
    """Request to create a breakpoint."""

    trigger: str
    target: str
    task_id: Optional[str] = None
    condition: Optional[str] = None
    action: str = "pause"


class StartRunRequest(BaseModel):
    """Request to start a new run."""

    mode: str  # "full_prd" or "quick_prompt"
    content: str  # PRD text for full_prd, or prompt for quick_prompt
    test_command: Optional[str] = None
    build_command: Optional[str] = None
    verification_profile: str = "none"  # none, python
    auto_approve_plans: bool = False
    auto_approve_changes: bool = False
    auto_approve_commits: bool = False


class StartRunResponse(BaseModel):
    """Response from starting a run."""

    success: bool
    message: str
    run_id: Optional[str] = None
    prd_path: Optional[str] = None


class ExecTaskRequest(BaseModel):
    """Request to execute a one-off task (exec command)."""

    prompt: str
    override_agents: bool = False
    context_files: Optional[str] = None  # Comma-separated file paths
    shift_minutes: int = 45
    heartbeat_seconds: int = 120


class ExecTaskResponse(BaseModel):
    """Response from executing a task."""

    success: bool
    message: str
    run_id: Optional[str] = None
    error: Optional[str] = None
