"""V2 Task API endpoints for the dynamic task board.

This module provides a FastAPI router with full CRUD, dependency management,
board views, and task generation endpoints.  It is mounted under ``/api/v2/tasks``
by the main ``create_app`` factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from ..task_engine.engine import TaskEngine
from ..task_engine.model import TaskStatus


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    task_type: str = "feature"
    priority: str = "P2"
    labels: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    effort: Optional[str] = None
    pipeline_template: Optional[str] = None
    source: str = "manual"
    created_by: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BulkCreateRequest(BaseModel):
    tasks: list[CreateTaskRequest]


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    labels: Optional[list[str]] = None
    acceptance_criteria: Optional[list[str]] = None
    context_files: Optional[list[str]] = None
    effort: Optional[str] = None
    pipeline_template: Optional[str] = None
    assignee: Optional[str] = None
    assignee_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TransitionRequest(BaseModel):
    status: str


class AssignRequest(BaseModel):
    assignee: str
    assignee_type: str = "agent"


class AddDependencyRequest(BaseModel):
    depends_on: str


class ReorderRequest(BaseModel):
    task_ids: list[str]


class RunTaskRequest(BaseModel):
    claimer: str = "manual_ui"
    assignee_type: str = "agent"
    run_id: Optional[str] = None


class RetryTaskRequest(BaseModel):
    reason: Optional[str] = None


class CancelTaskRequest(BaseModel):
    reason: Optional[str] = None


class TaskResponse(BaseModel):
    """Standard wrapper for task responses."""
    task: dict[str, Any]


class TaskListResponse(BaseModel):
    tasks: list[dict[str, Any]]
    total: int


class BoardResponse(BaseModel):
    columns: dict[str, list[dict[str, Any]]]


class DependencyGraphResponse(BaseModel):
    graph: dict[str, list[str]]


class ExecutionOrderResponse(BaseModel):
    batches: list[list[str]]


class StateMachineResponse(BaseModel):
    states: list[str]
    transitions: dict[str, list[str]]
    guards: dict[str, str]
    defaults: dict[str, Any]


class TaskEventsResponse(BaseModel):
    events: list[dict[str, Any]]
    total: int


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_task_router(get_engine: Any) -> APIRouter:
    """Create the v2 task API router.

    Parameters
    ----------
    get_engine:
        A callable ``(project_dir_param: str | None) -> TaskEngine`` that
        resolves the engine for the current request's project directory.
    """
    router = APIRouter(prefix="/api/v2/tasks", tags=["tasks-v2"])

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @router.get("", response_model=TaskListResponse)
    async def list_tasks(
        project_dir: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        priority: Optional[str] = Query(None),
        assignee: Optional[str] = Query(None),
        label: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        parent_id: Optional[str] = Query(None),
    ) -> TaskListResponse:
        engine = get_engine(project_dir)
        tasks = engine.list_tasks(
            status=status,
            task_type=task_type,
            priority=priority,
            assignee=assignee,
            label=label,
            search=search,
            parent_id=parent_id,
        )
        data = [t.to_dict() for t in tasks]
        return TaskListResponse(tasks=data, total=len(data))

    @router.post("", response_model=TaskResponse, status_code=201)
    async def create_task(
        body: CreateTaskRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        task = engine.create_task(**body.model_dump())
        return TaskResponse(task=task.to_dict())

    @router.post("/bulk", response_model=TaskListResponse, status_code=201)
    async def bulk_create(
        body: BulkCreateRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskListResponse:
        engine = get_engine(project_dir)
        from ..task_engine.model import Task
        tasks = []
        for req in body.tasks:
            t = engine.create_task(**req.model_dump())
            tasks.append(t)
        data = [t.to_dict() for t in tasks]
        return TaskListResponse(tasks=data, total=len(data))

    @router.get("/board", response_model=BoardResponse)
    async def get_board(
        project_dir: Optional[str] = Query(None),
    ) -> BoardResponse:
        engine = get_engine(project_dir)
        return BoardResponse(columns=engine.get_board())

    @router.get("/ready")
    async def get_ready_tasks(
        project_dir: Optional[str] = Query(None),
    ) -> TaskListResponse:
        engine = get_engine(project_dir)
        tasks = engine.get_ready_tasks()
        data = [t.to_dict() for t in tasks]
        return TaskListResponse(tasks=data, total=len(data))

    @router.get("/execution-order", response_model=ExecutionOrderResponse)
    async def get_execution_order(
        project_dir: Optional[str] = Query(None),
    ) -> ExecutionOrderResponse:
        engine = get_engine(project_dir)
        return ExecutionOrderResponse(batches=engine.get_execution_order())

    @router.get("/meta/state-machine", response_model=StateMachineResponse)
    async def get_state_machine(
        project_dir: Optional[str] = Query(None),
    ) -> StateMachineResponse:
        engine = get_engine(project_dir)
        transitions: dict[str, list[str]] = {
            "backlog": ["ready", "cancelled"],
            "ready": ["in_progress", "blocked", "backlog", "cancelled"],
            "in_progress": ["in_review", "blocked", "ready", "cancelled"],
            "in_review": ["done", "in_progress", "blocked", "cancelled"],
            "blocked": ["ready", "cancelled", "backlog"],
            "done": ["ready"],
            "cancelled": ["backlog"],
        }
        if engine.allow_auto_approve_review:
            transitions["in_progress"].append("done")
        guards = {
            "ready": "All blockers in blocked_by must be terminal before transition.",
            "in_progress": "All blockers in blocked_by must be terminal before transition.",
        }
        defaults = {
            "new_task_status": "backlog",
            "dependency_terminal_statuses": ["done", "cancelled"],
            "quick_action_promotion_behavior": "create_new_task",
            "feature_pipeline_steps": ["plan", "plan_impl", "implement", "verify", "review", "commit"],
            "human_review_required_default": not engine.allow_auto_approve_review,
            "allow_auto_approve_review": engine.allow_auto_approve_review,
            "auto_approve_env_var": "FEATURE_PRD_AUTO_APPROVE_REVIEW",
            "v2_tasks_default_enabled": True,
            "legacy_scheduler_opt_out_env_var": "FEATURE_PRD_DISABLE_V2_TASKS",
        }
        return StateMachineResponse(
            states=list(transitions.keys()),
            transitions=transitions,
            guards=guards,
            defaults=defaults,
        )

    @router.post("/reorder")
    async def reorder_tasks(
        body: ReorderRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, str]:
        engine = get_engine(project_dir)
        engine.reorder_tasks(body.task_ids)
        return {"status": "ok"}

    @router.get("/{task_id}", response_model=TaskResponse)
    async def get_task(
        task_id: str,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        task = engine.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.patch("/{task_id}", response_model=TaskResponse)
    async def update_task(
        task_id: str,
        body: UpdateTaskRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        changes = {k: v for k, v in body.model_dump().items() if v is not None}
        task = engine.update_task(task_id, changes)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.delete("/{task_id}")
    async def delete_task(
        task_id: str,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, str]:
        engine = get_engine(project_dir)
        if not engine.delete_task(task_id):
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {"status": "deleted"}

    # ------------------------------------------------------------------
    # Status transitions & assignment
    # ------------------------------------------------------------------

    @router.post("/{task_id}/transition", response_model=TaskResponse)
    async def transition_task(
        task_id: str,
        body: TransitionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        try:
            task = engine.transition_task(task_id, body.status)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.post("/{task_id}/run", response_model=TaskResponse)
    async def run_task(
        task_id: str,
        body: RunTaskRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        try:
            task = engine.claim_task(
                task_id,
                claimer=body.claimer,
                assignee_type=body.assignee_type,
                run_id=body.run_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.post("/{task_id}/retry", response_model=TaskResponse)
    async def retry_task(
        task_id: str,
        body: RetryTaskRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        try:
            task = engine.retry_task(task_id, reason=body.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.post("/{task_id}/cancel", response_model=TaskResponse)
    async def cancel_task(
        task_id: str,
        body: CancelTaskRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        try:
            task = engine.cancel_task(task_id, reason=body.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.post("/{task_id}/assign", response_model=TaskResponse)
    async def assign_task(
        task_id: str,
        body: AssignRequest,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        task = engine.assign_task(task_id, body.assignee, body.assignee_type)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    @router.post("/{task_id}/unassign", response_model=TaskResponse)
    async def unassign_task(
        task_id: str,
        project_dir: Optional[str] = Query(None),
    ) -> TaskResponse:
        engine = get_engine(project_dir)
        task = engine.unassign_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskResponse(task=task.to_dict())

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    @router.get("/{task_id}/dependencies", response_model=DependencyGraphResponse)
    async def get_task_dependencies(
        task_id: str,
        project_dir: Optional[str] = Query(None),
    ) -> DependencyGraphResponse:
        engine = get_engine(project_dir)
        graph = engine.get_dependency_graph(task_id)
        return DependencyGraphResponse(graph=graph)

    @router.post("/{task_id}/dependencies")
    async def add_dependency(
        task_id: str,
        body: AddDependencyRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, str]:
        engine = get_engine(project_dir)
        try:
            engine.add_dependency(task_id, body.depends_on)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "ok"}

    @router.delete("/{task_id}/dependencies/{dep_id}")
    async def remove_dependency(
        task_id: str,
        dep_id: str,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, str]:
        engine = get_engine(project_dir)
        engine.remove_dependency(task_id, dep_id)
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Task generators
    # ------------------------------------------------------------------

    @router.post("/generate/{source}", response_model=TaskListResponse)
    async def generate_tasks(
        source: str,
        project_dir: Optional[str] = Query(None),
        prd_content: Optional[str] = None,
    ) -> TaskListResponse:
        """Run a task generator and persist the resulting tasks."""
        from pathlib import Path as _Path
        from ..task_engine.sources.repo_review import RepoReviewGenerator
        from ..task_engine.sources.bug_scan import BugScanGenerator
        from ..task_engine.sources.prd_import import PrdImportGenerator
        from ..task_engine.sources.enhancement_brainstorm import EnhancementBrainstormGenerator

        engine = get_engine(project_dir)

        # Resolve project directory
        proj = _Path(project_dir) if project_dir else _Path.cwd()

        generators = {
            "repo_review": RepoReviewGenerator,
            "bug_scan": BugScanGenerator,
            "prd_import": PrdImportGenerator,
            "enhancement_brainstorm": EnhancementBrainstormGenerator,
        }

        gen_cls = generators.get(source)
        if gen_cls is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown generator: {source}. Available: {list(generators.keys())}",
            )

        gen = gen_cls()
        kwargs: dict[str, Any] = {}
        if source == "prd_import" and prd_content:
            kwargs["prd_content"] = prd_content

        new_tasks = gen.generate(proj, **kwargs)
        if new_tasks:
            engine.bulk_create(new_tasks)

        data = [t.to_dict() for t in new_tasks]
        return TaskListResponse(tasks=data, total=len(data))

    @router.get("/events/recent", response_model=TaskEventsResponse)
    async def get_recent_task_events(
        project_dir: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> TaskEventsResponse:
        engine = get_engine(project_dir)
        events = engine.get_recent_events(limit=limit)
        return TaskEventsResponse(events=events, total=len(events))

    @router.get("/{task_id}/events", response_model=TaskEventsResponse)
    async def get_task_events(
        task_id: str,
        project_dir: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> TaskEventsResponse:
        engine = get_engine(project_dir)
        if engine.get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        events = engine.get_task_events(task_id, limit=limit)
        return TaskEventsResponse(events=events, total=len(events))

    return router
