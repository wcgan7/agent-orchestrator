from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..domain.models import AgentRecord, QuickActionRun, Task, now_iso
from ..events.bus import EventBus
from ..orchestrator.service import OrchestratorService
from ..storage.container import V3Container


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    task_type: str = "feature"
    priority: str = "P2"
    labels: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    pipeline_template: list[str] = Field(default_factory=lambda: ["plan", "implement", "verify", "review"])
    approval_mode: str = "human_review"
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    labels: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = None
    approval_mode: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TransitionRequest(BaseModel):
    status: str


class AddDependencyRequest(BaseModel):
    depends_on: str


class PrdPreviewRequest(BaseModel):
    title: Optional[str] = None
    content: str
    default_priority: str = "P2"


class PrdCommitRequest(BaseModel):
    job_id: str


class QuickActionRequest(BaseModel):
    prompt: str


class PromoteQuickActionRequest(BaseModel):
    title: Optional[str] = None
    priority: str = "P2"


class OrchestratorControlRequest(BaseModel):
    action: str


class SpawnAgentRequest(BaseModel):
    role: str = "general"
    capacity: int = 1
    override_provider: Optional[str] = None


class ReviewActionRequest(BaseModel):
    guidance: Optional[str] = None


VALID_TRANSITIONS: dict[str, set[str]] = {
    "backlog": {"ready", "cancelled"},
    "ready": {"in_progress", "blocked", "backlog", "cancelled"},
    "in_progress": {"in_review", "blocked", "ready", "cancelled"},
    "in_review": {"done", "in_progress", "blocked", "cancelled"},
    "blocked": {"ready", "cancelled", "backlog"},
    "done": {"ready"},
    "cancelled": {"backlog"},
}


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def _execution_batches(tasks: list[Task]) -> list[list[str]]:
    by_id = {task.id: task for task in tasks}
    indegree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        refs = [dep_id for dep_id in task.blocked_by if dep_id in by_id]
        indegree[task.id] = len(refs)
        for dep_id in refs:
            dependents.setdefault(dep_id, []).append(task.id)

    ready = sorted([task_id for task_id, degree in indegree.items() if degree == 0], key=lambda tid: (_priority_rank(by_id[tid].priority), by_id[tid].created_at))
    batches: list[list[str]] = []
    while ready:
        batch = list(ready)
        batches.append(batch)
        next_ready: list[str] = []
        for task_id in batch:
            for dep_id in dependents.get(task_id, []):
                indegree[dep_id] -= 1
                if indegree[dep_id] == 0:
                    next_ready.append(dep_id)
        ready = sorted(next_ready, key=lambda tid: (_priority_rank(by_id[tid].priority), by_id[tid].created_at))
    return batches


def _has_unresolved_blockers(container: V3Container, task: Task) -> Optional[str]:
    for dep_id in task.blocked_by:
        dep = container.tasks.get(dep_id)
        if dep is None or dep.status not in {"done", "cancelled"}:
            return dep_id
    return None


def _parse_prd_into_tasks(content: str, default_priority: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line in content.splitlines():
        normalized = line.strip()
        if normalized.startswith("- ") or normalized.startswith("* "):
            title = normalized[2:].strip()
            if title:
                tasks.append({"title": title, "priority": default_priority})
        elif normalized.startswith("## "):
            title = normalized[3:].strip()
            if title:
                tasks.append({"title": title, "priority": default_priority})
    if not tasks:
        tasks.append({"title": "Imported PRD task", "priority": default_priority})
    return tasks


def create_v3_router(
    resolve_container: Any,
    resolve_orchestrator: Any,
    job_store: dict[str, dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/api/v3", tags=["v3"])

    def _ctx(project_dir: Optional[str]) -> tuple[V3Container, EventBus, OrchestratorService]:
        container: V3Container = resolve_container(project_dir)
        bus = EventBus(container.events, container.project_id)
        orchestrator: OrchestratorService = resolve_orchestrator(project_dir)
        return container, bus, orchestrator

    @router.get("/projects")
    async def list_projects(project_dir: Optional[str] = Query(None), include_non_git: bool = Query(False)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        pinned = list(cfg.get("pinned_projects") or [])
        discovered = []
        cwd = container.project_dir
        if (cwd / ".git").exists() or include_non_git:
            discovered.append({"id": cwd.name, "path": str(cwd), "source": "discovered", "is_git": (cwd / ".git").exists()})
        for item in pinned:
            p = Path(str(item.get("path") or "")).resolve()
            discovered.append({"id": item.get("id") or p.name, "path": str(p), "source": "pinned", "is_git": (p / ".git").exists()})
        dedup: dict[str, dict[str, Any]] = {entry["path"]: entry for entry in discovered}
        return {"projects": list(dedup.values())}

    @router.get("/projects/pinned")
    async def list_pinned_projects(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return {"items": list(cfg.get("pinned_projects") or [])}

    @router.post("/projects/pinned")
    async def pin_project(body: dict[str, Any], project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        path = Path(str(body.get("path") or "")).expanduser().resolve()
        allow_non_git = bool(body.get("allow_non_git", False))
        if not path.exists() or not path.is_dir() or not os_access(path):
            raise HTTPException(status_code=400, detail="Invalid project path")
        if not allow_non_git and not (path / ".git").exists():
            raise HTTPException(status_code=400, detail="Project path must contain .git unless allow_non_git=true")

        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        pinned = [entry for entry in list(cfg.get("pinned_projects") or []) if str(entry.get("path")) != str(path)]
        project_id = body.get("project_id") or f"pinned-{uuid.uuid4().hex[:8]}"
        pinned.append({"id": project_id, "path": str(path), "pinned_at": now_iso()})
        cfg["pinned_projects"] = pinned
        container.config.save(cfg)
        return {"project": {"id": project_id, "path": str(path)}}

    @router.delete("/projects/pinned/{project_id}")
    async def unpin_project(project_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        pinned = list(cfg.get("pinned_projects") or [])
        remaining = [entry for entry in pinned if entry.get("id") != project_id]
        cfg["pinned_projects"] = remaining
        container.config.save(cfg)
        return {"removed": len(remaining) != len(pinned)}

    @router.post("/tasks")
    async def create_task(body: CreateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = Task(
            title=body.title,
            description=body.description,
            task_type=body.task_type,
            priority=body.priority,
            labels=body.labels,
            blocked_by=body.blocked_by,
            parent_id=body.parent_id,
            pipeline_template=body.pipeline_template,
            approval_mode=body.approval_mode,
            source=body.source,
            metadata=body.metadata,
        )
        if task.parent_id:
            parent = container.tasks.get(task.parent_id)
            if parent and task.id not in parent.children_ids:
                parent.children_ids.append(task.id)
                container.tasks.upsert(parent)
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.created", entity_id=task.id, payload={"status": task.status})
        return {"task": task.to_dict()}

    @router.get("/tasks")
    async def list_tasks(
        project_dir: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        priority: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        tasks = container.tasks.list()
        filtered = []
        for task in tasks:
            if status and task.status != status:
                continue
            if task_type and task.task_type != task_type:
                continue
            if priority and task.priority != priority:
                continue
            filtered.append(task)
        filtered.sort(key=lambda t: (_priority_rank(t.priority), t.created_at))
        return {"tasks": [task.to_dict() for task in filtered], "total": len(filtered)}

    @router.get("/tasks/board")
    async def board(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        columns = {name: [] for name in ["backlog", "ready", "in_progress", "in_review", "blocked", "done", "cancelled"]}
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(task.to_dict())
        for key, items in columns.items():
            items.sort(key=lambda x: (_priority_rank(str(x.get("priority") or "P3")), str(x.get("created_at") or "")))
        return {"columns": columns}

    @router.get("/tasks/execution-order")
    async def execution_order(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        tasks = container.tasks.list()
        return {"batches": _execution_batches(tasks)}

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": task.to_dict()}

    @router.patch("/tasks/{task_id}")
    async def patch_task(task_id: str, body: UpdateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        updates = body.model_dump(exclude_none=True)
        if "status" in updates:
            raise HTTPException(
                status_code=400,
                detail="Task status cannot be changed via PATCH. Use /transition, /retry, /cancel, or review actions.",
            )
        for key, value in updates.items():
            setattr(task, key, value)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.updated", entity_id=task.id, payload={"status": task.status})
        return {"task": task.to_dict()}

    @router.post("/tasks/{task_id}/transition")
    async def transition_task(task_id: str, body: TransitionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        target = body.status
        valid = VALID_TRANSITIONS.get(task.status, set())
        if target not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid transition {task.status} -> {target}")
        if target in {"ready", "in_progress"}:
            unresolved = _has_unresolved_blockers(container, task)
            if unresolved is not None:
                raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        task.status = target
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.transitioned", entity_id=task.id, payload={"status": task.status})
        return {"task": task.to_dict()}

    @router.post("/tasks/{task_id}/run")
    async def run_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        try:
            task = orchestrator.run_task(task_id)
        except ValueError as exc:
            if "Task not found" in str(exc):
                raise HTTPException(status_code=404, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        return {"task": task.to_dict()}

    @router.post("/tasks/{task_id}/retry")
    async def retry_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        unresolved = _has_unresolved_blockers(container, task)
        if unresolved is not None:
            raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        task.retry_count += 1
        task.status = "ready"
        task.error = None
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.retry", entity_id=task.id, payload={"retry_count": task.retry_count})
        return {"task": task.to_dict()}

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.status = "cancelled"
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.cancelled", entity_id=task.id, payload={})
        return {"task": task.to_dict()}

    @router.post("/tasks/{task_id}/dependencies")
    async def add_dependency(task_id: str, body: AddDependencyRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        blocker = container.tasks.get(body.depends_on)
        if not task or not blocker:
            raise HTTPException(status_code=404, detail="Task or dependency not found")
        if body.depends_on not in task.blocked_by:
            task.blocked_by.append(body.depends_on)
        if task.id not in blocker.blocks:
            blocker.blocks.append(task.id)
        container.tasks.upsert(task)
        container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_added", entity_id=task.id, payload={"depends_on": body.depends_on})
        return {"task": task.to_dict()}

    @router.delete("/tasks/{task_id}/dependencies/{dep_id}")
    async def remove_dependency(task_id: str, dep_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.blocked_by = [item for item in task.blocked_by if item != dep_id]
        container.tasks.upsert(task)
        blocker = container.tasks.get(dep_id)
        if blocker:
            blocker.blocks = [item for item in blocker.blocks if item != task.id]
            container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_removed", entity_id=task.id, payload={"dep_id": dep_id})
        return {"task": task.to_dict()}

    @router.post("/import/prd/preview")
    async def preview_import(body: PrdPreviewRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        items = _parse_prd_into_tasks(body.content, body.default_priority)
        nodes = [{"id": f"n{idx + 1}", "title": str(item.get("title") or "Imported task"), "priority": str(item.get("priority") or body.default_priority)} for idx, item in enumerate(items)]
        edges = [{"from": nodes[idx]["id"], "to": nodes[idx + 1]["id"]} for idx in range(len(nodes) - 1)]
        job_id = f"imp-{uuid.uuid4().hex[:10]}"
        job_store[job_id] = {
            "id": job_id,
            "project_id": container.project_id,
            "title": body.title or "Imported PRD",
            "status": "preview_ready",
            "created_at": now_iso(),
            "tasks": items,
        }
        return {"job_id": job_id, "preview": {"nodes": nodes, "edges": edges}}

    @router.post("/import/prd/commit")
    async def commit_import(body: PrdCommitRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        job = job_store.get(body.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        created: list[str] = []
        previous: Optional[Task] = None
        for item in list(job.get("tasks", [])):
            if not isinstance(item, dict):
                continue
            task = Task(title=str(item.get("title") or "Imported task"), priority=str(item.get("priority") or "P2"), source="prd_import")
            task.status = "ready"
            if previous:
                task.blocked_by.append(previous.id)
                previous.blocks.append(task.id)
                container.tasks.upsert(previous)
            container.tasks.upsert(task)
            created.append(task.id)
            previous = task
        job["status"] = "committed"
        bus.emit(channel="tasks", event_type="import.committed", entity_id=body.job_id, payload={"created_task_ids": created})
        return {"job_id": body.job_id, "created_task_ids": created}

    @router.get("/import/{job_id}")
    async def get_import_job(job_id: str) -> dict[str, Any]:
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        return {"job": job}

    @router.post("/quick-actions")
    async def create_quick_action(body: QuickActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        run = QuickActionRun(prompt=body.prompt, status="completed", started_at=now_iso(), finished_at=now_iso(), result_summary="Quick action executed")
        container.quick_actions.upsert(run)
        bus.emit(channel="quick_actions", event_type="quick_action.completed", entity_id=run.id, payload={"status": run.status})
        return {"quick_action": run.to_dict()}

    @router.get("/quick-actions")
    async def list_quick_actions(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        runs = sorted(container.quick_actions.list(), key=lambda item: item.started_at or "", reverse=True)
        return {"quick_actions": [run.to_dict() for run in runs]}

    @router.get("/quick-actions/{quick_action_id}")
    async def get_quick_action(quick_action_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        run = container.quick_actions.get(quick_action_id)
        if not run:
            raise HTTPException(status_code=404, detail="Quick action not found")
        return {"quick_action": run.to_dict()}

    @router.post("/quick-actions/{quick_action_id}/promote")
    async def promote_quick_action(quick_action_id: str, body: PromoteQuickActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        run = container.quick_actions.get(quick_action_id)
        if not run:
            raise HTTPException(status_code=404, detail="Quick action not found")
        if run.promoted_task_id:
            task = container.tasks.get(run.promoted_task_id)
            return {"task": task.to_dict() if task else None, "already_promoted": True}
        title = body.title or f"Promoted quick action: {run.prompt[:50]}"
        task = Task(title=title, description=run.prompt, source="promoted_quick_action", priority=body.priority)
        container.tasks.upsert(task)
        run.promoted_task_id = task.id
        container.quick_actions.upsert(run)
        bus.emit(channel="quick_actions", event_type="quick_action.promoted", entity_id=run.id, payload={"task_id": task.id})
        return {"task": task.to_dict(), "already_promoted": False}

    @router.get("/review-queue")
    async def review_queue(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        items = [task.to_dict() for task in container.tasks.list() if task.status == "in_review"]
        return {"tasks": items, "total": len(items)}

    @router.post("/review/{task_id}/approve")
    async def approve_review(task_id: str, body: ReviewActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")
        task.status = "done"
        task.metadata["last_review_approval"] = {"ts": now_iso(), "guidance": body.guidance}
        container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.approved", entity_id=task.id, payload={})
        return {"task": task.to_dict()}

    @router.post("/review/{task_id}/request-changes")
    async def request_review_changes(task_id: str, body: ReviewActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")
        task.status = "ready"
        task.metadata["requested_changes"] = {"ts": now_iso(), "guidance": body.guidance}
        container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.changes_requested", entity_id=task.id, payload={"guidance": body.guidance})
        return {"task": task.to_dict()}

    @router.get("/orchestrator/status")
    async def orchestrator_status(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.status()

    @router.post("/orchestrator/control")
    async def orchestrator_control(body: OrchestratorControlRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.control(body.action)

    @router.get("/agents")
    async def list_agents(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        return {"agents": [agent.to_dict() for agent in container.agents.list()]}

    @router.post("/agents/spawn")
    async def spawn_agent(body: SpawnAgentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        agent = AgentRecord(role=body.role, capacity=body.capacity, override_provider=body.override_provider)
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.spawned", entity_id=agent.id, payload=agent.to_dict())
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "paused"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.paused", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "running"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.resumed", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/terminate")
    async def terminate_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "terminated"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.terminated", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    return router


def os_access(path: Path) -> bool:
    try:
        list(path.iterdir())
    except Exception:
        return False
    return True
