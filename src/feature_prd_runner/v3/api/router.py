from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...collaboration.modes import MODE_CONFIGS
from ...pipelines.registry import PipelineRegistry
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
    pipeline_template: Optional[list[str]] = None
    approval_mode: str = "human_review"
    hitl_mode: str = "autopilot"
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
    hitl_mode: Optional[str] = None
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


class ApproveGateRequest(BaseModel):
    gate: Optional[str] = None


class OrchestratorControlRequest(BaseModel):
    action: str


class OrchestratorSettingsRequest(BaseModel):
    concurrency: int = Field(2, ge=1, le=128)
    auto_deps: bool = True
    max_review_attempts: int = Field(3, ge=1, le=50)


class AgentRoutingSettingsRequest(BaseModel):
    default_role: str = "general"
    task_type_roles: dict[str, str] = Field(default_factory=dict)
    role_provider_overrides: dict[str, str] = Field(default_factory=dict)


class WorkerProviderSettingsRequest(BaseModel):
    type: str = "codex"
    command: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    num_ctx: Optional[int] = None


class WorkersSettingsRequest(BaseModel):
    default: str = "codex"
    routing: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, WorkerProviderSettingsRequest] = Field(default_factory=dict)


class QualityGateSettingsRequest(BaseModel):
    critical: int = Field(0, ge=0)
    high: int = Field(0, ge=0)
    medium: int = Field(0, ge=0)
    low: int = Field(0, ge=0)


class DefaultsSettingsRequest(BaseModel):
    quality_gate: QualityGateSettingsRequest = Field(default_factory=QualityGateSettingsRequest)


class LanguageCommandsRequest(BaseModel):
    test: Optional[str] = None
    lint: Optional[str] = None
    typecheck: Optional[str] = None
    format: Optional[str] = None


class ProjectSettingsRequest(BaseModel):
    commands: Optional[dict[str, LanguageCommandsRequest]] = None


class UpdateSettingsRequest(BaseModel):
    orchestrator: Optional[OrchestratorSettingsRequest] = None
    agent_routing: Optional[AgentRoutingSettingsRequest] = None
    defaults: Optional[DefaultsSettingsRequest] = None
    workers: Optional[WorkersSettingsRequest] = None
    project: Optional[ProjectSettingsRequest] = None


class SpawnAgentRequest(BaseModel):
    role: str = "general"
    capacity: int = 1
    override_provider: Optional[str] = None


class ReviewActionRequest(BaseModel):
    guidance: Optional[str] = None


class AddFeedbackRequest(BaseModel):
    task_id: str
    feedback_type: str = "general"
    priority: str = "should"
    summary: str
    details: str = ""
    target_file: Optional[str] = None


class AddCommentRequest(BaseModel):
    task_id: str
    file_path: str
    line_number: int = 0
    body: str
    line_type: Optional[str] = None
    parent_id: Optional[str] = None


VALID_TRANSITIONS: dict[str, set[str]] = {
    "backlog": {"ready", "cancelled"},
    "ready": {"in_progress", "blocked", "backlog", "cancelled"},
    "in_progress": {"in_review", "blocked", "ready", "cancelled"},
    "in_review": {"done", "in_progress", "blocked", "cancelled"},
    "blocked": {"ready", "cancelled", "backlog"},
    "done": {"ready"},
    "cancelled": {"backlog"},
}

IMPORT_JOB_TTL_SECONDS = 60 * 60 * 24
IMPORT_JOB_MAX_RECORDS = 200
QUICK_ACTION_MAX_PENDING = 32


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pruned_import_jobs(items: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    kept: list[dict[str, Any]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("id") or "").strip()
        if not job_id:
            continue
        created_at = _parse_iso_datetime(item.get("created_at"))
        if created_at and (now - created_at).total_seconds() > IMPORT_JOB_TTL_SECONDS:
            continue
        kept.append(item)

    kept.sort(key=lambda job: str(job.get("created_at") or ""), reverse=True)
    trimmed = kept[:IMPORT_JOB_MAX_RECORDS]
    return {str(job.get("id")): job for job in trimmed if str(job.get("id") or "").strip()}


def _coerce_int(value: Any, default: int, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def _normalize_str_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        k = str(key or "").strip()
        v = str(raw or "").strip()
        if k and v:
            out[k] = v
    return out


def _normalize_human_blocking_issues(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            summary = item.strip()
            if summary:
                out.append({"summary": summary})
            continue
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("issue") or "").strip()
        details = str(item.get("details") or item.get("rationale") or "").strip()
        if not summary and details:
            summary = details.splitlines()[0][:200].strip()
        if not summary:
            continue
        issue: dict[str, str] = {"summary": summary}
        if details:
            issue["details"] = details
        for key in ("category", "action", "blocking_on", "severity"):
            raw = item.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                issue[key] = text
        out.append(issue)
    return out[:20]


def _task_payload(task: Task) -> dict[str, Any]:
    payload = task.to_dict()
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    payload["human_blocking_issues"] = _normalize_human_blocking_issues(metadata.get("human_blocking_issues"))
    return payload


def _normalize_workers_providers(value: Any) -> dict[str, dict[str, Any]]:
    raw_providers = value if isinstance(value, dict) else {}
    providers: dict[str, dict[str, Any]] = {}
    for raw_name, raw_item in raw_providers.items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(raw_item, dict):
            continue
        provider_type = str(raw_item.get("type") or ("codex" if name == "codex" else "")).strip().lower()
        if provider_type == "local":
            provider_type = "ollama"
        if provider_type not in {"codex", "ollama"}:
            continue

        if provider_type == "codex":
            command = str(raw_item.get("command") or "codex").strip() or "codex"
            providers[name] = {"type": "codex", "command": command}
            continue

        endpoint = str(raw_item.get("endpoint") or "").strip()
        model = str(raw_item.get("model") or "").strip()
        provider: dict[str, Any] = {"type": "ollama"}
        if endpoint:
            provider["endpoint"] = endpoint
        if model:
            provider["model"] = model
        temperature = raw_item.get("temperature")
        if isinstance(temperature, (int, float)):
            provider["temperature"] = float(temperature)
        num_ctx = raw_item.get("num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0:
            provider["num_ctx"] = num_ctx
        providers[name] = provider

    codex = providers.get("codex")
    codex_command = "codex"
    if isinstance(codex, dict):
        codex_command = str(codex.get("command") or "codex").strip() or "codex"
    providers["codex"] = {"type": "codex", "command": codex_command}
    return providers


def _settings_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    orchestrator = dict(cfg.get("orchestrator") or {})
    routing = dict(cfg.get("agent_routing") or {})
    defaults = dict(cfg.get("defaults") or {})
    quality_gate = dict(defaults.get("quality_gate") or {})
    workers_cfg = dict(cfg.get("workers") or {})
    workers_providers = _normalize_workers_providers(workers_cfg.get("providers"))
    workers_default = str(workers_cfg.get("default") or "codex").strip() or "codex"
    if workers_default not in workers_providers:
        workers_default = "codex"
    return {
        "orchestrator": {
            "concurrency": _coerce_int(orchestrator.get("concurrency"), 2, minimum=1, maximum=128),
            "auto_deps": _coerce_bool(orchestrator.get("auto_deps"), True),
            "max_review_attempts": _coerce_int(orchestrator.get("max_review_attempts"), 3, minimum=1, maximum=50),
        },
        "agent_routing": {
            "default_role": str(routing.get("default_role") or "general"),
            "task_type_roles": _normalize_str_map(routing.get("task_type_roles")),
            "role_provider_overrides": _normalize_str_map(routing.get("role_provider_overrides")),
        },
        "defaults": {
            "quality_gate": {
                "critical": _coerce_int(quality_gate.get("critical"), 0, minimum=0),
                "high": _coerce_int(quality_gate.get("high"), 0, minimum=0),
                "medium": _coerce_int(quality_gate.get("medium"), 0, minimum=0),
                "low": _coerce_int(quality_gate.get("low"), 0, minimum=0),
            }
        },
        "workers": {
            "default": workers_default,
            "routing": _normalize_str_map(workers_cfg.get("routing")),
            "providers": workers_providers,
        },
        "project": {
            "commands": dict((cfg.get("project") or {}).get("commands") or {}),
        },
    }


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

    def _load_feedback_records(container: V3Container) -> list[dict[str, Any]]:
        cfg = container.config.load()
        raw = cfg.get("collaboration_feedback")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save_feedback_records(container: V3Container, items: list[dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["collaboration_feedback"] = items
        container.config.save(cfg)

    def _load_comment_records(container: V3Container) -> list[dict[str, Any]]:
        cfg = container.config.load()
        raw = cfg.get("collaboration_comments")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save_comment_records(container: V3Container, items: list[dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["collaboration_comments"] = items
        container.config.save(cfg)

    def _load_import_jobs(container: V3Container) -> dict[str, dict[str, Any]]:
        cfg = container.config.load()
        raw = cfg.get("import_jobs")
        jobs: dict[str, dict[str, Any]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    job_id = str(value.get("id") or key).strip()
                    if job_id:
                        item = dict(value)
                        item["id"] = job_id
                        jobs[job_id] = item
        elif isinstance(raw, list):
            for value in raw:
                if isinstance(value, dict):
                    job_id = str(value.get("id") or "").strip()
                    if job_id:
                        jobs[job_id] = dict(value)
        return _pruned_import_jobs(jobs)

    def _save_import_jobs(container: V3Container, jobs: dict[str, dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["import_jobs"] = list(_pruned_import_jobs(jobs).values())
        container.config.save(cfg)

    def _upsert_import_job(container: V3Container, job: dict[str, Any]) -> None:
        jobs = _load_import_jobs(container)
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            return
        jobs[job_id] = job
        _save_import_jobs(container, jobs)

    def _fetch_import_job(container: V3Container, job_id: str) -> Optional[dict[str, Any]]:
        jobs = _load_import_jobs(container)
        _save_import_jobs(container, jobs)
        return jobs.get(job_id)

    def _prune_in_memory_jobs() -> None:
        pruned = _pruned_import_jobs(dict(job_store))
        job_store.clear()
        job_store.update(pruned)

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

    @router.get("/projects/browse")
    async def browse_projects(
        project_dir: Optional[str] = Query(None),
        path: Optional[str] = Query(None),
        include_hidden: bool = Query(False),
        limit: int = Query(200, ge=1, le=1000),
    ) -> dict[str, Any]:
        _ctx(project_dir)
        target = Path(path).expanduser().resolve() if path else Path.home().resolve()
        if not target.exists() or not target.is_dir() or not os_access(target):
            raise HTTPException(status_code=400, detail="Invalid browse path")

        try:
            children = list(target.iterdir())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read browse path: {exc}") from exc

        directories: list[dict[str, Any]] = []
        for child in sorted(children, key=lambda item: item.name.lower()):
            if not child.is_dir():
                continue
            if not include_hidden and child.name.startswith("."):
                continue
            if not os_access(child):
                continue
            directories.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_git": (child / ".git").exists(),
                }
            )
            if len(directories) >= limit:
                break

        parent = target.parent if target.parent != target else None
        return {
            "path": str(target),
            "parent": str(parent) if parent else None,
            "current_is_git": (target / ".git").exists(),
            "directories": directories,
            "truncated": len(directories) >= limit,
        }

    @router.post("/tasks")
    async def create_task(body: CreateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        pipeline_steps = body.pipeline_template
        if pipeline_steps is None:
            registry = PipelineRegistry()
            template = registry.resolve_for_task_type(body.task_type)
            pipeline_steps = template.step_names()
        task = Task(
            title=body.title,
            description=body.description,
            task_type=body.task_type,
            priority=body.priority,
            labels=body.labels,
            blocked_by=body.blocked_by,
            parent_id=body.parent_id,
            pipeline_template=pipeline_steps,
            approval_mode=body.approval_mode,
            hitl_mode=body.hitl_mode,
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
        return {"task": _task_payload(task)}

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
        return {"tasks": [_task_payload(task) for task in filtered], "total": len(filtered)}

    @router.get("/tasks/board")
    async def board(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        columns = {name: [] for name in ["backlog", "ready", "in_progress", "in_review", "blocked", "done", "cancelled"]}
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(_task_payload(task))
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
        return {"task": _task_payload(task)}

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
        return {"task": _task_payload(task)}

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
        return {"task": _task_payload(task)}

    @router.post("/tasks/{task_id}/run")
    async def run_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        try:
            task = orchestrator.run_task(task_id)
        except ValueError as exc:
            if "Task not found" in str(exc):
                raise HTTPException(status_code=404, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        return {"task": _task_payload(task)}

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
        task.pending_gate = None
        if isinstance(task.metadata, dict):
            task.metadata.pop("human_blocking_issues", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.retry", entity_id=task.id, payload={"retry_count": task.retry_count})
        return {"task": _task_payload(task)}

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.status = "cancelled"
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.cancelled", entity_id=task.id, payload={})
        return {"task": _task_payload(task)}

    @router.post("/tasks/{task_id}/approve-gate")
    async def approve_gate(task_id: str, body: ApproveGateRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.pending_gate:
            raise HTTPException(status_code=400, detail="No pending gate on this task")
        if body.gate and body.gate != task.pending_gate:
            raise HTTPException(status_code=400, detail=f"Gate mismatch: pending={task.pending_gate}, requested={body.gate}")
        cleared_gate = task.pending_gate
        task.pending_gate = None
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.gate_approved", entity_id=task.id, payload={"gate": cleared_gate})
        return {"task": _task_payload(task), "cleared_gate": cleared_gate}

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
        return {"task": _task_payload(task)}

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
        return {"task": _task_payload(task)}

    @router.post("/tasks/analyze-dependencies")
    async def analyze_dependencies(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        orchestrator._maybe_analyze_dependencies()
        # Collect all inferred edges across tasks
        edges: list[dict[str, str]] = []
        for task in container.tasks.list():
            inferred = task.metadata.get("inferred_deps") if isinstance(task.metadata, dict) else None
            if isinstance(inferred, list):
                for dep in inferred:
                    if isinstance(dep, dict):
                        edges.append({"from": dep.get("from", ""), "to": task.id, "reason": dep.get("reason", "")})
        return {"edges": edges}

    @router.post("/tasks/{task_id}/reset-dep-analysis")
    async def reset_dep_analysis(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if isinstance(task.metadata, dict):
            # Remove inferred blocked_by entries and corresponding blocks on blockers
            inferred = task.metadata.get("inferred_deps")
            if isinstance(inferred, list):
                inferred_from_ids = {dep.get("from") for dep in inferred if isinstance(dep, dict)}
                task.blocked_by = [bid for bid in task.blocked_by if bid not in inferred_from_ids]
                for blocker_id in inferred_from_ids:
                    blocker = container.tasks.get(blocker_id)
                    if blocker:
                        blocker.blocks = [bid for bid in blocker.blocks if bid != task.id]
                        container.tasks.upsert(blocker)
            task.metadata.pop("deps_analyzed", None)
            task.metadata.pop("inferred_deps", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.dep_analysis_reset", entity_id=task.id, payload={})
        return {"task": _task_payload(task)}

    @router.post("/import/prd/preview")
    async def preview_import(body: PrdPreviewRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        _prune_in_memory_jobs()
        items = _parse_prd_into_tasks(body.content, body.default_priority)
        nodes = [{"id": f"n{idx + 1}", "title": str(item.get("title") or "Imported task"), "priority": str(item.get("priority") or body.default_priority)} for idx, item in enumerate(items)]
        edges = [{"from": nodes[idx]["id"], "to": nodes[idx + 1]["id"]} for idx in range(len(nodes) - 1)]
        job_id = f"imp-{uuid.uuid4().hex[:10]}"
        job = {
            "id": job_id,
            "project_id": container.project_id,
            "title": body.title or "Imported PRD",
            "status": "preview_ready",
            "created_at": now_iso(),
            "tasks": items,
        }
        job_store[job_id] = job
        _upsert_import_job(container, job)
        return {"job_id": job_id, "preview": {"nodes": nodes, "edges": edges}}

    @router.post("/import/prd/commit")
    async def commit_import(body: PrdCommitRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        _prune_in_memory_jobs()
        job = job_store.get(body.job_id)
        if not job:
            job = _fetch_import_job(container, body.job_id)
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
        job["created_task_ids"] = created
        job_store[body.job_id] = job
        _upsert_import_job(container, job)
        bus.emit(channel="tasks", event_type="import.committed", entity_id=body.job_id, payload={"created_task_ids": created})
        return {"job_id": body.job_id, "created_task_ids": created}

    @router.get("/import/{job_id}")
    async def get_import_job(job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _prune_in_memory_jobs()
        job = job_store.get(job_id)
        if not job:
            container: V3Container = resolve_container(project_dir)
            job = _fetch_import_job(container, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        return {"job": job}

    @router.get("/metrics")
    async def get_metrics(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        status = orchestrator.status()
        tasks = container.tasks.list()
        runs = container.runs.list()
        events = container.events.list_recent(limit=2000)
        phases_completed = sum(len(list(run.steps or [])) for run in runs)
        phases_total = sum(max(1, len(list(task.pipeline_template or []))) for task in tasks)
        wall_time_seconds = 0.0
        for run in runs:
            if not run.started_at:
                continue
            try:
                start = datetime.fromisoformat(str(run.started_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if run.finished_at:
                try:
                    end = datetime.fromisoformat(str(run.finished_at).replace("Z", "+00:00"))
                except ValueError:
                    continue
            else:
                end = datetime.now(timezone.utc)
            wall_time_seconds += max((end - start).total_seconds(), 0.0)
        api_calls = len(events)
        return {
            "tokens_used": 0,
            "api_calls": api_calls,
            "estimated_cost_usd": 0.0,
            "wall_time_seconds": int(wall_time_seconds),
            "phases_completed": phases_completed,
            "phases_total": phases_total,
            "files_changed": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "queue_depth": int(status.get("queue_depth", 0)),
            "in_progress": int(status.get("in_progress", 0)),
        }

    @router.get("/phases")
    async def get_phases(project_dir: Optional[str] = Query(None)) -> list[dict[str, Any]]:
        container, _, _ = _ctx(project_dir)
        phases: list[dict[str, Any]] = []
        for task in container.tasks.list():
            total_steps = max(1, len(list(task.pipeline_template or [])))
            completed_steps = 0
            if task.status == "done":
                completed_steps = total_steps
            elif task.status == "in_review":
                completed_steps = max(total_steps - 1, 1)
            elif task.status == "in_progress":
                completed_steps = 2
            elif task.status in {"ready", "blocked"}:
                completed_steps = 1
            progress = {
                "backlog": 0.0,
                "ready": 0.1,
                "blocked": 0.1,
                "in_progress": 0.6,
                "in_review": 0.9,
                "done": 1.0,
                "cancelled": 1.0,
            }.get(task.status, min(completed_steps / total_steps, 1.0))
            phases.append(
                {
                    "id": task.id,
                    "name": task.title,
                    "description": task.description,
                    "status": task.status,
                    "deps": list(task.blocked_by),
                    "progress": progress,
                }
            )
        return phases

    @router.get("/agents/types")
    async def get_agent_types(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        task_role_map = dict(routing.get("task_type_roles") or {})
        role_affinity: dict[str, list[str]] = {}
        for task_type, role in task_role_map.items():
            role_name = str(role or "")
            if not role_name:
                continue
            role_affinity.setdefault(role_name, []).append(str(task_type))
        roles = ["general", "implementer", "reviewer", "researcher", "tester", "planner", "debugger"]
        return {
            "types": [
                {
                    "role": role,
                    "display_name": role.replace("_", " ").title(),
                    "description": f"{role.replace('_', ' ').title()} agent",
                    "task_type_affinity": sorted(role_affinity.get(role, [])),
                    "allowed_steps": ["plan", "implement", "verify", "review"],
                    "limits": {"max_tokens": 0, "max_time_seconds": 0, "max_cost_usd": 0.0},
                }
                for role in roles
            ]
        }

    @router.get("/collaboration/modes")
    async def get_collaboration_modes() -> dict[str, Any]:
        return {"modes": [config.to_dict() for config in MODE_CONFIGS.values()]}

    @router.get("/collaboration/presence")
    async def get_collaboration_presence() -> dict[str, Any]:
        return {"users": []}

    @router.get("/collaboration/timeline/{task_id}")
    async def get_collaboration_timeline(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            return {"events": []}

        task_issues = _normalize_human_blocking_issues(
            task.metadata.get("human_blocking_issues") if isinstance(task.metadata, dict) else None
        )
        task_details = task.description or ""
        if task_issues:
            issue_summary = "; ".join(issue.get("summary", "") for issue in task_issues if issue.get("summary"))
            if issue_summary:
                task_details = (f"{task_details}\n\n" if task_details else "") + f"Human blockers: {issue_summary}"

        events: list[dict[str, Any]] = [
            {
                "id": f"task-{task.id}",
                "type": "status_change",
                "timestamp": task.updated_at or task.created_at,
                "actor": "system",
                "actor_type": "system",
                "summary": f"Task status: {task.status}",
                "details": task_details,
                "human_blocking_issues": task_issues,
            }
        ]
        for event in container.events.list_recent(limit=2000):
            if event.get("entity_id") != task_id:
                continue
            payload = event.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            issues = _normalize_human_blocking_issues(
                payload_dict.get("issues") if "issues" in payload_dict else payload_dict.get("human_blocking_issues")
            )
            details = str(payload_dict.get("error") or payload_dict.get("guidance") or "")
            if not details and issues:
                details = "; ".join(issue.get("summary", "") for issue in issues if issue.get("summary"))
            events.append(
                {
                    "id": str(event.get("id") or f"evt-{uuid.uuid4().hex[:10]}"),
                    "type": str(event.get("type") or "event"),
                    "timestamp": str(event.get("ts") or task.created_at),
                    "actor": "system",
                    "actor_type": "system",
                    "summary": str(event.get("type") or "event"),
                    "details": details,
                    "human_blocking_issues": issues,
                }
            )
        for item in _load_feedback_records(container):
            if item.get("task_id") != task_id:
                continue
            events.append(
                {
                    "id": f"feedback-{item.get('id')}",
                    "type": "feedback",
                    "timestamp": item.get("created_at") or task.created_at,
                    "actor": str(item.get("created_by") or "human"),
                    "actor_type": "human",
                    "summary": str(item.get("summary") or "Feedback added"),
                    "details": str(item.get("details") or ""),
                }
            )
        for item in _load_comment_records(container):
            if item.get("task_id") != task_id:
                continue
            events.append(
                {
                    "id": f"comment-{item.get('id')}",
                    "type": "comment",
                    "timestamp": item.get("created_at") or task.created_at,
                    "actor": str(item.get("author") or "human"),
                    "actor_type": "human",
                    "summary": str(item.get("body") or "Comment added"),
                    "details": "",
                }
            )
        events.sort(key=lambda event: str(event.get("timestamp") or ""), reverse=True)
        return {"events": events}

    @router.get("/collaboration/feedback/{task_id}")
    async def get_collaboration_feedback(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        items = [item for item in _load_feedback_records(container) if item.get("task_id") == task_id]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"feedback": items}

    @router.post("/collaboration/feedback")
    async def add_collaboration_feedback(body: AddFeedbackRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(body.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        item = {
            "id": f"fb-{uuid.uuid4().hex[:10]}",
            "task_id": body.task_id,
            "feedback_type": body.feedback_type,
            "priority": body.priority,
            "status": "active",
            "summary": body.summary,
            "details": body.details,
            "target_file": body.target_file,
            "action": f"{body.feedback_type}: {body.summary}",
            "created_by": "human",
            "created_at": now_iso(),
            "agent_response": None,
        }
        items = _load_feedback_records(container)
        items.append(item)
        _save_feedback_records(container, items)
        bus.emit(channel="review", event_type="feedback.added", entity_id=body.task_id, payload={"feedback_id": item["id"]})
        return {"feedback": item}

    @router.post("/collaboration/feedback/{feedback_id}/dismiss")
    async def dismiss_collaboration_feedback(feedback_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        items = _load_feedback_records(container)
        for item in items:
            if item.get("id") == feedback_id:
                item["status"] = "addressed"
                item["agent_response"] = item.get("agent_response") or "Dismissed by reviewer"
                _save_feedback_records(container, items)
                bus.emit(channel="review", event_type="feedback.dismissed", entity_id=str(item.get("task_id") or ""), payload={"feedback_id": feedback_id})
                return {"feedback": item}
        raise HTTPException(status_code=404, detail="Feedback not found")

    @router.get("/collaboration/comments/{task_id}")
    async def get_collaboration_comments(task_id: str, project_dir: Optional[str] = Query(None), file_path: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        items = []
        for item in _load_comment_records(container):
            if item.get("task_id") != task_id:
                continue
            if file_path and item.get("file_path") != file_path:
                continue
            items.append(item)
        items.sort(key=lambda item: str(item.get("created_at") or ""))
        return {"comments": items}

    @router.post("/collaboration/comments")
    async def add_collaboration_comment(body: AddCommentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(body.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        item = {
            "id": f"cm-{uuid.uuid4().hex[:10]}",
            "task_id": body.task_id,
            "file_path": body.file_path,
            "line_number": body.line_number,
            "line_type": body.line_type,
            "body": body.body,
            "author": "human",
            "created_at": now_iso(),
            "resolved": False,
            "parent_id": body.parent_id,
        }
        items = _load_comment_records(container)
        items.append(item)
        _save_comment_records(container, items)
        bus.emit(channel="review", event_type="comment.added", entity_id=body.task_id, payload={"comment_id": item["id"], "file_path": body.file_path})
        return {"comment": item}

    @router.post("/collaboration/comments/{comment_id}/resolve")
    async def resolve_collaboration_comment(comment_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        items = _load_comment_records(container)
        for item in items:
            if item.get("id") == comment_id:
                item["resolved"] = True
                _save_comment_records(container, items)
                bus.emit(channel="review", event_type="comment.resolved", entity_id=str(item.get("task_id") or ""), payload={"comment_id": comment_id})
                return {"comment": item}
        raise HTTPException(status_code=404, detail="Comment not found")

    @router.post("/quick-actions")
    async def create_quick_action(body: QuickActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        import asyncio
        from ..quick_actions.executor import QuickActionExecutor

        container, bus, _ = _ctx(project_dir)
        pending_runs = [run for run in container.quick_actions.list() if run.status in {"queued", "running"}]
        if len(pending_runs) >= QUICK_ACTION_MAX_PENDING:
            raise HTTPException(status_code=429, detail="Too many pending quick actions; wait for active runs to finish.")
        run = QuickActionRun(prompt=body.prompt, status="queued")
        container.quick_actions.upsert(run)
        bus.emit(channel="quick_actions", event_type="quick_action.queued", entity_id=run.id, payload={"status": run.status})

        response_snapshot = run.to_dict()

        executor = QuickActionExecutor(container, bus)
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, executor.execute, run)

        return {"quick_action": response_snapshot}

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
            return {"task": _task_payload(task) if task else None, "already_promoted": True}
        title = body.title or f"Promoted quick action: {run.prompt[:50]}"
        task = Task(title=title, description=run.prompt, source="promoted_quick_action", priority=body.priority)
        container.tasks.upsert(task)
        run.promoted_task_id = task.id
        container.quick_actions.upsert(run)
        bus.emit(channel="quick_actions", event_type="quick_action.promoted", entity_id=run.id, payload={"task_id": task.id})
        return {"task": _task_payload(task), "already_promoted": False}

    @router.get("/review-queue")
    async def review_queue(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        items = [_task_payload(task) for task in container.tasks.list() if task.status == "in_review"]
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
        return {"task": _task_payload(task)}

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
        return {"task": _task_payload(task)}

    @router.get("/orchestrator/status")
    async def orchestrator_status(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.status()

    @router.post("/orchestrator/control")
    async def orchestrator_control(body: OrchestratorControlRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        _, _, orchestrator = _ctx(project_dir)
        return orchestrator.control(body.action)

    @router.get("/settings")
    async def get_settings(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _settings_payload(cfg)

    @router.patch("/settings")
    async def patch_settings(body: UpdateSettingsRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        cfg = container.config.load()
        touched_sections: list[str] = []

        if body.orchestrator is not None:
            orchestrator_cfg = dict(cfg.get("orchestrator") or {})
            orchestrator_cfg.update(body.orchestrator.model_dump())
            cfg["orchestrator"] = orchestrator_cfg
            touched_sections.append("orchestrator")

        if body.agent_routing is not None:
            routing_cfg = dict(cfg.get("agent_routing") or {})
            routing_cfg.update(body.agent_routing.model_dump())
            cfg["agent_routing"] = routing_cfg
            touched_sections.append("agent_routing")

        if body.defaults is not None:
            defaults_cfg = dict(cfg.get("defaults") or {})
            incoming_defaults = body.defaults.model_dump()
            incoming_quality_gate = dict(incoming_defaults.get("quality_gate") or {})
            quality_gate_cfg = dict(defaults_cfg.get("quality_gate") or {})
            quality_gate_cfg.update(incoming_quality_gate)
            defaults_cfg["quality_gate"] = quality_gate_cfg
            cfg["defaults"] = defaults_cfg
            touched_sections.append("defaults.quality_gate")

        if body.workers is not None:
            workers_cfg = dict(cfg.get("workers") or {})
            incoming_workers = body.workers.model_dump(exclude_none=True, exclude_unset=True)

            if "default" in incoming_workers:
                workers_cfg["default"] = str(incoming_workers.get("default") or "codex")
            if "routing" in incoming_workers:
                workers_cfg["routing"] = dict(incoming_workers.get("routing") or {})
            if "providers" in incoming_workers:
                workers_cfg["providers"] = dict(incoming_workers.get("providers") or {})

            normalized_workers = _settings_payload({"workers": workers_cfg})["workers"]
            cfg["workers"] = normalized_workers
            touched_sections.append("workers")

        if body.project is not None and body.project.commands is not None:
            project_cfg = dict(cfg.get("project") or {})
            existing_commands = dict(project_cfg.get("commands") or {})
            for raw_lang, lang_req in body.project.commands.items():
                lang = raw_lang.strip().lower()
                if not lang:
                    continue
                lang_entry = dict(existing_commands.get(lang) or {})
                for field in ("test", "lint", "typecheck", "format"):
                    value = getattr(lang_req, field)
                    if value is None:
                        continue
                    if value == "":
                        lang_entry.pop(field, None)
                    else:
                        lang_entry[field] = value
                if lang_entry:
                    existing_commands[lang] = lang_entry
                else:
                    existing_commands.pop(lang, None)
            project_cfg["commands"] = existing_commands
            cfg["project"] = project_cfg
            touched_sections.append("project.commands")

        container.config.save(cfg)
        bus.emit(
            channel="system",
            event_type="settings.updated",
            entity_id=container.project_id,
            payload={"sections": touched_sections},
        )
        return _settings_payload(cfg)

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
