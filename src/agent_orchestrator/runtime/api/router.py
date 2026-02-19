from __future__ import annotations

import json
import subprocess
from hashlib import sha256
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...collaboration.modes import MODE_CONFIGS
from ...pipelines.registry import PipelineRegistry
from ...workers.config import WorkerProviderSpec, get_workers_runtime_config
from ...workers.diagnostics import test_worker
from ..domain.models import AgentRecord, QuickActionRun, Task, now_iso
from ..events.bus import EventBus
from ..orchestrator.service import OrchestratorService
from ..storage.container import Container


class CreateTaskRequest(BaseModel):
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
    dependency_policy: Optional[str] = None
    worker_model: Optional[str] = None
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
    timeout: Optional[int] = None


class PromoteQuickActionRequest(BaseModel):
    title: Optional[str] = None
    priority: str = "P2"


class PlanRefineRequest(BaseModel):
    base_revision_id: Optional[str] = None
    feedback: str
    instructions: Optional[str] = None
    priority: Literal["normal", "high"] = "normal"


class CommitPlanRequest(BaseModel):
    revision_id: str


class CreatePlanRevisionRequest(BaseModel):
    content: str
    parent_revision_id: Optional[str] = None
    feedback_note: Optional[str] = None


class GenerateTasksRequest(BaseModel):
    source: Optional[Literal["committed", "revision", "override", "latest"]] = None
    revision_id: Optional[str] = None
    plan_override: Optional[str] = None
    infer_deps: bool = True


class ApproveGateRequest(BaseModel):
    gate: Optional[str] = None


class OrchestratorControlRequest(BaseModel):
    action: str


class OrchestratorSettingsRequest(BaseModel):
    concurrency: int = Field(2, ge=1, le=128)
    auto_deps: bool = True
    max_review_attempts: int = Field(10, ge=1, le=50)


class AgentRoutingSettingsRequest(BaseModel):
    default_role: str = "general"
    task_type_roles: dict[str, str] = Field(default_factory=dict)
    role_provider_overrides: dict[str, str] = Field(default_factory=dict)


class WorkerProviderSettingsRequest(BaseModel):
    type: str = "codex"
    command: Optional[str] = None
    reasoning_effort: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    num_ctx: Optional[int] = None


class WorkersSettingsRequest(BaseModel):
    default: str = "codex"
    default_model: Optional[str] = None
    heartbeat_seconds: Optional[int] = Field(None, ge=1, le=3600)
    heartbeat_grace_seconds: Optional[int] = Field(None, ge=1, le=7200)
    routing: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, WorkerProviderSettingsRequest] = Field(default_factory=dict)


class QualityGateSettingsRequest(BaseModel):
    critical: int = Field(0, ge=0)
    high: int = Field(0, ge=0)
    medium: int = Field(0, ge=0)
    low: int = Field(0, ge=0)


class DefaultsSettingsRequest(BaseModel):
    quality_gate: QualityGateSettingsRequest = Field(default_factory=QualityGateSettingsRequest)
    dependency_policy: str = "prudent"


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


class RetryTaskRequest(BaseModel):
    guidance: Optional[str] = None
    start_from_step: Optional[str] = None


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
    "backlog": {"queued", "cancelled"},
    "queued": {"backlog", "cancelled"},
    "in_progress": {"cancelled"},
    "in_review": {"done", "blocked", "cancelled"},
    "blocked": {"queued", "in_review", "cancelled"},
    "done": set(),
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


def _build_execution_summary(task: Task, container: "Container") -> Optional[dict[str, Any]]:
    """Build execution summary from the latest RunRecord's step data."""
    if task.status not in ("in_review", "blocked", "done"):
        return None
    if not task.run_ids:
        return None
    # Find the latest run
    run = None
    for run_id in reversed(task.run_ids):
        run = container.runs.get(run_id)
        if run:
            break
    if not run or not run.steps:
        return None
    steps: list[dict[str, Any]] = []
    for step_data in run.steps:
        if not isinstance(step_data, dict):
            continue
        step_status = str(step_data.get("status") or "unknown")
        if step_status == "skipped":
            continue
        entry: dict[str, Any] = {
            "step": str(step_data.get("step") or "unknown"),
            "status": step_status,
            "summary": str(step_data.get("summary") or ""),
        }
        if step_data.get("open_counts") and isinstance(step_data["open_counts"], dict):
            entry["open_counts"] = step_data["open_counts"]
        if step_data.get("commit"):
            entry["commit"] = str(step_data["commit"])
        steps.append(entry)
    if not steps:
        return None
    # Derive a human-readable run summary
    status_labels = {
        "in_review": "Awaiting human review",
        "blocked": "Blocked",
        "done": "Completed",
        "error": "Failed",
        "running": "Running",
    }
    run_summary = status_labels.get(run.status) or status_labels.get(task.status) or run.status
    return {
        "run_id": run.id,
        "run_status": run.status,
        "run_summary": run_summary,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "steps": steps,
    }


def _task_payload(task: Task, container: Optional["Container"] = None) -> dict[str, Any]:
    payload = task.to_dict()
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    payload["human_blocking_issues"] = _normalize_human_blocking_issues(metadata.get("human_blocking_issues"))
    if container is not None:
        summary = _build_execution_summary(task, container)
        if summary is not None:
            payload["execution_summary"] = summary
    return payload


def _safe_state_path(raw_path: Any, state_root: Path) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        path = Path(raw_path).expanduser().resolve()
        root = state_root.resolve()
    except Exception:
        return None
    if path == root or root in path.parents:
        return path
    return None


def _read_tail(path: Optional[Path], max_chars: int) -> tuple[str, int]:
    """Read the tail of a file.

    Returns ``(text, tail_start_byte)`` where *tail_start_byte* is the byte
    offset in the file where the returned text begins.  When the entire file
    fits within *max_chars*, ``tail_start_byte`` is ``0``.
    """
    if not path or not path.exists() or not path.is_file() or max_chars <= 0:
        return "", 0
    try:
        size = path.stat().st_size
    except Exception:
        return "", 0
    if size <= 0:
        return "", 0

    # Read only a bounded tail window (overshoot bytes for multibyte chars).
    max_bytes = min(size, max_chars * 4)
    start_byte = max(0, size - max_bytes)
    prev_byte = b"\n"
    try:
        with open(path, "rb") as fh:
            if start_byte > 0:
                fh.seek(start_byte - 1)
                prev_byte = fh.read(1) or b"\n"
            fh.seek(start_byte)
            raw = fh.read(max_bytes)
    except Exception:
        return "", 0

    text = raw.decode("utf-8", errors="replace")
    if not text:
        return "", size

    cut_at = max(0, len(text) - max_chars)
    slice_starts_mid_line = False
    if cut_at > 0:
        slice_starts_mid_line = text[cut_at - 1] != "\n"
    elif start_byte > 0 and prev_byte != b"\n":
        slice_starts_mid_line = True

    # If the returned slice starts in the middle of a line, drop the orphaned
    # prefix so NDJSON/text consumers only see full lines.
    if slice_starts_mid_line:
        first_newline = text.find("\n", cut_at)
        if first_newline == -1:
            return "", size
        cut_at = first_newline + 1

    tail = text[cut_at:]
    tail_start_byte = start_byte + len(text[:cut_at].encode("utf-8"))
    return tail, tail_start_byte


def _read_from_offset(
    path: Optional[Path],
    offset: int,
    max_bytes: int,
    read_to: int = 0,
    *,
    align_start_to_line: bool = False,
) -> tuple[str, int, int]:
    """Read file content from *offset* bytes forward.

    Returns ``(text, new_offset, chunk_start_offset)`` where *new_offset* is the
    byte offset after the returned chunk and *chunk_start_offset* is the actual
    byte offset where the returned chunk begins.

    If *read_to* > 0, do not read past that byte position in the file.
    """
    if not path or not path.exists() or not path.is_file():
        return "", 0, 0
    try:
        size = path.stat().st_size
        end = min(size, read_to) if read_to > 0 else size
        start = max(offset, 0)
        if start >= end:
            bounded = min(start, size)
            return "", bounded, bounded

        aligned_start = start
        if align_start_to_line and start > 0:
            lookback = min(start, max(max_bytes, 8192))
            with open(path, "rb") as fh:
                fh.seek(start - lookback)
                prefix = fh.read(lookback)
            newline_idx = prefix.rfind(b"\n")
            if newline_idx >= 0:
                aligned_start = (start - lookback) + newline_idx + 1

        read_limit = end - aligned_start
        if read_limit <= 0:
            bounded = min(aligned_start, size)
            return "", bounded, bounded
        # Allow aligned reads to exceed ``max_bytes`` by at most one chunk size.
        capped = min(read_limit, max_bytes * 2 if align_start_to_line else max_bytes)
        with open(path, "rb") as fh:
            fh.seek(aligned_start)
            raw = fh.read(capped)
        text = raw.decode("utf-8", errors="replace")
        return text, aligned_start + len(raw), aligned_start
    except Exception:
        return "", 0, 0


def _logs_snapshot_id(logs_meta: dict[str, Any]) -> str:
    parts = [
        str(logs_meta.get("run_dir") or ""),
        str(logs_meta.get("stdout_path") or ""),
        str(logs_meta.get("stderr_path") or ""),
        str(logs_meta.get("started_at") or ""),
    ]
    material = "|".join(parts).strip("|")
    if not material:
        return ""
    return sha256(material.encode("utf-8", errors="ignore")).hexdigest()[:16]


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
        if provider_type not in {"codex", "ollama", "claude"}:
            continue

        if provider_type in {"codex", "claude"}:
            default_command = "codex exec" if provider_type == "codex" else "claude -p"
            command = str(raw_item.get("command") or default_command).strip() or default_command
            provider: dict[str, Any] = {"type": provider_type, "command": command}
            model = str(raw_item.get("model") or "").strip()
            if model:
                provider["model"] = model
            reasoning_effort = str(raw_item.get("reasoning_effort") or "").strip().lower()
            if reasoning_effort in {"low", "medium", "high"}:
                provider["reasoning_effort"] = reasoning_effort
            providers[name] = provider
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
    codex_command = "codex exec"
    codex_model = None
    codex_reasoning = None
    if isinstance(codex, dict):
        codex_command = str(codex.get("command") or "codex exec").strip() or "codex exec"
        codex_model = str(codex.get("model") or "").strip() or None
        raw_reasoning = str(codex.get("reasoning_effort") or "").strip().lower()
        codex_reasoning = raw_reasoning if raw_reasoning in {"low", "medium", "high"} else None
    providers["codex"] = {"type": "codex", "command": codex_command}
    if codex_model:
        providers["codex"]["model"] = codex_model
    if codex_reasoning:
        providers["codex"]["reasoning_effort"] = codex_reasoning
    return providers


def _settings_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    orchestrator = dict(cfg.get("orchestrator") or {})
    routing = dict(cfg.get("agent_routing") or {})
    defaults = dict(cfg.get("defaults") or {})
    quality_gate = dict(defaults.get("quality_gate") or {})
    workers_cfg = dict(cfg.get("workers") or {})
    workers_providers = _normalize_workers_providers(workers_cfg.get("providers"))
    workers_default = str(workers_cfg.get("default") or "codex").strip() or "codex"
    workers_default_model = str(workers_cfg.get("default_model") or "").strip()
    workers_heartbeat_seconds = _coerce_int(workers_cfg.get("heartbeat_seconds"), 60, minimum=1, maximum=3600)
    workers_heartbeat_grace_seconds = _coerce_int(
        workers_cfg.get("heartbeat_grace_seconds"), 240, minimum=1, maximum=7200
    )
    if workers_heartbeat_grace_seconds < workers_heartbeat_seconds:
        workers_heartbeat_grace_seconds = workers_heartbeat_seconds
    if workers_default not in workers_providers:
        workers_default = "codex"
    return {
        "orchestrator": {
            "concurrency": _coerce_int(orchestrator.get("concurrency"), 2, minimum=1, maximum=128),
            "auto_deps": _coerce_bool(orchestrator.get("auto_deps"), True),
            "max_review_attempts": _coerce_int(orchestrator.get("max_review_attempts"), 10, minimum=1, maximum=50),
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
            },
            "dependency_policy": str(defaults.get("dependency_policy") or "prudent") if str(defaults.get("dependency_policy") or "prudent") in ("permissive", "prudent", "strict") else "prudent",
        },
        "workers": {
            "default": workers_default,
            "default_model": workers_default_model,
            "heartbeat_seconds": workers_heartbeat_seconds,
            "heartbeat_grace_seconds": workers_heartbeat_grace_seconds,
            "routing": _normalize_str_map(workers_cfg.get("routing")),
            "providers": workers_providers,
        },
        "project": {
            "commands": dict((cfg.get("project") or {}).get("commands") or {}),
        },
    }


def _workers_health_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
    known_provider_names: list[str] = ["codex", "claude", "ollama"]
    for name in sorted(runtime.providers.keys()):
        if name not in known_provider_names:
            known_provider_names.append(name)

    providers: list[dict[str, Any]] = []
    checked_at = now_iso()
    for name in known_provider_names:
        spec = runtime.providers.get(name)
        if spec is None:
            # Probe common CLI providers even when not configured, so users can
            # see local availability before adding explicit settings.
            probe_spec: Optional[WorkerProviderSpec] = None
            if name == "codex":
                probe_spec = WorkerProviderSpec(name="codex", type="codex", command="codex exec")
            elif name == "claude":
                probe_spec = WorkerProviderSpec(name="claude", type="claude", command="claude -p")

            if probe_spec is not None:
                healthy, detail = test_worker(probe_spec)
                providers.append(
                    {
                        "name": name,
                        "type": probe_spec.type,
                        "configured": False,
                        "healthy": healthy,
                        "status": "connected" if healthy else "not_configured",
                        "detail": detail if healthy else f"Provider not configured. {detail}",
                        "checked_at": checked_at,
                        "command": probe_spec.command,
                    }
                )
                continue
            providers.append(
                {
                    "name": name,
                    "type": name if name in {"codex", "claude", "ollama"} else "unknown",
                    "configured": False,
                    "healthy": False,
                    "status": "not_configured",
                    "detail": "Provider is not configured.",
                    "checked_at": checked_at,
                }
            )
            continue

        healthy, detail = test_worker(spec)
        item: dict[str, Any] = {
            "name": spec.name,
            "type": spec.type,
            "configured": True,
            "healthy": healthy,
            "status": "connected" if healthy else "unavailable",
            "detail": detail,
            "checked_at": checked_at,
        }
        if spec.command:
            item["command"] = spec.command
        if spec.endpoint:
            item["endpoint"] = spec.endpoint
        if spec.model:
            item["model"] = spec.model
        providers.append(item)

    return {"providers": providers}


def _workers_routing_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
    default_provider = runtime.default_worker if runtime.default_worker in runtime.providers else "codex"
    canonical_steps = ["plan", "analyze", "generate_tasks", "implement", "verify", "review", "commit"]
    ordered_steps = list(dict.fromkeys(canonical_steps + sorted(runtime.routing.keys())))

    rows: list[dict[str, Any]] = []
    for step in ordered_steps:
        provider_name = runtime.routing.get(step) or default_provider
        provider = runtime.providers.get(provider_name)
        rows.append(
            {
                "step": step,
                "provider": provider_name,
                "provider_type": provider.type if provider else None,
                "source": "explicit" if step in runtime.routing else "default",
                "configured": provider is not None,
            }
        )

    return {"default": default_provider, "rows": rows}


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


def _has_unresolved_blockers(container: Container, task: Task) -> Optional[str]:
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


def create_router(
    resolve_container: Any,
    resolve_orchestrator: Any,
    job_store: dict[str, dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["api"])

    def _ctx(project_dir: Optional[str]) -> tuple[Container, EventBus, OrchestratorService]:
        container: Container = resolve_container(project_dir)
        bus = EventBus(container.events, container.project_id)
        orchestrator: OrchestratorService = resolve_orchestrator(project_dir)
        return container, bus, orchestrator

    def _load_feedback_records(container: Container) -> list[dict[str, Any]]:
        cfg = container.config.load()
        raw = cfg.get("collaboration_feedback")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save_feedback_records(container: Container, items: list[dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["collaboration_feedback"] = items
        container.config.save(cfg)

    def _load_comment_records(container: Container) -> list[dict[str, Any]]:
        cfg = container.config.load()
        raw = cfg.get("collaboration_comments")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save_comment_records(container: Container, items: list[dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["collaboration_comments"] = items
        container.config.save(cfg)

    def _load_import_jobs(container: Container) -> dict[str, dict[str, Any]]:
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

    def _save_import_jobs(container: Container, jobs: dict[str, dict[str, Any]]) -> None:
        cfg = container.config.load()
        cfg["import_jobs"] = list(_pruned_import_jobs(jobs).values())
        container.config.save(cfg)

    def _upsert_import_job(container: Container, job: dict[str, Any]) -> None:
        jobs = _load_import_jobs(container)
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            return
        jobs[job_id] = job
        _save_import_jobs(container, jobs)

    def _fetch_import_job(container: Container, job_id: str) -> Optional[dict[str, Any]]:
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
        dep_policy = body.dependency_policy.strip() if body.dependency_policy else ""
        if dep_policy not in ("permissive", "prudent", "strict"):
            cfg = container.config.load()
            dep_policy = str((cfg.get("defaults") or {}).get("dependency_policy") or "prudent")
            if dep_policy not in ("permissive", "prudent", "strict"):
                dep_policy = "prudent"
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
            dependency_policy=dep_policy,
            source=body.source,
            worker_model=(str(body.worker_model).strip() if body.worker_model else None),
            metadata=body.metadata,
        )
        if body.status in ("backlog", "queued"):
            task.status = body.status
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
        columns = {name: [] for name in ["backlog", "queued", "in_progress", "in_review", "blocked", "done", "cancelled"]}
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(_task_payload(task))
        for key, items in columns.items():
            items.sort(key=lambda x: (_priority_rank(str(x.get("priority") or "P3")), str(x.get("created_at") or "")))
        return {"columns": columns}

    @router.get("/tasks/execution-order")
    async def execution_order(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        tasks = container.tasks.list()
        terminal = {"done", "cancelled"}
        pending = [t for t in tasks if t.status not in terminal]
        completed = [t for t in tasks if t.status in terminal]
        completed.sort(key=lambda t: t.updated_at, reverse=True)
        return {
            "batches": _execution_batches(pending),
            "completed": [t.id for t in completed],
        }

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": _task_payload(task, container)}

    @router.get("/tasks/{task_id}/diff")
    async def get_task_diff(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the git diff for a task's latest commit."""
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Find the commit SHA from the latest run's steps.
        commit_sha: str | None = None
        for run_id in reversed(task.run_ids):
            for run in container.runs.list():
                if run.id == run_id:
                    for step in reversed(run.steps or []):
                        if isinstance(step, dict) and step.get("step") == "commit" and step.get("commit"):
                            commit_sha = str(step["commit"])
                            break
                    if commit_sha:
                        break
            if commit_sha:
                break

        if not commit_sha:
            return {"commit": None, "files": [], "diff": "", "stat": ""}

        git_dir = container.project_dir
        try:
            stat_result = subprocess.run(
                ["git", "show", "--stat", "--format=", commit_sha],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
            diff_result = subprocess.run(
                ["git", "diff", f"{commit_sha}~1..{commit_sha}"],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read diff: {exc}") from exc

        # Parse stat lines into structured file list.
        files: list[dict[str, str]] = []
        for line in stat_result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("Showing") or " | " not in line:
                continue
            parts = line.split(" | ", 1)
            file_path = parts[0].strip()
            change_info = parts[1].strip() if len(parts) > 1 else ""
            files.append({"path": file_path, "changes": change_info})

        return {
            "commit": commit_sha,
            "files": files,
            "diff": diff_result.stdout,
            "stat": stat_result.stdout,
        }

    @router.get("/tasks/{task_id}/logs")
    async def get_task_logs(
        task_id: str,
        project_dir: Optional[str] = Query(None),
        max_chars: int = Query(12000, ge=200, le=2000000),
        stdout_offset: int = Query(0, ge=0),
        stderr_offset: int = Query(0, ge=0),
        backfill: bool = Query(False),
        stdout_read_to: int = Query(0, ge=0),
        stderr_read_to: int = Query(0, ge=0),
        step: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        # When a specific step is requested, look up its log paths from run.steps.
        step_logs_meta: Optional[dict[str, Any]] = None
        if step:
            for run_id in reversed(task.run_ids):
                for run in container.runs.list():
                    if run.id == run_id:
                        # Find the last matching step entry (latest attempt).
                        for entry in reversed(run.steps or []):
                            if isinstance(entry, dict) and entry.get("step") == step and entry.get("stdout_path"):
                                step_logs_meta = entry
                                break
                        break
                if step_logs_meta:
                    break

        if step_logs_meta:
            logs_meta = step_logs_meta
            mode = "history"
        else:
            active_meta = metadata.get("active_logs") if isinstance(metadata.get("active_logs"), dict) else None
            last_meta = metadata.get("last_logs") if isinstance(metadata.get("last_logs"), dict) else None
            logs_meta = active_meta or last_meta or {}
            mode = "active" if active_meta else ("last" if last_meta else "none")

        stdout_path = _safe_state_path(logs_meta.get("stdout_path"), container.state_root)
        stderr_path = _safe_state_path(logs_meta.get("stderr_path"), container.state_root)
        progress_path = _safe_state_path(logs_meta.get("progress_path"), container.state_root)

        progress_payload: dict[str, Any] = {}
        if progress_path and progress_path.exists():
            try:
                raw = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    progress_payload = raw
            except Exception:
                progress_payload = {}

        use_incremental = backfill or stdout_offset > 0 or stderr_offset > 0
        stdout_tail_start = 0
        stderr_tail_start = 0
        stdout_chunk_start = 0
        stderr_chunk_start = 0
        if use_incremental:
            stdout_text, new_stdout_offset, stdout_chunk_start = _read_from_offset(
                stdout_path,
                stdout_offset,
                max_chars,
                read_to=stdout_read_to,
                align_start_to_line=backfill,
            )
            stderr_text, new_stderr_offset, stderr_chunk_start = _read_from_offset(
                stderr_path,
                stderr_offset,
                max_chars,
                read_to=stderr_read_to,
                align_start_to_line=backfill,
            )
        else:
            stdout_text, stdout_tail_start = _read_tail(stdout_path, max_chars)
            stderr_text, stderr_tail_start = _read_tail(stderr_path, max_chars)
            new_stdout_offset = stdout_path.stat().st_size if stdout_path and stdout_path.exists() else 0
            new_stderr_offset = stderr_path.stat().st_size if stderr_path and stderr_path.exists() else 0
            stdout_chunk_start = stdout_tail_start
            stderr_chunk_start = stderr_tail_start

        # Collect which steps have stored logs and count executions.
        available_steps: list[str] = []
        step_execution_counts: dict[str, int] = {}
        seen: set[str] = set()
        for run_id in task.run_ids[-1:]:
            for run in container.runs.list():
                if run.id == run_id:
                    for entry in (run.steps or []):
                        if isinstance(entry, dict) and entry.get("stdout_path"):
                            s = str(entry.get("step") or "")
                            if s:
                                step_execution_counts[s] = step_execution_counts.get(s, 0) + 1
                                if s not in seen:
                                    available_steps.append(s)
                                    seen.add(s)
                    break
        # Include the currently running step so users can switch between
        # earlier completed steps and the live step while a task is in progress.
        active_step = (metadata.get("active_logs") or {}).get("step") if isinstance(metadata.get("active_logs"), dict) else None
        if not active_step:
            active_step = task.current_step
        if active_step and active_step not in seen:
            available_steps.append(str(active_step))

        return {
            "mode": mode,
            "task_status": task.status,
            "step": str(logs_meta.get("step") or task.current_step or ""),
            "current_step": str(task.current_step or ""),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_offset": new_stdout_offset,
            "stderr_offset": new_stderr_offset,
            "stdout_chunk_start": stdout_chunk_start,
            "stderr_chunk_start": stderr_chunk_start,
            "stdout_tail_start": stdout_tail_start,
            "stderr_tail_start": stderr_tail_start,
            "started_at": logs_meta.get("started_at"),
            "finished_at": logs_meta.get("finished_at"),
            "log_id": _logs_snapshot_id(logs_meta),
            "progress": progress_payload,
            "available_steps": available_steps,
            "step_execution_counts": step_execution_counts,
        }

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
        return {"task": _task_payload(task, container)}

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
        if target == "queued":
            unresolved = _has_unresolved_blockers(container, task)
            if unresolved is not None:
                raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        task.status = target
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.transitioned", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, container)}

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
    async def retry_task(task_id: str, body: Optional[RetryTaskRequest] = None, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        unresolved = _has_unresolved_blockers(container, task)
        if unresolved is not None:
            raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        # Capture previous error before clearing it.
        previous_error = task.error or ""
        task.retry_count += 1
        task.status = "queued"
        task.error = None
        task.pending_gate = None
        task.metadata = task.metadata if isinstance(task.metadata, dict) else {}
        task.metadata.pop("human_blocking_issues", None)
        guidance = (body.guidance if body else None) or ""
        if guidance.strip() or previous_error.strip():
            task.metadata["retry_guidance"] = {
                "ts": now_iso(),
                "guidance": guidance.strip(),
                "previous_error": previous_error.strip(),
            }
        start_from = (body.start_from_step if body else None) or ""
        if start_from.strip():
            task.metadata["retry_from_step"] = start_from.strip()
        else:
            task.metadata.pop("retry_from_step", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.retry", entity_id=task.id, payload={"retry_count": task.retry_count})
        return {"task": _task_payload(task, container)}

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.status = "cancelled"
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.cancelled", entity_id=task.id, payload={})
        return {"task": _task_payload(task, container)}

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
        return {"task": _task_payload(task, container), "cleared_gate": cleared_gate}

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
        return {"task": _task_payload(task, container)}

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
        return {"task": _task_payload(task, container)}

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
        return {"task": _task_payload(task, container)}

    @router.get("/tasks/{task_id}/plan")
    async def get_task_plan(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = resolve_orchestrator(project_dir)
        try:
            return orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/tasks/{task_id}/plan/refine")
    async def refine_task_plan(
        task_id: str,
        body: PlanRefineRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.feedback or "").strip():
            raise HTTPException(status_code=400, detail="feedback is required")
        try:
            job = orchestrator.queue_plan_refine_job(
                task_id=task_id,
                base_revision_id=body.base_revision_id,
                feedback=body.feedback,
                instructions=body.instructions,
                priority=body.priority,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job": job.to_dict()}

    @router.get("/tasks/{task_id}/plan/jobs")
    async def list_plan_refine_jobs(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            jobs = orchestrator.list_plan_refine_jobs(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"jobs": [job.to_dict() for job in jobs]}

    @router.get("/tasks/{task_id}/plan/jobs/{job_id}")
    async def get_plan_refine_job(task_id: str, job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            job = orchestrator.get_plan_refine_job(task_id, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"job": job.to_dict()}

    @router.post("/tasks/{task_id}/plan/commit")
    async def commit_plan_revision(
        task_id: str,
        body: CommitPlanRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            committed_revision_id = orchestrator.commit_plan_revision(task_id, body.revision_id)
            plan_doc = orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "committed_revision_id": committed_revision_id,
            "latest_revision_id": plan_doc.get("latest_revision_id"),
            "committed_plan_revision_id": plan_doc.get("committed_revision_id"),
        }

    @router.post("/tasks/{task_id}/plan/revisions")
    async def create_plan_revision(
        task_id: str,
        body: CreatePlanRevisionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.content or "").strip():
            raise HTTPException(status_code=400, detail="content is required")
        try:
            revision = orchestrator.create_plan_revision(
                task_id=task_id,
                content=body.content,
                source="human_edit",
                parent_revision_id=body.parent_revision_id,
                feedback_note=body.feedback_note,
                step=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"revision": revision.to_dict()}

    @router.post("/tasks/{task_id}/generate-tasks")
    async def generate_tasks_from_plan(
        task_id: str, body: GenerateTasksRequest, project_dir: Optional[str] = Query(None)
    ) -> dict[str, Any]:
        container, _, orchestrator = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        source = body.source
        if source is None:
            # Backward compatibility: previous API accepted only optional plan_override.
            source = "override" if str(body.plan_override or "").strip() else "latest"
        if source == "revision" and not body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is required when source=revision")
        if source == "override" and not str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is required when source=override")
        if source != "revision" and body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is only valid when source=revision")
        if source != "override" and str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is only valid when source=override")
        try:
            plan_text, resolved_revision_id = orchestrator.resolve_plan_text_for_generation(
                task_id=task_id,
                source=source,
                revision_id=body.revision_id,
                plan_override=body.plan_override,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            created_ids = orchestrator.generate_tasks_from_plan(task_id, plan_text, infer_deps=body.infer_deps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Re-fetch parent to get updated children_ids
        updated_task = container.tasks.get(task_id)
        children = [_task_payload(container.tasks.get(cid)) for cid in created_ids if container.tasks.get(cid)]
        return {
            "task": _task_payload(updated_task) if updated_task else None,
            "created_task_ids": created_ids,
            "children": children,
            "source": source,
            "source_revision_id": resolved_revision_id,
        }

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
            task.status = "queued"
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
            container: Container = resolve_container(project_dir)
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
            elif task.status in {"queued", "blocked"}:
                completed_steps = 1
            progress = {
                "backlog": 0.0,
                "queued": 0.1,
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
        timeout_seconds: Optional[int] = None
        if body.timeout is not None:
            timeout_seconds = max(10, min(600, body.timeout))
        run = QuickActionRun(prompt=body.prompt, status="queued", timeout_seconds=timeout_seconds)
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

    @router.get("/quick-actions/{quick_action_id}/logs")
    async def get_quick_action_logs(
        quick_action_id: str,
        project_dir: Optional[str] = Query(None),
        stdout_offset: int = Query(0),
        stderr_offset: int = Query(0),
        max_chars: int = Query(65536),
    ) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        run = container.quick_actions.get(quick_action_id)
        if not run:
            raise HTTPException(status_code=404, detail="Quick action not found")
        max_chars = min(max(1024, max_chars), 262144)
        max_bytes = max_chars * 2

        stdout_path = _safe_state_path(run.stdout_path, container.state_root)
        stderr_path = _safe_state_path(run.stderr_path, container.state_root)

        stdout_text, new_stdout_offset, _ = _read_from_offset(stdout_path, stdout_offset, max_bytes)
        stderr_text, new_stderr_offset, _ = _read_from_offset(stderr_path, stderr_offset, max_bytes)

        return {
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_offset": new_stdout_offset,
            "stderr_offset": new_stderr_offset,
            "status": run.status,
            "finished_at": run.finished_at,
        }

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
        return {"task": _task_payload(task, container)}

    @router.post("/review/{task_id}/request-changes")
    async def request_review_changes(task_id: str, body: ReviewActionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "in_review":
            raise HTTPException(status_code=400, detail=f"Task {task_id} is not in_review")
        task.status = "queued"
        task.metadata["requested_changes"] = {"ts": now_iso(), "guidance": body.guidance}
        task.metadata["retry_from_step"] = "implement"
        container.tasks.upsert(task)
        bus.emit(channel="review", event_type="task.changes_requested", entity_id=task.id, payload={"guidance": body.guidance})
        return {"task": _task_payload(task, container)}

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

    @router.get("/workers/health")
    async def workers_health(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _workers_health_payload(cfg)

    @router.get("/workers/routing")
    async def workers_routing(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = _ctx(project_dir)
        cfg = container.config.load()
        return _workers_routing_payload(cfg)

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
            dep_policy = str(incoming_defaults.get("dependency_policy") or "").strip()
            if dep_policy in ("permissive", "prudent", "strict"):
                defaults_cfg["dependency_policy"] = dep_policy
            cfg["defaults"] = defaults_cfg
            touched_sections.append("defaults")

        if body.workers is not None:
            workers_cfg = dict(cfg.get("workers") or {})
            incoming_workers = body.workers.model_dump(exclude_none=True, exclude_unset=True)

            if "default" in incoming_workers:
                workers_cfg["default"] = str(incoming_workers.get("default") or "codex")
            if "default_model" in incoming_workers:
                default_model = str(incoming_workers.get("default_model") or "").strip()
                if default_model:
                    workers_cfg["default_model"] = default_model
                else:
                    workers_cfg.pop("default_model", None)
            if "heartbeat_seconds" in incoming_workers:
                workers_cfg["heartbeat_seconds"] = incoming_workers.get("heartbeat_seconds")
            if "heartbeat_grace_seconds" in incoming_workers:
                workers_cfg["heartbeat_grace_seconds"] = incoming_workers.get("heartbeat_grace_seconds")
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

    @router.delete("/agents/{agent_id}")
    async def remove_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}

    @router.post("/agents/{agent_id}/remove")
    async def remove_agent_post(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, bus, _ = _ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}

    return router


def os_access(path: Path) -> bool:
    try:
        list(path.iterdir())
    except Exception:
        return False
    return True
