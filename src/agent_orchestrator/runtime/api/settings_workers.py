"""Settings and worker payload helpers for runtime API routes."""

from __future__ import annotations

from typing import Any, Optional

from ...workers.config import WorkerProviderSpec, get_workers_runtime_config
from ...workers.diagnostics import test_worker
from ..domain.models import now_iso
from .helpers import _coerce_bool, _coerce_int, _normalize_str_map


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
        ollama_provider: dict[str, Any] = {"type": "ollama"}
        if endpoint:
            ollama_provider["endpoint"] = endpoint
        if model:
            ollama_provider["model"] = model
        temperature = raw_item.get("temperature")
        if isinstance(temperature, (int, float)):
            ollama_provider["temperature"] = float(temperature)
        num_ctx = raw_item.get("num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0:
            ollama_provider["num_ctx"] = num_ctx
        providers[name] = ollama_provider

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
            "dependency_policy": str(defaults.get("dependency_policy") or "prudent")
            if str(defaults.get("dependency_policy") or "prudent") in ("permissive", "prudent", "strict")
            else "prudent",
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
