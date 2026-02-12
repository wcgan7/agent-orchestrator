"""Parse worker provider configuration and resolve routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

WorkerProviderType = Literal["codex", "ollama"]


@dataclass(frozen=True)
class WorkerProviderSpec:
    name: str
    type: WorkerProviderType
    # codex
    command: Optional[str] = None
    # ollama
    endpoint: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    num_ctx: Optional[int] = None


@dataclass(frozen=True)
class WorkersRuntimeConfig:
    """Resolved worker configuration for a run."""

    default_worker: str
    routing: dict[str, str]
    providers: dict[str, WorkerProviderSpec]
    cli_worker_override: Optional[str] = None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _step_key(step: str) -> str:
    return str(step or "").strip()


def get_workers_runtime_config(
    *,
    config: dict[str, Any],
    codex_command_fallback: str,
    cli_worker: Optional[str] = None,
) -> WorkersRuntimeConfig:
    workers_cfg = _as_dict(config.get("workers"))
    routing = _as_dict(workers_cfg.get("routing"))
    providers_cfg = _as_dict(workers_cfg.get("providers"))

    default_worker = str(workers_cfg.get("default") or "codex").strip() or "codex"

    providers: dict[str, WorkerProviderSpec] = {}

    # Always provide a built-in codex provider; config can override fields.
    codex_cfg = _as_dict(providers_cfg.get("codex"))
    codex_command = str(codex_cfg.get("command") or codex_command_fallback).strip()
    providers["codex"] = WorkerProviderSpec(name="codex", type="codex", command=codex_command)

    for name, raw in providers_cfg.items():
        if not isinstance(name, str) or not name.strip():
            continue
        item = _as_dict(raw)
        typ = str(item.get("type") or "").strip().lower()
        if typ == "local":
            typ = "ollama"
        if typ not in {"codex", "ollama"}:
            continue
        if typ == "codex":
            cmd = str(item.get("command") or codex_command_fallback).strip()
            providers[name] = WorkerProviderSpec(name=name, type="codex", command=cmd)
            continue

        endpoint = str(item.get("endpoint") or "").strip() or None
        model = str(item.get("model") or "").strip() or None
        temperature = item.get("temperature")
        num_ctx = item.get("num_ctx")
        providers[name] = WorkerProviderSpec(
            name=name,
            type="ollama",
            endpoint=endpoint,
            model=model,
            temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
            num_ctx=int(num_ctx) if isinstance(num_ctx, int) else None,
        )

    # Normalize routing values to strings.
    routing_out: dict[str, str] = {}
    for k, v in routing.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        routing_out[k.strip()] = v.strip()

    return WorkersRuntimeConfig(
        default_worker=default_worker,
        routing=routing_out,
        providers=providers,
        cli_worker_override=cli_worker.strip() if isinstance(cli_worker, str) and cli_worker.strip() else None,
    )


def resolve_worker_for_step(runtime: WorkersRuntimeConfig, step: str) -> WorkerProviderSpec:
    """Resolve which worker provider should handle a given task step.

    Note: plan tasks are routed via the special key `"plan"` (since planning is
    represented by task.type="plan" rather than a dedicated TaskStep).
    """
    if runtime.cli_worker_override:
        name = runtime.cli_worker_override
    else:
        name = runtime.routing.get(_step_key(step)) or runtime.default_worker

    if name not in runtime.providers:
        available = ", ".join(sorted(runtime.providers.keys()))
        raise ValueError(f"Unknown worker '{name}' (available: {available})")
    spec = runtime.providers[name]

    if spec.type == "codex":
        if not spec.command:
            raise ValueError(f"Worker '{spec.name}' missing required 'command'")
        return spec
    if spec.type == "ollama":
        if not spec.endpoint or not spec.model:
            raise ValueError(f"Worker '{spec.name}' missing required 'endpoint' and/or 'model'")
        return spec
    raise ValueError(f"Unsupported worker type '{spec.type}'")
