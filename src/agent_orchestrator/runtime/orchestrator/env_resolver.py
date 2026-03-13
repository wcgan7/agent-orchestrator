"""Resolve worker environment variables from multiple sources.

Env vars are resolved with a 4-layer precedence model:

1. Auto-detected (lowest) — scanned from project files (.env, prisma, compose).
2. Process env — inherited from ``os.environ``.
3. Project-level — ``workers.environment.env_vars`` in runtime config.
4. Task-level (highest) — ``task.metadata["env_vars"]``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..domain.models import Task
from .venv_detector import detect_python_venv

# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

_DOTENV_FILES = (".env", ".env.local", ".env.test")

_PRISMA_ENV_RE = re.compile(r'env\("([^"]+)"\)')


def _parse_dotenv_files(project_dir: Path) -> dict[str, str]:
    """Parse .env, .env.local, .env.test (lowest to highest priority)."""
    result: dict[str, str] = {}
    for name in _DOTENV_FILES:
        path = project_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Strip optional 'export ' prefix
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            # Strip surrounding quotes (no inline comment stripping inside quotes)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Strip inline comments for unquoted values
                comment_idx = value.find(" #")
                if comment_idx >= 0:
                    value = value[:comment_idx].rstrip()
            result[key] = value
    return result


def _detect_prisma_required_vars(
    project_dir: Path, found: dict[str, str | None],
) -> dict[str, str | None]:
    """Extract ``env()`` references from ``prisma/schema.prisma``."""
    schema_path = project_dir / "prisma" / "schema.prisma"
    if not schema_path.is_file():
        return {}
    try:
        text = schema_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    result: dict[str, str | None] = {}
    for match in _PRISMA_ENV_RE.finditer(text):
        name = match.group(1)
        if name not in found:
            result[name] = None
    return result


def _detect_compose_env_vars(
    project_dir: Path, found: dict[str, str | None],
) -> dict[str, str | None]:
    """Extract environment vars from ``docker-compose.yml``."""
    result: dict[str, str | None] = {}
    for filename in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        compose_path = project_dir / filename
        if not compose_path.is_file():
            continue
        try:
            # Use yaml if available, otherwise skip
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            return {}
        try:
            text = compose_path.read_text(encoding="utf-8", errors="replace")
            doc = yaml.safe_load(text)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        services = doc.get("services")
        if not isinstance(services, dict):
            continue
        for _svc_name, svc_def in services.items():
            if not isinstance(svc_def, dict):
                continue
            environment = svc_def.get("environment")
            if isinstance(environment, list):
                for item in environment:
                    item_str = str(item or "")
                    if "=" in item_str:
                        key, _, value = item_str.partition("=")
                        key = key.strip()
                        if key and key not in found:
                            result[key] = value.strip() or None
                    else:
                        key = item_str.strip()
                        if key and key not in found:
                            result[key] = None
            elif isinstance(environment, dict):
                for key, value in environment.items():
                    key_str = str(key or "").strip()
                    if key_str and key_str not in found:
                        if value is not None:
                            result[key_str] = str(value)
                        else:
                            result[key_str] = None
        break  # Only process the first compose file found
    return result


def auto_detect_env_vars(project_dir: Path) -> dict[str, str | None]:
    """Scan project files for declared env var requirements.

    Returns dict of ``{VAR_NAME: value_or_None}``. ``None`` means the var is
    required but no value was found in project files.
    """
    result: dict[str, str | None] = {}
    result.update(_parse_dotenv_files(project_dir))
    result.update(_detect_prisma_required_vars(project_dir, found=result))
    result.update(_detect_compose_env_vars(project_dir, found=result))
    return result


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_env_vars_from_config(cfg: dict[str, Any]) -> dict[str, str]:
    """Extract ``env_vars`` from ``workers.environment`` config section."""
    workers = cfg.get("workers")
    if not isinstance(workers, dict):
        return {}
    environment = workers.get("environment")
    if not isinstance(environment, dict):
        return {}
    env_vars = environment.get("env_vars")
    if not isinstance(env_vars, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in env_vars.items():
        k = str(key or "").strip()
        if k and isinstance(value, str):
            result[k] = value
    return result


def _extract_env_vars_from_task(task: Task) -> dict[str, str]:
    """Extract ``env_vars`` from ``task.metadata``."""
    meta = task.metadata
    if not isinstance(meta, dict):
        return {}
    env_vars = meta.get("env_vars")
    if not isinstance(env_vars, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in env_vars.items():
        k = str(key or "").strip()
        if k and isinstance(value, str):
            result[k] = value
    return result


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_env_vars(
    *, project_dir: Path, cfg: dict[str, Any], task: Task,
) -> dict[str, str]:
    """Build the final merged env dict for worker execution.

    Precedence: auto-detected < process env < project config < task metadata.
    Only includes keys that have non-None values after merging.
    """
    merged = dict(os.environ)

    # Layer 1: auto-detected (only fills in keys not already in process env)
    auto = auto_detect_env_vars(project_dir)
    for k, v in auto.items():
        if v is not None and k not in merged:
            merged[k] = v

    # Layer 1.5: auto-detected venv — VIRTUAL_ENV and PATH prepend
    venv_info = detect_python_venv(project_dir)
    if venv_info is not None:
        if "VIRTUAL_ENV" not in merged:
            merged["VIRTUAL_ENV"] = str(venv_info.path)
        bin_dir_str = str(venv_info.bin_dir)
        current_path = merged.get("PATH", "")
        if bin_dir_str not in current_path.split(os.pathsep):
            merged["PATH"] = bin_dir_str + os.pathsep + current_path if current_path else bin_dir_str

    # Layer 3: project-level config (overrides process env)
    merged.update(_extract_env_vars_from_config(cfg))

    # Layer 4: task-level (highest priority)
    merged.update(_extract_env_vars_from_task(task))

    return merged


# ---------------------------------------------------------------------------
# API view
# ---------------------------------------------------------------------------


def resolved_env_vars_view(
    *, project_dir: Path, cfg: dict[str, Any],
) -> list[dict[str, str | None | bool]]:
    """Build a display-safe resolved view for the API.

    Returns list of ``{key, source, has_value}`` for each known env var.
    Sources: ``"auto"``, ``"process"``, ``"config"``, ``"required"``
    (detected but no value).
    Values are never included — only whether a value exists.
    """
    auto = auto_detect_env_vars(project_dir)
    config_vars = _extract_env_vars_from_config(cfg)

    # Collect all known keys
    all_keys: dict[str, tuple[str, bool]] = {}  # key -> (source, has_value)

    # Auto-detected
    for key, value in auto.items():
        if value is not None:
            all_keys[key] = ("auto", True)
        else:
            all_keys[key] = ("required", False)

    # Venv-detected VIRTUAL_ENV
    venv_info = detect_python_venv(project_dir)
    if venv_info is not None:
        if "VIRTUAL_ENV" not in all_keys:
            all_keys["VIRTUAL_ENV"] = ("venv", True)

    # Process env overrides auto
    for key in all_keys:
        if key in os.environ:
            all_keys[key] = ("process", True)

    # Config overrides all
    for key in config_vars:
        all_keys[key] = ("config", True)

    result: list[dict[str, str | None | bool]] = []
    for key in sorted(all_keys):
        source, has_value = all_keys[key]
        result.append({"key": key, "source": source, "has_value": has_value})
    return result
