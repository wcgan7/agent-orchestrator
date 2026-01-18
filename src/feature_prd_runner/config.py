"""Load optional runner configuration from `.prd_runner/config.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import CONFIG_FILE, STATE_DIR_NAME
from .io_utils import _load_data_with_error


def load_runner_config(project_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Load the optional runner config file.

    Args:
        project_dir: Repository root directory.

    Returns:
        A tuple of `(config, error_message)`. If the file is missing, returns `({}, None)`.
    """
    project_dir = project_dir.resolve()
    path = project_dir / STATE_DIR_NAME / CONFIG_FILE
    data, err = _load_data_with_error(path, {})
    if not path.exists():
        return {}, None
    if err:
        return {}, err
    return data, None


def _get_nested(config: dict[str, Any], *keys: str) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def get_verify_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the verify configuration block from the runner config.

    Args:
        config: Runner configuration dictionary.

    Returns:
        The `verify` config mapping, or an empty dict if not present.
    """
    raw = _get_nested(config, "verify")
    return raw if isinstance(raw, dict) else {}


def get_workers_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the workers configuration block from the runner config.

    Args:
        config: Runner configuration dictionary.

    Returns:
        The `workers` config mapping, or an empty dict if not present.
    """
    raw = _get_nested(config, "workers")
    return raw if isinstance(raw, dict) else {}
