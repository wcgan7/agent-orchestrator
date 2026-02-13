from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .file_repos import FileConfigRepository

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


V3_FILES = {
    "tasks": "tasks.yaml",
    "runs": "runs.yaml",
    "review_cycles": "review_cycles.yaml",
    "agents": "agents.yaml",
    "quick_actions": "quick_actions.yaml",
    "events": "events.jsonl",
    "config": "config.yaml",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _schema_version(path: Path) -> int | None:
    if yaml is None or not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    value = raw.get("schema_version")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _needs_archive(base: Path, v3_root: Path) -> bool:
    if not base.exists():
        return False
    if not v3_root.exists():
        # Contract: if .prd_runner exists and is not v3, archive before fresh init.
        return True
    return _schema_version(v3_root / "config.yaml") != 3


def ensure_v3_state_root(project_dir: Path) -> Path:
    base = project_dir / ".prd_runner"
    v3_root = base / "v3"

    if _needs_archive(base, v3_root):
        archive_target = project_dir / f".prd_runner_legacy_{_utc_stamp()}"
        base.rename(archive_target)
        base.mkdir(parents=True, exist_ok=True)

    v3_root.mkdir(parents=True, exist_ok=True)

    for file_name in V3_FILES.values():
        target = v3_root / file_name
        if file_name.endswith(".yaml") and not target.exists():
            target.write_text("version: 3\n", encoding="utf-8")
        if file_name.endswith(".jsonl") and not target.exists():
            target.touch()

    config_repo = FileConfigRepository(v3_root / "config.yaml", v3_root / "config.lock")
    config = config_repo.load()
    config["schema_version"] = 3
    config.setdefault("pinned_projects", [])
    config.setdefault("orchestrator", {"status": "running", "concurrency": 2, "max_review_attempts": 3})
    config.setdefault("defaults", {"approval_mode": "human_review", "quality_gate": {"critical": 0, "high": 0, "medium": 0, "low": 0}})
    config.setdefault("project", {"commands": {}})
    config_repo.save(config)

    return v3_root
