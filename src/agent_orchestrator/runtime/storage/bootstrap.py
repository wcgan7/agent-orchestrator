from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .file_repos import FileConfigRepository

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


STATE_FILES = {
    "tasks": "tasks.yaml",
    "runs": "runs.yaml",
    "review_cycles": "review_cycles.yaml",
    "agents": "agents.yaml",
    "quick_actions": "quick_actions.yaml",
    "plan_revisions": "plan_revisions.yaml",
    "plan_refine_jobs": "plan_refine_jobs.yaml",
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


def _needs_archive(base: Path) -> bool:
    if not base.exists():
        return False
    if not (base / "config.yaml").exists():
        return True
    return _schema_version(base / "config.yaml") != 3


def _ensure_gitignored(project_dir: Path) -> None:
    """Add .agent_orchestrator/ and .workdoc.md to the project's .gitignore if not already present."""
    gitignore = project_dir / ".gitignore"
    entries = [".agent_orchestrator/", ".workdoc.md"]
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        existing_stripped = {line.strip() for line in content.splitlines()}
        missing = [e for e in entries if e not in existing_stripped and e.rstrip("/") not in existing_stripped]
        if not missing:
            return
        if content and not content.endswith("\n"):
            content += "\n"
        # Only add the comment header if not already present.
        if "# Agent Orchestrator runtime data" not in content:
            content += "\n# Agent Orchestrator runtime data\n"
        for e in missing:
            content += f"{e}\n"
        gitignore.write_text(content, encoding="utf-8")
    else:
        lines = "# Agent Orchestrator runtime data\n"
        for e in entries:
            lines += f"{e}\n"
        gitignore.write_text(lines, encoding="utf-8")


def ensure_state_root(project_dir: Path) -> Path:
    base = project_dir / ".agent_orchestrator"
    state_root = base

    if _needs_archive(base):
        archive_target = project_dir / f".agent_orchestrator_legacy_{_utc_stamp()}"
        base.rename(archive_target)
        base.mkdir(parents=True, exist_ok=True)

    state_root.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(project_dir)

    for file_name in STATE_FILES.values():
        target = state_root / file_name
        if file_name.endswith(".yaml") and not target.exists():
            target.write_text("version: 3\n", encoding="utf-8")
        if file_name.endswith(".jsonl") and not target.exists():
            target.touch()

    (state_root / "workdocs").mkdir(parents=True, exist_ok=True)

    config_repo = FileConfigRepository(state_root / "config.yaml", state_root / "config.lock")
    config = config_repo.load()
    config["schema_version"] = 3
    config.setdefault("pinned_projects", [])
    config.setdefault("orchestrator", {"status": "running", "concurrency": 2, "max_review_attempts": 10, "max_verify_fix_attempts": 3})
    config.setdefault("defaults", {"approval_mode": "human_review", "quality_gate": {"critical": 0, "high": 0, "medium": 0, "low": 0}, "dependency_policy": "prudent"})
    config.setdefault("project", {"commands": {}})
    config_repo.save(config)

    return state_root
