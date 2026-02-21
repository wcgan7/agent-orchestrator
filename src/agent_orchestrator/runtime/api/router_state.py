"""Persistence helpers for router-scoped collaboration/import state."""

from __future__ import annotations

from typing import Any, Optional

from .helpers import _pruned_import_jobs
from ..storage.container import Container


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


def _prune_in_memory_jobs(job_store: dict[str, dict[str, Any]]) -> None:
    pruned = _pruned_import_jobs(dict(job_store))
    job_store.clear()
    job_store.update(pruned)
