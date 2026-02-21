"""PRD import route registration for the runtime API."""

from __future__ import annotations

from typing import Any, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query

from ..domain.models import Task, now_iso
from ..storage.container import Container
from .deps import RouteDeps
from . import router_impl as impl


PrdCommitRequest = impl.PrdCommitRequest
PrdPreviewRequest = impl.PrdPreviewRequest
_apply_generated_dep_links = impl._apply_generated_dep_links
_generated_tasks_from_parsed_prd = impl._generated_tasks_from_parsed_prd
_ingest_prd = impl._ingest_prd


def register_import_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register PRD import preview, commit, and lookup routes."""
    @router.post("/import/prd/preview")
    async def preview_import(body: PrdPreviewRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Ingest PRD content and return a preview graph of candidate tasks.
        
        Args:
            body: PRD preview request with raw content and default priority.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the temporary import job id and preview graph.
        """
        container, _, _ = deps.ctx(project_dir)
        deps.prune_in_memory_jobs()
        ingestion = _ingest_prd(body.content, body.default_priority)
        parsed_prd = ingestion["parsed_prd"]
        original_prd = ingestion["original_prd"]
        items = list(parsed_prd.get("task_candidates") or [])
        nodes = [
            {
                "id": f"n{idx + 1}",
                "title": str(item.get("title") or "Imported task"),
                "priority": str(item.get("priority") or body.default_priority),
            }
            for idx, item in enumerate(items)
        ]
        edges = [{"from": nodes[idx]["id"], "to": nodes[idx + 1]["id"]} for idx in range(len(nodes) - 1)]
        job_id = f"imp-{uuid.uuid4().hex[:10]}"
        job = {
            "id": job_id,
            "project_id": container.project_id,
            "title": body.title or "Imported PRD",
            "status": "preview_ready",
            "created_at": now_iso(),
            "tasks": items,
            "original_prd": original_prd,
            "parsed_prd": parsed_prd,
        }
        deps.job_store[job_id] = job
        deps.upsert_import_job(container, job)
        return {
            "job_id": job_id,
            "preview": {
                "nodes": nodes,
                "edges": edges,
                "chunk_count": int(parsed_prd.get("chunk_count") or 0),
                "strategy": str(parsed_prd.get("strategy") or "unknown"),
                "ambiguity_warnings": list(parsed_prd.get("ambiguity_warnings") or []),
            },
        }

    @router.post("/import/prd/commit")
    async def commit_import(body: PrdCommitRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Commit a previewed PRD import and run initiative generation.
        
        Args:
            body: PRD commit request identifying the preview job to finalize.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with created task identifiers and the parent initiative task id.
        
        Raises:
            HTTPException: If the import job is missing, invalid, or execution fails.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        deps.prune_in_memory_jobs()
        job = deps.job_store.get(body.job_id)
        if not job:
            job = deps.fetch_import_job(container, body.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        if not isinstance(job.get("parsed_prd"), dict):
            raise HTTPException(status_code=400, detail="Import job is missing parsed PRD artifacts.")
        parsed_prd = dict(job.get("parsed_prd") or {})
        generated_tasks = _generated_tasks_from_parsed_prd(parsed_prd)
        if not generated_tasks:
            raise HTTPException(status_code=400, detail="Parsed PRD produced no task candidates to generate.")

        # Execute the existing initiative pipeline end-to-end using the ingested PRD artifacts.
        prd_parent = Task(
            title=str(job.get("title") or "Imported PRD"),
            description=str((job.get("original_prd") or {}).get("normalized_content") or ""),
            task_type="initiative_plan",
            priority="P2",
            source="prd_import",
            metadata={
                "prd_import_job_id": body.job_id,
                "original_prd": job.get("original_prd"),
                "parsed_prd": parsed_prd,
            },
        )
        prd_parent.status = "queued"
        container.tasks.upsert(prd_parent)
        bus.emit(
            channel="tasks",
            event_type="task.created",
            entity_id=prd_parent.id,
            payload={"source": "prd_import", "import_job_id": body.job_id, "task_type": "initiative_plan"},
        )

        try:
            orchestrator.run_task(prd_parent.id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        refreshed_parent = container.tasks.get(prd_parent.id)
        created = list(refreshed_parent.children_ids if refreshed_parent else [])
        _apply_generated_dep_links(container, created)
        for child_id in created:
            child = container.tasks.get(child_id)
            if not child:
                continue
            child.source = "prd_import"
            container.tasks.upsert(child)

        job["status"] = "committed"
        job["created_task_ids"] = created
        job["parent_task_id"] = prd_parent.id
        deps.job_store[body.job_id] = job
        deps.upsert_import_job(container, job)
        bus.emit(channel="tasks", event_type="import.committed", entity_id=body.job_id, payload={"created_task_ids": created})
        return {"job_id": body.job_id, "created_task_ids": created, "parent_task_id": prd_parent.id}

    @router.get("/import/{job_id}")
    async def get_import_job(job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the current state of a PRD import job.
        
        Args:
            job_id: Identifier of the import job.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the import job record.
        
        Raises:
            HTTPException: If the job cannot be found.
        """
        deps.prune_in_memory_jobs()
        job = deps.job_store.get(job_id)
        if not job:
            container: Container = deps.resolve_container(project_dir)
            job = deps.fetch_import_job(container, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        return {"job": job}
