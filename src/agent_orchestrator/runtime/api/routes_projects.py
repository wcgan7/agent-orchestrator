"""Project route registration for the runtime API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query

from ..domain.models import now_iso
from .deps import RouteDeps
from . import router_impl as impl


os_access = impl.os_access


def register_project_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register project and workspace browsing routes."""
    @router.get("/projects")
    async def list_projects(project_dir: Optional[str] = Query(None), include_non_git: bool = Query(False)) -> dict[str, Any]:
        """List discoverable and pinned projects for the current workspace.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
            include_non_git: Whether to include directories without a `.git` repository.
        
        Returns:
            A payload containing deduplicated project entries.
        """
        container, _, _ = deps.ctx(project_dir)
        cfg = container.config.load()
        pinned = list(cfg.get("pinned_projects") or [])
        discovered = []
        cwd = container.project_dir
        if (cwd / ".git").exists() or include_non_git:
            discovered.append({"id": cwd.name, "path": str(cwd), "source": "discovered", "is_git": (cwd / ".git").exists()})
        for item in pinned:
            p = Path(str(item.get("path") or "")).resolve()
            discovered.append({"id": item.get("id") or p.name, "path": str(p), "source": "pinned", "is_git": (p / ".git").exists()})
        dedup: dict[str, dict[str, Any]] = {str(entry.get("path") or ""): entry for entry in discovered}
        return {"projects": list(dedup.values())}

    @router.get("/projects/pinned")
    async def list_pinned_projects(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return projects pinned in persisted runtime configuration.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with the list of pinned project records.
        """
        container, _, _ = deps.ctx(project_dir)
        cfg = container.config.load()
        return {"items": list(cfg.get("pinned_projects") or [])}

    @router.post("/projects/pinned")
    async def pin_project(body: dict[str, Any], project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Persist a project path in the pinned project list.
        
        Args:
            body: Request payload containing the path and optional pin metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload describing the pinned project record.
        
        Raises:
            HTTPException: If the requested path is invalid or fails pinning policy checks.
        """
        path = Path(str(body.get("path") or "")).expanduser().resolve()
        allow_non_git = bool(body.get("allow_non_git", False))
        if not path.exists() or not path.is_dir() or not os_access(path):
            raise HTTPException(status_code=400, detail="Invalid project path")
        if not allow_non_git and not (path / ".git").exists():
            raise HTTPException(status_code=400, detail="Project path must contain .git unless allow_non_git=true")

        container, _, _ = deps.ctx(project_dir)
        cfg = container.config.load()
        pinned = [entry for entry in list(cfg.get("pinned_projects") or []) if str(entry.get("path")) != str(path)]
        project_id = body.get("project_id") or f"pinned-{uuid.uuid4().hex[:8]}"
        pinned.append({"id": project_id, "path": str(path), "pinned_at": now_iso()})
        cfg["pinned_projects"] = pinned
        container.config.save(cfg)
        return {"project": {"id": project_id, "path": str(path)}}

    @router.delete("/projects/pinned/{project_id}")
    async def unpin_project(project_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Remove a pinned project entry by its identifier.
        
        Args:
            project_id: Identifier of the pinned project to remove.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating whether a matching project was removed.
        """
        container, _, _ = deps.ctx(project_dir)
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
        """Browse filesystem directories for project selection.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
            path: Optional directory path to browse; defaults to the user's home directory.
            include_hidden: Whether to include hidden directories.
            limit: Maximum number of directories returned.
        
        Returns:
            A payload with the current path, parent path, and child directory entries.
        
        Raises:
            HTTPException: If the browse path is invalid or unreadable.
        """
        deps.ctx(project_dir)
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

