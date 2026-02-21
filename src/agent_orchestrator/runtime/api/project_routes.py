"""Project-related route registration for runtime API."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional, Callable

from fastapi import APIRouter, HTTPException, Query

from ..domain.models import now_iso


def os_access(path: Path) -> bool:
    """Best-effort readability/executability check for filesystem browsing.

    Falls back to ``iterdir`` probe when POSIX mode checks are unavailable.
    """
    try:
        import os
        import stat

        st = path.stat()
        mode = st.st_mode
        uid = os.getuid()
        gid = os.getgid()
        groups = set(os.getgroups())
        groups.add(gid)
        if uid == st.st_uid:
            read = bool(mode & stat.S_IRUSR)
            exec_ = bool(mode & stat.S_IXUSR)
        elif st.st_gid in groups:
            read = bool(mode & stat.S_IRGRP)
            exec_ = bool(mode & stat.S_IXGRP)
        else:
            read = bool(mode & stat.S_IROTH)
            exec_ = bool(mode & stat.S_IXOTH)
        return read and exec_
    except Exception:
        try:
            next(path.iterdir(), None)
            return True
        except Exception:
            return False


def register_project_routes(
    router: APIRouter,
    ctx: Callable[[Optional[str]], tuple[Any, Any, Any]],
) -> None:
    """Register project browsing and pinning endpoints on the shared router."""

    @router.get("/projects")
    async def list_projects(project_dir: Optional[str] = Query(None), include_non_git: bool = Query(False)) -> dict[str, Any]:
        container, _, _ = ctx(project_dir)
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
        container, _, _ = ctx(project_dir)
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

        container, _, _ = ctx(project_dir)
        cfg = container.config.load()
        pinned = [entry for entry in list(cfg.get("pinned_projects") or []) if str(entry.get("path")) != str(path)]
        project_id = body.get("project_id") or f"pinned-{uuid.uuid4().hex[:8]}"
        pinned.append({"id": project_id, "path": str(path), "pinned_at": now_iso()})
        cfg["pinned_projects"] = pinned
        container.config.save(cfg)
        return {"project": {"id": project_id, "path": str(path)}}

    @router.delete("/projects/pinned/{project_id}")
    async def unpin_project(project_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        container, _, _ = ctx(project_dir)
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
        ctx(project_dir)
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
