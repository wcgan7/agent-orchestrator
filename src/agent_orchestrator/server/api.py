"""FastAPI app wiring for orchestrator-first runtime."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional, cast

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ..runtime.api import create_router
from ..runtime.events import EventBus
from ..runtime.events import hub
from ..runtime.orchestrator import OrchestratorService, WorkerAdapter, create_orchestrator
from ..runtime.storage import Container


def create_app(
    project_dir: Optional[Path] = None,
    enable_cors: bool = True,
    worker_adapter: Optional[WorkerAdapter] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        project_dir (Optional[Path]): Default project directory used when request-level
            ``project_dir`` query parameters are not provided.
        enable_cors (bool): Whether to install permissive CORS middleware for browser
            clients.
        worker_adapter (Optional[WorkerAdapter]): Optional worker adapter forwarded to
            newly created orchestrator instances.

    Returns:
        FastAPI: Configured application instance with router endpoints, websocket
        bridge, and per-project container/orchestrator caches stored on
        ``app.state``.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.attach_loop(asyncio.get_running_loop())
        try:
            yield
        finally:
            orchestrators = list(getattr(app.state, "orchestrators", {}).values())
            for orchestrator in orchestrators:
                try:
                    orchestrator.shutdown(timeout=10.0)
                except Exception:
                    pass
            app.state.orchestrators = {}
            app.state.containers = {}
            app.state.import_jobs = {}

    app = FastAPI(
        title="Agent Orchestrator",
        description="Orchestrator-first AI engineering control center",
        version="3.0.0",
        lifespan=_lifespan,
    )

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.default_project_dir = project_dir
    app.state.containers = {}
    app.state.orchestrators = {}
    app.state.import_jobs = {}

    def _resolve_project_dir(project_dir_param: Optional[str] = None) -> Path:
        if project_dir_param:
            return Path(project_dir_param).expanduser().resolve()
        if app.state.default_project_dir:
            return Path(app.state.default_project_dir).resolve()
        return Path.cwd().resolve()

    def _resolve_container(project_dir_param: Optional[str] = None) -> Container:
        resolved = _resolve_project_dir(project_dir_param)
        key = str(resolved)
        cache = cast(dict[str, Container], app.state.containers)
        if key not in cache:
            cache[key] = Container(resolved)
        return cache[key]

    def _resolve_orchestrator(project_dir_param: Optional[str] = None) -> OrchestratorService:
        resolved = _resolve_project_dir(project_dir_param)
        key = str(resolved)
        cache = cast(dict[str, OrchestratorService], app.state.orchestrators)
        if key not in cache:
            container = _resolve_container(project_dir_param)
            bus_factory = cast(Any, app.state.bus_factory)
            cache[key] = create_orchestrator(container, bus=bus_factory(container), worker_adapter=worker_adapter)
        return cache[key]

    app.state.bus_factory = lambda container: EventBus(container.events, container.project_id)

    app.include_router(create_router(_resolve_container, _resolve_orchestrator, app.state.import_jobs))

    @app.get("/")
    async def root(project_dir: Optional[str] = Query(None)) -> dict[str, object]:
        """Return basic service metadata for the selected project context.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing service metadata and project identity fields.
        """
        container = _resolve_container(project_dir)
        return {
            "name": "Agent Orchestrator",
            "version": "3.0.0",
            "project": str(container.project_dir),
            "project_id": container.project_id,
            "schema_version": 3,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        """Expose liveness status for process-level health checks.
        
        Returns:
            A payload indicating the API process is running.
        """
        return {"status": "ok", "version": "3.0.0"}

    @app.get("/readyz")
    async def readyz(project_dir: Optional[str] = Query(None)) -> dict[str, object]:
        """Expose readiness status for the selected project context.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating readiness and current orchestrator cache size.
        """
        container = _resolve_container(project_dir)
        return {
            "status": "ready",
            "project": str(container.project_dir),
            "project_id": container.project_id,
            "orchestrators": len(app.state.orchestrators),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Bridge websocket clients to the shared event hub handler.
        
        Args:
            websocket: Active websocket connection accepted by FastAPI.
        
        Returns:
            ``None`` after the connection lifecycle ends.
        """
        await hub.handle_connection(websocket)

    return app


app = create_app()
