"""FastAPI web server for Feature PRD Runner dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..io_utils import _load_data
from ..phase_utils import _normalize_phases
from ..tasks import _normalize_tasks
from .models import (
    ControlAction,
    ControlResponse,
    PhaseInfo,
    ProjectInfo,
    ProjectStatus,
    RunDetail,
    RunInfo,
    RunMetrics,
    TaskInfo,
)
from .websocket import manager, watch_logs, watch_run_progress


def create_app(
    project_dir: Optional[Path] = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        project_dir: Default project directory.
        enable_cors: Whether to enable CORS.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Feature PRD Runner Dashboard",
        description="Web dashboard for monitoring and controlling Feature PRD Runner",
        version="1.0.0",
    )

    # Enable CORS for development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Store default project directory
    app.state.default_project_dir = project_dir

    def _get_project_dir(project_dir_param: Optional[str] = None) -> Path:
        """Get project directory from parameter or default."""
        if project_dir_param:
            return Path(project_dir_param)
        if app.state.default_project_dir:
            return app.state.default_project_dir
        return Path.cwd()

    def _get_paths(project_dir: Path) -> dict[str, Path]:
        """Get standard paths for a project."""
        state_dir = project_dir / ".prd_runner"
        return {
            "state_dir": state_dir,
            "run_state": state_dir / "run_state.yaml",
            "task_queue": state_dir / "task_queue.yaml",
            "phase_plan": state_dir / "phase_plan.yaml",
            "events": state_dir / "artifacts" / "events.jsonl",
            "runs": state_dir / "runs",
            "artifacts": state_dir / "artifacts",
        }

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Feature PRD Runner Dashboard",
            "version": "1.0.0",
            "status": "running",
        }

    @app.get("/api/projects")
    async def list_projects(
        search_path: Optional[str] = Query(None, description="Path to search for projects"),
    ) -> list[ProjectInfo]:
        """List available projects with .prd_runner state.

        Args:
            search_path: Optional path to search for projects (defaults to common locations).

        Returns:
            List of discovered projects.
        """
        projects = []

        # Default search paths
        search_paths = []
        if search_path:
            search_paths.append(Path(search_path))
        else:
            # Search in common locations
            home = Path.home()
            search_paths.extend([
                Path.cwd(),  # Current directory
                home / "Documents",
                home / "Projects",
                home,
            ])

        # Scan for projects
        for base_path in search_paths:
            if not base_path.exists():
                continue

            try:
                # Check if base path itself is a project
                if (base_path / ".prd_runner").exists():
                    project_info = _get_project_info(base_path)
                    if project_info and project_info not in projects:
                        projects.append(project_info)

                # Search subdirectories (1 level deep)
                for subdir in base_path.iterdir():
                    if subdir.is_dir() and (subdir / ".prd_runner").exists():
                        project_info = _get_project_info(subdir)
                        if project_info and project_info not in projects:
                            projects.append(project_info)

            except Exception as e:
                logger.warning(f"Error scanning {base_path}: {e}")
                continue

        # Remove duplicates based on path
        seen_paths = set()
        unique_projects = []
        for proj in projects:
            if proj.path not in seen_paths:
                seen_paths.add(proj.path)
                unique_projects.append(proj)

        return unique_projects

    def _get_project_info(project_path: Path) -> ProjectInfo | None:
        """Get project information from a directory.

        Args:
            project_path: Path to project directory.

        Returns:
            ProjectInfo or None if project is invalid.
        """
        try:
            state_dir = project_path / ".prd_runner"
            if not state_dir.exists():
                return None

            # Load run state to get status
            run_state_path = state_dir / "run_state.yaml"
            run_state = _load_data(run_state_path, {})

            # Load phase plan to get counts
            phase_plan_path = state_dir / "phase_plan.yaml"
            phase_plan = _load_data(phase_plan_path, {})
            phases = phase_plan.get("phases", [])

            # Count completed phases
            task_queue_path = state_dir / "task_queue.yaml"
            task_queue = _load_data(task_queue_path, {})
            tasks = task_queue.get("tasks", [])

            completed_phases = set()
            for task in tasks:
                if task.get("lifecycle") == "done" and task.get("phase_id"):
                    completed_phases.add(task["phase_id"])

            # Get last run
            runs_dir = state_dir / "runs"
            last_run = None
            if runs_dir.exists():
                run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                if run_dirs:
                    latest = max(run_dirs, key=lambda d: d.name)
                    last_run = latest.name

            # Determine status
            status = run_state.get("status", "idle")
            if status == "running":
                status = "active"

            return ProjectInfo(
                name=project_path.name,
                path=str(project_path.resolve()),
                status=status,
                last_run=last_run,
                phases_total=len(phases),
                phases_completed=len(completed_phases),
            )

        except Exception as e:
            logger.warning(f"Error getting project info for {project_path}: {e}")
            return None

    @app.get("/api/status")
    async def get_status(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ProjectStatus:
        """Get current project status.

        Args:
            project_dir: Project directory path.

        Returns:
            Project status overview.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        # Check if state directory exists
        if not paths["state_dir"].exists():
            raise HTTPException(
                status_code=404,
                detail=f"No .prd_runner state found in {proj_dir}",
            )

        # Load state
        run_state = _load_data(paths["run_state"], {})
        queue = _load_data(paths["task_queue"], {})
        plan = _load_data(paths["phase_plan"], {})

        tasks = _normalize_tasks(queue)
        phases = _normalize_phases(plan)

        # Count task statuses
        tasks_ready = sum(1 for t in tasks if t.get("lifecycle") == "ready")
        tasks_running = sum(1 for t in tasks if t.get("lifecycle") == "running")
        tasks_done = sum(1 for t in tasks if t.get("lifecycle") == "done")
        tasks_blocked = sum(1 for t in tasks if t.get("lifecycle") == "waiting_human")

        # Count completed phases
        phases_completed = sum(1 for p in phases if p.get("status") == "done")

        return ProjectStatus(
            project_dir=str(proj_dir),
            status=run_state.get("status", "idle"),
            current_task_id=run_state.get("current_task_id"),
            current_phase_id=run_state.get("current_phase_id"),
            run_id=run_state.get("run_id"),
            last_error=run_state.get("last_error"),
            phases_completed=phases_completed,
            phases_total=len(phases),
            tasks_ready=tasks_ready,
            tasks_running=tasks_running,
            tasks_done=tasks_done,
            tasks_blocked=tasks_blocked,
        )

    @app.get("/api/phases")
    async def get_phases(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> list[PhaseInfo]:
        """Get all phases with their status.

        Args:
            project_dir: Project directory path.

        Returns:
            List of phases.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        if not paths["phase_plan"].exists():
            return []

        plan = _load_data(paths["phase_plan"], {})
        phases = _normalize_phases(plan)

        return [
            PhaseInfo(
                id=p.get("id", ""),
                name=p.get("name", ""),
                description=p.get("description", ""),
                status=p.get("status", "pending"),
                deps=p.get("deps", []),
                progress=_calculate_phase_progress(p),
            )
            for p in phases
        ]

    @app.get("/api/tasks")
    async def get_tasks(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        phase_id: Optional[str] = Query(None, description="Filter by phase ID"),
    ) -> list[TaskInfo]:
        """Get all tasks, optionally filtered by phase.

        Args:
            project_dir: Project directory path.
            phase_id: Optional phase ID to filter by.

        Returns:
            List of tasks.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        if not paths["task_queue"].exists():
            return []

        queue = _load_data(paths["task_queue"], {})
        tasks = _normalize_tasks(queue)

        # Filter by phase if specified
        if phase_id:
            tasks = [t for t in tasks if t.get("phase_id") == phase_id]

        return [
            TaskInfo(
                id=t.get("id", ""),
                type=t.get("type", ""),
                phase_id=t.get("phase_id"),
                step=t.get("step", ""),
                lifecycle=t.get("lifecycle", ""),
                status=t.get("status", ""),
                branch=t.get("branch"),
                last_error=t.get("last_error"),
                last_run_id=t.get("last_run_id"),
                worker_attempts=t.get("worker_attempts", 0),
            )
            for t in tasks
        ]

    @app.get("/api/runs")
    async def list_runs(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        limit: int = Query(50, ge=1, le=500, description="Maximum number of runs to return"),
    ) -> list[RunInfo]:
        """List recent runs.

        Args:
            project_dir: Project directory path.
            limit: Maximum number of runs to return.

        Returns:
            List of recent runs.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        if not paths["runs"].exists():
            return []

        # List all run directories
        run_dirs = sorted(
            [d for d in paths["runs"].iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:limit]

        runs = []
        for run_dir in run_dirs:
            progress_file = run_dir / "progress.json"
            if not progress_file.exists():
                continue

            progress = _load_data(progress_file, {})
            runs.append(
                RunInfo(
                    run_id=progress.get("run_id", run_dir.name),
                    task_id=progress.get("task_id", ""),
                    phase=progress.get("phase", ""),
                    step=progress.get("step", ""),
                    status=progress.get("status", "unknown"),
                    started_at=progress.get("started_at", ""),
                    updated_at=progress.get("updated_at", ""),
                )
            )

        return runs

    @app.get("/api/runs/{run_id}")
    async def get_run_details(
        run_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> RunDetail:
        """Get detailed information about a specific run.

        Args:
            run_id: Run ID.
            project_dir: Project directory path.

        Returns:
            Detailed run information.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        run_dir = paths["runs"] / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        progress_file = run_dir / "progress.json"
        if not progress_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Progress file not found for run {run_id}",
            )

        progress = _load_data(progress_file, {})

        # Load current state
        queue = _load_data(paths["task_queue"], {})
        plan = _load_data(paths["phase_plan"], {})
        run_state = _load_data(paths["run_state"], {})

        tasks = _normalize_tasks(queue)
        phases = _normalize_phases(plan)

        return RunDetail(
            run_id=run_id,
            task_id=progress.get("task_id", ""),
            phase=progress.get("phase", ""),
            step=progress.get("step", ""),
            status=run_state.get("status", "unknown"),
            started_at=progress.get("started_at", ""),
            updated_at=progress.get("updated_at", ""),
            current_task_id=run_state.get("current_task_id"),
            current_phase_id=run_state.get("current_phase_id"),
            last_error=run_state.get("last_error"),
            phases=[
                PhaseInfo(
                    id=p.get("id", ""),
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    status=p.get("status", "pending"),
                    deps=p.get("deps", []),
                    progress=_calculate_phase_progress(p),
                )
                for p in phases
            ],
            tasks=[
                TaskInfo(
                    id=t.get("id", ""),
                    type=t.get("type", ""),
                    phase_id=t.get("phase_id"),
                    step=t.get("step", ""),
                    lifecycle=t.get("lifecycle", ""),
                    status=t.get("status", ""),
                    branch=t.get("branch"),
                    last_error=t.get("last_error"),
                    last_run_id=t.get("last_run_id"),
                    worker_attempts=t.get("worker_attempts", 0),
                )
                for t in tasks
            ],
        )

    @app.get("/api/logs/{run_id}")
    async def get_run_logs(
        run_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        lines: int = Query(100, ge=1, le=10000, description="Number of lines to return"),
    ) -> dict[str, Any]:
        """Get logs for a specific run.

        Args:
            run_id: Run ID.
            project_dir: Project directory path.
            lines: Number of log lines to return.

        Returns:
            Log content.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        run_dir = paths["runs"] / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Try to find log files
        log_files = list(run_dir.glob("*.log")) + list(run_dir.glob("*.txt"))

        if not log_files:
            return {"run_id": run_id, "logs": []}

        # Read the most recent log file
        log_file = max(log_files, key=lambda f: f.stat().st_mtime)

        try:
            log_lines = log_file.read_text().splitlines()
            # Return last N lines
            log_lines = log_lines[-lines:]
        except Exception as e:
            logger.error("Failed to read log file {}: {}", log_file, e)
            log_lines = []

        return {
            "run_id": run_id,
            "log_file": str(log_file.name),
            "logs": log_lines,
        }

    @app.post("/api/control")
    async def control_run(
        action: ControlAction,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Control run execution (pause, resume, skip, retry, stop).

        Args:
            action: Control action to perform.
            project_dir: Project directory path.

        Returns:
            Control response with success status and message.
        """
        from .control import ControlError, create_controller

        proj_dir = _get_project_dir(project_dir)

        logger.info(
            "Control action received: {} for task={} (project={})",
            action.action,
            action.task_id,
            proj_dir,
        )

        try:
            controller = create_controller(proj_dir)

            if action.action == "retry":
                if not action.task_id:
                    return ControlResponse(
                        success=False,
                        message="task_id required for retry action",
                        data={},
                    )
                step = action.params.get("step", "plan_impl") if action.params else "plan_impl"
                result = controller.retry_task(action.task_id, step)

            elif action.action == "skip":
                if not action.task_id:
                    return ControlResponse(
                        success=False,
                        message="task_id required for skip action",
                        data={},
                    )
                step = action.params.get("step") if action.params else None
                result = controller.skip_step(action.task_id, step)

            elif action.action == "resume":
                if not action.task_id:
                    return ControlResponse(
                        success=False,
                        message="task_id required for resume action",
                        data={},
                    )
                step = action.params.get("step") if action.params else None
                result = controller.resume_task(action.task_id, step)

            elif action.action == "stop":
                result = controller.stop_run()

            else:
                return ControlResponse(
                    success=False,
                    message=f"Unknown action: {action.action}",
                    data={"action": action.action},
                )

            return ControlResponse(
                success=result["success"],
                message=result["message"],
                data=result,
            )

        except ControlError as e:
            logger.error("Control action failed: {}", e)
            return ControlResponse(
                success=False,
                message=str(e),
                data={"action": action.action, "task_id": action.task_id},
            )
        except Exception as e:
            logger.error("Unexpected error in control action: {}", e)
            return ControlResponse(
                success=False,
                message=f"Internal error: {e}",
                data={"action": action.action, "task_id": action.task_id},
            )

    @app.get("/api/metrics")
    async def get_metrics(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> RunMetrics:
        """Get aggregated metrics for the project.

        Args:
            project_dir: Project directory path.

        Returns:
            Aggregated metrics with real calculations from events, git, and state files.
        """
        from .metrics import calculate_metrics

        proj_dir = _get_project_dir(project_dir)

        # Calculate real metrics from available data
        metrics = calculate_metrics(proj_dir)

        return RunMetrics(
            tokens_used=metrics.tokens_used,
            api_calls=metrics.api_calls,
            estimated_cost_usd=metrics.estimated_cost_usd,
            wall_time_seconds=metrics.wall_time_seconds,
            phases_completed=metrics.phases_completed,
            phases_total=metrics.phases_total,
            files_changed=metrics.files_changed,
            lines_added=metrics.lines_added,
            lines_removed=metrics.lines_removed,
        )

    @app.websocket("/ws/runs/{run_id}")
    async def websocket_run_updates(
        websocket: WebSocket,
        run_id: str,
        project_dir: Optional[str] = Query(None),
    ):
        """WebSocket endpoint for real-time run updates.

        Args:
            websocket: WebSocket connection.
            run_id: Run ID to subscribe to.
            project_dir: Project directory path.
        """
        proj_dir = _get_project_dir(project_dir)

        await manager.connect(websocket, run_id)
        try:
            await watch_run_progress(websocket, run_id, proj_dir)
        except WebSocketDisconnect:
            manager.disconnect(websocket, run_id)
        except Exception as e:
            logger.error("WebSocket error for run {}: {}", run_id, e)
            manager.disconnect(websocket, run_id)

    @app.websocket("/ws/logs/{run_id}")
    async def websocket_log_stream(
        websocket: WebSocket,
        run_id: str,
        project_dir: Optional[str] = Query(None),
    ):
        """WebSocket endpoint for streaming logs.

        Args:
            websocket: WebSocket connection.
            run_id: Run ID.
            project_dir: Project directory path.
        """
        proj_dir = _get_project_dir(project_dir)

        await websocket.accept()
        try:
            await watch_logs(websocket, run_id, proj_dir)
        except WebSocketDisconnect:
            logger.info("Client disconnected from log stream for run {}", run_id)
        except Exception as e:
            logger.error("WebSocket error for logs {}: {}", run_id, e)

    def _calculate_phase_progress(phase: dict[str, Any]) -> float:
        """Calculate progress for a phase (0.0 to 1.0)."""
        status = phase.get("status", "pending")
        if status == "done":
            return 1.0
        elif status == "running":
            return 0.5
        else:
            return 0.0

    return app


# Create default app instance
app = create_app()
