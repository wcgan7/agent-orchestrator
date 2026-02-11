"""FastAPI web server for Feature PRD Runner dashboard."""

from __future__ import annotations

import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..io_utils import _load_data
from ..tasks import _normalize_phases
from ..tasks import _normalize_tasks
from .models import (
    ApprovalAction,
    ApprovalGateInfo,
    AuthStatus,
    BreakpointCreateRequest,
    BreakpointInfo,
    ChatMessage,
    ControlAction,
    ControlResponse,
    CorrectionRequest,
    DoctorResponse,
    DryRunResponse,
    ExecTaskRequest,
    ExecTaskResponse,
    ExplainResponse,
    FileChange,
    FileReviewRequest,
    InspectResponse,
    LoginRequest,
    LoginResponse,
    PhaseInfo,
    ProjectInfo,
    ProjectStatus,
    PromoteQuickRunRequest,
    PromoteQuickRunResponse,
    QuickRunCreateRequest,
    QuickRunEventRecord,
    QuickRunExecuteResponse,
    QuickRunEventsResponse,
    QuickRunRecord,
    RequirementRequest,
    RunDetail,
    RunInfo,
    RunMetrics,
    SendMessageRequest,
    StartRunRequest,
    StartRunResponse,
    TaskInfo,
    TaskLogsResponse,
    WorkerInfo,
    WorkerTestResponse,
    WorkersListResponse,
)
from .websocket import manager, watch_logs, watch_run_progress


def _find_codex_command() -> str | None:
    """Auto-detect codex command on the system.

    Returns:
        The codex command string if found, None otherwise.
    """
    is_windows = platform.system() == "Windows"

    # Try to find codex in PATH first
    codex_path = shutil.which("codex")
    if codex_path:
        logger.info("Found codex in PATH: {}", codex_path)
        return f"{codex_path} exec -"

    # On Windows, try common installation locations
    if is_windows:
        common_paths = [
            Path(r"C:\Program Files\nodejs\codex.cmd"),
            Path(r"C:\Program Files (x86)\nodejs\codex.cmd"),
            Path.home() / "AppData" / "Roaming" / "npm" / "codex.cmd",
            Path.home() / "AppData" / "Local" / "Programs" / "Claude" / "codex.cmd",
            Path(r"C:\Program Files\Claude\codex.cmd"),
        ]

        for path in common_paths:
            if path.exists():
                logger.info("Found codex at: {}", path)
                return f'"{path}" exec -'

    # Try alternative names (claude, claude-code, etc.)
    alternative_names = ["claude", "claude-code"]
    for name in alternative_names:
        alt_path = shutil.which(name)
        if alt_path:
            logger.info("Found {} in PATH: {}", name, alt_path)
            return f"{alt_path} exec -"

    return None


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

    def _quick_runs_path(project_dir: Path) -> Path:
        return _get_paths(project_dir)["state_dir"] / "quick_runs.json"

    def _quick_run_events_path(project_dir: Path) -> Path:
        return _get_paths(project_dir)["state_dir"] / "artifacts" / "quick_run_events.jsonl"

    def _append_quick_run_event(
        project_dir: Path,
        *,
        event_type: str,
        quick_run_id: str,
        status: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        path = _quick_run_events_path(project_dir)
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "quick_run_id": quick_run_id,
            "status": status,
            "details": details or {},
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            logger.exception("Failed to append quick-run event {} for {}", event_type, quick_run_id)

    def _load_quick_run_events(project_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
        path = _quick_run_events_path(project_dir)
        if limit < 1 or not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            selected = lines[-limit:]
            events: list[dict[str, Any]] = []
            for line in selected:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
            return events
        except Exception:
            logger.warning("Failed to read quick-run events from {}", path)
            return []

    def _load_quick_runs(project_dir: Path) -> list[dict[str, Any]]:
        path = _quick_runs_path(project_dir)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except Exception:
            logger.warning("Failed to read quick runs from {}", path)
            return []

    def _save_quick_runs(project_dir: Path, records: list[dict[str, Any]]) -> None:
        path = _quick_runs_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))

    def _find_quick_run(project_dir: Path, quick_run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        records = _load_quick_runs(project_dir)
        target = next((r for r in records if r.get("id") == quick_run_id), None)
        return records, target

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Feature PRD Runner Dashboard",
            "version": "1.0.0",
            "status": "running",
        }

    @app.post("/api/auth/login")
    async def login(request: LoginRequest) -> LoginResponse:
        """Login endpoint.

        Args:
            request: Login credentials.

        Returns:
            Access token and user info.
        """
        from datetime import timedelta

        from .auth import auth_config, create_access_token, verify_credentials

        if not verify_credentials(request.username, request.password):
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        access_token_expires = timedelta(minutes=auth_config.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": request.username}, expires_delta=access_token_expires
        )

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            username=request.username,
        )

    @app.get("/api/auth/status")
    async def auth_status() -> AuthStatus:
        """Get authentication status.

        Returns:
            Auth status including whether auth is enabled.
        """
        from .auth import auth_config

        return AuthStatus(
            enabled=auth_config.enabled,
            authenticated=not auth_config.enabled,  # Always authenticated if auth disabled
            username=auth_config.username if not auth_config.enabled else None,
        )

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

    @app.get("/api/breakpoints")
    async def list_breakpoints(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> list[BreakpointInfo]:
        """List configured breakpoints for a project."""
        from ..breakpoints import BreakpointManager

        proj_dir = _get_project_dir(project_dir)
        state_dir = proj_dir / ".prd_runner"
        if not state_dir.exists():
            return []

        manager = BreakpointManager(state_dir)
        return [BreakpointInfo(**bp.to_dict()) for bp in manager.list_breakpoints()]

    @app.post("/api/breakpoints")
    async def create_breakpoint(
        request: BreakpointCreateRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Create a breakpoint."""
        from ..breakpoints import BreakpointManager

        proj_dir = _get_project_dir(project_dir)
        state_dir = proj_dir / ".prd_runner"
        if not state_dir.exists():
            return ControlResponse(
                success=False,
                message="No .prd_runner state directory found",
                data={},
            )

        manager = BreakpointManager(state_dir)
        bp = manager.add_breakpoint(
            trigger=request.trigger,
            target=request.target,
            task_id=request.task_id,
            condition=request.condition,
            action=request.action,
        )
        return ControlResponse(
            success=True,
            message="Breakpoint created",
            data=bp.to_dict(),
        )

    @app.post("/api/breakpoints/{breakpoint_id}/toggle")
    async def toggle_breakpoint(
        breakpoint_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Toggle breakpoint enabled status."""
        from ..breakpoints import BreakpointManager

        proj_dir = _get_project_dir(project_dir)
        state_dir = proj_dir / ".prd_runner"
        if not state_dir.exists():
            return ControlResponse(
                success=False,
                message="No .prd_runner state directory found",
                data={},
            )

        manager = BreakpointManager(state_dir)
        enabled = manager.toggle_breakpoint(breakpoint_id)
        if enabled is None:
            return ControlResponse(
                success=False,
                message=f"Breakpoint not found: {breakpoint_id}",
                data={"breakpoint_id": breakpoint_id},
            )

        bp = next((b for b in manager.list_breakpoints() if b.id == breakpoint_id), None)
        return ControlResponse(
            success=True,
            message=f"Breakpoint {'enabled' if enabled else 'disabled'}",
            data=bp.to_dict() if bp else {"breakpoint_id": breakpoint_id, "enabled": enabled},
        )

    @app.delete("/api/breakpoints/{breakpoint_id}")
    async def delete_breakpoint(
        breakpoint_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Delete a breakpoint."""
        from ..breakpoints import BreakpointManager

        proj_dir = _get_project_dir(project_dir)
        state_dir = proj_dir / ".prd_runner"
        if not state_dir.exists():
            return ControlResponse(
                success=False,
                message="No .prd_runner state directory found",
                data={},
            )

        manager = BreakpointManager(state_dir)
        removed = manager.remove_breakpoint(breakpoint_id)
        if not removed:
            return ControlResponse(
                success=False,
                message=f"Breakpoint not found: {breakpoint_id}",
                data={"breakpoint_id": breakpoint_id},
            )

        return ControlResponse(
            success=True,
            message="Breakpoint deleted",
            data={"breakpoint_id": breakpoint_id},
        )

    @app.delete("/api/breakpoints")
    async def clear_breakpoints(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Clear all breakpoints."""
        from ..breakpoints import BreakpointManager

        proj_dir = _get_project_dir(project_dir)
        state_dir = proj_dir / ".prd_runner"
        if not state_dir.exists():
            return ControlResponse(
                success=False,
                message="No .prd_runner state directory found",
                data={},
            )

        manager = BreakpointManager(state_dir)
        count = manager.clear_all()
        return ControlResponse(
            success=True,
            message=f"Cleared {count} breakpoint(s)",
            data={"count": count},
        )

    @app.get("/api/approvals")
    async def get_approvals(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> list[ApprovalGateInfo]:
        """Get pending approval requests.

        Args:
            project_dir: Project directory path.

        Returns:
            List of pending approval requests.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        # Resolve current run and read approval state from progress.json
        run_state = _load_data(paths["run_state"], {})
        run_id = run_state.get("run_id")
        if not run_id:
            return []

        progress_path: Optional[Path] = None
        for run_dir in paths["runs"].glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path:
            return []

        progress = _load_data(progress_path, {})
        approval = progress.get("approval_pending")
        if not isinstance(approval, dict):
            return []

        context = approval.get("context") if isinstance(approval.get("context"), dict) else {}
        return [
            ApprovalGateInfo(
                request_id=str(approval.get("id", "")),
                gate_type=str(approval.get("gate_type", "")),
                message=str(approval.get("message", "")),
                task_id=context.get("task_id"),
                phase_id=context.get("phase_id"),
                created_at=str(approval.get("created_at", "")),
                timeout=approval.get("timeout"),
                context=context,
                show_diff=bool(context.get("show_diff", False)),
                show_plan=bool(context.get("show_plan", False)),
                show_tests=bool(context.get("show_tests", False)),
                show_review=bool(context.get("show_review", False)),
            )
        ]

    @app.post("/api/approvals/respond")
    async def respond_to_approval(
        action: ApprovalAction,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Respond to an approval request.

        Args:
            action: Approval action (approve/reject with optional feedback).
            project_dir: Project directory path.

        Returns:
            Response indicating success or failure.
        """
        from ..messaging import ApprovalResponse, MessageBus

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        run_state = _load_data(paths["run_state"], {})
        run_id = run_state.get("run_id")
        if not run_id:
            return ControlResponse(success=False, message="No active run found", data={})

        progress_path: Optional[Path] = None
        for run_dir in paths["runs"].glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path:
            return ControlResponse(
                success=False,
                message=f"No progress.json found for run {run_id}",
                data={},
            )

        progress = _load_data(progress_path, {})
        pending = progress.get("approval_pending")
        if not isinstance(pending, dict):
            return ControlResponse(
                success=False,
                message="No pending approval request found",
                data={},
            )

        pending_id = str(pending.get("id", ""))
        if pending_id != action.request_id:
            return ControlResponse(
                success=False,
                message=f"Approval request {action.request_id} not found",
                data={"pending_request_id": pending_id},
            )

        response = ApprovalResponse(
            request_id=action.request_id,
            approved=action.approved,
            feedback=action.feedback,
            responded_at=datetime.now(timezone.utc).isoformat(),
        )
        MessageBus(progress_path).respond_to_approval(response)

        logger.info(
            "Approval request {} {}: {}",
            action.request_id,
            "approved" if action.approved else "rejected",
            action.feedback or "no feedback",
        )

        return ControlResponse(
            success=True,
            message=f"Approval request {'approved' if action.approved else 'rejected'}",
            data={
                "request_id": action.request_id,
                "approved": action.approved,
                "feedback": action.feedback,
            },
        )

    @app.get("/api/messages")
    async def get_messages(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        run_id: Optional[str] = Query(None, description="Run ID"),
    ) -> list[ChatMessage]:
        """Get chat messages between human and worker.

        Args:
            project_dir: Project directory path.
            run_id: Optional run ID (auto-detect if not specified).

        Returns:
            List of chat messages.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        # Auto-detect run_id if not specified
        if not run_id:
            run_state_path = paths["run_state"]
            if run_state_path.exists():
                run_state = _load_data(run_state_path, {})
                run_id = run_state.get("run_id")

        if not run_id:
            return []

        # Find progress.json for this run
        runs_dir = paths["runs"]
        progress_path = None
        for run_dir in runs_dir.glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path or not progress_path.exists():
            return []

        try:
            progress_data = _load_data(progress_path, {})
            messages = []

            # Get messages from human
            for msg in progress_data.get("messages_from_human", []):
                messages.append(
                    ChatMessage(
                        id=msg.get("id", ""),
                        type=msg.get("type", ""),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp", ""),
                        from_human=True,
                        metadata=msg.get("metadata", {}),
                    )
                )

            # Get messages to human (from worker)
            for msg in progress_data.get("messages_to_human", []):
                messages.append(
                    ChatMessage(
                        id=msg.get("id", ""),
                        type=msg.get("type", ""),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp", ""),
                        from_human=False,
                        metadata=msg.get("metadata", {}),
                    )
                )

            # Sort by timestamp
            messages.sort(key=lambda m: m.timestamp)

            return messages
        except Exception as e:
            logger.error("Failed to read messages: {}", e)
            return []

    @app.post("/api/messages")
    async def send_message(
        request: SendMessageRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        run_id: Optional[str] = Query(None, description="Run ID"),
    ) -> ControlResponse:
        """Send a message to the worker.

        Args:
            request: Message content and metadata.
            project_dir: Project directory path.
            run_id: Optional run ID (auto-detect if not specified).

        Returns:
            Response indicating success or failure.
        """
        from ..messaging import Message, MessageBus

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        # Auto-detect run_id if not specified
        if not run_id:
            run_state_path = paths["run_state"]
            if run_state_path.exists():
                run_state = _load_data(run_state_path, {})
                run_id = run_state.get("run_id")

        if not run_id:
            return ControlResponse(
                success=False,
                message="No active run found",
                data={},
            )

        # Find progress.json for this run
        runs_dir = paths["runs"]
        progress_path = None
        for run_dir in runs_dir.glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path:
            return ControlResponse(
                success=False,
                message=f"No progress.json found for run {run_id}",
                data={},
            )

        try:
            import time

            # Create and send message
            msg = Message(
                id=f"{request.type}-{int(time.time() * 1000)}",
                type=request.type,
                content=request.content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=request.metadata,
            )

            bus = MessageBus(progress_path)
            bus.send_to_worker(msg)

            logger.info("Message sent to worker: type={} content={}", msg.type, msg.content[:100])

            return ControlResponse(
                success=True,
                message="Message sent to worker",
                data={"message_id": msg.id, "type": msg.type},
            )
        except Exception as e:
            logger.error("Failed to send message: {}", e)
            return ControlResponse(
                success=False,
                message=f"Error sending message: {e}",
                data={},
            )

    @app.get("/api/file-changes")
    async def get_file_changes(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        task_id: Optional[str] = Query(None, description="Task ID"),
    ) -> list[FileChange]:
        """Get file changes for review.

        Args:
            project_dir: Project directory path.
            task_id: Optional task ID (uses current working tree if not specified).

        Returns:
            List of file changes with diffs.
        """
        import subprocess

        proj_dir = _get_project_dir(project_dir)

        # Check if in a git repository
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=proj_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            return []

        files = []

        try:
            # Get list of changed files
            if task_id:
                # Try to find task branch
                state_dir = proj_dir / ".prd_runner"
                if state_dir.exists():
                    task_queue_path = state_dir / "task_queue.yaml"
                    tasks = _load_data(task_queue_path, {})

                    branch = None
                    for t in tasks.get("tasks", []):
                        if t.get("id") == task_id:
                            branch = t.get("branch")
                            break

                    if branch:
                        # Get diff from main to branch
                        result = subprocess.run(
                            ["git", "diff", f"main...{branch}", "--numstat"],
                            cwd=proj_dir,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                    else:
                        result = subprocess.run(
                            ["git", "diff", "--numstat"],
                            cwd=proj_dir,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                else:
                    result = subprocess.run(
                        ["git", "diff", "--numstat"],
                        cwd=proj_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
            else:
                # Current working tree changes
                result = subprocess.run(
                    ["git", "diff", "--numstat"],
                    cwd=proj_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        additions = parts[0]
                        deletions = parts[1]
                        file_path = parts[2]

                        # Get diff for this file
                        if task_id and branch:
                            diff_result = subprocess.run(
                                ["git", "diff", f"main...{branch}", "--", file_path],
                                cwd=proj_dir,
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                        else:
                            diff_result = subprocess.run(
                                ["git", "diff", "--", file_path],
                                cwd=proj_dir,
                                capture_output=True,
                                text=True,
                                check=False,
                            )

                        # Determine status
                        try:
                            add_count = int(additions) if additions != "-" else 0
                            del_count = int(deletions) if deletions != "-" else 0
                        except ValueError:
                            add_count = 0
                            del_count = 0

                        if add_count > 0 and del_count == 0:
                            status = "added"
                        elif add_count == 0 and del_count > 0:
                            status = "deleted"
                        else:
                            status = "modified"

                        files.append(
                            FileChange(
                                file_path=file_path,
                                status=status,
                                additions=add_count,
                                deletions=del_count,
                                diff=diff_result.stdout,
                                approved=None,
                                comments=[],
                            )
                        )

        except Exception as e:
            logger.error("Failed to get file changes: {}", e)

        return files

    @app.post("/api/file-review")
    async def submit_file_review(
        review: FileReviewRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Submit file review (approve/reject).

        Args:
            review: File review action.
            project_dir: Project directory path.

        Returns:
            Response indicating success or failure.
        """
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        # Store review in file_reviews.json
        reviews_path = paths["state_dir"] / "file_reviews.json"

        try:
            if reviews_path.exists():
                reviews_data = _load_data(reviews_path, {})
            else:
                reviews_data = {"reviews": []}

            # Add or update review
            review_dict = {
                "file_path": review.file_path,
                "approved": review.approved,
                "comment": review.comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Remove existing review for this file
            reviews_data["reviews"] = [
                r for r in reviews_data.get("reviews", []) if r.get("file_path") != review.file_path
            ]

            reviews_data["reviews"].append(review_dict)

            _save_data(reviews_path, reviews_data)

            logger.info(
                "File review submitted: file={} approved={}",
                review.file_path,
                review.approved,
            )

            return ControlResponse(
                success=True,
                message=f"File {'approved' if review.approved else 'rejected'}",
                data=review_dict,
            )

        except Exception as e:
            logger.error("Failed to submit file review: {}", e)
            return ControlResponse(
                success=False,
                message=f"Error submitting review: {e}",
                data={},
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

    @app.post("/api/runs/start")
    async def start_run(
        request: StartRunRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> StartRunResponse:
        """Start a new run from the web UI.

        Args:
            request: Run configuration (PRD content or prompt, test/build commands, etc.).
            project_dir: Project directory path.

        Returns:
            Response with run ID and status.
        """
        import subprocess
        import tempfile
        import uuid
        from datetime import datetime

        proj_dir = _get_project_dir(project_dir)

        try:
            # Generate or use provided PRD
            prd_content = ""
            if request.mode == "quick_prompt":
                # Generate PRD from prompt using codex worker
                logger.info("Generating PRD from prompt: {}", request.content[:100])
                from ..custom_execution import execute_custom_prompt
                from ..config import load_runner_config

                # Load project config to get codex command
                config, config_err = load_runner_config(proj_dir)
                if config_err:
                    logger.warning("Failed to load config: {}", config_err)

                # Get codex command from config, auto-detect, or use default
                codex_cmd = None
                if config and "codex_command" in config:
                    codex_cmd = config["codex_command"]
                    logger.info("Using codex command from config: {}", codex_cmd)

                    # Validate that the command includes stdin or placeholder
                    has_stdin = " -" in codex_cmd or codex_cmd.endswith("-")
                    has_placeholder = "{prompt_file}" in codex_cmd or "{prompt}" in codex_cmd
                    if not has_stdin and not has_placeholder:
                        return StartRunResponse(
                            success=False,
                            message=(
                                f"Invalid codex_command in config: {codex_cmd}\n\n"
                                f"The command must include one of:\n"
                                f"  - 'exec -' to read from stdin (recommended)\n"
                                f"  - '{{prompt_file}}' placeholder\n"
                                f"  - '{{prompt}}' placeholder\n\n"
                                f"Example: codex_command: 'C:/Program Files/nodejs/codex.ps1 exec -'"
                            ),
                            run_id=None,
                            prd_path=None,
                        )
                else:
                    # Try to auto-detect
                    codex_cmd = _find_codex_command()
                    if codex_cmd:
                        logger.info("Auto-detected codex command: {}", codex_cmd)
                    else:
                        logger.warning("Could not auto-detect codex command, using default")
                        codex_cmd = "codex exec -"

                # Create a temporary file path for the generated PRD
                generated_prd_path = proj_dir / ".prd_runner" / "temp_generated_prd.md"
                generated_prd_path.parent.mkdir(parents=True, exist_ok=True)

                prd_gen_prompt = f"""You are a product requirements document (PRD) generator. Generate a clear, well-structured PRD based on the following user request:

{request.content}

Format the PRD in markdown with these sections:
# Feature: [Brief Title]

## Overview
[What is this feature and why is it needed?]

## Requirements
[Detailed functional requirements as a numbered list]

## Technical Approach
[High-level technical implementation notes]

## Success Criteria
[How to verify the feature works correctly]

Be specific, actionable, and include all necessary details for implementation.

Write the generated PRD to the file: {generated_prd_path}"""

                try:
                    # Use codex to generate the PRD
                    success, error_msg = execute_custom_prompt(
                        user_prompt=prd_gen_prompt,
                        project_dir=proj_dir,
                        codex_command=codex_cmd,
                        heartbeat_seconds=60,
                        heartbeat_grace_seconds=120,
                        shift_minutes=5,
                        override_agents=False,
                    )

                    if not success:
                        logger.error("Failed to generate PRD: {}", error_msg)
                        # Check if it's a codex not found error
                        error_details = error_msg or ""
                        if "cannot find the file" in error_details.lower() or "no such file" in error_details.lower():
                            error_details += (
                                f"\n\nCodex command used: {codex_cmd}\n"
                                f"Make sure codex is installed and in your PATH.\n"
                                f"Alternatively, configure 'codex_command' in .prd_runner/config.yaml with the full path.\n"
                                f"Example (Windows): codex_command: 'C:\\Program Files\\nodejs\\codex.cmd exec -'"
                            )
                        return StartRunResponse(
                            success=False,
                            message=f"Failed to generate PRD from prompt: {error_details}",
                            run_id=None,
                            prd_path=None,
                        )

                    # Read the generated PRD
                    if not generated_prd_path.exists():
                        logger.error("Generated PRD file not found at {}", generated_prd_path)
                        return StartRunResponse(
                            success=False,
                            message="PRD generation completed but output file not found",
                            run_id=None,
                            prd_path=None,
                        )

                    prd_content = generated_prd_path.read_text()
                    logger.info("Generated PRD: {} chars", len(prd_content))

                    # Clean up temporary file
                    generated_prd_path.unlink()

                except Exception as e:
                    logger.error("Failed to generate PRD: {}", e)
                    error_msg = str(e)
                    # Check if it's a codex not found error
                    if "cannot find the file" in error_msg.lower() or "no such file" in error_msg.lower():
                        error_msg += (
                            f"\n\nCodex command used: {codex_cmd}\n"
                            f"Make sure codex is installed and in your PATH.\n"
                            f"Alternatively, configure 'codex_command' in .prd_runner/config.yaml with the full path.\n"
                            f"Example (Windows): codex_command: 'C:\\Program Files\\nodejs\\codex.cmd exec -'"
                        )
                    return StartRunResponse(
                        success=False,
                        message=f"Failed to generate PRD from prompt: {error_msg}",
                        run_id=None,
                        prd_path=None,
                    )
            elif request.mode == "full_prd":
                prd_content = request.content
            else:
                return StartRunResponse(
                    success=False,
                    message=f"Invalid mode: {request.mode}. Must be 'full_prd' or 'quick_prompt'",
                    run_id=None,
                    prd_path=None,
                )

            # Save PRD to file
            prd_dir = proj_dir / ".prd_runner" / "generated_prds"
            prd_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prd_filename = f"prd_{timestamp}.md"
            prd_path = prd_dir / prd_filename

            prd_path.write_text(prd_content)
            logger.info("Saved PRD to: {}", prd_path)

            # Build command arguments
            run_id = f"web-{uuid.uuid4().hex[:8]}"

            cmd_args = [
                "feature-prd-runner",
                "run",
                "--project-dir",
                str(proj_dir),
                "--prd-file",
                str(prd_path),
            ]

            # Add optional commands
            if request.test_command:
                cmd_args.extend(["--test-command", request.test_command])
            if request.build_command:
                cmd_args.extend(["--build-cmd", request.build_command])
            if request.verification_profile:
                cmd_args.extend(["--verification-profile", request.verification_profile])

            # Advanced run options (Batch 6)
            if request.language:
                cmd_args.extend(["--language", request.language])
            if request.reset_state:
                cmd_args.append("--reset-state")
            if not request.require_clean:
                cmd_args.append("--no-require-clean")
            if not request.commit_enabled:
                cmd_args.append("--no-commit")
            if not request.push_enabled:
                cmd_args.append("--no-push")
            if request.interactive:
                cmd_args.append("--interactive")
            if request.parallel:
                cmd_args.append("--parallel")
            if request.max_workers != 3:
                cmd_args.extend(["--max-workers", str(request.max_workers)])
            if request.ensure_ruff != "off":
                cmd_args.extend(["--ensure-ruff", request.ensure_ruff])
            if request.ensure_deps != "off":
                cmd_args.extend(["--ensure-deps", request.ensure_deps])
            if request.ensure_deps_command:
                cmd_args.extend(["--ensure-deps-command", request.ensure_deps_command])
            if request.shift_minutes != 45:
                cmd_args.extend(["--shift-minutes", str(request.shift_minutes)])
            if request.max_task_attempts != 5:
                cmd_args.extend(["--max-task-attempts", str(request.max_task_attempts)])
            if request.max_review_attempts != 10:
                cmd_args.extend(["--max-review-attempts", str(request.max_review_attempts)])
            if request.worker:
                cmd_args.extend(["--worker", request.worker])
            if request.codex_command:
                cmd_args.extend(["--codex-command", request.codex_command])

            logger.info("Starting run with command: {}", " ".join(cmd_args))

            # Spawn subprocess in background
            # Use nohup or similar to detach from parent process
            subprocess.Popen(
                cmd_args,
                cwd=proj_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
            )

            logger.info("Started run {} for project {}", run_id, proj_dir)

            return StartRunResponse(
                success=True,
                message=f"Run started successfully",
                run_id=run_id,
                prd_path=str(prd_path),
            )

        except Exception as e:
            logger.error("Failed to start run: {}", e)
            return StartRunResponse(
                success=False,
                message=f"Failed to start run: {e}",
                run_id=None,
                prd_path=None,
            )

    @app.post("/api/runs/exec")
    async def exec_task(
        request: ExecTaskRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ExecTaskResponse:
        """Execute a one-off task using the exec command.

        This endpoint runs a task immediately and terminates upon completion,
        skipping the full workflow (plan → implement → verify → review → commit).

        Args:
            request: Task execution request (prompt, options).
            project_dir: Project directory path.

        Returns:
            Response with execution status and run ID.
        """
        from ..custom_execution import execute_custom_prompt

        proj_dir = _get_project_dir(project_dir)

        try:
            logger.info("Executing one-off task: {}", request.prompt[:100])

            # Build context if files are provided
            context = None
            if request.context_files:
                context = {"files": request.context_files.split(",")}

            # Execute the custom prompt (synchronous, blocks until complete)
            success, error_message = execute_custom_prompt(
                user_prompt=request.prompt,
                project_dir=proj_dir,
                override_agents=request.override_agents,
                context=context,
                shift_minutes=request.shift_minutes,
                heartbeat_seconds=request.heartbeat_seconds,
                heartbeat_grace_seconds=300,
                then_continue=False,
            )

            if success:
                logger.info("Task executed successfully")
                return ExecTaskResponse(
                    success=True,
                    message="Task executed successfully",
                    run_id=None,  # execute_custom_prompt doesn't return run_id easily
                    error=None,
                )
            else:
                logger.error("Task execution failed: {}", error_message)
                return ExecTaskResponse(
                    success=False,
                    message="Task execution failed",
                    run_id=None,
                    error=error_message,
                )

        except Exception as e:
            logger.error("Failed to execute task: {}", e)
            return ExecTaskResponse(
                success=False,
                message=f"Failed to execute task: {e}",
                run_id=None,
                error=str(e),
            )

    @app.post("/api/v2/quick-runs", response_model=QuickRunExecuteResponse)
    async def create_quick_run(
        request: QuickRunCreateRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> QuickRunExecuteResponse:
        """Execute a quick one-off run and store its record."""
        from ..custom_execution import execute_custom_prompt

        proj_dir = _get_project_dir(project_dir)
        run_id = f"qrun-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex()}"
        started_at = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "id": run_id,
            "prompt": request.prompt,
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "result_summary": None,
            "error": None,
            "promoted_task_id": None,
        }

        records = _load_quick_runs(proj_dir)
        records.append(record)
        _save_quick_runs(proj_dir, records)
        _append_quick_run_event(
            proj_dir,
            event_type="quick_run.started",
            quick_run_id=run_id,
            status=record["status"],
        )

        try:
            context = None
            if request.context_files:
                context = {"files": request.context_files.split(",")}

            success, error_message = execute_custom_prompt(
                user_prompt=request.prompt,
                project_dir=proj_dir,
                override_agents=request.override_agents,
                context=context,
                shift_minutes=request.shift_minutes,
                heartbeat_seconds=request.heartbeat_seconds,
                heartbeat_grace_seconds=300,
                then_continue=False,
            )

            record["status"] = "completed" if success else "failed"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["result_summary"] = (
                "Quick action executed successfully"
                if success
                else "Quick action execution failed"
            )
            record["error"] = None if success else error_message
            _save_quick_runs(proj_dir, records)
            _append_quick_run_event(
                proj_dir,
                event_type="quick_run.completed" if success else "quick_run.failed",
                quick_run_id=run_id,
                status=record["status"],
                details={"error": error_message} if not success and error_message else {},
            )

            return QuickRunExecuteResponse(
                success=success,
                message=record["result_summary"],
                quick_run=QuickRunRecord(**record),
                error=record["error"],
            )
        except Exception as e:
            record["status"] = "failed"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["result_summary"] = "Quick action execution failed"
            record["error"] = str(e)
            _save_quick_runs(proj_dir, records)
            _append_quick_run_event(
                proj_dir,
                event_type="quick_run.failed",
                quick_run_id=run_id,
                status=record["status"],
                details={"error": str(e)},
            )
            logger.error("Failed to execute quick run {}: {}", run_id, e)
            return QuickRunExecuteResponse(
                success=False,
                message="Quick action execution failed",
                quick_run=QuickRunRecord(**record),
                error=str(e),
            )

    @app.get("/api/v2/quick-runs", response_model=list[QuickRunRecord])
    async def list_quick_runs(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        limit: int = Query(20, ge=1, le=200),
    ) -> list[QuickRunRecord]:
        proj_dir = _get_project_dir(project_dir)
        records = _load_quick_runs(proj_dir)
        records_sorted = sorted(
            records,
            key=lambda item: str(item.get("started_at") or ""),
            reverse=True,
        )
        return [QuickRunRecord(**item) for item in records_sorted[:limit]]

    @app.get("/api/v2/quick-runs/{quick_run_id}", response_model=QuickRunRecord)
    async def get_quick_run(
        quick_run_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> QuickRunRecord:
        proj_dir = _get_project_dir(project_dir)
        _records, target = _find_quick_run(proj_dir, quick_run_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Quick run {quick_run_id} not found")
        return QuickRunRecord(**target)

    @app.get("/api/v2/quick-runs/events/recent", response_model=QuickRunEventsResponse)
    async def get_recent_quick_run_events(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        limit: int = Query(100, ge=1, le=1000),
    ) -> QuickRunEventsResponse:
        proj_dir = _get_project_dir(project_dir)
        events = _load_quick_run_events(proj_dir, limit=limit)
        parsed = [QuickRunEventRecord(**item) for item in events]
        return QuickRunEventsResponse(events=parsed, total=len(parsed))

    @app.get("/api/v2/quick-runs/{quick_run_id}/events", response_model=QuickRunEventsResponse)
    async def get_quick_run_events(
        quick_run_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        limit: int = Query(100, ge=1, le=1000),
    ) -> QuickRunEventsResponse:
        proj_dir = _get_project_dir(project_dir)
        _records, target = _find_quick_run(proj_dir, quick_run_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Quick run {quick_run_id} not found")
        events = _load_quick_run_events(proj_dir, limit=max(limit * 5, limit))
        filtered = [item for item in events if str(item.get("quick_run_id")) == quick_run_id]
        parsed = [QuickRunEventRecord(**item) for item in filtered[-limit:]]
        return QuickRunEventsResponse(events=parsed, total=len(parsed))

    @app.post("/api/v2/quick-runs/{quick_run_id}/promote", response_model=PromoteQuickRunResponse)
    async def promote_quick_run(
        quick_run_id: str,
        body: PromoteQuickRunRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> PromoteQuickRunResponse:
        from ..task_engine.engine import TaskEngine

        proj_dir = _get_project_dir(project_dir)
        records, target = _find_quick_run(proj_dir, quick_run_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Quick run {quick_run_id} not found")

        if target.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail="Only completed quick runs can be promoted",
            )

        existing_task_id = target.get("promoted_task_id")
        if isinstance(existing_task_id, str) and existing_task_id:
            return PromoteQuickRunResponse(
                success=True,
                message=f"Quick run already promoted to task {existing_task_id}",
                task_id=existing_task_id,
                quick_run=QuickRunRecord(**target),
            )

        prompt = str(target.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Quick run has no prompt to promote")

        first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "Quick action follow-up")
        title = body.title.strip() if body.title else first_line
        if len(title) > 80:
            title = f"{title[:77].rstrip()}..."

        state_dir = proj_dir / ".prd_runner"
        state_dir.mkdir(parents=True, exist_ok=True)
        engine = TaskEngine(state_dir)
        task = engine.create_task(
            title=title,
            description=f"Promoted from Quick Action.\n\nOriginal prompt:\n{prompt}",
            task_type=body.task_type,
            priority=body.priority,
            source="promoted_quick_action",
            metadata={
                "origin": "quick_action",
                "quick_run_id": quick_run_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        target["promoted_task_id"] = task.id
        _save_quick_runs(proj_dir, records)
        _append_quick_run_event(
            proj_dir,
            event_type="quick_run.promoted",
            quick_run_id=quick_run_id,
            status=str(target.get("status") or "completed"),
            details={"task_id": task.id},
        )

        return PromoteQuickRunResponse(
            success=True,
            message=f"Promoted quick run to task {task.id}",
            task_id=task.id,
            quick_run=QuickRunRecord(**target),
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

    # ------------------------------------------------------------------
    # Batch 1: Explain + Inspect
    # ------------------------------------------------------------------

    @app.get("/api/tasks/{task_id}/explain")
    async def explain_task(
        task_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ExplainResponse:
        """Explain why a task is blocked or errored."""
        from ..debug import ErrorAnalyzer

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)
        if not paths["state_dir"].exists():
            raise HTTPException(status_code=404, detail="No .prd_runner state found")

        analyzer = ErrorAnalyzer(proj_dir)
        explanation = analyzer.explain_blocking(task_id)
        is_blocked = "blocked" in explanation.lower() or "waiting_human" in explanation.lower()

        return ExplainResponse(
            task_id=task_id,
            explanation=explanation,
            is_blocked=is_blocked,
        )

    @app.get("/api/tasks/{task_id}/inspect")
    async def inspect_task(
        task_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> InspectResponse:
        """Inspect full task state."""
        from ..debug import ErrorAnalyzer

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)
        if not paths["state_dir"].exists():
            raise HTTPException(status_code=404, detail="No .prd_runner state found")

        analyzer = ErrorAnalyzer(proj_dir)
        snapshot = analyzer.inspect_state(task_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return InspectResponse(
            task_id=snapshot.task_id,
            lifecycle=snapshot.lifecycle,
            step=snapshot.step,
            status=snapshot.status,
            worker_attempts=snapshot.worker_attempts,
            last_error=snapshot.last_error,
            last_error_type=snapshot.last_error_type,
            context=snapshot.context,
            metadata=snapshot.metadata,
        )

    @app.get("/api/tasks/{task_id}/trace")
    async def trace_task(
        task_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        limit: int = Query(50, ge=1, le=1000, description="Max events"),
    ) -> list[dict[str, Any]]:
        """Get event history for a task (trace)."""
        from ..debug import ErrorAnalyzer

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)
        if not paths["state_dir"].exists():
            return []

        analyzer = ErrorAnalyzer(proj_dir)
        events = analyzer.trace_history(task_id)
        if limit > 0:
            events = events[-limit:]
        return events

    # ------------------------------------------------------------------
    # Batch 2: Dry-Run + Doctor
    # ------------------------------------------------------------------

    @app.get("/api/dry-run")
    async def dry_run(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        prd_file: Optional[str] = Query(None, description="PRD file path"),
    ) -> DryRunResponse:
        """Preview what the next run action would do without writing anything."""
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        warnings: list[str] = []
        errors: list[str] = []
        next_action: dict[str, Any] | None = None
        would_spawn_codex = False
        would_run_tests = False
        would_checkout_branch = False

        if not paths["state_dir"].exists():
            return DryRunResponse(
                project_dir=str(proj_dir),
                state_dir=str(paths["state_dir"]),
                next={"action": "init", "description": "Initialize .prd_runner state directory"},
                warnings=["No state directory found — a fresh run would initialize it."],
            )

        # Load state files
        run_state = _load_data(paths["run_state"], {})
        queue = _load_data(paths["task_queue"], {})
        plan = _load_data(paths["phase_plan"], {})
        tasks = _normalize_tasks(queue)
        phases = _normalize_phases(plan)

        if not phases:
            warnings.append("No phases found in phase_plan.yaml")
        if not tasks:
            warnings.append("No tasks found in task_queue.yaml")

        # Check for active run
        if run_state.get("status") == "running":
            warnings.append(f"An active run exists: {run_state.get('run_id', 'unknown')}")

        # Validate PRD if specified
        if prd_file:
            prd_path = Path(prd_file)
            if not prd_path.exists():
                errors.append(f"PRD file not found: {prd_file}")

        # Find next task
        ready_tasks = [t for t in tasks if t.get("lifecycle") == "ready"]
        if ready_tasks:
            task = ready_tasks[0]
            step = task.get("step", "plan_impl")
            would_spawn_codex = step in ("plan_impl", "implement", "review")
            would_run_tests = step == "verify"
            would_checkout_branch = bool(task.get("phase_id")) and task.get("type") != "plan"

            next_action = {
                "action": "process_task",
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "step": step,
                "phase_id": task.get("phase_id"),
            }
        elif all(t.get("lifecycle") == "done" for t in tasks) and tasks:
            next_action = {"action": "complete", "description": "All tasks done"}
        else:
            blocked = [t for t in tasks if t.get("lifecycle") == "waiting_human"]
            if blocked:
                warnings.append(f"{len(blocked)} task(s) waiting for human intervention")

        return DryRunResponse(
            project_dir=str(proj_dir),
            state_dir=str(paths["state_dir"]),
            would_write_repo_files=would_spawn_codex,
            would_spawn_codex=would_spawn_codex,
            would_run_tests=would_run_tests,
            would_checkout_branch=would_checkout_branch,
            next=next_action,
            warnings=warnings,
            errors=errors,
        )

    @app.get("/api/doctor")
    async def doctor(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        check_codex: bool = Query(False, description="Check codex availability"),
    ) -> DoctorResponse:
        """Run diagnostic checks on the project state."""
        import subprocess

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        checks: dict[str, Any] = {}
        warnings: list[str] = []
        errors: list[str] = []

        # Check state directory
        if paths["state_dir"].exists():
            checks["state_dir"] = {"status": "pass", "path": str(paths["state_dir"])}
        else:
            checks["state_dir"] = {"status": "fail", "path": str(paths["state_dir"])}
            errors.append("No .prd_runner state directory found")

        # Check git status
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=proj_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                checks["git"] = {"status": "pass"}
                # Check for uncommitted changes
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=proj_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if status_result.stdout.strip():
                    warnings.append("Uncommitted changes detected in working tree")
            else:
                checks["git"] = {"status": "fail"}
                warnings.append("Not a git repository")
        except Exception:
            checks["git"] = {"status": "fail"}
            warnings.append("Unable to check git status")

        # Check state files
        for name, path in [
            ("run_state", paths["run_state"]),
            ("task_queue", paths["task_queue"]),
            ("phase_plan", paths["phase_plan"]),
        ]:
            if path.exists():
                data = _load_data(path, None)
                if data is not None:
                    checks[name] = {"status": "pass"}
                else:
                    checks[name] = {"status": "fail"}
                    errors.append(f"Failed to parse {name}: {path.name}")
            else:
                checks[name] = {"status": "skip", "reason": "File not found"}

        # Check codex availability
        if check_codex:
            codex_cmd = _find_codex_command()
            if codex_cmd:
                checks["codex"] = {"status": "pass", "command": codex_cmd}
            else:
                checks["codex"] = {"status": "fail"}
                errors.append("Codex command not found in PATH")

        exit_code = 1 if errors else 0
        return DoctorResponse(
            checks=checks,
            warnings=warnings,
            errors=errors,
            exit_code=exit_code,
        )

    # ------------------------------------------------------------------
    # Batch 3: Workers Management
    # ------------------------------------------------------------------

    @app.get("/api/workers")
    async def list_workers(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> WorkersListResponse:
        """List configured worker providers."""
        from ..config import load_runner_config
        from ..workers import get_workers_runtime_config

        proj_dir = _get_project_dir(project_dir)
        config, config_err = load_runner_config(proj_dir)

        if config_err:
            return WorkersListResponse(
                default_worker="codex",
                config_error=config_err,
            )

        try:
            codex_cmd = _find_codex_command() or "codex exec -"
            runtime = get_workers_runtime_config(config=config, codex_command_fallback=codex_cmd, cli_worker=None)

            providers = []
            for name, spec in runtime.providers.items():
                providers.append(WorkerInfo(
                    name=name,
                    type=spec.type,
                    detail=spec.command or spec.endpoint or "",
                    model=spec.model,
                    endpoint=spec.endpoint,
                    command=spec.command,
                ))

            return WorkersListResponse(
                default_worker=runtime.default_worker,
                routing=dict(runtime.routing),
                providers=providers,
            )
        except Exception as e:
            logger.error("Failed to load workers config: {}", e)
            return WorkersListResponse(
                default_worker="codex",
                config_error=str(e),
            )

    @app.post("/api/workers/{worker_name}/test")
    async def test_worker(
        worker_name: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> WorkerTestResponse:
        """Test a worker provider."""
        from ..config import load_runner_config
        from ..workers import get_workers_runtime_config

        proj_dir = _get_project_dir(project_dir)
        config, config_err = load_runner_config(proj_dir)
        if config_err:
            return WorkerTestResponse(worker=worker_name, success=False, message=config_err)

        try:
            codex_cmd = _find_codex_command() or "codex exec -"
            runtime = get_workers_runtime_config(config=config, codex_command_fallback=codex_cmd, cli_worker=None)

            if worker_name not in runtime.providers:
                return WorkerTestResponse(
                    worker=worker_name,
                    success=False,
                    message=f"Provider '{worker_name}' not found",
                )

            spec = runtime.providers[worker_name]
            if spec.type == "codex":
                import shutil
                cmd = (spec.command or "").split()[0] if spec.command else "codex"
                found = shutil.which(cmd)
                if found:
                    return WorkerTestResponse(worker=worker_name, success=True, message=f"Found: {found}")
                return WorkerTestResponse(worker=worker_name, success=False, message=f"Command not found: {cmd}")
            elif spec.type == "ollama":
                return WorkerTestResponse(
                    worker=worker_name,
                    success=bool(spec.endpoint and spec.model),
                    message=f"Endpoint: {spec.endpoint}, Model: {spec.model}" if spec.endpoint else "Missing endpoint or model",
                )
            else:
                return WorkerTestResponse(worker=worker_name, success=False, message=f"Unknown type: {spec.type}")

        except Exception as e:
            return WorkerTestResponse(worker=worker_name, success=False, message=str(e))

    # ------------------------------------------------------------------
    # Batch 4: Structured Correct + Require
    # ------------------------------------------------------------------

    @app.post("/api/tasks/{task_id}/correct")
    async def correct_task(
        task_id: str,
        request: CorrectionRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Send a structured correction to the running worker."""
        import time

        from ..messaging import Message, MessageBus

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        run_state = _load_data(paths["run_state"], {})
        run_id = run_state.get("run_id")
        if not run_id:
            return ControlResponse(success=False, message="No active run found", data={})

        # Find progress.json
        progress_path = None
        for run_dir in paths["runs"].glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path:
            return ControlResponse(success=False, message=f"No progress.json for run {run_id}", data={})

        metadata: dict[str, Any] = {"task_id": task_id, "issue": request.issue}
        if request.file_path:
            metadata["file"] = request.file_path
        if request.suggested_fix:
            metadata["suggested_fix"] = request.suggested_fix

        msg = Message(
            id=f"correction-{int(time.time() * 1000)}",
            type="correction",
            content=request.issue,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

        MessageBus(progress_path).send_to_worker(msg)

        return ControlResponse(
            success=True,
            message="Correction sent to worker",
            data={"message_id": msg.id, "task_id": task_id},
        )

    @app.post("/api/requirements")
    async def add_requirement(
        request: RequirementRequest,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
    ) -> ControlResponse:
        """Inject a structured requirement into the running worker."""
        import time

        from ..messaging import Message, MessageBus

        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        run_state = _load_data(paths["run_state"], {})
        run_id = run_state.get("run_id")
        if not run_id:
            return ControlResponse(success=False, message="No active run found", data={})

        progress_path = None
        for run_dir in paths["runs"].glob(f"{run_id}*"):
            candidate = run_dir / "progress.json"
            if candidate.exists():
                progress_path = candidate
                break

        if not progress_path:
            return ControlResponse(success=False, message=f"No progress.json for run {run_id}", data={})

        metadata: dict[str, Any] = {"priority": request.priority}
        if request.task_id:
            metadata["task_id"] = request.task_id

        msg = Message(
            id=f"requirement-{int(time.time() * 1000)}",
            type="requirement",
            content=request.requirement,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

        MessageBus(progress_path).send_to_worker(msg)

        return ControlResponse(
            success=True,
            message="Requirement sent to worker",
            data={"message_id": msg.id, "priority": request.priority},
        )

    # ------------------------------------------------------------------
    # Batch 5: Logs by Task + Metrics Export
    # ------------------------------------------------------------------

    @app.get("/api/tasks/{task_id}/logs")
    async def get_task_logs(
        task_id: str,
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        step: Optional[str] = Query(None, description="Filter by step"),
        lines: int = Query(100, ge=1, le=10000, description="Max lines per file"),
    ) -> TaskLogsResponse:
        """Get logs for a specific task."""
        proj_dir = _get_project_dir(project_dir)
        paths = _get_paths(proj_dir)

        if not paths["runs"].exists():
            return TaskLogsResponse(task_id=task_id)

        # Search for matching run by task_id in progress.json
        matching_run_dir = None
        matching_run_id = None
        for run_dir in sorted(paths["runs"].iterdir(), key=lambda d: d.name, reverse=True):
            if not run_dir.is_dir():
                continue
            progress_file = run_dir / "progress.json"
            if progress_file.exists():
                progress = _load_data(progress_file, {})
                if progress.get("task_id") == task_id:
                    matching_run_dir = run_dir
                    matching_run_id = progress.get("run_id", run_dir.name)
                    break

        if not matching_run_dir:
            # Fall back: check current run_state
            run_state = _load_data(paths["run_state"], {})
            run_id = run_state.get("run_id")
            if run_id:
                for run_dir in paths["runs"].glob(f"{run_id}*"):
                    if run_dir.is_dir():
                        matching_run_dir = run_dir
                        matching_run_id = run_id
                        break

        if not matching_run_dir:
            return TaskLogsResponse(task_id=task_id)

        logs: dict[str, list[str]] = {}

        # Determine which files to read based on step
        if step == "verify":
            log_patterns = ["verify_output*.txt", "pytest_failures*.txt"]
        elif step:
            log_patterns = [f"*{step}*.log", f"*{step}*.txt"]
        else:
            log_patterns = ["*.log", "*.txt"]

        for pattern in log_patterns:
            for log_file in matching_run_dir.glob(pattern):
                try:
                    file_lines = log_file.read_text().splitlines()
                    logs[log_file.name] = file_lines[-lines:]
                except Exception:
                    logs[log_file.name] = ["[Error reading file]"]

        return TaskLogsResponse(
            task_id=task_id,
            run_id=matching_run_id,
            logs=logs,
        )

    @app.get("/api/metrics/export")
    async def export_metrics(
        project_dir: Optional[str] = Query(None, description="Project directory path"),
        format: str = Query("csv", description="Export format: csv or html"),
    ):
        """Export metrics as CSV or HTML."""
        from fastapi.responses import StreamingResponse

        from .metrics import calculate_metrics

        proj_dir = _get_project_dir(project_dir)
        metrics = calculate_metrics(proj_dir)
        data = metrics.to_dict()

        if format == "html":
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items()
            )
            html = f"<html><body><table border='1'><tr><th>Metric</th><th>Value</th></tr>{rows}</table></body></html>"
            return StreamingResponse(
                iter([html]),
                media_type="text/html",
                headers={"Content-Disposition": "attachment; filename=metrics.html"},
            )
        else:
            header = ",".join(data.keys())
            values = ",".join(str(v) for v in data.values())
            csv_content = f"{header}\n{values}\n"
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=metrics.csv"},
            )

    # ------------------------------------------------------------------
    # Batch 6: Advanced Run Options (extend StartRunRequest mapping)
    # ------------------------------------------------------------------
    # Note: StartRunRequest model extended in models.py;
    # the start_run endpoint above already handles the base fields.
    # Additional fields are mapped in _build_advanced_cmd_args below.

    def _calculate_phase_progress(phase: dict[str, Any]) -> float:
        """Calculate progress for a phase (0.0 to 1.0)."""
        status = phase.get("status", "pending")
        if status == "done":
            return 1.0
        elif status == "running":
            return 0.5
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Multiplexed WebSocket Hub
    # ------------------------------------------------------------------
    from .ws_hub import hub as _ws_hub, web_notifications as _web_notifications

    @app.websocket("/ws")
    async def ws_hub_endpoint(websocket: WebSocket) -> None:
        """Single multiplexed WebSocket for all real-time state updates."""
        await _ws_hub.handle_connection(websocket)

    # Store hub on app state so other modules can broadcast
    app.state.ws_hub = _ws_hub

    # ------------------------------------------------------------------
    # V2 Task Engine API
    # ------------------------------------------------------------------
    from ..task_engine.engine import TaskEngine as _TaskEngine
    from .import_api import create_import_router
    from .task_api import create_task_router

    _engine_cache: dict[str, _TaskEngine] = {}
    allow_auto_approve_review = os.getenv("FEATURE_PRD_AUTO_APPROVE_REVIEW", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    def _get_task_engine(project_dir_param: Optional[str] = None) -> _TaskEngine:
        proj = _get_project_dir(project_dir_param)
        state_dir = proj / ".prd_runner"
        state_dir.mkdir(parents=True, exist_ok=True)
        key = str(state_dir)
        if key not in _engine_cache:
            _engine_cache[key] = _TaskEngine(
                state_dir,
                allow_auto_approve_review=allow_auto_approve_review,
            )
        return _engine_cache[key]

    app.include_router(create_task_router(_get_task_engine))
    app.include_router(create_import_router(_get_task_engine))

    # ------------------------------------------------------------------
    # V2 Agent Pool API
    # ------------------------------------------------------------------
    from ..agents.registry import AgentRegistry as _AgentRegistry
    from ..agents.pool import AgentPool as _AgentPool
    from .agent_api import create_agent_router

    _agent_registry = _AgentRegistry()

    def _agent_event_handler(aid: str, evt: str, data: dict) -> None:
        """Handle agent pool events — broadcast via WebSocket + fire notifications."""
        _ws_hub.broadcast_sync("agents", evt, {"agent_id": aid, **data})

        if evt == "spawned":
            _web_notifications.agent_spawned(aid, data.get("role", ""))
        elif evt == "failed":
            _web_notifications.agent_error(aid, data.get("error", "Unknown error"))
        elif evt == "terminated":
            _web_notifications._push("info", f"Agent terminated: {aid}", "Agent has been stopped.", "info")
        elif evt == "auto_restarted":
            retries = data.get("retries", 0)
            _web_notifications._push("warning", f"Agent restarted: {aid}", f"Auto-restart attempt {retries}.", "warning")
        elif evt == "progress":
            # Check budget limits on progress updates
            try:
                agent = _agent_pool.get(aid)
                if agent:
                    atype = _agent_registry.get_type(agent.agent_type)
                    if atype.limits.max_cost_usd > 0 and agent.cost_usd > 0:
                        pct = (agent.cost_usd / atype.limits.max_cost_usd) * 100
                        if pct >= 80:
                            _web_notifications.budget_warning(aid, pct)
            except Exception:
                pass

    _agent_pool = _AgentPool(
        registry=_agent_registry,
        on_agent_event=_agent_event_handler,
    )

    app.state.agent_registry = _agent_registry
    app.state.agent_pool = _agent_pool

    # Reasoning store (shared between agents and collaboration)
    from ..collaboration.reasoning import ReasoningStore as _ReasoningStore
    _reasoning_store = _ReasoningStore()
    app.state.reasoning_store = _reasoning_store

    app.include_router(create_agent_router(
        get_pool=lambda: _agent_pool,
        get_registry=lambda: _agent_registry,
        get_reasoning_store=lambda: _reasoning_store,
    ))

    # ------------------------------------------------------------------
    # V2 Collaboration API (Feedback, Comments, HITL Modes, Timeline)
    # ------------------------------------------------------------------
    from ..collaboration.feedback import FeedbackStore as _FeedbackStore
    from ..collaboration.timeline import StateChangeStore as _StateChangeStore
    from .collaboration_api import create_collaboration_router

    _feedback_store = _FeedbackStore()
    app.state.feedback_store = _feedback_store

    _state_change_store = _StateChangeStore()
    app.state.state_change_store = _state_change_store

    from .users import UserStore as _UserStore, PresenceTracker as _PresenceTracker
    _user_store = _UserStore()
    _presence_tracker = _PresenceTracker()
    app.state.user_store = _user_store
    app.state.presence_tracker = _presence_tracker

    app.include_router(create_collaboration_router(
        get_feedback_store=lambda: _feedback_store,
        get_reasoning_store=lambda: _reasoning_store,
        get_user_store=lambda: _user_store,
        get_presence=lambda: _presence_tracker,
        get_state_change_store=lambda: _state_change_store,
        get_web_notifications=lambda: _web_notifications,
    ))

    return app


# Create default app instance
app = create_app()
