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
    ExecTaskRequest,
    ExecTaskResponse,
    FileChange,
    FileReviewRequest,
    LoginRequest,
    LoginResponse,
    PhaseInfo,
    ProjectInfo,
    ProjectStatus,
    RunDetail,
    RunInfo,
    RunMetrics,
    SendMessageRequest,
    StartRunRequest,
    StartRunResponse,
    TaskInfo,
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

            # Add auto-approve flags (they're False by default in CLI, so we only set them if True)
            # Note: The CLI doesn't have these exact flags, so we'll need to use interactive mode
            # and rely on the approval gates being configured
            if not (request.auto_approve_plans and request.auto_approve_changes and request.auto_approve_commits):
                # If not all auto-approve, we might want interactive mode
                # But for now, let's just run without interactive mode
                pass

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
    from .task_api import create_task_router

    _engine_cache: dict[str, _TaskEngine] = {}

    def _get_task_engine(project_dir_param: Optional[str] = None) -> _TaskEngine:
        proj = _get_project_dir(project_dir_param)
        state_dir = proj / ".prd_runner"
        state_dir.mkdir(parents=True, exist_ok=True)
        key = str(state_dir)
        if key not in _engine_cache:
            _engine_cache[key] = _TaskEngine(state_dir)
        return _engine_cache[key]

    app.include_router(create_task_router(_get_task_engine))

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
