"""Terminal route registration for the runtime API."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from .deps import RouteDeps
from . import router_impl as impl


StartTerminalSessionRequest = impl.StartTerminalSessionRequest
StopTerminalSessionRequest = impl.StopTerminalSessionRequest
TerminalInputRequest = impl.TerminalInputRequest
TerminalResizeRequest = impl.TerminalResizeRequest


def register_terminal_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register managed terminal session routes."""
    @router.post("/terminal/session")
    async def start_terminal_session(
        body: StartTerminalSessionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Start a managed terminal session for the current project.
        
        Args:
            body: Terminal session configuration payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created session and project identifier.
        """
        container, _, terminal = deps.terminal_ctx(project_dir)
        session = terminal.start_session(shell=body.shell, cols=body.cols or 120, rows=body.rows or 36)
        return {"session": session.to_dict(), "project_id": container.project_id}

    @router.get("/terminal/session")
    async def get_terminal_session(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the currently active terminal session for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the active session, if any.
        """
        container, _, terminal = deps.terminal_ctx(project_dir)
        session = terminal.get_active_session(container.project_id)
        return {"session": session.to_dict() if session else None}

    @router.post("/terminal/session/{session_id}/input")
    async def write_terminal_input(
        session_id: str,
        body: TerminalInputRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Write input bytes to a managed terminal session.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Input payload containing data to write.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = deps.terminal_ctx(project_dir)
        try:
            session = terminal.write_input(session_id=session_id, data=body.data)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.post("/terminal/session/{session_id}/resize")
    async def resize_terminal_session(
        session_id: str,
        body: TerminalResizeRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Resize a managed terminal session pseudo-terminal.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Resize payload with column and row counts.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = deps.terminal_ctx(project_dir)
        try:
            session = terminal.resize(session_id=session_id, cols=body.cols, rows=body.rows)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.post("/terminal/session/{session_id}/stop")
    async def stop_terminal_session(
        session_id: str,
        body: StopTerminalSessionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Stop a managed terminal session with an optional signal.
        
        Args:
            session_id: Identifier of the terminal session.
            body: Stop payload containing the signal name.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the stopped session state.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = deps.terminal_ctx(project_dir)
        try:
            session = terminal.stop_session(session_id=session_id, signal_name=body.signal)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session": session.to_dict()}

    @router.get("/terminal/session/{session_id}/logs")
    async def get_terminal_session_logs(
        session_id: str,
        project_dir: Optional[str] = Query(None),
        offset: int = Query(0),
        max_bytes: int = Query(65536),
    ) -> dict[str, Any]:
        """Read terminal session output from a byte offset.
        
        Args:
            session_id: Identifier of the terminal session.
            project_dir: Optional project directory used to resolve runtime state.
            offset: Starting byte offset in the output stream.
            max_bytes: Maximum number of bytes to read.
        
        Returns:
            A payload containing output data, new offset, and session status.
        
        Raises:
            HTTPException: If the terminal session cannot be found.
        """
        _, _, terminal = deps.terminal_ctx(project_dir)
        try:
            output, new_offset = terminal.read_output(session_id=session_id, offset=offset, max_bytes=max_bytes)
            session = terminal.get_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "output": output,
            "offset": new_offset,
            "status": session.status if session else "unknown",
            "finished_at": session.finished_at if session else None,
        }

