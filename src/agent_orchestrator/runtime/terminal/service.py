"""Terminal session orchestration and PTY I/O persistence."""

from __future__ import annotations

import fcntl
import logging
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..domain.models import TerminalSession, now_iso
from ..events.bus import EventBus
from ..storage.container import Container

logger = logging.getLogger(__name__)

_MAX_LOG_BYTES = 50 * 1024 * 1024
_READ_CHUNK_BYTES = 8192


@dataclass
class _LiveTerminal:
    session_id: str
    proc: subprocess.Popen[bytes]
    master_fd: int
    output_log_path: Path
    audit_log_path: Path
    lock: threading.RLock


class TerminalService:
    """Manage interactive PTY sessions and persist their logs/metadata."""
    def __init__(self, container: Container, bus: EventBus) -> None:
        self._container = container
        self._bus = bus
        self._lock = threading.RLock()
        self._live: dict[str, _LiveTerminal] = {}
        self._recover_stale_sessions()

    def _recover_stale_sessions(self) -> None:
        for session in self._container.terminal_sessions.list():
            if session.status in {"starting", "running"}:
                session.status = "error"
                session.finished_at = now_iso()
                session.last_error = "Terminal service restarted; session no longer attached."
                self._container.terminal_sessions.upsert(session)

    def _resolve_shell(self, requested_shell: Optional[str]) -> str:
        shell = (requested_shell or "").strip()
        if shell:
            return shell
        env_shell = str(os.environ.get("SHELL") or "").strip()
        if env_shell:
            return env_shell
        return "/bin/zsh" if os.name == "posix" and Path("/bin/zsh").exists() else "/bin/bash"

    def _session_logs_dir(self, session_id: str) -> Path:
        path = self._container.state_root / "terminal_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path / session_id

    def _trim_file_if_needed(self, path: Path, max_bytes: int = _MAX_LOG_BYTES) -> None:
        try:
            if not path.exists():
                return
            size = path.stat().st_size
            if size <= max_bytes:
                return
            keep = max_bytes // 2
            with path.open("rb") as fh:
                fh.seek(max(0, size - keep))
                data = fh.read()
            with path.open("wb") as fh:
                fh.write(data)
        except Exception:
            logger.debug("Failed to trim terminal log file %s", path, exc_info=True)

    def _append_bytes(self, path: Path, data: bytes) -> int:
        self._trim_file_if_needed(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(data)
        return path.stat().st_size if path.exists() else 0

    def _set_winsize(self, fd: int, rows: int, cols: int) -> None:
        winsize = struct.pack("HHHH", max(rows, 2), max(cols, 2), 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    def _find_active_session(self, project_id: str) -> Optional[TerminalSession]:
        for session in reversed(self._container.terminal_sessions.list()):
            if session.project_id != project_id:
                continue
            if session.status in {"starting", "running"}:
                return session
        return None

    def _emit(self, event_type: str, session: TerminalSession, payload: dict[str, object]) -> None:
        data = {"session_id": session.id, "status": session.status, **payload}
        try:
            self._bus.emit(
                channel="terminal",
                event_type=event_type,
                entity_id=session.id,
                payload=data,
            )
        except Exception:
            logger.debug("Failed to emit terminal event %s", event_type, exc_info=True)

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """Fetch a terminal session by id from persistent storage."""
        return self._container.terminal_sessions.get(session_id)

    def get_active_session(self, project_id: Optional[str] = None) -> Optional[TerminalSession]:
        """Fetch the active terminal session for a project, if one is still live."""
        pid = project_id or self._container.project_id
        with self._lock:
            session = self._find_active_session(pid)
            if not session:
                return None
            live = self._live.get(session.id)
            if live and live.proc.poll() is None:
                return session
            # stale running record; mark exited
            if session.status in {"starting", "running"}:
                session.status = "exited"
                session.finished_at = now_iso()
                session.exit_code = live.proc.poll() if live else session.exit_code
                self._container.terminal_sessions.upsert(session)
            return None

    def _reader_loop(self, session_id: str) -> None:
        with self._lock:
            live = self._live.get(session_id)
        if not live:
            return
        while True:
            try:
                readable, _, _ = select.select([live.master_fd], [], [], 0.2)
            except Exception:
                readable = []
            if readable:
                try:
                    data = os.read(live.master_fd, _READ_CHUNK_BYTES)
                except OSError:
                    data = b""
                if data:
                    with live.lock:
                        output_offset = self._append_bytes(live.output_log_path, data)
                        self._append_bytes(live.audit_log_path, b"[OUT] " + data)
                    text = data.decode("utf-8", errors="replace")
                    session = self._container.terminal_sessions.get(session_id)
                    if session:
                        self._emit("terminal.output", session, {"chunk": text, "offset": output_offset})
            if live.proc.poll() is not None:
                break
        exit_code = live.proc.poll()
        session = self._container.terminal_sessions.get(session_id)
        if session:
            session.status = "exited"
            session.finished_at = now_iso()
            session.exit_code = exit_code
            self._container.terminal_sessions.upsert(session)
            self._emit("terminal.exited", session, {"exit_code": exit_code})
        with self._lock:
            self._live.pop(session_id, None)
        try:
            os.close(live.master_fd)
        except OSError:
            pass

    def start_session(self, *, shell: Optional[str] = None, cols: int = 120, rows: int = 36) -> TerminalSession:
        """Start a new shell session or return the current active one."""
        with self._lock:
            active = self.get_active_session(self._container.project_id)
            if active:
                return active
            resolved_shell = self._resolve_shell(shell)
            logs_base = self._container.state_root / "terminal_logs"
            logs_base.mkdir(parents=True, exist_ok=True)
            session = TerminalSession(
                project_id=self._container.project_id,
                status="starting",
                shell=resolved_shell,
                cwd=str(self._container.project_dir),
                started_at=now_iso(),
                cols=max(cols, 2),
                rows=max(rows, 2),
            )
            log_stem = self._session_logs_dir(session.id)
            session.audit_log_path = str(log_stem.with_suffix(".audit.log"))
            session.output_log_path = str(log_stem.with_suffix(".output.log"))
            self._container.terminal_sessions.upsert(session)

            master_fd, slave_fd = pty.openpty()
            env = dict(os.environ)
            env.setdefault("TERM", "xterm-256color")
            try:
                proc = subprocess.Popen(
                    [resolved_shell],
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=str(self._container.project_dir),
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
            except Exception as exc:
                os.close(master_fd)
                os.close(slave_fd)
                session.status = "error"
                session.finished_at = now_iso()
                session.last_error = f"Failed to start shell: {exc}"
                self._container.terminal_sessions.upsert(session)
                self._emit("terminal.error", session, {"error": session.last_error})
                return session
            finally:
                try:
                    os.close(slave_fd)
                except OSError:
                    pass

            self._set_winsize(master_fd, session.rows, session.cols)
            session.status = "running"
            session.pid = proc.pid
            self._container.terminal_sessions.upsert(session)
            live = _LiveTerminal(
                session_id=session.id,
                proc=proc,
                master_fd=master_fd,
                output_log_path=Path(session.output_log_path or ""),
                audit_log_path=Path(session.audit_log_path or ""),
                lock=threading.RLock(),
            )
            self._live[session.id] = live
            self._emit("terminal.started", session, {"pid": session.pid, "shell": session.shell, "cwd": session.cwd})

            thread = threading.Thread(target=self._reader_loop, args=(session.id,), daemon=True)
            thread.start()
            return session

    def write_input(self, *, session_id: str, data: str) -> TerminalSession:
        """Write user input to a live terminal session."""
        if not data:
            session = self.get_session(session_id)
            if not session:
                raise ValueError("Terminal session not found")
            return session
        with self._lock:
            session = self._container.terminal_sessions.get(session_id)
            if not session:
                raise ValueError("Terminal session not found")
            live = self._live.get(session_id)
            if not live or live.proc.poll() is not None:
                raise ValueError("Terminal session is not active")
            raw = data.encode("utf-8", errors="replace")
            os.write(live.master_fd, raw)
            with live.lock:
                self._append_bytes(live.audit_log_path, b"[IN] " + raw)
            return session

    def resize(self, *, session_id: str, cols: int, rows: int) -> TerminalSession:
        """Resize the PTY for an active session."""
        with self._lock:
            session = self._container.terminal_sessions.get(session_id)
            if not session:
                raise ValueError("Terminal session not found")
            live = self._live.get(session_id)
            if not live or live.proc.poll() is not None:
                raise ValueError("Terminal session is not active")
            session.cols = max(cols, 2)
            session.rows = max(rows, 2)
            self._set_winsize(live.master_fd, session.rows, session.cols)
            self._container.terminal_sessions.upsert(session)
            return session

    def stop_session(self, *, session_id: str, signal_name: str = "TERM") -> TerminalSession:
        """Send a termination signal to an active terminal session."""
        with self._lock:
            session = self._container.terminal_sessions.get(session_id)
            if not session:
                raise ValueError("Terminal session not found")
            live = self._live.get(session_id)
            if not live:
                return session
            sig = signal.SIGKILL if signal_name.upper() == "KILL" else signal.SIGTERM
            try:
                os.killpg(live.proc.pid, sig)
            except ProcessLookupError:
                pass
            except Exception:
                logger.debug("Failed to stop terminal session %s", session_id, exc_info=True)
            return session

    def read_output(self, *, session_id: str, offset: int = 0, max_bytes: int = 65536) -> tuple[str, int]:
        """Read session output from the persisted log starting at byte offset."""
        session = self._container.terminal_sessions.get(session_id)
        if not session:
            raise ValueError("Terminal session not found")
        # Always read from the canonical session log location under state_root,
        # instead of trusting persisted path fields.
        output_path = self._session_logs_dir(session.id).with_suffix(".output.log")
        if not output_path.exists() or not output_path.is_file():
            return "", 0
        size = output_path.stat().st_size
        if offset < 0:
            offset = 0
        if offset > size:
            offset = size
        read_size = max(1, min(max_bytes, 262144))
        with output_path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(read_size)
            new_offset = fh.tell()
        return data.decode("utf-8", errors="replace"), new_offset
