from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .io_utils import _heartbeat_from_progress, _read_log_tail
from .utils import _now_iso


def _stream_pipe(pipe: Any, file_path: Path, label: str, to_stderr: bool, quiet: bool = True) -> None:
    prefix = f"[codex {label}] "
    with open(file_path, "w") as handle:
        for line in iter(pipe.readline, ""):
            handle.write(line)
            handle.flush()
            if quiet:
                continue
            if to_stderr:
                sys.stderr.write(prefix + line)
                sys.stderr.flush()
            else:
                sys.stdout.write(prefix + line)
                sys.stdout.flush()
    try:
        pipe.close()
    except Exception:
        pass


def _latest_mtime(paths: list[Path]) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        stamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
        if latest is None or stamp > latest:
            latest = stamp
    return latest


def _run_codex_worker(
    command: str,
    prompt: str,
    project_dir: Path,
    run_dir: Path,
    timeout_seconds: int,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
    progress_path: Path,
    expected_run_id: Optional[str] = None,
    on_spawn: Optional[Callable[[int], None]] = None,
) -> dict[str, Any]:
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    start_wall = datetime.now(timezone.utc)
    try:
        formatted_command = command.format(
            prompt_file=str(prompt_path),
            project_dir=str(project_dir),
            run_dir=str(run_dir),
            prompt=prompt,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in codex command: {exc}") from exc

    command_parts = shlex.split(formatted_command)
    uses_prompt_placeholder = "{prompt_file}" in command or "{prompt}" in command
    expects_stdin = "-" in command_parts
    if not uses_prompt_placeholder and not expects_stdin:
        raise ValueError(
            "Codex command must include {prompt_file}, {prompt}, or '-' to accept stdin input."
        )

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    start_time = time.monotonic()
    start_iso = _now_iso()
    timed_out = False
    no_heartbeat = False
    last_heartbeat = None

    process = subprocess.Popen(
        command_parts,
        cwd=project_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if on_spawn:
        try:
            on_spawn(process.pid)
        except Exception:
            pass

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, stdout_path, "stdout", False),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, stderr_path, "stderr", True),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    if not uses_prompt_placeholder and expects_stdin:
        if process.stdin:
            try:
                process.stdin.write(prompt)
                process.stdin.flush()
                process.stdin.close()
            except BrokenPipeError:
                pass

    poll_interval = max(5, min(heartbeat_seconds // 2, 30))
    heartbeat_tolerance = timedelta(seconds=poll_interval)

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break

        heartbeat = _heartbeat_from_progress(progress_path, expected_run_id)
        if heartbeat and heartbeat >= (start_wall - heartbeat_tolerance):
            last_heartbeat = heartbeat

        # Treat new stdout/stderr output as liveness even if progress heartbeats
        # are delayed or skipped by the worker.
        log_activity = _latest_mtime([stdout_path, stderr_path])

        now = datetime.now(timezone.utc)
        last_activity = start_wall
        if last_heartbeat and last_heartbeat > last_activity:
            last_activity = last_heartbeat
        if log_activity and log_activity >= (start_wall - heartbeat_tolerance) and log_activity > last_activity:
            last_activity = log_activity

        age = (now - last_activity).total_seconds()
        if age > heartbeat_grace_seconds:
            no_heartbeat = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break

        try:
            process.wait(timeout=poll_interval)
            break
        except subprocess.TimeoutExpired:
            continue

    exit_code = process.poll()
    if exit_code is None:
        exit_code = -1

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    end_iso = _now_iso()
    runtime_seconds = int(time.monotonic() - start_time)

    return {
        "command": formatted_command,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "no_heartbeat": no_heartbeat,
        "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
    }


def _run_command(
    command: str,
    project_dir: Path,
    log_path: Path,
    *,
    timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with open(log_path, "w") as handle:
        try:
            result = subprocess.run(
                command,
                cwd=project_dir,
                shell=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            handle.write(f"\n[runner] Command timed out after {timeout_seconds}s\n")
            return {
                "command": command,
                "exit_code": 124,
                "log_path": str(log_path),
                "timed_out": True,
            }
    return {
        "command": command,
        "exit_code": result.returncode,
        "log_path": str(log_path),
        "timed_out": timed_out,
    }


def _capture_test_result_snapshot(
    *,
    command: str,
    exit_code: int,
    log_path: Path,
    max_tail_chars: int = 4000,
) -> dict[str, Any]:
    return {
        "command": command,
        "exit_code": int(exit_code),
        "log_path": str(log_path),
        "log_tail": _read_log_tail(log_path, max_chars=max_tail_chars),
        "captured_at": _now_iso(),
    }
