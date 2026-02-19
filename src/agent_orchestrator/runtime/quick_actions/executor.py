"""Quick action execution engine.

Dispatches quick action runs to either a shortcut (direct shell command)
or an agent fallback via the workers subsystem.
"""
from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...workers.run import run_worker
from ..domain.models import QuickActionRun, now_iso
from ..events.bus import EventBus
from ..storage.container import Container
from .shortcuts import load_shortcuts, match_prompt

_MAX_OUTPUT = 4000
_UNSAFE_SHELL_TOKENS = {"|", "&", ";", "<", ">", "`", "$", "\n", "\r"}


def _contains_unsafe_shell_tokens(command: str) -> bool:
    return any(token in command for token in _UNSAFE_SHELL_TOKENS)


class QuickActionExecutor:
    def __init__(self, container: Container, bus: EventBus) -> None:
        self._container = container
        self._bus = bus

    def execute(self, run: QuickActionRun) -> QuickActionRun:
        """Execute a quick action run. Blocks until completion."""
        # Mark running
        run.status = "running"
        run.started_at = now_iso()
        self._container.quick_actions.upsert(run)
        self._bus.emit(
            channel="quick_actions",
            event_type="quick_action.started",
            entity_id=run.id,
            payload={"status": run.status},
        )

        # Match prompt against shortcuts
        project_dir = self._container.project_dir
        rules = load_shortcuts(project_dir)
        match = match_prompt(run.prompt, rules, project_dir)

        if match.matched:
            return self._execute_shortcut(run, match)
        return self._execute_agent(run)

    def _execute_shortcut(self, run: QuickActionRun, match) -> QuickActionRun:
        run.kind = "shortcut"
        run.command = match.command
        project_dir = self._container.project_dir
        timeout = run.timeout_seconds or 120

        # Create log directory and files
        run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"
        run.stdout_path = str(stdout_log)
        run.stderr_path = str(stderr_log)
        self._container.quick_actions.upsert(run)

        try:
            if _contains_unsafe_shell_tokens(match.command):
                run.exit_code = -1
                run.result_summary = "Shortcut rejected: shell metacharacters are not allowed."
                run.status = "failed"
                raise RuntimeError("unsafe shortcut command")

            command_parts = shlex.split(match.command)
            if not command_parts:
                run.exit_code = -1
                run.result_summary = "Shortcut rejected: empty command."
                run.status = "failed"
                raise RuntimeError("empty shortcut command")

            with open(stdout_log, "wb") as stdout_fh, open(stderr_log, "wb") as stderr_fh:
                proc = subprocess.Popen(
                    command_parts,
                    shell=False,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    cwd=str(project_dir),
                )
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                    run.exit_code = -1
                    run.result_summary = f"Command timed out after {timeout} seconds"
                    run.status = "failed"
                    run.finished_at = now_iso()
                    self._container.quick_actions.upsert(run)
                    self._bus.emit(
                        channel="quick_actions",
                        event_type="quick_action.failed",
                        entity_id=run.id,
                        payload={"status": run.status, "exit_code": run.exit_code},
                    )
                    return run

            stdout_text = stdout_log.read_text(errors="replace").strip()
            stderr_text = stderr_log.read_text(errors="replace").strip()
            output = (stdout_text + "\n" + stderr_text).strip() if stderr_text else stdout_text
            run.exit_code = proc.returncode
            run.result_summary = output[:_MAX_OUTPUT] if output else "(no output)"
            run.status = "completed" if proc.returncode == 0 else "failed"
        except RuntimeError:
            # result_summary/status are already set for rejected shortcut commands.
            pass
        except Exception as exc:
            run.exit_code = -1
            run.result_summary = f"Execution error: {exc}"
            run.status = "failed"

        run.finished_at = now_iso()
        self._container.quick_actions.upsert(run)

        event_type = "quick_action.completed" if run.status == "completed" else "quick_action.failed"
        self._bus.emit(
            channel="quick_actions",
            event_type=event_type,
            entity_id=run.id,
            payload={"status": run.status, "exit_code": run.exit_code},
        )
        return run

    def _execute_agent(self, run: QuickActionRun) -> QuickActionRun:
        """Dispatch to the workers subsystem (codex / ollama)."""
        run.kind = "agent"
        timeout = run.timeout_seconds or 120
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, "implement")
            workers_cfg = cfg.get("workers") if isinstance(cfg, dict) else {}
            workers_cfg = workers_cfg if isinstance(workers_cfg, dict) else {}
            try:
                heartbeat_seconds = max(1, int(workers_cfg.get("heartbeat_seconds", 60)))
            except (TypeError, ValueError):
                heartbeat_seconds = 60
            try:
                heartbeat_grace_seconds = max(1, int(workers_cfg.get("heartbeat_grace_seconds", 240)))
            except (TypeError, ValueError):
                heartbeat_grace_seconds = 240
            if heartbeat_grace_seconds < heartbeat_seconds:
                heartbeat_grace_seconds = heartbeat_seconds

            available, reason = test_worker(spec)
            if not available:
                run.status = "failed"
                run.finished_at = now_iso()
                run.result_summary = f"No worker available: {reason}"
                self._container.quick_actions.upsert(run)
                self._bus.emit(
                    channel="quick_actions",
                    event_type="quick_action.failed",
                    entity_id=run.id,
                    payload={"status": run.status},
                )
                return run

            run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
            progress_path = run_dir / "progress.json"
            run.stdout_path = str(run_dir / "stdout.log")
            run.stderr_path = str(run_dir / "stderr.log")
            self._container.quick_actions.upsert(run)

            result = run_worker(
                spec=spec,
                prompt=run.prompt,
                project_dir=self._container.project_dir,
                run_dir=run_dir,
                timeout_seconds=timeout,
                heartbeat_seconds=heartbeat_seconds,
                heartbeat_grace_seconds=heartbeat_grace_seconds,
                progress_path=progress_path,
            )

            if result.timed_out:
                run.exit_code = result.exit_code
                run.status = "failed"
                run.result_summary = "Worker timed out"
            else:
                output = result.response_text
                if not output and result.stdout_path:
                    try:
                        output = Path(result.stdout_path).read_text(errors="replace")
                    except Exception:
                        output = ""
                run.exit_code = result.exit_code
                run.status = "completed" if result.exit_code == 0 else "failed"
                run.result_summary = (output[:_MAX_OUTPUT] if output else "(no output)")

        except ValueError as exc:
            run.status = "failed"
            run.result_summary = f"No worker configured: {exc}"
        except Exception as exc:
            run.status = "failed"
            run.result_summary = f"Agent error: {exc}"

        run.finished_at = now_iso()
        self._container.quick_actions.upsert(run)
        event_type = "quick_action.completed" if run.status == "completed" else "quick_action.failed"
        self._bus.emit(
            channel="quick_actions",
            event_type=event_type,
            entity_id=run.id,
            payload={"status": run.status, "exit_code": run.exit_code},
        )
        return run
