"""Quick action execution engine.

Dispatches quick action runs to either a shortcut (direct shell command)
or an agent fallback via the workers subsystem.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...workers.run import run_worker
from ..domain.models import QuickActionRun, now_iso
from ..events.bus import EventBus
from ..storage.container import V3Container
from .shortcuts import load_shortcuts, match_prompt

_MAX_OUTPUT = 4000


class QuickActionExecutor:
    def __init__(self, container: V3Container, bus: EventBus) -> None:
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

        try:
            result = subprocess.run(
                match.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(project_dir),
            )
            output = (result.stdout + result.stderr).strip()
            run.exit_code = result.returncode
            run.result_summary = output[:_MAX_OUTPUT] if output else "(no output)"
            run.status = "completed" if result.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            run.exit_code = -1
            run.result_summary = "Command timed out after 120 seconds"
            run.status = "failed"
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
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex")
            spec = resolve_worker_for_step(runtime, "implement")

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

            run_dir = Path(tempfile.mkdtemp(dir=str(self._container.v3_root)))
            progress_path = run_dir / "progress.json"

            result = run_worker(
                spec=spec,
                prompt=run.prompt,
                project_dir=self._container.project_dir,
                run_dir=run_dir,
                timeout_seconds=120,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=15,
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
