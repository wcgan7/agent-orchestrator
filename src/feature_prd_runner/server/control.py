"""Run control operations for the web dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock


class ControlError(Exception):
    """Error during control operation."""

    pass


class RunController:
    """Control running tasks via the web API."""

    def __init__(self, project_dir: Path):
        """Initialize controller with project directory.

        Args:
            project_dir: Path to the project root directory.
        """
        self.project_dir = project_dir
        self.state_dir = project_dir / ".prd_runner"
        self.lock_path = self.state_dir / ".lock"
        self.run_state_path = self.state_dir / "run_state.yaml"
        self.task_queue_path = self.state_dir / "task_queue.yaml"
        self.runs_dir = self.state_dir / "runs"

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file safely.

        Args:
            path: Path to YAML file.

        Returns:
            Parsed YAML data.

        Raises:
            ControlError: If file cannot be loaded.
        """
        if not path.exists():
            return {}

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except Exception as e:
            raise ControlError(f"Failed to load {path.name}: {e}")

    def _save_yaml(self, path: Path, data: dict[str, Any]) -> None:
        """Save data to YAML file safely.

        Args:
            path: Path to YAML file.
            data: Data to save.

        Raises:
            ControlError: If file cannot be saved.
        """
        try:
            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ControlError(f"Failed to save {path.name}: {e}")

    def _now_iso(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO formatted timestamp string.
        """
        return datetime.now(timezone.utc).isoformat()

    def _find_task(self, tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        """Find task by ID.

        Args:
            tasks: List of task dictionaries.
            task_id: Task identifier.

        Returns:
            Task dictionary or None if not found.
        """
        for task in tasks:
            if task.get("id") == task_id:
                return task
        return None

    def _is_run_active(self) -> bool:
        """Check if a run is currently active.

        Returns:
            True if a run is active, False otherwise.
        """
        if not self.runs_dir.exists():
            return False

        # Check for recent run directories
        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return False

        # Find most recent run
        latest_run = max(run_dirs, key=lambda d: d.name)
        progress_path = latest_run / "progress.json"

        if not progress_path.exists():
            return False

        import json

        try:
            with open(progress_path) as f:
                progress = json.load(f)

            # Check heartbeat - if within last 60 seconds, consider active
            heartbeat = progress.get("heartbeat")
            if heartbeat:
                from datetime import datetime

                heartbeat_time = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = (now - heartbeat_time).total_seconds()
                return delta < 60  # Active if heartbeat within last minute

        except Exception:
            pass

        return False

    def retry_task(self, task_id: str, step: str = "plan_impl") -> dict[str, Any]:
        """Retry a task from a specific step.

        Args:
            task_id: Task identifier.
            step: Step to retry from (plan_impl, implement, verify, review, commit).

        Returns:
            Result dictionary with success status.

        Raises:
            ControlError: If operation fails.
        """
        if not self.state_dir.exists():
            raise ControlError(f"No state directory found at {self.state_dir}")

        with FileLock(self.lock_path):
            # Check if run is active
            if self._is_run_active():
                raise ControlError(
                    "Cannot retry while run is active. Stop the run first or use --force."
                )

            # Load task queue
            queue = self._load_yaml(self.task_queue_path)
            tasks = queue.get("tasks", [])

            # Find target task
            target = self._find_task(tasks, task_id)
            if not target:
                raise ControlError(f"Task not found: {task_id}")

            # Reset task state
            target["lifecycle"] = "ready"
            target["step"] = step
            target["status"] = step
            target["prompt_mode"] = None
            target["last_error"] = None
            target["last_error_type"] = None
            target["block_reason"] = None
            target["human_blocking_issues"] = []
            target["human_next_steps"] = []
            target["last_updated_at"] = self._now_iso()
            target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1

            # Add context
            context = target.get("context", [])
            context.append(f"Manual retry requested (step={step}).")
            target["context"] = context

            # Save updated queue
            queue["tasks"] = tasks
            queue["updated_at"] = target["last_updated_at"]
            self._save_yaml(self.task_queue_path, queue)

        return {
            "success": True,
            "task_id": task_id,
            "step": step,
            "message": f"Task {task_id} queued for retry at step {step}",
        }

    def skip_step(self, task_id: str, step: str | None = None) -> dict[str, Any]:
        """Skip current step and move to next.

        Args:
            task_id: Task identifier.
            step: Specific step to skip (optional, uses current step if not provided).

        Returns:
            Result dictionary with success status.

        Raises:
            ControlError: If operation fails.
        """
        # Step order
        step_order = ["plan_impl", "implement", "verify", "review", "commit"]

        if not self.state_dir.exists():
            raise ControlError(f"No state directory found at {self.state_dir}")

        with FileLock(self.lock_path):
            # Check if run is active
            if self._is_run_active():
                raise ControlError(
                    "Cannot skip step while run is active. Stop the run first or use --force."
                )

            # Load task queue
            queue = self._load_yaml(self.task_queue_path)
            tasks = queue.get("tasks", [])

            # Find target task
            target = self._find_task(tasks, task_id)
            if not target:
                raise ControlError(f"Task not found: {task_id}")

            # Determine current and next step
            current_step = step if step else target.get("step", "plan_impl")

            try:
                current_idx = step_order.index(current_step)
                if current_idx >= len(step_order) - 1:
                    raise ControlError(f"Cannot skip {current_step} - it's the last step")
                next_step = step_order[current_idx + 1]
            except ValueError:
                raise ControlError(f"Unknown step: {current_step}")

            # Update task to next step
            target["lifecycle"] = "ready"
            target["step"] = next_step
            target["status"] = next_step
            target["last_updated_at"] = self._now_iso()

            # Add context
            context = target.get("context", [])
            context.append(f"Skipped step {current_step}, moving to {next_step}.")
            target["context"] = context

            # Save updated queue
            queue["tasks"] = tasks
            queue["updated_at"] = target["last_updated_at"]
            self._save_yaml(self.task_queue_path, queue)

        return {
            "success": True,
            "task_id": task_id,
            "skipped_step": current_step,
            "next_step": next_step,
            "message": f"Skipped {current_step}, moving to {next_step}",
        }

    def resume_task(self, task_id: str, step: str | None = None) -> dict[str, Any]:
        """Resume a blocked task.

        Args:
            task_id: Task identifier.
            step: Step to resume from (optional, keeps current step if not provided).

        Returns:
            Result dictionary with success status.

        Raises:
            ControlError: If operation fails.
        """
        if not self.state_dir.exists():
            raise ControlError(f"No state directory found at {self.state_dir}")

        with FileLock(self.lock_path):
            # Load task queue
            queue = self._load_yaml(self.task_queue_path)
            tasks = queue.get("tasks", [])

            # Find target task
            target = self._find_task(tasks, task_id)
            if not target:
                raise ControlError(f"Task not found: {task_id}")

            # Update task state to ready
            if step:
                target["step"] = step
                target["status"] = step

            target["lifecycle"] = "ready"
            target["block_reason"] = None
            target["human_blocking_issues"] = []
            target["human_next_steps"] = []
            target["last_error"] = None
            target["last_error_type"] = None
            target["last_updated_at"] = self._now_iso()
            target["manual_resume_attempts"] = int(target.get("manual_resume_attempts", 0)) + 1

            # Add context
            context = target.get("context", [])
            context.append(f"Manual resume requested{' at step ' + step if step else ''}.")
            target["context"] = context

            # Save updated queue
            queue["tasks"] = tasks
            queue["updated_at"] = target["last_updated_at"]
            self._save_yaml(self.task_queue_path, queue)

        return {
            "success": True,
            "task_id": task_id,
            "step": target["step"],
            "message": f"Task {task_id} resumed",
        }

    def stop_run(self) -> dict[str, Any]:
        """Stop the currently active run.

        This creates a stop signal that the runner should detect.

        Returns:
            Result dictionary with success status.

        Raises:
            ControlError: If operation fails.
        """
        if not self.state_dir.exists():
            raise ControlError(f"No state directory found at {self.state_dir}")

        with FileLock(self.lock_path):
            # Update run state to request stop
            run_state = self._load_yaml(self.run_state_path)
            run_state["stop_requested"] = True
            run_state["stop_requested_at"] = self._now_iso()
            self._save_yaml(self.run_state_path, run_state)

        return {
            "success": True,
            "message": "Stop signal sent to running process",
        }


def create_controller(project_dir: Path) -> RunController:
    """Create a run controller instance.

    Args:
        project_dir: Path to the project directory.

    Returns:
        RunController instance.
    """
    return RunController(project_dir)
