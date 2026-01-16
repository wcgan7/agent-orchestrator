"""Enhanced error analysis and debugging tools.

This module provides comprehensive error analysis, debugging capabilities,
and rich error formatting for better troubleshooting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .constants import STATE_DIR_NAME
from .io_utils import _load_data


@dataclass
class ErrorReport:
    """Comprehensive error analysis report."""

    task_id: str
    error_type: str
    error_detail: str
    root_cause: Optional[str] = None
    files_involved: list[str] = None
    suggested_actions: list[dict[str, str]] = None
    quick_fixes: list[dict[str, str]] = None
    severity: str = "error"  # error, warning, critical

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.files_involved is None:
            self.files_involved = []
        if self.suggested_actions is None:
            self.suggested_actions = []
        if self.quick_fixes is None:
            self.quick_fixes = []


@dataclass
class StateSnapshot:
    """Task state snapshot for inspection."""

    task_id: str
    lifecycle: str
    step: str
    status: str
    attempts: int
    last_error: Optional[str]
    last_error_type: Optional[str]
    context: list[str]
    metadata: dict[str, Any]


class ErrorAnalyzer:
    """Analyze and explain errors with actionable guidance."""

    def __init__(self, project_dir: Path):
        """Initialize error analyzer.

        Args:
            project_dir: Project directory path.
        """
        self.project_dir = project_dir.resolve()
        self.state_dir = self.project_dir / STATE_DIR_NAME
        self.console = Console()

    def analyze_error(
        self,
        task_id: str,
        error_type: str,
        error_detail: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ErrorReport:
        """Generate comprehensive error report.

        Args:
            task_id: Task identifier.
            error_type: Type of error.
            error_detail: Detailed error message.
            context: Additional context information.

        Returns:
            ErrorReport with analysis and suggestions.
        """
        context = context or {}

        report = ErrorReport(
            task_id=task_id,
            error_type=error_type,
            error_detail=error_detail,
        )

        # Analyze based on error type
        if error_type == "test_failed":
            self._analyze_test_failure(report, context)
        elif error_type == "worker_failed":
            self._analyze_worker_failure(report, context)
        elif error_type == "allowlist_violation":
            self._analyze_allowlist_violation(report, context)
        elif error_type == "no_progress":
            self._analyze_no_progress(report, context)
        elif error_type == "review_failed":
            self._analyze_review_failure(report, context)
        elif error_type == "git_push_failed":
            self._analyze_git_failure(report, context)
        else:
            self._analyze_generic_error(report, context)

        return report

    def _analyze_test_failure(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze test failure errors."""
        report.root_cause = "Tests failed during verification."

        # Extract test info from context
        failed_tests = context.get("failed_tests", [])
        test_output = context.get("test_output", "")

        if failed_tests:
            report.files_involved = failed_tests[:10]  # Limit to first 10

        report.suggested_actions = [
            {
                "action": "Review test failures",
                "command": f"feature-prd-runner logs {report.task_id} --step verify",
            },
            {
                "action": "Retry after reviewing",
                "command": f"feature-prd-runner retry {report.task_id}",
            },
            {
                "action": "Skip verification (not recommended)",
                "command": f"feature-prd-runner skip-step {report.task_id} --step verify",
            },
        ]

        report.quick_fixes = [
            {"label": "View full test output", "command": f"cat .prd_runner/runs/*/tests_{report.task_id}.log"},
            {"label": "Debug interactively", "command": f"feature-prd-runner debug {report.task_id}"},
        ]

    def _analyze_worker_failure(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze worker failure errors."""
        report.root_cause = "The Codex worker failed to complete the task."

        run_id = context.get("run_id")
        step = context.get("step", "unknown")

        report.suggested_actions = [
            {
                "action": "Check worker logs",
                "command": f"feature-prd-runner logs {report.task_id} --step {step}",
            },
            {
                "action": "Retry the task",
                "command": f"feature-prd-runner retry {report.task_id}",
            },
            {
                "action": "Resume with different step",
                "command": f"feature-prd-runner resume {report.task_id} --step implement",
            },
        ]

        if run_id:
            report.quick_fixes = [
                {"label": "View stdout", "command": f"cat .prd_runner/runs/{run_id}/stdout.log"},
                {"label": "View stderr", "command": f"cat .prd_runner/runs/{run_id}/stderr.log"},
            ]

    def _analyze_allowlist_violation(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze allowlist violation errors."""
        disallowed = context.get("disallowed_files", [])

        report.root_cause = f"Worker modified {len(disallowed)} file(s) not in the allowlist."
        report.files_involved = disallowed

        report.suggested_actions = [
            {
                "action": "Review disallowed changes",
                "command": f"git diff {' '.join(disallowed[:5])}",
            },
            {
                "action": "Rerun plan_impl to update allowlist",
                "command": f"feature-prd-runner rerun-step {report.task_id} --step plan_impl",
            },
            {
                "action": "Force retry (careful!)",
                "command": f"feature-prd-runner retry {report.task_id} --force",
            },
        ]

    def _analyze_no_progress(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze no progress errors."""
        report.root_cause = "Worker completed but introduced no code changes."
        report.severity = "warning"

        report.suggested_actions = [
            {
                "action": "Check if task is actually complete",
                "command": "git status",
            },
            {
                "action": "Retry with more guidance",
                "command": f"feature-prd-runner retry {report.task_id}",
            },
            {
                "action": "Skip if task is not needed",
                "command": f"feature-prd-runner skip-step {report.task_id} --step implement",
            },
        ]

    def _analyze_review_failure(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze review failure errors."""
        issues = context.get("issues", [])

        report.root_cause = f"Code review found {len(issues)} blocking issue(s)."
        report.severity = "warning" if len(issues) < 3 else "error"

        # Extract files from issues
        for issue in issues:
            if "file" in issue:
                report.files_involved.append(issue["file"])

        report.suggested_actions = [
            {
                "action": "Review detailed issues",
                "command": f"cat .prd_runner/artifacts/review_{report.task_id}.json",
            },
            {
                "action": "Retry to address issues",
                "command": f"feature-prd-runner retry {report.task_id}",
            },
        ]

    def _analyze_git_failure(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze git push failure errors."""
        report.root_cause = "Failed to push changes to remote repository."

        report.suggested_actions = [
            {
                "action": "Check git remote configuration",
                "command": "git remote -v",
            },
            {
                "action": "Verify authentication",
                "command": "git fetch",
            },
            {
                "action": "Retry commit step",
                "command": f"feature-prd-runner rerun-step {report.task_id} --step commit",
            },
        ]

    def _analyze_generic_error(
        self,
        report: ErrorReport,
        context: dict[str, Any],
    ) -> None:
        """Analyze generic/unknown errors."""
        report.root_cause = f"Task failed with error type: {report.error_type}"

        report.suggested_actions = [
            {
                "action": "View task details",
                "command": f"feature-prd-runner inspect {report.task_id}",
            },
            {
                "action": "View event history",
                "command": f"feature-prd-runner trace {report.task_id}",
            },
            {
                "action": "Retry task",
                "command": f"feature-prd-runner retry {report.task_id}",
            },
        ]

    def format_error_report(self, report: ErrorReport, verbose: bool = False) -> str:
        """Format error report as rich text.

        Args:
            report: Error report to format.
            verbose: Include additional details.

        Returns:
            Formatted error message string.
        """
        console = Console(record=True, width=80)

        # Header
        severity_colors = {
            "critical": "bright_red",
            "error": "red",
            "warning": "yellow",
        }
        color = severity_colors.get(report.severity, "red")

        console.print()
        console.print(f"[{color}]❌ Error: {report.task_id}[/{color}]", style="bold")
        console.print("━" * 80, style=color)

        # Error type and detail
        console.print(f"\n[bold]Error Type:[/bold] {report.error_type}")
        if len(report.error_detail) > 200 and not verbose:
            detail = report.error_detail[:200] + "..."
        else:
            detail = report.error_detail
        console.print(f"[dim]{detail}[/dim]")

        # Root cause
        if report.root_cause:
            console.print(f"\n[bold]Root Cause:[/bold]")
            console.print(f"  {report.root_cause}")

        # Files involved
        if report.files_involved:
            console.print(f"\n[bold]Files Involved:[/bold]")
            for i, file in enumerate(report.files_involved[:10], 1):
                console.print(f"  {i}. {file}")
            if len(report.files_involved) > 10:
                console.print(f"  ... and {len(report.files_involved) - 10} more")

        # Suggested actions
        if report.suggested_actions:
            console.print(f"\n[bold cyan]Suggested Actions:[/bold cyan]")
            for i, action in enumerate(report.suggested_actions, 1):
                console.print(f"\n  [{i}] {action['action']}")
                console.print(f"      [dim]$ {action['command']}[/dim]")

        # Quick fixes
        if report.quick_fixes:
            console.print(f"\n[bold green]Quick Fixes:[/bold green]")
            for fix in report.quick_fixes:
                console.print(f"  • {fix['label']}")
                console.print(f"    [dim]$ {fix['command']}[/dim]")

        console.print("\n" + "━" * 80, style=color)

        return console.export_text()

    def explain_blocking(self, task_id: str) -> str:
        """Generate human-readable explanation of why task is blocked.

        Args:
            task_id: Task identifier.

        Returns:
            Explanation text.
        """
        # Load task data
        task_queue_path = self.state_dir / "task_queue.yaml"
        queue = _load_data(task_queue_path, {})
        tasks = queue.get("tasks", [])

        task = None
        for t in tasks:
            if t.get("id") == task_id:
                task = t
                break

        if not task:
            return f"Task {task_id} not found."

        if task.get("lifecycle") != "waiting_human":
            return f"Task {task_id} is not blocked (lifecycle={task.get('lifecycle')})."

        # Build explanation
        lines = [
            f"Task '{task_id}' is blocked and requires human intervention.",
            "",
            f"Status: {task.get('status')}",
            f"Step: {task.get('step')}",
            f"Attempts: {task.get('attempts', 0)}",
            "",
        ]

        if task.get("last_error"):
            lines.append("Last Error:")
            lines.append(f"  {task.get('last_error')}")
            lines.append("")

        if task.get("last_error_type"):
            lines.append(f"Error Type: {task.get('last_error_type')}")
            lines.append("")

        if task.get("block_reason"):
            lines.append("Blocking Reason:")
            lines.append(f"  {task.get('block_reason')}")
            lines.append("")

        if task.get("human_blocking_issues"):
            lines.append("Blocking Issues:")
            for issue in task.get("human_blocking_issues", []):
                lines.append(f"  • {issue}")
            lines.append("")

        if task.get("human_next_steps"):
            lines.append("Suggested Next Steps:")
            for step in task.get("human_next_steps", []):
                lines.append(f"  • {step}")
            lines.append("")

        intent = task.get("blocked_intent", {})
        if intent:
            lines.append("To Resume:")
            lines.append(f"  feature-prd-runner resume {task_id}")
            if intent.get("step"):
                lines.append(f"  (will resume at step: {intent.get('step')})")
            lines.append("")

        return "\n".join(lines)

    def inspect_state(self, task_id: str) -> Optional[StateSnapshot]:
        """Inspect full task state.

        Args:
            task_id: Task identifier.

        Returns:
            StateSnapshot or None if not found.
        """
        task_queue_path = self.state_dir / "task_queue.yaml"
        queue = _load_data(task_queue_path, {})
        tasks = queue.get("tasks", [])

        for task in tasks:
            if task.get("id") == task_id:
                return StateSnapshot(
                    task_id=task_id,
                    lifecycle=task.get("lifecycle", "unknown"),
                    step=task.get("step", "unknown"),
                    status=task.get("status", "unknown"),
                    attempts=task.get("attempts", 0),
                    last_error=task.get("last_error"),
                    last_error_type=task.get("last_error_type"),
                    context=task.get("context", []),
                    metadata={
                        "type": task.get("type"),
                        "phase_id": task.get("phase_id"),
                        "prompt_mode": task.get("prompt_mode"),
                        "auto_resume_count": task.get("auto_resume_count", 0),
                        "manual_resume_attempts": task.get("manual_resume_attempts", 0),
                    },
                )

        return None

    def trace_history(self, task_id: str) -> list[dict[str, Any]]:
        """Show full event history for task.

        Args:
            task_id: Task identifier.

        Returns:
            List of events related to this task.
        """
        events_path = self.state_dir / "artifacts" / "events.jsonl"

        if not events_path.exists():
            return []

        events = []
        try:
            with events_path.open("r") as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        # Filter events for this task
                        if event.get("task_id") == task_id:
                            events.append(event)
        except Exception as e:
            logger.warning("Failed to read events: {}", e)

        return events

    def format_state_snapshot(self, snapshot: StateSnapshot) -> str:
        """Format state snapshot as rich text.

        Args:
            snapshot: State snapshot to format.

        Returns:
            Formatted text.
        """
        console = Console(record=True, width=80)

        console.print()
        console.print(f"[bold]Task State: {snapshot.task_id}[/bold]")
        console.print("━" * 80)

        table = Table(show_header=False, box=None)
        table.add_row("Lifecycle:", snapshot.lifecycle)
        table.add_row("Step:", snapshot.step)
        table.add_row("Status:", snapshot.status)
        table.add_row("Attempts:", str(snapshot.attempts))

        console.print(table)

        if snapshot.last_error:
            console.print(f"\n[bold red]Last Error:[/bold red]")
            console.print(f"[dim]{snapshot.last_error}[/dim]")

        if snapshot.last_error_type:
            console.print(f"\n[bold]Error Type:[/bold] {snapshot.last_error_type}")

        if snapshot.context:
            console.print(f"\n[bold]Context:[/bold]")
            for ctx in snapshot.context[:5]:
                console.print(f"  • {ctx}")
            if len(snapshot.context) > 5:
                console.print(f"  ... and {len(snapshot.context) - 5} more")

        if snapshot.metadata:
            console.print(f"\n[bold]Metadata:[/bold]")
            for key, value in snapshot.metadata.items():
                if value is not None:
                    console.print(f"  {key}: {value}")

        console.print("\n" + "━" * 80)

        return console.export_text()
