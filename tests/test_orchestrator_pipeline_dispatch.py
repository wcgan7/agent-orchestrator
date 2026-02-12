"""Tests verifying template-driven pipeline dispatch per task type."""
from __future__ import annotations

from pathlib import Path

from feature_prd_runner.v3.domain.models import Task
from feature_prd_runner.v3.events import EventBus
from feature_prd_runner.v3.orchestrator import OrchestratorService
from feature_prd_runner.v3.storage.container import V3Container


def _service(tmp_path: Path) -> tuple[V3Container, OrchestratorService, EventBus]:
    container = V3Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service, bus


def _step_names(container: V3Container, task_id: str) -> list[str]:
    """Return the step names recorded in the run for the given task."""
    runs = container.runs.list()
    for run in runs:
        if run.task_id == task_id:
            return [step["step"] for step in (run.steps or [])]
    return []


# ---------------------------------------------------------------------------
# 1. Feature runs full pipeline
# ---------------------------------------------------------------------------


def test_feature_runs_full_pipeline(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Feature task",
        task_type="feature",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["plan", "plan_impl", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 2. Bug fix runs correct steps
# ---------------------------------------------------------------------------


def test_bug_fix_runs_correct_steps(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Bug fix task",
        task_type="bug",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["reproduce", "diagnose", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 3. Research runs without review or commit
# ---------------------------------------------------------------------------


def test_research_runs_without_review_or_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Research task",
        task_type="research",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["gather", "analyze", "summarize", "report"]
    # No review or commit in steps
    assert "review" not in steps
    assert "commit" not in steps


# ---------------------------------------------------------------------------
# 4. Security audit runs scan steps
# ---------------------------------------------------------------------------


def test_security_audit_runs_scan_steps(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Security audit task",
        task_type="security",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["scan_deps", "scan_code", "report", "generate_tasks"]


# ---------------------------------------------------------------------------
# 5. Repo review skips review and commit
# ---------------------------------------------------------------------------


def test_repo_review_skips_review_and_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Repo review task",
        task_type="repo_review",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["scan", "analyze", "generate_tasks"]


# ---------------------------------------------------------------------------
# 6. Review pipeline has review but no commit
# ---------------------------------------------------------------------------


def test_review_pipeline_has_review_but_no_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Code review task",
        task_type="review",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    # review pipeline: analyze, review, report — review loop fires, then done (no commit)
    assert "analyze" in steps
    assert "review" in steps
    assert "report" in steps
    assert "commit" not in steps


# ---------------------------------------------------------------------------
# 7. Performance pipeline
# ---------------------------------------------------------------------------


def test_performance_pipeline(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Performance task",
        task_type="performance",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["profile", "plan", "implement", "benchmark", "review", "commit"]


# ---------------------------------------------------------------------------
# 8. Unknown type falls back to feature
# ---------------------------------------------------------------------------


def test_unknown_type_falls_back_to_feature(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Unknown type task",
        task_type="unknown_thing",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    # Falls back to feature pipeline
    assert steps == ["plan", "plan_impl", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 9. Custom pipeline_template honored
# ---------------------------------------------------------------------------


def test_custom_pipeline_template_honored(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Custom pipeline task",
        task_type="feature",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
        pipeline_template=["plan", "implement", "commit"],
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["plan", "implement", "commit"]


# ---------------------------------------------------------------------------
# 10. Pipeline template stored on task
# ---------------------------------------------------------------------------


def test_pipeline_template_stored_on_task(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Template storage test",
        task_type="bug",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    # pipeline_template starts empty — should be resolved from registry
    assert not task.pipeline_template
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    # After execution, the stored template should match the bug_fix pipeline
    assert result.pipeline_template == ["reproduce", "diagnose", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 11. Review findings passed to implement_fix
# ---------------------------------------------------------------------------


def test_review_findings_passed_to_implement_fix(tmp_path: Path) -> None:
    """When review requests changes, open findings are attached to
    task.metadata['review_findings'] before implement_fix runs."""
    from unittest.mock import MagicMock

    from feature_prd_runner.v3.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult

    captured_metadata: list[dict] = []
    real_adapter = DefaultWorkerAdapter()

    def spy_run_step(*, task, step, attempt):
        if step == "implement_fix":
            # Capture the metadata at the moment implement_fix is called
            captured_metadata.append(dict(task.metadata))
        return real_adapter.run_step(task=task, step=step, attempt=attempt)

    adapter = MagicMock()
    adapter.run_step = spy_run_step

    container = V3Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Fix findings task",
        task_type="feature",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
        metadata={
            "scripted_findings": [
                # First review: one high-severity finding (triggers changes_requested)
                [{"severity": "high", "summary": "Missing null check", "file": "main.py", "line": 42}],
                # Second review: clean (approved)
                [],
            ]
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    # implement_fix should have been called once (after first review)
    assert len(captured_metadata) == 1
    # The metadata should contain the open findings from the first review
    review_findings = captured_metadata[0].get("review_findings")
    assert isinstance(review_findings, list)
    assert len(review_findings) == 1
    assert review_findings[0]["summary"] == "Missing null check"
    assert review_findings[0]["file"] == "main.py"
    assert review_findings[0]["line"] == 42
    # After the run completes, review_findings should be cleaned up
    assert "review_findings" not in result.metadata


# ---------------------------------------------------------------------------
# 12. generate_tasks creates child tasks
# ---------------------------------------------------------------------------


def test_generate_tasks_creates_child_tasks(tmp_path: Path) -> None:
    """The generate_tasks step output creates child tasks linked to the parent."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Security audit parent",
        task_type="security",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
        metadata={
            "scripted_generated_tasks": [
                {"title": "Upgrade lodash", "task_type": "bug", "priority": "P1",
                 "description": "CVE-2024-1234"},
                {"title": "Fix SQL injection in auth", "task_type": "bug", "priority": "P0",
                 "labels": ["security"]},
            ]
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    # Parent should have two children
    assert len(result.children_ids) == 2

    # Verify child tasks exist and are linked correctly
    all_tasks = container.tasks.list()
    children = [t for t in all_tasks if t.parent_id == task.id]
    assert len(children) == 2

    titles = sorted(c.title for c in children)
    assert titles == ["Fix SQL injection in auth", "Upgrade lodash"]

    lodash_task = next(c for c in children if c.title == "Upgrade lodash")
    assert lodash_task.task_type == "bug"
    assert lodash_task.priority == "P1"
    assert lodash_task.description == "CVE-2024-1234"
    assert lodash_task.source == "generated"
    assert lodash_task.parent_id == task.id

    sqli_task = next(c for c in children if c.title == "Fix SQL injection in auth")
    assert sqli_task.priority == "P0"
    assert sqli_task.labels == ["security"]


# ---------------------------------------------------------------------------
# 13. generate_tasks with no output creates no children
# ---------------------------------------------------------------------------


def test_generate_tasks_no_output_no_children(tmp_path: Path) -> None:
    """When generate_tasks returns no tasks, no children are created."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Clean repo review",
        task_type="repo_review",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    assert result.children_ids == []


# ---------------------------------------------------------------------------
# 14. Hotfix pipeline — skip diagnosis, straight to fix
# ---------------------------------------------------------------------------


def test_hotfix_skips_diagnosis(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Urgent fix",
        task_type="hotfix",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 15. Spike pipeline — prototype + report, no commit
# ---------------------------------------------------------------------------


def test_spike_runs_without_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Explore caching options",
        task_type="spike",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["gather", "prototype", "summarize", "report"]
    assert "commit" not in steps
    assert "review" not in steps


# ---------------------------------------------------------------------------
# 16. Chore pipeline — no plan, no review
# ---------------------------------------------------------------------------


def test_chore_skips_plan_and_review(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Run formatter",
        task_type="chore",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["implement", "verify", "commit"]
    assert "plan" not in steps
    assert "review" not in steps


# ---------------------------------------------------------------------------
# 17. Plan-only pipeline — analyze + plan + report, no code
# ---------------------------------------------------------------------------


def test_plan_only_produces_report_no_code(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Design auth system",
        task_type="plan_only",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "plan", "report"]
    assert "implement" not in steps
    assert "commit" not in steps


# ---------------------------------------------------------------------------
# 18. Decompose pipeline — breaks into subtasks
# ---------------------------------------------------------------------------


def test_decompose_generates_child_tasks(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Break down auth epic",
        task_type="decompose",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
        metadata={
            "scripted_generated_tasks": [
                {"title": "Add login endpoint", "task_type": "feature"},
                {"title": "Add session middleware", "task_type": "feature"},
                {"title": "Write auth tests", "task_type": "test"},
            ]
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "plan", "generate_tasks"]
    assert len(result.children_ids) == 3

    children = [t for t in container.tasks.list() if t.parent_id == task.id]
    assert len(children) == 3
    types = sorted(c.task_type for c in children)
    assert types == ["feature", "feature", "test"]


# ---------------------------------------------------------------------------
# 19. Verify-only pipeline — just check current state
# ---------------------------------------------------------------------------


def test_verify_only_runs_checks_no_changes(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Check CI status",
        task_type="verify_only",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["verify", "report"]
    assert "implement" not in steps
    assert "commit" not in steps
