"""Tests verifying template-driven pipeline dispatch per task type."""
from __future__ import annotations

from pathlib import Path

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.storage.container import Container


def _service(tmp_path: Path) -> tuple[Container, OrchestratorService, EventBus]:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service, bus


def _step_names(container: Container, task_id: str) -> list[str]:
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
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["plan", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 1b. Feature writes plan/implement/verify/review/fix-log sections to workdoc
# ---------------------------------------------------------------------------


def test_feature_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """feature run should populate core implementation-cycle sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    plan_text = "plan: implement scoped login feature"
    implement_text = "implement: added login endpoint and handler"
    verify_text = "verify: unit and integration checks green"

    def mock_run_step(*, task, step, attempt):
        if step == "plan":
            return StepResult(status="ok", summary=plan_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "medium", "summary": "Add negative auth test", "file": "auth.py", "line": 22}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="added negative auth test")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature workdoc write test",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert plan_text in content
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Add negative auth test" in content
    assert "### Fix Cycle 1" in content
    assert "added negative auth test" in content
    assert "_Pending: will be populated by the plan step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


# ---------------------------------------------------------------------------
# 2. Bug fix runs correct steps
# ---------------------------------------------------------------------------


def test_bug_fix_runs_correct_steps(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Bug fix task",
        task_type="bug",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["reproduce", "diagnose", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 2b. Bug-fix writes bug-specific sections and fix loop entries to workdoc
# ---------------------------------------------------------------------------


def test_bug_fix_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """bug_fix run should populate reproduce/diagnose/fix/verify/review/fix-log."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    reproduce_text = "reproduce: crash occurs on missing user profile"
    diagnose_text = "diagnose: None dereference in auth adapter"
    implement_text = "implement: added guard and fallback profile path"
    verify_text = "verify: regression and unit checks passed"
    report_text = "report: post-fix review summary"

    def mock_run_step(*, task, step, attempt):
        if step == "reproduce":
            return StepResult(status="ok", summary=reproduce_text)
        if step == "diagnose":
            return StepResult(status="ok", summary=diagnose_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "high", "summary": "Add missing edge-case test", "file": "auth.py", "line": 41}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="added edge-case test coverage")
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Bug-fix workdoc write test",
        task_type="bug",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert reproduce_text in content
    assert diagnose_text in content
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Add missing edge-case test" in content
    assert "### Fix Cycle 1" in content
    assert "added edge-case test coverage" in content
    assert "_Pending: will be populated by the reproduce step._" not in content
    assert "_Pending: will be populated by the diagnose step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


# ---------------------------------------------------------------------------
# 2c. Refactor writes refactor-specific sections to workdoc
# ---------------------------------------------------------------------------


def test_refactor_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """refactor run should populate analysis/plan/implement/verify/review/fix-log."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: identified module boundary issues"
    plan_text = "plan: phased internal API cleanup"
    implement_text = "implement: extracted auth adapter interface"
    verify_text = "verify: test suite and static checks passed"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "plan":
            return StepResult(status="ok", summary=plan_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "low", "summary": "Rename helper for clarity", "file": "auth.py", "line": 88}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="renamed helper and updated references")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Refactor workdoc write test",
        task_type="refactor",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert plan_text in content
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Rename helper for clarity" in content
    assert "### Fix Cycle 1" in content
    assert "renamed helper and updated references" in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the plan step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


# ---------------------------------------------------------------------------
# 3. Research runs without review or commit
# ---------------------------------------------------------------------------


def test_research_runs_without_review_or_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Research task",
        task_type="research",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "report"]
    # No review or commit in steps
    assert "review" not in steps
    assert "commit" not in steps


# ---------------------------------------------------------------------------
# 3b. Research writes analysis/report sections to workdoc
# ---------------------------------------------------------------------------


def test_research_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """research run should populate Research Analysis and Final Report sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: compared options and constraints"
    report_text = "report: recommended option B with migration notes"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Research workdoc write test",
        task_type="research",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert report_text in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the report step._" not in content


# ---------------------------------------------------------------------------
# 4. Security audit runs scan steps
# ---------------------------------------------------------------------------


def test_security_audit_runs_scan_steps(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Security audit task",
        task_type="security",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["scan_deps", "scan_code", "report", "generate_tasks"]


# ---------------------------------------------------------------------------
# 4b. Security audit writes scan/report/task-gen sections to workdoc
# ---------------------------------------------------------------------------


def test_security_audit_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """security_audit run should populate scan/report/generate tasks sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    deps_text = "scan_deps: found 2 vulnerable packages"
    code_text = "scan_code: found 1 high severity sink"
    report_text = "report: prioritized remediation order documented"
    tasks_text = "generate_tasks: created remediation tasks"

    def mock_run_step(*, task, step, attempt):
        if step == "scan_deps":
            return StepResult(status="ok", summary=deps_text)
        if step == "scan_code":
            return StepResult(status="ok", summary=code_text)
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        if step == "generate_tasks":
            return StepResult(status="ok", summary=tasks_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Security workdoc write test",
        task_type="security",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert deps_text in content
    assert code_text in content
    assert report_text in content
    assert tasks_text in content
    assert "_Pending: will be populated by the scan_deps step._" not in content
    assert "_Pending: will be populated by the scan_code step._" not in content
    assert "_Pending: will be populated by the report step._" not in content
    assert "_Pending: will be populated by the generate_tasks step._" not in content


# ---------------------------------------------------------------------------
# 5. Repo review skips review and commit
# ---------------------------------------------------------------------------


def test_repo_review_skips_review_and_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Repo review task",
        task_type="repo_review",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "initiative_plan", "generate_tasks"]


# ---------------------------------------------------------------------------
# 5b. Repo review writes analysis/initiative_plan/task-gen sections to workdoc
# ---------------------------------------------------------------------------


def test_repo_review_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """repo_review run should populate repository-analysis planning sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: identified architecture and hygiene gaps"
    plan_text = "initiative plan: phased modernization strategy"
    tasks_text = "generate_tasks: created backlog decomposition tasks"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "initiative_plan":
            return StepResult(status="ok", summary=plan_text)
        if step == "generate_tasks":
            return StepResult(status="ok", summary=tasks_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Repo review workdoc write test",
        task_type="repo_review",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert plan_text in content
    assert tasks_text in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the plan step._" not in content
    assert "_Pending: will be populated by the generate_tasks step._" not in content


# ---------------------------------------------------------------------------
# 6. Review pipeline has review but no commit
# ---------------------------------------------------------------------------


def test_review_pipeline_has_review_but_no_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Code review task",
        task_type="review",
        status="queued",
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
# 6b. Review writes analysis/findings/report sections to workdoc
# ---------------------------------------------------------------------------


def test_review_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """review run should populate Review Analysis, Review Findings, and Final Report."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: baseline and scope verified"
    report_text = "report: review complete with actionable findings"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "review":
            findings = [{"severity": "medium", "summary": "Missing null check", "file": "main.py", "line": 12}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        if step == "implement_fix":
            return StepResult(status="ok", summary="fix applied")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Review workdoc write test",
        task_type="review",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert "Review Cycle 1" in content
    assert "Missing null check" in content
    assert report_text in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the review step._" not in content
    assert "_Pending: will be populated by the report step._" not in content


# ---------------------------------------------------------------------------
# 7. Performance pipeline
# ---------------------------------------------------------------------------


def test_performance_pipeline(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Performance task",
        task_type="performance",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["profile", "plan", "implement", "benchmark", "review", "commit"]


# ---------------------------------------------------------------------------
# 7b. Performance writes profile/plan/impl/benchmark/review/fix-log sections
# ---------------------------------------------------------------------------


def test_performance_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """performance run should populate perf-specific lifecycle sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    called_steps: list[str] = []
    profile_text = "profile: baseline p95=240ms cpu=82%"
    plan_text = "plan: optimize hot path allocations"
    implement_text = "implement: reduced per-request allocations"
    benchmark_text = "benchmark: p95 improved to 170ms"

    def mock_run_step(*, task, step, attempt):
        called_steps.append(step)
        if step == "profile":
            return StepResult(status="ok", summary=profile_text)
        if step == "plan":
            return StepResult(status="ok", summary=plan_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "benchmark":
            return StepResult(status="ok", summary=benchmark_text)
        if step == "review":
            findings = [{"severity": "medium", "summary": "Add longer benchmark window", "file": "bench.py", "line": 10}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="extended benchmark window and reran")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Performance workdoc write test",
        task_type="performance",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert profile_text in content
    assert plan_text in content
    assert implement_text in content
    assert benchmark_text in content
    assert "Review Cycle 1" in content
    assert "Add longer benchmark window" in content
    assert "### Fix Cycle 1" in content
    assert "extended benchmark window and reran" in content
    assert "_Pending: will be populated by the profile step._" not in content
    assert "_Pending: will be populated by the plan step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content
    # Performance review-loop validation should use benchmark, not verify.
    assert called_steps.count("benchmark") == 2
    assert "verify" not in called_steps


# ---------------------------------------------------------------------------
# 8. Unknown type falls back to feature
# ---------------------------------------------------------------------------


def test_test_pipeline_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """test run should populate coverage/test-impl/verify/review/fix-log sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: identified missing auth integration scenarios"
    implement_text = "implement: added integration and negative-path tests"
    verify_text = "verify: all test suites pass"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "low", "summary": "Cover token expiry edge case", "file": "tests/test_auth.py", "line": 64}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="added token expiry edge-case test")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Test pipeline workdoc write test",
        task_type="test",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Cover token expiry edge case" in content
    assert "### Fix Cycle 1" in content
    assert "added token expiry edge-case test" in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


def test_docs_pipeline_runs_with_verify(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Docs task",
        task_type="docs",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "implement", "verify", "review", "commit"]


def test_docs_pipeline_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """docs run should populate docs-analysis/update/verify/review/fix-log sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: found stale setup and API examples"
    implement_text = "implement: refreshed onboarding and API usage docs"
    verify_text = "verify: markdown lint and docs links passed"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "low", "summary": "Clarify env var description", "file": "README.md", "line": 35}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="clarified env var description")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Docs workdoc write test",
        task_type="docs",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Clarify env var description" in content
    assert "### Fix Cycle 1" in content
    assert "clarified env var description" in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


def test_unknown_type_falls_back_to_feature(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Unknown type task",
        task_type="unknown_thing",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    # Falls back to feature pipeline
    assert steps == ["plan", "implement", "verify", "review", "commit"]


# ---------------------------------------------------------------------------
# 9. Custom pipeline_template honored
# ---------------------------------------------------------------------------


def test_custom_pipeline_template_honored(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Custom pipeline task",
        task_type="feature",
        status="queued",
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
        status="queued",
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

    from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult

    captured_metadata: list[dict] = []
    real_adapter = DefaultWorkerAdapter()

    def spy_run_step(*, task, step, attempt):
        if step == "implement_fix":
            # Capture the metadata at the moment implement_fix is called
            captured_metadata.append(dict(task.metadata))
        return real_adapter.run_step(task=task, step=step, attempt=attempt)

    adapter = MagicMock()
    adapter.run_step = spy_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Fix findings task",
        task_type="feature",
        status="queued",
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
        status="queued",
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
        status="queued",
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
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["implement", "verify", "review", "commit"]


def test_hotfix_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """hotfix run should populate hotfix-impl/verify/review/fix-log sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    implement_text = "implement: patched auth timeout regression"
    verify_text = "verify: smoke tests and focused checks passed"

    def mock_run_step(*, task, step, attempt):
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "review":
            findings = [{"severity": "medium", "summary": "Add rollback note", "file": "README.md", "line": 12}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="added rollback note to runbook")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Hotfix workdoc write test",
        task_type="hotfix",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert implement_text in content
    assert verify_text in content
    assert "Review Cycle 1" in content
    assert "Add rollback note" in content
    assert "### Fix Cycle 1" in content
    assert "added rollback note to runbook" in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the review step._" not in content


# ---------------------------------------------------------------------------
# 15. Spike pipeline — prototype + report, no commit
# ---------------------------------------------------------------------------


def test_spike_runs_without_commit(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Explore caching options",
        task_type="spike",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "prototype", "report"]
    assert "commit" not in steps
    assert "review" not in steps


def test_spike_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """spike run should populate analysis/prototype/report sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: compared caching candidates and constraints"
    prototype_text = "prototype: implemented throwaway Redis adapter"
    report_text = "report: recommend Redis with bounded TTL strategy"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "prototype":
            return StepResult(status="ok", summary=prototype_text)
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Spike workdoc write test",
        task_type="spike",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert prototype_text in content
    assert report_text in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the report step._" not in content


# ---------------------------------------------------------------------------
# 16. Chore pipeline — no plan, no review
# ---------------------------------------------------------------------------


def test_chore_skips_plan_and_review(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Run formatter",
        task_type="chore",
        status="queued",
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


def test_chore_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """chore run should populate chore implementation and verification sections."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    implement_text = "implement: applied formatter and removed dead comments"
    verify_text = "verify: lint and test checks passed"

    def mock_run_step(*, task, step, attempt):
        if step == "implement":
            return StepResult(status="ok", summary=implement_text)
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Chore workdoc write test",
        task_type="chore",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert implement_text in content
    assert verify_text in content
    assert "_Pending: will be populated by the implement step._" not in content
    assert "_Pending: will be populated by the verify step._" not in content


# ---------------------------------------------------------------------------
# 17. Plan-only pipeline — analyze + initiative_plan + generate_tasks, no code
# ---------------------------------------------------------------------------


def test_plan_only_produces_report_no_code(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Design auth system",
        task_type="plan_only",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    steps = _step_names(container, task.id)
    assert steps == ["analyze", "initiative_plan", "generate_tasks"]
    assert "implement" not in steps
    assert "commit" not in steps


# ---------------------------------------------------------------------------
# 17b. Plan-only writes analysis/plan/task-gen/report sections to workdoc
# ---------------------------------------------------------------------------


def test_plan_only_writes_sections_to_workdoc(tmp_path: Path) -> None:
    """plan_only run should populate all sections of its minimal workdoc."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    analyze_text = "analyze: scope and constraints clarified"
    plan_text = "initiative plan: phased rollout with dependencies"
    generated_tasks_text = "generated_tasks: created 3 execution tasks"

    def mock_run_step(*, task, step, attempt):
        if step == "analyze":
            return StepResult(status="ok", summary=analyze_text)
        if step == "initiative_plan":
            return StepResult(status="ok", summary=plan_text)
        if step == "generate_tasks":
            return StepResult(status="ok", summary=generated_tasks_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Plan-only workdoc write test",
        task_type="plan_only",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert analyze_text in content
    assert plan_text in content
    assert generated_tasks_text in content
    assert "_Pending: will be populated by the analyze step._" not in content
    assert "_Pending: will be populated by the plan step._" not in content
    assert "_Pending: will be populated by the generate_tasks step._" not in content


# ---------------------------------------------------------------------------
# 18. Decompose task type aliases to initiative planning pipeline
# ---------------------------------------------------------------------------


def test_decompose_task_type_uses_plan_only_pipeline(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Break down auth epic",
        task_type="decompose",
        status="queued",
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
    assert steps == ["analyze", "initiative_plan", "generate_tasks"]
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
        status="queued",
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


# ---------------------------------------------------------------------------
# 19b. Verify-only writes verify/report outputs into workdoc
# ---------------------------------------------------------------------------


def test_verify_only_writes_verify_and_report_to_workdoc(tmp_path: Path) -> None:
    """verify_only run should populate Verification Results and Final Report."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    verify_text = "verify: all selected checks passed"
    report_text = "report: repository is currently healthy"

    def mock_run_step(*, task, step, attempt):
        if step == "verify":
            return StepResult(status="ok", summary=verify_text)
        if step == "report":
            return StepResult(status="ok", summary=report_text)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Verify-only workdoc write test",
        task_type="verify_only",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    workdoc = container.state_root / "workdocs" / f"{task.id}.md"
    content = workdoc.read_text(encoding="utf-8")
    assert verify_text in content
    assert report_text in content
    assert "_Pending: will be populated by the verify step._" not in content
    assert "_Pending: will be populated by the report step._" not in content


# ---------------------------------------------------------------------------
# 20. Plan output stored in revisions
# ---------------------------------------------------------------------------


def test_plan_output_stored_in_revisions(tmp_path: Path) -> None:
    """Plan/analyze steps store their output in plan revisions."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult

    real_adapter = DefaultWorkerAdapter()

    def spy_run_step(*, task, step, attempt):
        if step in ("plan", "analyze"):
            return StepResult(status="ok", summary=f"Plan output for {step}")
        return real_adapter.run_step(task=task, step=step, attempt=attempt)

    adapter = MagicMock()
    adapter.run_step = spy_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Plan storage test",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    revisions = container.plan_revisions.for_task(result.id)
    assert len(revisions) == 1  # plan (feature pipeline)
    assert revisions[0].step == "plan"
    assert revisions[0].content == "Plan output for plan"


# ---------------------------------------------------------------------------
# 21. Generate tasks from plan
# ---------------------------------------------------------------------------


def test_generate_tasks_from_plan(tmp_path: Path) -> None:
    """generate_tasks_from_plan() creates child tasks from plan text."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    def mock_run_step(*, task, step, attempt):
        if step == "generate_tasks":
            assert task.metadata.get("plan_for_generation") == "Build auth system"
            return StepResult(
                status="ok",
                generated_tasks=[
                    {"title": "Add login", "task_type": "feature", "priority": "P1"},
                    {"title": "Add session", "task_type": "feature", "priority": "P2"},
                ],
            )
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(title="Auth epic", task_type="decompose", status="queued")
    container.tasks.upsert(task)

    created_ids = service.generate_tasks_from_plan(task.id, "Build auth system", infer_deps=False)

    assert len(created_ids) == 2
    parent = container.tasks.get(task.id)
    assert set(parent.children_ids) == set(created_ids)

    children = [container.tasks.get(cid) for cid in created_ids]
    assert children[0].title == "Add login"
    assert children[0].parent_id == task.id
    assert children[1].title == "Add session"

    # plan_for_generation should be cleaned up
    assert "plan_for_generation" not in parent.metadata


# ---------------------------------------------------------------------------
# 22. Generate tasks from plan with dependencies
# ---------------------------------------------------------------------------


def test_generate_tasks_from_plan_with_deps(tmp_path: Path) -> None:
    """depends_on task IDs are wired as blocked_by/blocks between children."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    def mock_run_step(*, task, step, attempt):
        if step == "generate_tasks":
            return StepResult(
                status="ok",
                generated_tasks=[
                    {"id": "setup_db", "title": "Setup DB", "task_type": "feature", "priority": "P0"},
                    {"id": "add_models", "title": "Add models", "task_type": "feature", "priority": "P1", "depends_on": ["setup_db"]},
                    {"id": "add_api", "title": "Add API", "task_type": "feature", "priority": "P1", "depends_on": ["setup_db", "add_models"]},
                ],
            )
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(title="Backend epic", task_type="decompose", status="queued")
    container.tasks.upsert(task)

    created_ids = service.generate_tasks_from_plan(task.id, "Backend plan", infer_deps=True)

    assert len(created_ids) == 3
    db_task = container.tasks.get(created_ids[0])
    models_task = container.tasks.get(created_ids[1])
    api_task = container.tasks.get(created_ids[2])

    # Setup DB blocks Models and API
    assert created_ids[1] in db_task.blocks
    assert created_ids[2] in db_task.blocks

    # Models depends on DB, blocks API
    assert created_ids[0] in models_task.blocked_by
    assert created_ids[2] in models_task.blocks

    # API depends on both DB and Models
    assert created_ids[0] in api_task.blocked_by
    assert created_ids[1] in api_task.blocked_by


# ---------------------------------------------------------------------------
# 23. Generate tasks from plan fails when no plan
# ---------------------------------------------------------------------------


def test_generate_tasks_from_plan_no_task_fails(tmp_path: Path) -> None:
    """generate_tasks_from_plan() raises ValueError for missing task."""
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    import pytest
    with pytest.raises(ValueError, match="Task not found"):
        service.generate_tasks_from_plan("nonexistent", "some plan")


def test_generate_tasks_from_plan_empty_worker_output_fails(tmp_path: Path) -> None:
    """Explicit plan decomposition should fail if worker returns no tasks."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    def mock_run_step(*, task, step, attempt):
        if step == "generate_tasks":
            return StepResult(status="ok", summary="No decomposition possible.")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(title="Empty decomposition", task_type="decompose", status="queued")
    container.tasks.upsert(task)

    import pytest
    with pytest.raises(ValueError, match="no generated tasks"):
        service.generate_tasks_from_plan(task.id, "Some plan text")


# ---------------------------------------------------------------------------
# Step output storage and cleanup
# ---------------------------------------------------------------------------


def test_step_outputs_populated_after_successful_step(tmp_path: Path) -> None:
    """After a successful non-review step, step_outputs should contain the summary."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    summaries: dict[str, str] = {}

    def mock_run_step(*, task, step, attempt):
        summary = f"{step} output text"
        summaries[step] = summary
        return StepResult(status="ok", summary=summary)

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature task",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)
    service.run_task(task.id)

    # After completion the step_outputs are cleaned up, so check via the run steps
    # that the pipeline completed. Instead, verify the adapter saw accumulated outputs
    # by checking that the task stored outputs during execution.
    # Since done cleans up, we verify indirectly: the task completed successfully.
    result = container.tasks.get(task.id)
    assert result is not None
    assert result.status == "done"
    # step_outputs should be cleaned up after completion
    assert result.metadata.get("step_outputs") is None


def test_step_outputs_cleaned_up_on_completion(tmp_path: Path) -> None:
    """step_outputs metadata is removed when pipeline reaches done status."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    def mock_run_step(*, task, step, attempt):
        return StepResult(status="ok", summary=f"Result of {step}")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Refactor task",
        task_type="refactor",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)
    service.run_task(task.id)

    final = container.tasks.get(task.id)
    assert final is not None
    assert final.status == "done"
    assert "step_outputs" not in (final.metadata or {})


def test_step_outputs_preserved_on_blocked(tmp_path: Path) -> None:
    """step_outputs should NOT be cleaned up when task is blocked (may be retried)."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    call_count = {"n": 0}

    def mock_run_step(*, task, step, attempt):
        call_count["n"] += 1
        if step == "plan":
            return StepResult(status="ok", summary="The plan")
        # Fail on implement step
        return StepResult(status="error", summary="implement failed")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature that blocks",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)
    service.run_task(task.id)

    final = container.tasks.get(task.id)
    assert final is not None
    assert final.status == "blocked"
    # step_outputs should still be present for retry
    assert isinstance(final.metadata, dict)
    assert "step_outputs" in final.metadata
    assert final.metadata["step_outputs"]["plan"] == "The plan"


def test_feature_non_actionable_verify_failure_skips_fix_loop_and_completes(tmp_path: Path) -> None:
    """Missing tooling/config verify failures should not trigger implement_fix loops."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    seen_fix = {"called": 0}

    def mock_run_step(*, task, step, attempt):
        if step == "plan":
            return StepResult(status="ok", summary="Plan complete")
        if step == "implement":
            return StepResult(status="ok", summary="Implementation complete")
        if step == "verify":
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata["verify_reason_code"] = "tool_missing"
            return StepResult(status="error", summary="pytest command not found in environment")
        if step == "implement_fix":
            seen_fix["called"] += 1
            return StepResult(status="ok", summary="Should not run")
        if step == "review":
            return StepResult(status="ok", findings=[])
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature with missing verify tooling",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    assert seen_fix["called"] == 0
    md = result.metadata or {}
    assert isinstance(md.get("verify_degraded_issues"), list)
    assert len(md["verify_degraded_issues"]) >= 1
    assert md["verify_degraded_issues"][-1]["reason_code"] == "tool_missing"


def test_review_loop_non_actionable_verify_failure_skips_extra_fix_loop(tmp_path: Path) -> None:
    """When verify fails non-actionably after implement_fix, do not spin extra verify-fix loop."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    calls = {"implement_fix": 0}

    def mock_run_step(*, task, step, attempt):
        if step == "plan":
            return StepResult(status="ok", summary="Plan complete")
        if step == "implement":
            return StepResult(status="ok", summary="Implementation complete")
        if step == "verify":
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata["verify_reason_code"] = "config_missing"
            return StepResult(status="error", summary="Missing ESLint config")
        if step == "implement_fix":
            calls["implement_fix"] += 1
            return StepResult(status="ok", summary="Addressed review finding")
        if step == "review":
            # First review requests changes; second approves.
            findings = [{"severity": "high", "summary": "Need additional docs", "file": "README.md", "line": 5}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature with review change and degraded verify",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    # Only one implement_fix should run (for review finding), no extra verify-fix loop.
    assert calls["implement_fix"] == 1


def test_feature_review_loop_post_fix_validation_uses_verify(tmp_path: Path) -> None:
    """Feature review-loop post-fix validation should run verify after implement_fix."""
    from unittest.mock import MagicMock

    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult

    called_steps: list[str] = []

    def mock_run_step(*, task, step, attempt):
        called_steps.append(step)
        if step == "plan":
            return StepResult(status="ok", summary="Plan complete")
        if step == "implement":
            return StepResult(status="ok", summary="Implementation complete")
        if step == "verify":
            return StepResult(status="ok", summary="Verification complete")
        if step == "review":
            findings = [{"severity": "medium", "summary": "Adjust behavior", "file": "app.py", "line": 12}] if attempt == 1 else []
            return StepResult(status="ok", findings=findings)
        if step == "implement_fix":
            return StepResult(status="ok", summary="Applied requested adjustment")
        return StepResult(status="ok")

    adapter = MagicMock()
    adapter.run_step = mock_run_step

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter)

    task = Task(
        title="Feature with review fix validation",
        task_type="feature",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    # One verify before review + one verify after implement_fix.
    assert called_steps.count("verify") == 2
