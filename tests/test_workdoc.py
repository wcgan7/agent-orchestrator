"""Tests for the workdoc (working document) feature."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_orchestrator.runtime.domain.models import ReviewCycle, ReviewFinding, Task
from agent_orchestrator.runtime.events.bus import EventBus
from agent_orchestrator.runtime.orchestrator.service import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter
from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
from agent_orchestrator.runtime.storage.container import Container
from agent_orchestrator.pipelines.registry import BUILTIN_TEMPLATES


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    ensure_state_root(tmp_path)
    return tmp_path


@pytest.fixture()
def service(project_dir: Path) -> OrchestratorService:
    container = Container(project_dir)
    bus = EventBus(container.events, container.project_id)
    return OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())


@pytest.fixture()
def task() -> Task:
    return Task(
        title="Add login page",
        description="Build a login form with validation",
        task_type="feature",
        priority="P1",
    )


# ------------------------------------------------------------------
# _init_workdoc
# ------------------------------------------------------------------


def test_init_creates_canonical_and_worktree_copy(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    canonical = service._init_workdoc(task, project_dir)

    assert canonical.exists()
    assert canonical == service.container.state_root / "workdocs" / f"{task.id}.md"

    worktree = project_dir / ".workdoc.md"
    assert worktree.exists()

    # Contents should match
    assert canonical.read_text() == worktree.read_text()

    # Template should be rendered
    content = canonical.read_text()
    assert task.title in content
    assert task.id in content
    assert task.task_type in content
    assert "## Plan" in content
    assert "## Implementation Log" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Analysis" not in content
    assert "## Profiling Baseline" not in content
    assert "## Final Report" not in content

    # Metadata should be set
    assert task.metadata["workdoc_path"] == str(canonical)


def test_init_verify_only_uses_minimal_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Verify only",
        description="Run checks only",
        task_type="verify_only",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Verification Results" in content
    assert "## Final Report" in content
    assert "## Plan" not in content
    assert "## Analysis" not in content
    assert "## Implementation Log" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content


def test_init_plan_only_uses_planning_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Initiative plan",
        description="Define initiative scope and split to tasks",
        task_type="plan_only",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Analysis" in content
    assert "## Plan" in content
    assert "## Generated Tasks" in content
    assert "## Final Report" in content
    assert "## Implementation Log" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content


def test_init_security_audit_uses_scan_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Security sweep",
        description="Run dependency and code scans",
        task_type="security",
        priority="P1",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Dependency Scan Findings" in content
    assert "## Code Scan Findings" in content
    assert "## Security Report" in content
    assert "## Generated Remediation Tasks" in content
    assert "## Plan" not in content
    assert "## Implementation Log" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content


def test_init_repo_review_uses_review_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Repository review",
        description="Assess repo health and plan improvements",
        task_type="repo_review",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Repository Analysis" in content
    assert "## Initiative Plan" in content
    assert "## Generated Tasks" in content
    assert "## Implementation Log" not in content
    assert "## Verification Results" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content


def test_init_research_uses_minimal_research_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Research topic",
        description="Investigate architecture alternatives",
        task_type="research",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Research Analysis" in content
    assert "## Final Report" in content
    assert "## Plan" not in content
    assert "## Implementation Log" not in content
    assert "## Verification Results" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content


def test_init_review_uses_review_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Code review",
        description="Review changes and report findings",
        task_type="review",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Review Analysis" in content
    assert "## Review Findings" in content
    assert "## Final Report" in content
    assert "## Implementation Log" not in content
    assert "## Verification Results" not in content
    assert "## Fix Log" not in content


def test_init_bug_fix_uses_bug_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Fix auth crash",
        description="Resolve null pointer in auth flow",
        task_type="bug",
        priority="P1",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Reproduction Evidence" in content
    assert "## Diagnosis" in content
    assert "## Fix Implementation" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Plan" not in content
    assert "## Final Report" not in content


def test_init_refactor_uses_refactor_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Refactor auth flow",
        description="Restructure auth internals without behavior changes",
        task_type="refactor",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Refactor Analysis" in content
    assert "## Refactor Plan" in content
    assert "## Refactor Implementation" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Final Report" not in content
    assert "## Profiling Baseline" not in content


def test_init_performance_uses_performance_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Optimize API latency",
        description="Improve p95 response time under load",
        task_type="performance",
        priority="P1",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Profiling Baseline" in content
    assert "## Optimization Plan" in content
    assert "## Optimization Implementation" in content
    assert "## Benchmark Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Final Report" not in content
    assert "## Verification Results" not in content


def test_init_test_uses_test_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Improve auth tests",
        description="Add missing integration coverage",
        task_type="test",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Coverage Analysis" in content
    assert "## Test Implementation" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Plan" not in content
    assert "## Final Report" not in content


def test_init_docs_uses_docs_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Docs refresh",
        description="Update onboarding and API usage docs",
        task_type="docs",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Documentation Analysis" in content
    assert "## Documentation Updates" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Plan" not in content
    assert "## Final Report" not in content


def test_init_hotfix_uses_hotfix_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Urgent auth fix",
        description="Patch production login issue",
        task_type="hotfix",
        priority="P0",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Hotfix Implementation" in content
    assert "## Verification Results" in content
    assert "## Review Findings" in content
    assert "## Fix Log" in content
    assert "## Plan" not in content
    assert "## Analysis" not in content
    assert "## Final Report" not in content


def test_init_chore_uses_chore_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Formatting cleanup",
        description="Apply standard formatting rules",
        task_type="chore",
        priority="P3",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Chore Implementation" in content
    assert "## Verification Results" in content
    assert "## Plan" not in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content
    assert "## Final Report" not in content


def test_init_spike_uses_spike_template(service: OrchestratorService, project_dir: Path) -> None:
    task = Task(
        title="Evaluate caching strategy",
        description="Timebox an exploratory prototype",
        task_type="spike",
        priority="P2",
    )
    canonical = service._init_workdoc(task, project_dir)
    content = canonical.read_text()
    assert "## Spike Analysis" in content
    assert "## Prototype Notes" in content
    assert "## Final Report" in content
    assert "## Review Findings" not in content
    assert "## Fix Log" not in content
    assert "## Verification Results" not in content


def test_builtin_pipelines_use_dedicated_templates(service: OrchestratorService) -> None:
    """All built-in pipelines should avoid falling back to generic workdoc template."""
    representative_task_types = {
        "feature": "feature",
        "bug_fix": "bug",
        "refactor": "refactor",
        "research": "research",
        "docs": "docs",
        "test": "test",
        "repo_review": "repo_review",
        "security_audit": "security",
        "review": "review",
        "performance": "performance",
        "hotfix": "hotfix",
        "spike": "spike",
        "chore": "chore",
        "plan_only": "initiative_plan",
        "verify_only": "verify_only",
    }
    assert set(representative_task_types.keys()) == set(BUILTIN_TEMPLATES.keys())

    for pipeline_id, task_type in representative_task_types.items():
        task = Task(title=f"Template check {pipeline_id}", task_type=task_type)
        template = service._workdoc_template_for_task(task)
        assert template != service._GENERIC_WORKDOC_TEMPLATE


def test_bootstrap_adds_workdoc_gitignore_entry(project_dir: Path) -> None:
    """Bootstrap's _ensure_gitignored adds .workdoc.md to .gitignore."""
    gitignore = project_dir / ".gitignore"
    assert gitignore.exists()
    lines = gitignore.read_text().splitlines()
    assert ".workdoc.md" in [line.strip() for line in lines]


def test_bootstrap_gitignore_idempotent(project_dir: Path) -> None:
    """Running ensure_state_root twice should not duplicate .gitignore entries."""
    from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
    ensure_state_root(project_dir)  # second call (first was in fixture)
    gitignore = project_dir / ".gitignore"
    content = gitignore.read_text()
    assert content.count(".workdoc.md") == 1


def test_bootstrap_gitignore_preserves_existing(tmp_path: Path) -> None:
    """Existing .gitignore content should be preserved."""
    from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n.env\n", encoding="utf-8")
    ensure_state_root(tmp_path)
    content = gitignore.read_text()
    assert "node_modules/" in content
    assert ".env" in content
    assert ".workdoc.md" in content


# ------------------------------------------------------------------
# _cleanup_workdoc_for_commit
# ------------------------------------------------------------------


def test_cleanup_removes_worktree_copy(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)
    worktree = project_dir / ".workdoc.md"
    assert worktree.exists()

    service._cleanup_workdoc_for_commit(project_dir)
    assert not worktree.exists()


def test_cleanup_noop_when_no_worktree_copy(project_dir: Path) -> None:
    """Should not raise when .workdoc.md doesn't exist."""
    OrchestratorService._cleanup_workdoc_for_commit(project_dir)
    assert not (project_dir / ".workdoc.md").exists()


# ------------------------------------------------------------------
# _refresh_workdoc
# ------------------------------------------------------------------


def test_refresh_copies_canonical_to_worktree(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)

    # Modify canonical to simulate orchestrator update
    updated = canonical.read_text().replace("_Pending: will be populated by the plan step._", "The actual plan.")
    canonical.write_text(updated, encoding="utf-8")

    service._refresh_workdoc(task, project_dir)

    assert worktree.read_text() == updated


def test_refresh_noop_when_no_canonical(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """If canonical doesn't exist, refresh should be a no-op."""
    service._refresh_workdoc(task, project_dir)
    # No exception raised, worktree not created
    assert not (project_dir / ".workdoc.md").exists()


# ------------------------------------------------------------------
# _sync_workdoc
# ------------------------------------------------------------------


def test_sync_accepts_worker_changes(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)
    worktree = service._workdoc_worktree_path(project_dir)
    canonical = service._workdoc_canonical_path(task.id)

    # Worker modifies worktree
    modified = worktree.read_text().replace(
        "_Pending: will be populated by the plan step._",
        "# My detailed plan\n1. Do X\n2. Do Y",
    )
    worktree.write_text(modified, encoding="utf-8")

    service._sync_workdoc(task, "plan", project_dir, "some summary")

    # Canonical should now match the worker's version
    assert canonical.read_text() == modified


def test_sync_fallback_appends_summary(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """When worker didn't change the file, fallback should replace the placeholder."""
    service._init_workdoc(task, project_dir)

    service._sync_workdoc(task, "plan", project_dir, "Step 1: Do the thing\nStep 2: Verify")

    canonical = service._workdoc_canonical_path(task.id)
    content = canonical.read_text()
    assert "Step 1: Do the thing" in content
    assert "_Pending: will be populated by the plan step._" not in content


def test_sync_fallback_writes_analyze_to_analysis_section(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """Analyze summaries should be persisted into ## Analysis."""
    task = Task(
        title="Refactor auth module",
        description="Consolidate auth checks",
        task_type="refactor",
        priority="P2",
    )
    service._init_workdoc(task, project_dir)

    service._sync_workdoc(task, "analyze", project_dir, "Current state: duplicated auth checks in two modules.")

    canonical = service._workdoc_canonical_path(task.id)
    content = canonical.read_text()
    assert "Current state: duplicated auth checks in two modules." in content
    assert "_Pending: will be populated by the analyze step._" not in content


def test_sync_fallback_appends_under_existing_content(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """When placeholder is already replaced, summary appends under the heading."""
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)

    # Replace placeholder first
    text = canonical.read_text().replace(
        "_Pending: will be populated by the verify step._",
        "All tests passed.",
    )
    canonical.write_text(text, encoding="utf-8")
    worktree.write_text(text, encoding="utf-8")

    service._sync_workdoc(task, "verify", project_dir, "Lint clean.")

    content = canonical.read_text()
    assert "All tests passed." in content
    assert "Lint clean." in content


def test_sync_verify_ignores_worker_changes_and_uses_summary(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """Verify step should not accept worker edits to workdoc; orchestrator writes summary."""
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)
    before = canonical.read_text()

    # Worker attempts to mutate verification section directly.
    modified = worktree.read_text().replace(
        "_Pending: will be populated by the verify step._",
        "worker-edited verification text",
    )
    worktree.write_text(modified, encoding="utf-8")

    service._sync_workdoc(task, "verify", project_dir, "orchestrator verify summary")

    content = canonical.read_text()
    assert content != modified
    assert "orchestrator verify summary" in content
    assert "worker-edited verification text" not in content
    assert content != before


def test_sync_implement_fix_ignores_worker_changes_and_formats_cycles(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """implement_fix writes should be orchestrator-managed with cycle headings."""
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)

    # Worker tries to write directly to Fix Log; should be ignored.
    modified = worktree.read_text().replace(
        "_Pending: will be populated as needed._",
        "worker-edited fix log",
    )
    worktree.write_text(modified, encoding="utf-8")

    service._sync_workdoc(task, "implement_fix", project_dir, "Fixed null check in parser.", attempt=2)
    service._sync_workdoc(task, "implement_fix", project_dir, "Adjusted edge-case tests.", attempt=3)

    content = canonical.read_text()
    assert "worker-edited fix log" not in content
    assert "### Fix Cycle 2" in content
    assert "Fixed null check in parser." in content
    assert "### Fix Cycle 3" in content
    assert "Adjusted edge-case tests." in content


def test_sync_report_ignores_worker_changes_and_uses_summary(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """Report step should be orchestrator-managed and append into Final Report."""
    task = Task(
        title="Research report",
        description="Investigate DB options",
        task_type="research",
        priority="P2",
    )
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)
    before = canonical.read_text()

    modified = worktree.read_text().replace(
        "_Pending: will be populated by the report step._",
        "worker-edited report text",
    )
    worktree.write_text(modified, encoding="utf-8")

    service._sync_workdoc(task, "report", project_dir, "Final status: done with known risk A.")

    content = canonical.read_text()
    assert content != modified
    assert "Final status: done with known risk A." in content
    assert "worker-edited report text" not in content
    assert content != before


def test_sync_profile_ignores_worker_changes_and_uses_summary(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """Profile step should be orchestrator-managed and append into Profiling Baseline."""
    task = Task(
        title="Performance baseline",
        description="Profile request latency",
        task_type="performance",
        priority="P1",
    )
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    worktree = service._workdoc_worktree_path(project_dir)
    before = canonical.read_text()

    modified = worktree.read_text().replace(
        "_Pending: will be populated by the profile step._",
        "worker-edited profile text",
    )
    worktree.write_text(modified, encoding="utf-8")

    service._sync_workdoc(task, "profile", project_dir, "Baseline p95=220ms, cpu=78% under load test.")

    content = canonical.read_text()
    assert content != modified
    assert "Baseline p95=220ms, cpu=78% under load test." in content
    assert "worker-edited profile text" not in content
    assert content != before


def test_sync_noop_when_no_canonical(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """When canonical doesn't exist, sync should be a no-op."""
    service._sync_workdoc(task, "plan", project_dir, "some summary")
    # No exception raised


def test_sync_noop_when_no_summary_and_no_worker_change(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """When worker didn't change and summary is empty, nothing changes."""
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    before = canonical.read_text()

    service._sync_workdoc(task, "plan", project_dir, "")

    assert canonical.read_text() == before


def test_sync_no_heading_for_unknown_step(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """Steps not in the section map should not modify the workdoc."""
    service._init_workdoc(task, project_dir)
    canonical = service._workdoc_canonical_path(task.id)
    before = canonical.read_text()

    service._sync_workdoc(task, "commit", project_dir, "committed SHA abc123")

    assert canonical.read_text() == before


def test_commit_plan_revision_updates_workdoc_plan_section(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    """Committing a plan revision should refresh the workdoc ## Plan section."""
    service.container.tasks.upsert(task)
    service._init_workdoc(task, project_dir)

    base = service.create_plan_revision(
        task_id=task.id,
        content="Initial plan text",
        source="worker_plan",
        step="plan",
    )
    manual = service.create_plan_revision(
        task_id=task.id,
        content="Manually edited committed plan",
        source="human_edit",
        parent_revision_id=base.id,
    )

    service.commit_plan_revision(task.id, manual.id)

    content = service._workdoc_canonical_path(task.id).read_text()
    assert "Manually edited committed plan" in content
    assert "_Pending: will be populated by the plan step._" not in content


# ------------------------------------------------------------------
# _sync_workdoc_review
# ------------------------------------------------------------------


def test_review_sync_appends_findings(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)

    findings = [
        ReviewFinding(
            id="f1", task_id=task.id, severity="high",
            category="security", summary="SQL injection risk",
            file="app/db.py", line=42,
        ),
        ReviewFinding(
            id="f2", task_id=task.id, severity="low",
            category="style", summary="Missing docstring",
        ),
    ]
    cycle = ReviewCycle(
        task_id=task.id,
        attempt=1,
        findings=findings,
        open_counts={"critical": 0, "high": 1, "medium": 0, "low": 1},
        decision="changes_requested",
    )

    service._sync_workdoc_review(task, cycle, project_dir)

    canonical = service._workdoc_canonical_path(task.id)
    content = canonical.read_text()
    assert "### Review Cycle 1" in content
    assert "changes_requested" in content
    assert "Open findings: critical=0, high=1, medium=0, low=1" in content
    assert "SQL injection risk" in content
    assert "[security]" in content
    assert "app/db.py:42" in content
    assert "Missing docstring" in content
    assert "_Pending: will be populated by the review step._" not in content


def test_review_sync_file_without_line(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    """Finding with file but no line should render file only, not 'file:None'."""
    service._init_workdoc(task, project_dir)

    findings = [
        ReviewFinding(
            id="f1", task_id=task.id, severity="medium",
            category="quality", summary="Missing tests",
            file="app/views.py", line=None,
        ),
    ]
    cycle = ReviewCycle(
        task_id=task.id,
        attempt=1,
        findings=findings,
        open_counts={"critical": 0, "high": 0, "medium": 1, "low": 0},
        decision="changes_requested",
    )

    service._sync_workdoc_review(task, cycle, project_dir)

    content = service._workdoc_canonical_path(task.id).read_text()
    assert "(app/views.py)" in content
    assert "None" not in content


def test_review_sync_appends_multiple_cycles(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)

    for attempt in (1, 2):
        findings = [
            ReviewFinding(
                id=f"f{attempt}", task_id=task.id, severity="medium",
                category="quality", summary=f"Issue from cycle {attempt}",
            ),
        ]
        cycle = ReviewCycle(
            task_id=task.id,
            attempt=attempt,
            findings=findings,
            open_counts={"critical": 0, "high": 0, "medium": 1, "low": 0},
            decision="changes_requested" if attempt == 1 else "approved",
        )
        service._sync_workdoc_review(task, cycle, project_dir)

    content = service._workdoc_canonical_path(task.id).read_text()
    assert "### Review Cycle 1" in content
    assert "### Review Cycle 2" in content
    assert "Issue from cycle 1" in content
    assert "Issue from cycle 2" in content


def test_review_sync_includes_suggested_fix_and_status(
    service: OrchestratorService, task: Task, project_dir: Path
) -> None:
    service._init_workdoc(task, project_dir)
    findings = [
        ReviewFinding(
            id="f1",
            task_id=task.id,
            severity="high",
            category="correctness",
            summary="Null access in edge path",
            file="app/core.py",
            line=7,
            suggested_fix="Guard for None before dereference.",
            status="resolved",
        ),
    ]
    cycle = ReviewCycle(
        task_id=task.id,
        attempt=1,
        findings=findings,
        open_counts={"critical": 0, "high": 0, "medium": 0, "low": 0},
        decision="approved",
    )
    service._sync_workdoc_review(task, cycle, project_dir)

    content = service._workdoc_canonical_path(task.id).read_text()
    assert "[correctness]" in content
    assert "Suggested fix: Guard for None before dereference." in content
    assert "Status: resolved" in content


# ------------------------------------------------------------------
# get_workdoc
# ------------------------------------------------------------------


def test_get_workdoc_returns_content(service: OrchestratorService, task: Task, project_dir: Path) -> None:
    service._init_workdoc(task, project_dir)

    result = service.get_workdoc(task.id)
    assert result["task_id"] == task.id
    assert result["exists"] is True
    assert task.title in result["content"]


def test_get_workdoc_returns_not_exists(service: OrchestratorService, task: Task) -> None:
    result = service.get_workdoc(task.id)
    assert result["task_id"] == task.id
    assert result["exists"] is False
    assert result["content"] is None
