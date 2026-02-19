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

    # Metadata should be set
    assert task.metadata["workdoc_path"] == str(canonical)


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
    assert "SQL injection risk" in content
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
