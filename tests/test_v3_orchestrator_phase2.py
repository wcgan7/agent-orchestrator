from __future__ import annotations

import subprocess
from pathlib import Path

from feature_prd_runner.v3.domain.models import AgentRecord, Task
from feature_prd_runner.v3.events import EventBus
from feature_prd_runner.v3.orchestrator import OrchestratorService
from feature_prd_runner.v3.storage.container import V3Container


def _service(tmp_path: Path) -> tuple[V3Container, OrchestratorService]:
    container = V3Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service


def test_review_loop_retries_until_findings_clear(tmp_path: Path) -> None:
    container, service = _service(tmp_path)

    task = Task(
        title="Loop task",
        status="ready",
        approval_mode="auto_approve",
        metadata={
            "scripted_findings": [
                [{"severity": "high", "summary": "Fix me"}],
                [],
            ]
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    assert result.retry_count >= 1
    cycles = container.reviews.for_task(task.id)
    assert len(cycles) == 2
    assert cycles[0].decision == "changes_requested"
    assert cycles[1].decision == "approved"


def test_review_loop_cap_moves_task_to_blocked(tmp_path: Path) -> None:
    container, service = _service(tmp_path)
    cfg = container.config.load()
    orchestrator_cfg = dict(cfg.get("orchestrator") or {})
    orchestrator_cfg["max_review_attempts"] = 2
    cfg["orchestrator"] = orchestrator_cfg
    container.config.save(cfg)

    task = Task(
        title="Cap task",
        status="ready",
        approval_mode="auto_approve",
        metadata={
            "scripted_findings": [
                [{"severity": "high", "summary": "Fix me"}],
                [{"severity": "medium", "summary": "Still broken"}],
            ]
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "blocked"
    assert "Review attempt cap exceeded" in str(result.error)
    cycles = container.reviews.for_task(task.id)
    assert len(cycles) == 2


def test_agent_role_routing_and_provider_override(tmp_path: Path) -> None:
    container, service = _service(tmp_path)
    cfg = container.config.load()
    cfg["agent_routing"] = {
        "default_role": "general",
        "task_type_roles": {"feature": "implementer"},
        "role_provider_overrides": {"implementer": "codex"},
    }
    container.config.save(cfg)

    impl = AgentRecord(role="implementer", status="running")
    other = AgentRecord(role="reviewer", status="running")
    container.agents.upsert(impl)
    container.agents.upsert(other)

    task = Task(title="Route task", status="ready", approval_mode="auto_approve")
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.current_agent_id == impl.id
    assert result.metadata.get("provider_override") == "codex"


def test_single_run_branch_receives_per_task_commits(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True)

    container, service = _service(tmp_path)

    first = Task(title="First task", status="ready", approval_mode="auto_approve")
    second = Task(title="Second task", status="ready", approval_mode="auto_approve")
    container.tasks.upsert(first)
    container.tasks.upsert(second)

    one = service.run_task(first.id)
    two = service.run_task(second.id)

    assert one.status == "done"
    assert two.status == "done"
    assert service.status()["run_branch"] is not None

    branch = subprocess.run(["git", "branch", "--show-current"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    assert branch == service.status()["run_branch"]

    log = subprocess.run(["git", "log", "--pretty=%s", "-n", "2"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.splitlines()
    assert len(log) == 2
    assert log[0].startswith(f"task({second.id})")
    assert log[1].startswith(f"task({first.id})")


def test_scheduler_respects_priority_dependency_and_repo_conflict(tmp_path: Path) -> None:
    container, _ = _service(tmp_path)
    high = Task(title="High", status="ready", priority="P0", metadata={"repo_path": "/repo/a"})
    mid = Task(title="Mid", status="ready", priority="P1", metadata={"repo_path": "/repo/b"})
    low = Task(title="Low", status="ready", priority="P2", metadata={"repo_path": "/repo/c"})
    blocked = Task(title="Blocked", status="ready", priority="P0", blocked_by=["missing-task"])
    container.tasks.upsert(high)
    container.tasks.upsert(mid)
    container.tasks.upsert(low)
    container.tasks.upsert(blocked)

    claimed_first = container.tasks.claim_next_runnable(max_in_progress=3, repo_conflicts={"/repo/a"})
    assert claimed_first is not None
    assert claimed_first.id == mid.id

    claimed_second = container.tasks.claim_next_runnable(max_in_progress=3, repo_conflicts=set())
    assert claimed_second is not None
    assert claimed_second.id == high.id


def test_scheduler_enforces_concurrency_cap(tmp_path: Path) -> None:
    container, _ = _service(tmp_path)
    running = Task(title="Running", status="in_progress")
    queued = Task(title="Queued", status="ready")
    container.tasks.upsert(running)
    container.tasks.upsert(queued)

    claimed = container.tasks.claim_next_runnable(max_in_progress=1, repo_conflicts=set())
    assert claimed is None
