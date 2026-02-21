from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent_orchestrator.runtime.domain.models import AgentRecord, Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult
from agent_orchestrator.runtime.storage.container import Container


def _service(tmp_path: Path) -> tuple[Container, OrchestratorService]:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service


def test_review_loop_retries_until_findings_clear(tmp_path: Path) -> None:
    container, service = _service(tmp_path)

    task = Task(
        title="Loop task",
        status="queued",
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
        status="queued",
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

    task = Task(title="Route task", status="queued", approval_mode="auto_approve")
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

    first = Task(title="First task", status="queued", approval_mode="auto_approve", metadata={"scripted_files": {"first.txt": "first"}})
    second = Task(title="Second task", status="queued", approval_mode="auto_approve", metadata={"scripted_files": {"second.txt": "second"}})
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


def test_single_run_branch_uses_fast_forward_even_when_merge_ff_disabled(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "merge.ff", "false"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True)

    container, service = _service(tmp_path)

    first = Task(title="First task", status="queued", approval_mode="auto_approve", metadata={"scripted_files": {"first.txt": "first"}})
    second = Task(title="Second task", status="queued", approval_mode="auto_approve", metadata={"scripted_files": {"second.txt": "second"}})
    container.tasks.upsert(first)
    container.tasks.upsert(second)

    one = service.run_task(first.id)
    two = service.run_task(second.id)

    assert one.status == "done"
    assert two.status == "done"

    log = subprocess.run(["git", "log", "--pretty=%s", "-n", "2"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.splitlines()
    assert len(log) == 2
    assert log[0].startswith(f"task({second.id})")
    assert log[1].startswith(f"task({first.id})")


def test_scheduler_respects_priority_and_dependency(tmp_path: Path) -> None:
    container, _ = _service(tmp_path)
    high = Task(title="High", status="queued", priority="P0")
    mid = Task(title="Mid", status="queued", priority="P1")
    low = Task(title="Low", status="queued", priority="P2")
    blocked = Task(title="Blocked", status="queued", priority="P0", blocked_by=["missing-task"])
    container.tasks.upsert(high)
    container.tasks.upsert(mid)
    container.tasks.upsert(low)
    container.tasks.upsert(blocked)

    claimed_first = container.tasks.claim_next_runnable(max_in_progress=5)
    assert claimed_first is not None
    assert claimed_first.id == high.id

    claimed_second = container.tasks.claim_next_runnable(max_in_progress=5)
    assert claimed_second is not None
    assert claimed_second.id == mid.id


def test_scheduler_enforces_concurrency_cap(tmp_path: Path) -> None:
    container, _ = _service(tmp_path)
    running = Task(title="Running", status="in_progress")
    queued = Task(title="Queued", status="queued")
    container.tasks.upsert(running)
    container.tasks.upsert(queued)

    claimed = container.tasks.claim_next_runnable(max_in_progress=1)
    assert claimed is None


def test_human_blocking_step_sets_pending_gate_and_blocks_task(tmp_path: Path) -> None:
    container, service = _service(tmp_path)
    task = Task(
        title="Needs decision",
        status="queued",
        approval_mode="auto_approve",
        metadata={
            "scripted_steps": {
                "plan": {
                    "status": "human_blocked",
                    "summary": "Need API token",
                    "human_blocking_issues": [{"summary": "Need API token"}],
                }
            }
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "blocked"
    assert result.pending_gate == "human_intervention"
    assert result.error == "Need API token"
    assert result.metadata.get("human_blocking_issues") == [{"summary": "Need API token"}]


def test_review_human_blocking_creates_step_log(tmp_path: Path) -> None:
    """When review returns human_blocking_issues, a step log entry with
    stdout_path should still be created so historical logs are retrievable."""
    container, service = _service(tmp_path)
    task = Task(
        title="Review block",
        status="queued",
        task_type="feature",
        approval_mode="auto_approve",
        metadata={
            "scripted_steps": {
                "review": {
                    "status": "human_blocked",
                    "summary": "Needs human decision",
                    "human_blocking_issues": [{"summary": "Ambiguous requirement"}],
                }
            }
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"

    # The run should have a step entry for "review" even though it was blocked early
    runs = [r for r in container.runs.list() if r.task_id == task.id]
    assert len(runs) >= 1
    last_run = runs[-1]
    review_entries = [e for e in last_run.steps if isinstance(e, dict) and e.get("step") == "review"]
    assert len(review_entries) >= 1, "Review step log entry should exist even on human_blocking early exit"
    assert review_entries[-1]["status"] == "blocked"


def test_review_error_status_creates_step_log(tmp_path: Path) -> None:
    """When review returns a non-ok status, a step log entry should still
    be created so historical logs are retrievable."""
    container, service = _service(tmp_path)
    task = Task(
        title="Review error",
        status="queued",
        task_type="feature",
        approval_mode="auto_approve",
        metadata={
            "scripted_steps": {
                "review": {
                    "status": "error",
                    "summary": "Worker crashed during review",
                }
            }
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"

    runs = [r for r in container.runs.list() if r.task_id == task.id]
    assert len(runs) >= 1
    last_run = runs[-1]
    review_entries = [e for e in last_run.steps if isinstance(e, dict) and e.get("step") == "review"]
    assert len(review_entries) >= 1, "Review step log entry should exist even on error status"
    assert review_entries[-1]["status"] == "error"


def test_implement_fix_sets_pipeline_phase(tmp_path: Path) -> None:
    """pipeline_phase metadata should track the logical pipeline step,
    even when current_step is a sub-step like implement_fix."""
    captured_phases: list[dict[str, Any]] = []

    class SpyAdapter(DefaultWorkerAdapter):
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            captured_phases.append({
                "step": step,
                "attempt": attempt,
                "current_step": task.current_step,
                "pipeline_phase": task.metadata.get("pipeline_phase"),
            })
            return super().run_step(task=task, step=step, attempt=attempt)

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=SpyAdapter())

    task = Task(
        title="Pipeline phase test",
        status="queued",
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

    # During implement_fix calls, pipeline_phase should be "review"
    fix_calls = [c for c in captured_phases if c["step"] == "implement_fix"]
    assert len(fix_calls) >= 1, "Expected at least one implement_fix call"
    for call in fix_calls:
        assert call["pipeline_phase"] == "review", (
            f"implement_fix should have pipeline_phase='review', got {call['pipeline_phase']}"
        )

    # During review calls, pipeline_phase should be "review"
    review_calls = [c for c in captured_phases if c["step"] == "review"]
    assert len(review_calls) >= 1
    for call in review_calls:
        assert call["pipeline_phase"] == "review"

    # After completion, pipeline_phase should be cleared
    assert "pipeline_phase" not in result.metadata
