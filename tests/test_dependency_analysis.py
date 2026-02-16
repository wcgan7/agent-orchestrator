"""Tests for automatic dependency analysis feature."""
from __future__ import annotations

from pathlib import Path

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult
from agent_orchestrator.runtime.storage.container import Container


def _service(tmp_path: Path, *, auto_deps: bool = True, adapter=None) -> tuple[Container, OrchestratorService, EventBus]:
    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"auto_deps": auto_deps, "concurrency": 2}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter or DefaultWorkerAdapter())
    return container, service, bus


# ---------------------------------------------------------------------------
# 1. Independent tasks get no deps
# ---------------------------------------------------------------------------


def test_independent_tasks_get_no_deps(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    t1 = Task(title="Task A", status="queued", metadata={"scripted_dependency_edges": []})
    t2 = Task(title="Task B", status="queued", metadata={"scripted_dependency_edges": []})
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    service._maybe_analyze_dependencies()

    t1r = container.tasks.get(t1.id)
    t2r = container.tasks.get(t2.id)
    assert t1r.metadata.get("deps_analyzed") is True
    assert t2r.metadata.get("deps_analyzed") is True
    assert t1r.blocked_by == []
    assert t2r.blocked_by == []


# ---------------------------------------------------------------------------
# 2. Scripted edges applied
# ---------------------------------------------------------------------------


def test_scripted_edges_applied(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    t1 = Task(title="Add auth", status="queued")
    t2 = Task(title="Add profile", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Use a custom adapter that returns edges
    class EdgeAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            return StepResult(status="ok", dependency_edges=[
                {"from": t1.id, "to": t2.id, "reason": "Profile needs auth"}
            ])

    service.worker_adapter = EdgeAdapter()
    service._maybe_analyze_dependencies()

    t1r = container.tasks.get(t1.id)
    t2r = container.tasks.get(t2.id)
    assert t1.id in t2r.blocked_by
    assert t2.id in t1r.blocks
    assert isinstance(t2r.metadata.get("inferred_deps"), list)
    assert len(t2r.metadata["inferred_deps"]) == 1
    assert t2r.metadata["inferred_deps"][0]["from"] == t1.id


# ---------------------------------------------------------------------------
# 3. Single task skips analysis
# ---------------------------------------------------------------------------


def test_single_task_skips_analysis(tmp_path: Path) -> None:
    call_count = 0

    class CountingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal call_count
            call_count += 1
            return StepResult(status="ok", dependency_edges=[])

    container, service, _ = _service(tmp_path, adapter=CountingAdapter())
    t1 = Task(title="Solo task", status="queued")
    container.tasks.upsert(t1)

    service._maybe_analyze_dependencies()

    assert call_count == 0
    t1r = container.tasks.get(t1.id)
    assert t1r.metadata.get("deps_analyzed") is True


# ---------------------------------------------------------------------------
# 4. Already analyzed not reanalyzed
# ---------------------------------------------------------------------------


def test_already_analyzed_not_reanalyzed(tmp_path: Path) -> None:
    call_count = 0

    class CountingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal call_count
            call_count += 1
            return StepResult(status="ok", dependency_edges=[])

    container, service, _ = _service(tmp_path, adapter=CountingAdapter())
    t1 = Task(title="Task A", status="queued")
    t2 = Task(title="Task B", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    service._maybe_analyze_dependencies()
    assert call_count == 1

    # Second call should not invoke the adapter
    service._maybe_analyze_dependencies()
    assert call_count == 1


# ---------------------------------------------------------------------------
# 5. Cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    t1 = Task(title="Task A", status="queued")
    t2 = Task(title="Task B", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Adapter returns bidirectional edges (would create cycle)
    class CycleAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            return StepResult(status="ok", dependency_edges=[
                {"from": t1.id, "to": t2.id, "reason": "A before B"},
                {"from": t2.id, "to": t1.id, "reason": "B before A"},
            ])

    service.worker_adapter = CycleAdapter()
    service._maybe_analyze_dependencies()

    t1r = container.tasks.get(t1.id)
    t2r = container.tasks.get(t2.id)
    # First edge should be applied, second should be rejected
    assert t1.id in t2r.blocked_by
    assert t2.id not in t1r.blocked_by


# ---------------------------------------------------------------------------
# 6. PRD import tasks skipped
# ---------------------------------------------------------------------------


def test_prd_import_tasks_skipped(tmp_path: Path) -> None:
    call_count = 0

    class CountingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal call_count
            call_count += 1
            return StepResult(status="ok", dependency_edges=[])

    container, service, _ = _service(tmp_path, adapter=CountingAdapter())
    t1 = Task(title="PRD Task A", status="queued", source="prd_import")
    t2 = Task(title="PRD Task B", status="queued", source="prd_import")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    service._maybe_analyze_dependencies()
    assert call_count == 0


# ---------------------------------------------------------------------------
# 7. tick_once integrates dep analysis
# ---------------------------------------------------------------------------


def test_tick_once_integrates_dep_analysis(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    t1 = Task(title="Foundation", status="queued", task_type="chore",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Dependent", status="queued", task_type="chore",
              approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Set up adapter that infers dependency then does normal work
    analysis_done = False

    class DepThenWorkAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal analysis_done
            if step == "analyze_deps":
                analysis_done = True
                return StepResult(status="ok", dependency_edges=[
                    {"from": t1.id, "to": t2.id, "reason": "t2 depends on t1"}
                ])
            return StepResult(status="ok")

    service.worker_adapter = DepThenWorkAdapter()

    # First tick should run analysis then claim t1 (since t2 is now blocked)
    service.tick_once()
    assert analysis_done is True

    # t2 should be blocked by t1
    t2r = container.tasks.get(t2.id)
    assert t1.id in t2r.blocked_by


# ---------------------------------------------------------------------------
# 8. Analysis failure graceful
# ---------------------------------------------------------------------------


def test_analysis_failure_graceful(tmp_path: Path) -> None:
    container, service, _ = _service(tmp_path)
    t1 = Task(title="Task A", status="queued")
    t2 = Task(title="Task B", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    class FailingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            raise RuntimeError("LLM unavailable")

    service.worker_adapter = FailingAdapter()
    service._maybe_analyze_dependencies()

    # Tasks should still be marked analyzed
    t1r = container.tasks.get(t1.id)
    t2r = container.tasks.get(t2.id)
    assert t1r.metadata.get("deps_analyzed") is True
    assert t2r.metadata.get("deps_analyzed") is True
    # No deps should have been added
    assert t1r.blocked_by == []
    assert t2r.blocked_by == []


# ---------------------------------------------------------------------------
# 9. Auto deps disabled via config
# ---------------------------------------------------------------------------


def test_auto_deps_disabled_via_config(tmp_path: Path) -> None:
    call_count = 0

    class CountingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal call_count
            call_count += 1
            return StepResult(status="ok", dependency_edges=[])

    container, service, _ = _service(tmp_path, auto_deps=False, adapter=CountingAdapter())
    t1 = Task(title="Task A", status="queued")
    t2 = Task(title="Task B", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    service._maybe_analyze_dependencies()
    assert call_count == 0


# ---------------------------------------------------------------------------
# 10. New tasks trigger reanalysis
# ---------------------------------------------------------------------------


def test_new_tasks_trigger_reanalysis(tmp_path: Path) -> None:
    call_count = 0

    class CountingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            nonlocal call_count
            call_count += 1
            return StepResult(status="ok", dependency_edges=[])

    container, service, _ = _service(tmp_path, adapter=CountingAdapter())
    t1 = Task(title="Task A", status="queued")
    t2 = Task(title="Task B", status="queued")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    service._maybe_analyze_dependencies()
    assert call_count == 1

    # Add two new unanalyzed tasks
    t3 = Task(title="Task C", status="queued")
    t4 = Task(title="Task D", status="queued")
    container.tasks.upsert(t3)
    container.tasks.upsert(t4)

    service._maybe_analyze_dependencies()
    assert call_count == 2
