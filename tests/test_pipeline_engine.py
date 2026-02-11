"""Tests for pipeline execution engine."""

import asyncio
import pytest
from pathlib import Path

from feature_prd_runner.pipelines.engine import PipelineEngine, PipelineExecution
from feature_prd_runner.pipelines.registry import PipelineRegistry
from feature_prd_runner.pipelines.steps.base import (
    PipelineStep,
    StepContext,
    StepOutcome,
    StepResult,
    step_registry,
)

# Ensure standard steps are loaded
from feature_prd_runner.pipelines.steps import standard  # noqa: F401


def _make_ctx(task_type: str = "feature") -> StepContext:
    return StepContext(
        task_id="task-test",
        task_type=task_type,
        task_title="Test task",
        task_description="Test description",
        project_dir=Path("/tmp/test"),
        state_dir=Path("/tmp/test/.prd_runner"),
        run_id="run-test",
    )


class TestStepRegistry:
    def test_standard_steps_registered(self):
        names = step_registry.list_steps()
        assert "plan" in names
        assert "implement" in names
        assert "verify" in names
        assert "review" in names
        assert "commit" in names
        assert "reproduce" in names
        assert "diagnose" in names
        assert "gather" in names
        assert "analyze" in names
        assert "summarize" in names
        assert "report" in names
        assert "scan_deps" in names
        assert "scan_code" in names
        assert "scan" in names
        assert "generate_tasks" in names

    def test_get_step(self):
        step = step_registry.get("plan")
        assert step.name == "plan"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown step"):
            step_registry.get("nonexistent_step")

    def test_has_step(self):
        assert step_registry.has("implement")
        assert not step_registry.has("nonexistent_step")


class TestPipelineEngine:
    def test_resolve_template(self):
        engine = PipelineEngine()
        tmpl = engine.resolve_template("feature")
        assert tmpl.id == "feature"

    def test_resolve_template_override(self):
        engine = PipelineEngine()
        tmpl = engine.resolve_template("feature", template_override="bug_fix")
        assert tmpl.id == "bug_fix"

    def test_create_execution(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)
        assert exe.task_id == "task-1"
        assert exe.template_id == "feature"
        assert len(exe.steps) == 6
        assert exe.status == "pending"

    def test_execute_all_steps(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)
        ctx = _make_ctx()

        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"
        assert result.progress == 1.0
        for step in result.steps:
            assert step.status == "completed"

    def test_execute_research_pipeline(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("research")
        exe = engine.create_execution("task-2", tmpl)
        ctx = _make_ctx(task_type="research")

        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"
        assert len(result.steps) == 4

    def test_execution_to_dict(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)
        d = exe.to_dict()
        assert d["task_id"] == "task-1"
        assert d["template_id"] == "feature"
        assert d["status"] == "pending"
        assert len(d["steps"]) == 6

    def test_skip_step(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)

        engine.skip_step(exe, 3)  # skip verify
        assert exe.steps[3].status == "skipped"

    def test_insert_step(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)

        original_count = len(exe.steps)
        engine.insert_step(exe, "analyze", after_index=1)
        assert len(exe.steps) == original_count + 1
        assert exe.steps[2].name == "analyze"

    def test_progress_tracking(self):
        engine = PipelineEngine()
        tmpl = engine.pipelines.get("feature")
        exe = engine.create_execution("task-1", tmpl)

        assert exe.progress == 0.0
        exe.steps[0].status = "completed"
        exe.steps[1].status = "completed"
        assert exe.progress == pytest.approx(2/6, rel=0.01)


class TestScheduler:
    def test_basic_scheduling(self):
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.scheduler import Scheduler

        registry = AgentRegistry()
        pool = AgentPool(registry=registry, max_agents=5)
        scheduler = Scheduler(pool, registry)

        # Spawn some idle agents
        pool.spawn("implementer")
        pool.spawn("reviewer")

        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "t2", "task_type": "review", "priority": "P0", "created_at": "2024-01-01T00:00:00Z"},
        ]

        assignments = scheduler.schedule(tasks)
        assert len(assignments) == 2

        # P0 task should be assigned first
        assert assignments[0].task_id == "t2"

    def test_affinity_matching(self):
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.scheduler import Scheduler

        registry = AgentRegistry()
        pool = AgentPool(registry=registry, max_agents=5)
        scheduler = Scheduler(pool, registry)

        pool.spawn("reviewer")
        pool.spawn("implementer")

        tasks = [
            {"id": "t1", "task_type": "review", "priority": "P2", "created_at": "2024-01-01T00:00:00Z"},
        ]

        assignments = scheduler.schedule(tasks)
        assert len(assignments) == 1
        assert assignments[0].agent_role == "reviewer"

    def test_auto_spawn(self):
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.scheduler import Scheduler

        registry = AgentRegistry()
        pool = AgentPool(registry=registry, max_agents=3)
        scheduler = Scheduler(pool, registry)

        # No agents spawned yet
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1", "created_at": "2024-01-01T00:00:00Z"},
        ]

        assignments = scheduler.schedule(tasks)
        assert len(assignments) == 1
        assert pool.active_count == 1
