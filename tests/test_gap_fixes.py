"""Tests for gap-fix modules: reasoning, timeline, HITL enforcement, debugger agent,
security/performance audits, missing pipeline steps, feedback injection, JSON schema,
effectiveness tracking, and collaboration API endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Reasoning store
# ---------------------------------------------------------------------------

class TestReasoningStore:
    def test_get_or_create_new(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        r = store.get_or_create("task-1", "agent-abc", "implementer")
        assert r.agent_id == "agent-abc"
        assert r.agent_role == "implementer"
        assert r.task_id == "task-1"
        assert r.steps == []

    def test_get_or_create_existing(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        r1 = store.get_or_create("task-1", "agent-abc", "implementer")
        r2 = store.get_or_create("task-1", "agent-abc", "implementer")
        assert r1 is r2

    def test_get_or_create_different_agents(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        r1 = store.get_or_create("task-1", "agent-a", "implementer")
        r2 = store.get_or_create("task-1", "agent-b", "reviewer")
        assert r1 is not r2
        assert len(store.get_for_task("task-1")) == 2

    def test_get_for_task_empty(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        assert store.get_for_task("unknown") == []

    def test_start_step(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        step = store.start_step("task-1", "agent-a", "implementer", "plan", "planning approach")
        assert step.step_name == "plan"
        assert step.status == "running"
        assert step.reasoning == "planning approach"
        assert step.started_at is not None

        entry = store.get_for_task("task-1")[0]
        assert entry.current_step == "plan"
        assert len(entry.steps) == 1

    def test_complete_step(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        store.start_step("task-1", "agent-a", "implementer", "plan", "reasoning")
        ok = store.complete_step("task-1", "agent-a", "plan", status="completed", output="done")
        assert ok is True

        entry = store.get_for_task("task-1")[0]
        assert entry.steps[0].status == "completed"
        assert entry.steps[0].output == "done"
        assert entry.steps[0].completed_at is not None
        assert entry.current_step is None

    def test_complete_step_nonexistent(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        ok = store.complete_step("task-1", "agent-a", "plan")
        assert ok is False

    def test_clear_task(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        store = ReasoningStore()
        store.start_step("task-1", "agent-a", "implementer", "plan")
        store.clear_task("task-1")
        assert store.get_for_task("task-1") == []

    def test_reasoning_step_duration(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStep

        step = ReasoningStep(step_name="test", started_at=100.0, completed_at=100.5)
        assert step.duration_ms == 500.0

    def test_reasoning_step_no_duration(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStep

        step = ReasoningStep(step_name="test")
        assert step.duration_ms is None

    def test_reasoning_step_to_dict(self):
        from feature_prd_runner.collaboration.reasoning import ReasoningStep

        step = ReasoningStep(step_name="plan", status="completed", reasoning="r", output="o")
        d = step.to_dict()
        assert d["step_name"] == "plan"
        assert d["status"] == "completed"

    def test_agent_reasoning_to_dict(self):
        from feature_prd_runner.collaboration.reasoning import AgentReasoning

        r = AgentReasoning(agent_id="a1", agent_role="implementer", task_id="t1")
        d = r.to_dict()
        assert d["agent_id"] == "a1"
        assert d["steps"] == []


# ---------------------------------------------------------------------------
# Timeline aggregator
# ---------------------------------------------------------------------------

class TestTimelineAggregator:
    def _make_stores(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.collaboration.reasoning import ReasoningStore

        return FeedbackStore(), ReasoningStore()

    def test_empty_timeline(self):
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1")
        assert events == []

    def test_feedback_in_timeline(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackType, FeedbackPriority
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        fb_store.add_feedback(Feedback(
            task_id="task-1",
            summary="fix this",
            details="details here",
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.MUST,
        ))

        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1")
        assert len(events) == 1
        assert events[0].type == "feedback"
        assert "general" in events[0].summary.lower()

    def test_comment_in_timeline(self):
        from feature_prd_runner.collaboration.feedback import ReviewComment
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        fb_store.add_comment(ReviewComment(
            task_id="task-1",
            file_path="main.py",
            line_number=10,
            body="nit: fix style",
        ))

        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1")
        assert len(events) == 1
        assert events[0].type == "comment"
        assert "main.py" in events[0].summary

    def test_reasoning_in_timeline(self):
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        reasoning_store.start_step("task-1", "agent-1", "implementer", "plan", "thinking")

        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1")
        assert len(events) == 1
        assert events[0].type == "reasoning"
        assert events[0].actor == "agent-1"
        assert events[0].actor_type == "agent"

    def test_timeline_sorted_newest_first(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackType, FeedbackPriority
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        fb_store.add_feedback(Feedback(
            task_id="task-1",
            summary="first",
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.SUGGESTION,
        ))
        fb_store.add_feedback(Feedback(
            task_id="task-1",
            summary="second",
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.SUGGESTION,
        ))

        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1")
        assert len(events) == 2
        assert events[0].timestamp >= events[1].timestamp

    def test_timeline_limit(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackType, FeedbackPriority
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, reasoning_store = self._make_stores()
        for i in range(10):
            fb_store.add_feedback(Feedback(
                task_id="task-1",
                summary=f"item-{i}",
                feedback_type=FeedbackType.GENERAL,
                priority=FeedbackPriority.SUGGESTION,
            ))

        agg = TimelineAggregator(fb_store, reasoning_store)
        events = agg.get_timeline("task-1", limit=3)
        assert len(events) == 3

    def test_timeline_without_reasoning_store(self):
        from feature_prd_runner.collaboration.timeline import TimelineAggregator

        fb_store, _ = self._make_stores()
        agg = TimelineAggregator(fb_store, reasoning_store=None)
        events = agg.get_timeline("task-1")
        assert events == []


# ---------------------------------------------------------------------------
# HITL mode enforcement in pipeline engine
# ---------------------------------------------------------------------------

class TestHITLEnforcement:
    def test_should_gate_autopilot_no_gates(self):
        from feature_prd_runner.collaboration.modes import should_gate

        assert should_gate("autopilot", "before_plan") is False
        assert should_gate("autopilot", "before_implement") is False
        assert should_gate("autopilot", "before_commit") is False
        assert should_gate("autopilot", "after_implement") is False

    def test_should_gate_supervised_all_gates(self):
        from feature_prd_runner.collaboration.modes import should_gate

        assert should_gate("supervised", "before_plan") is True
        assert should_gate("supervised", "before_implement") is True
        assert should_gate("supervised", "before_commit") is True

    def test_should_gate_collaborative(self):
        from feature_prd_runner.collaboration.modes import should_gate

        assert should_gate("collaborative", "before_plan") is False
        assert should_gate("collaborative", "after_implement") is True
        assert should_gate("collaborative", "before_commit") is True

    def test_should_gate_review_only(self):
        from feature_prd_runner.collaboration.modes import should_gate

        assert should_gate("review_only", "after_implement") is True
        assert should_gate("review_only", "before_commit") is True
        assert should_gate("review_only", "before_plan") is False

    def test_engine_gate_mapping(self):
        from feature_prd_runner.pipelines.engine import PipelineEngine

        engine = PipelineEngine()
        assert engine._step_gate("plan") == "before_plan"
        assert engine._step_gate("implement") == "before_implement"
        assert engine._step_gate("commit") == "before_commit"
        assert engine._step_gate("review") == "after_implement"
        assert engine._step_gate("verify") is None

    def test_engine_blocks_on_supervised_mode(self):
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef

        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-gate",
            display_name="Test Gate",
            description="Test",
            task_types=("feature",),
            steps=(StepDef(name="plan"),),
        )
        reg.register(template)

        engine = PipelineEngine(pipeline_registry=reg, hitl_mode="supervised")
        execution = engine.create_execution("task-1", template)

        from feature_prd_runner.pipelines.steps.base import StepContext
        ctx = StepContext(task_id="task-1", task_type="feature", task_title="Test", task_description="", project_dir=Path("/tmp"), state_dir=Path("/tmp"), run_id="run-1")
        result = asyncio.run(engine.execute(execution, ctx))

        assert result.status == "failed"
        assert result.steps[0].status == "failed"
        assert "Approval required" in result.steps[0].result.message

    def test_engine_no_block_autopilot(self):
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef

        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-nogate",
            display_name="Test No Gate",
            description="Test",
            task_types=("feature",),
            steps=(StepDef(name="plan"),),
        )
        reg.register(template)

        engine = PipelineEngine(pipeline_registry=reg, hitl_mode="autopilot")
        execution = engine.create_execution("task-1", template)

        from feature_prd_runner.pipelines.steps.base import StepContext
        ctx = StepContext(task_id="task-1", task_type="feature", task_title="Test", task_description="", project_dir=Path("/tmp"), state_dir=Path("/tmp"), run_id="run-1")
        result = asyncio.run(engine.execute(execution, ctx))

        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Debugger agent type
# ---------------------------------------------------------------------------

class TestDebuggerAgent:
    def test_debugger_in_builtin_types(self):
        from feature_prd_runner.agents.registry import BUILTIN_AGENT_TYPES, AgentRole

        assert "debugger" in BUILTIN_AGENT_TYPES
        debugger = BUILTIN_AGENT_TYPES["debugger"]
        assert debugger.role == AgentRole.DEBUGGER
        assert "reproduce" in debugger.allowed_steps
        assert "diagnose" in debugger.allowed_steps
        assert "implement" in debugger.allowed_steps
        assert "verify" in debugger.allowed_steps

    def test_debugger_task_affinity(self):
        from feature_prd_runner.agents.registry import BUILTIN_AGENT_TYPES

        debugger = BUILTIN_AGENT_TYPES["debugger"]
        assert "bug" in debugger.task_type_affinity

    def test_registry_finds_debugger_for_bug(self):
        from feature_prd_runner.agents.registry import AgentRegistry

        reg = AgentRegistry()
        role = reg.best_role_for_task_type("bug")
        # Could be debugger or implementer (implementer also has 'bug' affinity)
        assert role in ("debugger", "implementer")

    def test_create_debugger_instance(self):
        from feature_prd_runner.agents.registry import AgentRegistry

        reg = AgentRegistry()
        instance = reg.create_instance("debugger")
        assert instance.agent_type == "debugger"
        assert instance.display_name == "Debugger"


# ---------------------------------------------------------------------------
# Missing pipeline steps (reproduce, diagnose, analyze_coverage, plan_tests)
# ---------------------------------------------------------------------------

class TestMissingPipelineSteps:
    def test_reproduce_step_registered(self):
        from feature_prd_runner.pipelines.steps.base import step_registry
        # Ensure standard is imported
        import feature_prd_runner.pipelines.steps.standard  # noqa

        assert step_registry.has("reproduce")

    def test_diagnose_step_registered(self):
        from feature_prd_runner.pipelines.steps.base import step_registry
        import feature_prd_runner.pipelines.steps.standard  # noqa

        assert step_registry.has("diagnose")

    def test_analyze_coverage_step_registered(self):
        from feature_prd_runner.pipelines.steps.base import step_registry
        import feature_prd_runner.pipelines.steps.standard  # noqa

        assert step_registry.has("analyze_coverage")

    def test_plan_tests_step_registered(self):
        from feature_prd_runner.pipelines.steps.base import step_registry
        import feature_prd_runner.pipelines.steps.standard  # noqa

        assert step_registry.has("plan_tests")

    def test_analyze_coverage_execute(self):
        from feature_prd_runner.pipelines.steps.base import step_registry, StepContext, StepOutcome
        import feature_prd_runner.pipelines.steps.standard  # noqa

        step = step_registry.get("analyze_coverage")
        ctx = StepContext(task_id="t1", task_type="test", task_title="T", task_description="", project_dir=Path("/tmp"), state_dir=Path("/tmp"), run_id="r1")
        result = asyncio.run(step.execute(ctx))
        assert result.outcome == StepOutcome.SUCCESS

    def test_plan_tests_execute(self):
        from feature_prd_runner.pipelines.steps.base import step_registry, StepContext, StepOutcome
        import feature_prd_runner.pipelines.steps.standard  # noqa

        step = step_registry.get("plan_tests")
        ctx = StepContext(task_id="t1", task_type="test", task_title="T", task_description="", project_dir=Path("/tmp"), state_dir=Path("/tmp"), run_id="r1")
        result = asyncio.run(step.execute(ctx))
        assert result.outcome == StepOutcome.SUCCESS


# ---------------------------------------------------------------------------
# Security audit generator
# ---------------------------------------------------------------------------

class TestSecurityAuditGenerator:
    def test_scan_secrets_finds_api_key(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        # Create a file with a hardcoded secret
        src = tmp_path / "config.py"
        src.write_text('API_KEY = "sk-1234567890abcdef"\n')

        gen = SecurityAuditGenerator()
        tasks = gen._scan_secrets(tmp_path)
        assert len(tasks) == 1
        assert "secret" in tasks[0].title.lower() or "hardcoded" in tasks[0].title.lower()

    def test_scan_secrets_ignores_short_values(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        src = tmp_path / "config.py"
        src.write_text('password = "short"\n')

        gen = SecurityAuditGenerator()
        tasks = gen._scan_secrets(tmp_path)
        assert len(tasks) == 0

    def test_scan_secrets_ignores_node_modules(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "config.js").write_text('const API_KEY = "sk-1234567890abcdef"\n')

        gen = SecurityAuditGenerator()
        tasks = gen._scan_secrets(tmp_path)
        assert len(tasks) == 0

    def test_scan_antipatterns_finds_eval(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        src = tmp_path / "bad.py"
        src.write_text('result = eval(user_input)\n')

        gen = SecurityAuditGenerator()
        tasks = gen._scan_antipatterns(tmp_path)
        assert len(tasks) == 1
        assert "anti-pattern" in tasks[0].title.lower() or "security" in tasks[0].title.lower()

    def test_scan_antipatterns_finds_innerhtml(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        src = tmp_path / "app.tsx"
        src.write_text('el.innerHTML = data;\n')

        gen = SecurityAuditGenerator()
        tasks = gen._scan_antipatterns(tmp_path)
        assert len(tasks) == 1

    def test_generate_calls_all_scans(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.security_audit import SecurityAuditGenerator

        gen = SecurityAuditGenerator()
        progress_calls = []
        tasks = gen.generate(tmp_path, on_progress=lambda msg, frac: progress_calls.append((msg, frac)))
        assert len(progress_calls) == 4
        assert progress_calls[-1][1] == 1.0


# ---------------------------------------------------------------------------
# Performance audit generator
# ---------------------------------------------------------------------------

class TestPerformanceAuditGenerator:
    def test_check_large_files(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.performance_audit import PerformanceAuditGenerator

        large = tmp_path / "big.py"
        large.write_text("\n".join(f"line_{i} = {i}" for i in range(600)))

        gen = PerformanceAuditGenerator()
        tasks = gen._check_large_files(tmp_path)
        assert len(tasks) == 1
        assert "large" in tasks[0].title.lower() or "500" in tasks[0].description

    def test_check_large_files_under_threshold(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.performance_audit import PerformanceAuditGenerator

        small = tmp_path / "small.py"
        small.write_text("x = 1\n")

        gen = PerformanceAuditGenerator()
        tasks = gen._check_large_files(tmp_path)
        assert len(tasks) == 0

    def test_check_antipatterns_wildcard_import(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.performance_audit import PerformanceAuditGenerator

        src = tmp_path / "mod.py"
        src.write_text("import * from os\n")

        gen = PerformanceAuditGenerator()
        tasks = gen._check_antipatterns(tmp_path)
        assert len(tasks) == 1

    def test_check_antipatterns_json_clone(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.performance_audit import PerformanceAuditGenerator

        src = tmp_path / "app.ts"
        src.write_text("const copy = JSON.parse(JSON.stringify(obj));\n")

        gen = PerformanceAuditGenerator()
        tasks = gen._check_antipatterns(tmp_path)
        assert len(tasks) == 1

    def test_generate_calls_all_checks(self, tmp_path: Path):
        from feature_prd_runner.task_engine.sources.performance_audit import PerformanceAuditGenerator

        gen = PerformanceAuditGenerator()
        progress_calls = []
        tasks = gen.generate(tmp_path, on_progress=lambda msg, frac: progress_calls.append((msg, frac)))
        assert len(progress_calls) == 5
        assert progress_calls[-1][1] == 1.0


# ---------------------------------------------------------------------------
# Feedback injection in prompts
# ---------------------------------------------------------------------------

class TestFeedbackInjection:
    def test_human_feedback_block_no_store(self):
        from feature_prd_runner.prompts_local import _human_feedback_block

        result = _human_feedback_block("task-1")
        assert result == ""

    def test_impl_prompt_includes_feedback_call(self):
        """Verify build_local_implement_prompt references _human_feedback_block."""
        import inspect
        from feature_prd_runner.prompts_local import build_local_implement_prompt

        src = inspect.getsource(build_local_implement_prompt)
        assert "_human_feedback_block" in src

    def test_impl_plan_prompt_includes_feedback_call(self):
        """Verify build_local_impl_plan_prompt references _human_feedback_block."""
        import inspect
        from feature_prd_runner.prompts_local import build_local_impl_plan_prompt

        src = inspect.getsource(build_local_impl_plan_prompt)
        assert "_human_feedback_block" in src

    def test_review_prompt_includes_feedback_call(self):
        """Verify build_local_review_prompt references _human_feedback_block."""
        import inspect
        from feature_prd_runner.prompts_local import build_local_review_prompt

        src = inspect.getsource(build_local_review_prompt)
        assert "_human_feedback_block" in src

    def test_build_implement_runs(self):
        """Verify the implement prompt builds without error."""
        from feature_prd_runner.prompts_local import build_local_implement_prompt

        result = build_local_implement_prompt(
            prd_path=Path("test.md"),
            phase={"id": "phase-1", "name": "Test", "description": "d"},
            task={"id": "task-1"},
            impl_plan_path=Path("impl.json"),
            impl_plan_text="{}",
            allowed_files=["a.py"],
            agents_text="",
            repo_context_files={},
            user_prompt=None,
        )
        assert "Implement" in result

    def test_build_review_runs(self):
        """Verify the review prompt builds without error."""
        from feature_prd_runner.prompts_local import build_local_review_prompt

        result = build_local_review_prompt(
            phase={"id": "phase-1", "name": "Test"},
            review_path=Path("review.json"),
            prd_path=Path("test.md"),
            prd_text="PRD",
            prd_markers=[],
            user_prompt=None,
            agents_text="",
            changed_files=[],
            diff_text="",
            diff_stat="",
            status_text="",
            impl_plan_text="{}",
            tests_snapshot=None,
            simple_review=False,
        )
        assert "Review" in result

    def test_build_impl_plan_runs(self):
        """Verify the impl_plan prompt builds without error."""
        from feature_prd_runner.prompts_local import build_local_impl_plan_prompt

        result = build_local_impl_plan_prompt(
            phase={"id": "phase-1", "name": "Test"},
            prd_path=Path("test.md"),
            prd_text="PRD",
            prd_markers=[],
            impl_plan_path=Path("impl.json"),
            user_prompt=None,
            agents_text="",
            repo_file_list="",
            test_command=None,
        )
        assert "implementation plan" in result.lower()


# ---------------------------------------------------------------------------
# B1: AgentRegistry.load_from_yaml
# ---------------------------------------------------------------------------

class TestAgentRegistryYAML:
    def test_load_override_builtin(self, tmp_path: Path):
        """YAML can override fields on a built-in agent type."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - role: implementer\n"
            "    model_override: gpt-4\n"
            "    max_tokens: 50000\n"
            "    max_cost_usd: 5.0\n"
        )

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        impl = reg.get_type("implementer")
        assert impl.model_override == "gpt-4"
        assert impl.limits.max_tokens == 50000
        assert impl.limits.max_cost_usd == 5.0
        # Other fields should be preserved from built-in
        assert impl.display_name == "Implementer"
        assert "feature" in impl.task_type_affinity

    def test_load_custom_agent_type(self, tmp_path: Path):
        """YAML can define a brand-new agent type not in the enum."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - role: custom_agent\n"
            "    display_name: Custom Agent\n"
            "    description: A custom agent type\n"
            "    system_prompt: You are a custom agent\n"
            "    task_type_affinity: [custom]\n"
        )

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        assert reg.has_type("custom_agent")
        custom = reg.get_type("custom_agent")
        assert custom.display_name == "Custom Agent"
        assert custom.description == "A custom agent type"
        assert custom.system_prompt == "You are a custom agent"
        assert "custom" in custom.task_type_affinity

    def test_load_multiple_agents(self, tmp_path: Path):
        """YAML with multiple agent entries registers all of them."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - role: implementer\n"
            "    model_override: gpt-4\n"
            "  - role: custom_one\n"
            "    display_name: Custom One\n"
            "    description: First custom\n"
            "  - role: custom_two\n"
            "    display_name: Custom Two\n"
            "    description: Second custom\n"
        )

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        assert reg.get_type("implementer").model_override == "gpt-4"
        assert reg.has_type("custom_one")
        assert reg.has_type("custom_two")

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Loading from a nonexistent file should not raise."""
        from feature_prd_runner.agents.registry import AgentRegistry

        reg = AgentRegistry()
        reg.load_from_yaml(tmp_path / "does_not_exist.yaml")
        # Should still have all built-in types
        assert reg.has_type("implementer")

    def test_load_invalid_yaml_no_agents_key(self, tmp_path: Path):
        """YAML without an 'agents' list should not raise."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("something_else: true\n")

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        assert reg.has_type("implementer")

    def test_load_skips_entries_without_role(self, tmp_path: Path):
        """Entries missing the 'role' key are silently skipped."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - display_name: No Role\n"
            "  - role: reviewer\n"
            "    model_override: claude-3\n"
        )

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        assert reg.get_type("reviewer").model_override == "claude-3"

    def test_load_preserves_builtin_count(self, tmp_path: Path):
        """Loading an override does not add extra types."""
        from feature_prd_runner.agents.registry import AgentRegistry

        reg = AgentRegistry()
        initial_count = len(reg.list_types())

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - role: implementer\n"
            "    model_override: gpt-4\n"
        )
        reg.load_from_yaml(yaml_file)
        assert len(reg.list_types()) == initial_count

    def test_load_extra_keys_in_metadata(self, tmp_path: Path):
        """Unknown keys end up in metadata."""
        from feature_prd_runner.agents.registry import AgentRegistry

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n"
            "  - role: custom_meta\n"
            "    display_name: Meta Agent\n"
            "    description: Has metadata\n"
            "    custom_key: custom_value\n"
        )

        reg = AgentRegistry()
        reg.load_from_yaml(yaml_file)
        meta_agent = reg.get_type("custom_meta")
        assert meta_agent.metadata.get("custom_key") == "custom_value"


# ---------------------------------------------------------------------------
# B6: PipelineRegistry.load_from_yaml
# ---------------------------------------------------------------------------

class TestPipelineRegistryYAML:
    def test_load_single_file(self, tmp_path: Path):
        """Load a pipeline template from a single YAML file."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "id: custom_pipeline\n"
            "display_name: Custom Pipeline\n"
            "description: A custom pipeline\n"
            "task_types: [custom]\n"
            "steps:\n"
            "  - name: analyze\n"
            "    display_name: Analyze\n"
            "  - name: implement\n"
            "    display_name: Implement\n"
        )

        reg = PipelineRegistry()
        reg.load_from_yaml(yaml_file)
        tmpl = reg.get("custom_pipeline")
        assert tmpl.display_name == "Custom Pipeline"
        assert tmpl.description == "A custom pipeline"
        assert len(tmpl.steps) == 2
        assert tmpl.steps[0].name == "analyze"
        assert tmpl.steps[1].name == "implement"
        assert "custom" in tmpl.task_types

    def test_load_directory(self, tmp_path: Path):
        """Load multiple pipeline templates from a directory."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()

        (pipelines_dir / "a.yaml").write_text(
            "id: pipeline_a\n"
            "display_name: Pipeline A\n"
            "description: First\n"
            "steps:\n"
            "  - name: plan\n"
        )
        (pipelines_dir / "b.yml").write_text(
            "id: pipeline_b\n"
            "display_name: Pipeline B\n"
            "description: Second\n"
            "steps:\n"
            "  - name: implement\n"
        )

        reg = PipelineRegistry()
        reg.load_from_yaml(pipelines_dir)
        assert reg.get("pipeline_a").display_name == "Pipeline A"
        assert reg.get("pipeline_b").display_name == "Pipeline B"

    def test_load_resolves_task_type(self, tmp_path: Path):
        """A loaded pipeline's task_types should be resolvable."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "id: my_custom\n"
            "display_name: My Custom\n"
            "description: desc\n"
            "task_types: [my_custom_type]\n"
            "steps:\n"
            "  - name: plan\n"
        )

        reg = PipelineRegistry()
        reg.load_from_yaml(yaml_file)
        resolved = reg.resolve_for_task_type("my_custom_type")
        assert resolved.id == "my_custom"

    def test_load_nonexistent_path(self, tmp_path: Path):
        """Loading from nonexistent path should not raise."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        reg = PipelineRegistry()
        reg.load_from_yaml(tmp_path / "nope")
        # Still has built-in templates
        assert len(reg.list_templates()) > 0

    def test_load_yaml_missing_id(self, tmp_path: Path):
        """YAML without 'id' is silently skipped."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            "display_name: No ID\n"
            "description: Missing id\n"
            "steps:\n"
            "  - name: plan\n"
        )

        reg = PipelineRegistry()
        initial_count = len(reg.list_templates())
        reg.load_from_yaml(yaml_file)
        assert len(reg.list_templates()) == initial_count

    def test_load_step_fields(self, tmp_path: Path):
        """Step definition fields are correctly parsed from YAML."""
        from feature_prd_runner.pipelines.registry import PipelineRegistry

        yaml_file = tmp_path / "detailed.yaml"
        yaml_file.write_text(
            "id: detailed\n"
            "display_name: Detailed\n"
            "description: d\n"
            "steps:\n"
            "  - name: custom_step\n"
            "    display_name: Custom Step\n"
            "    required: false\n"
            "    timeout_seconds: 120\n"
            "    retry_limit: 1\n"
        )

        reg = PipelineRegistry()
        reg.load_from_yaml(yaml_file)
        tmpl = reg.get("detailed")
        step = tmpl.steps[0]
        assert step.name == "custom_step"
        assert step.display_name == "Custom Step"
        assert step.required is False
        assert step.timeout_seconds == 120
        assert step.retry_limit == 1


# ---------------------------------------------------------------------------
# C3: ReasoningStore wired into PipelineEngine
# ---------------------------------------------------------------------------

class TestReasoningStoreInEngine:
    def test_engine_without_reasoning_store(self):
        """Engine without reasoning_store works exactly as before."""
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.steps.base import StepContext

        engine = PipelineEngine()
        assert engine.reasoning_store is None

        tmpl = engine.pipelines.get("research")
        exe = engine.create_execution("task-1", tmpl)
        ctx = StepContext(
            task_id="task-1", task_type="research", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"

    def test_engine_with_reasoning_store_records_steps(self):
        """Engine with reasoning_store records start/complete for each step."""
        from feature_prd_runner.collaboration.reasoning import ReasoningStore
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef
        from feature_prd_runner.pipelines.steps.base import StepContext

        store = ReasoningStore()
        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-reasoning",
            display_name="Test Reasoning",
            description="Test",
            task_types=("feature",),
            steps=(
                StepDef(name="plan"),
                StepDef(name="implement"),
            ),
        )
        reg.register(template)

        engine = PipelineEngine(
            pipeline_registry=reg,
            hitl_mode="autopilot",
            reasoning_store=store,
        )
        exe = engine.create_execution("task-r1", template)
        ctx = StepContext(
            task_id="task-r1", task_type="feature", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"

        # Verify reasoning was recorded
        entries = store.get_for_task("task-r1")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.agent_id == "pipeline"
        assert len(entry.steps) == 2
        assert entry.steps[0].step_name == "plan"
        assert entry.steps[0].status == "success"
        assert entry.steps[1].step_name == "implement"
        assert entry.steps[1].status == "success"

    def test_engine_with_reasoning_store_uses_ctx_agent_id(self):
        """Engine uses ctx.agent_id when available instead of default 'pipeline'."""
        from feature_prd_runner.collaboration.reasoning import ReasoningStore
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef
        from feature_prd_runner.pipelines.steps.base import StepContext

        store = ReasoningStore()
        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-agent-id",
            display_name="Test Agent ID",
            description="Test",
            task_types=("feature",),
            steps=(StepDef(name="plan"),),
        )
        reg.register(template)

        engine = PipelineEngine(
            pipeline_registry=reg,
            hitl_mode="autopilot",
            reasoning_store=store,
        )
        exe = engine.create_execution("task-a1", template)
        ctx = StepContext(
            task_id="task-a1", task_type="feature", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
            agent_id="agent-custom-42",
        )
        asyncio.run(engine.execute(exe, ctx))

        entries = store.get_for_task("task-a1")
        assert len(entries) == 1
        assert entries[0].agent_id == "agent-custom-42"

    def test_engine_reasoning_records_on_failure(self):
        """Reasoning is recorded even when a step outcome is not SUCCESS."""
        from feature_prd_runner.collaboration.reasoning import ReasoningStore
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef
        from feature_prd_runner.pipelines.steps.base import StepContext

        store = ReasoningStore()
        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-record-all",
            display_name="Test Record",
            description="Test",
            task_types=("feature",),
            steps=(
                StepDef(name="plan"),
                StepDef(name="verify"),
            ),
        )
        reg.register(template)

        engine = PipelineEngine(
            pipeline_registry=reg,
            hitl_mode="autopilot",
            reasoning_store=store,
        )
        exe = engine.create_execution("task-f1", template)
        ctx = StepContext(
            task_id="task-f1", task_type="feature", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"

        entries = store.get_for_task("task-f1")
        assert len(entries) == 1
        # Both steps should have been recorded
        assert len(entries[0].steps) == 2
        # All steps should be completed (current_step should be None)
        assert entries[0].current_step is None


# ---------------------------------------------------------------------------
# B2: Pool manager enhancements (reassign, auto-restart, reap, per-role limits)
# ---------------------------------------------------------------------------

class TestPoolManagerEnhancements:
    def setup_method(self):
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.agents.pool import AgentPool

        self.registry = AgentRegistry()
        self.events = []
        self.pool = AgentPool(
            registry=self.registry,
            max_agents=5,
            on_agent_event=lambda aid, evt, data: self.events.append((aid, evt, data)),
        )

    def test_reassign_moves_task(self):
        from feature_prd_runner.agents.pool import AgentPool

        agent = self.pool.spawn("implementer", task_id="t1")
        old = self.pool.reassign(agent.id, "t2")
        assert old == "t1"
        assert agent.task_id == "t2"

    def test_reassign_from_idle(self):
        agent = self.pool.spawn("implementer")
        old = self.pool.reassign(agent.id, "t1")
        assert old is None
        assert agent.task_id == "t1"

    def test_auto_restart_failed_agent(self):
        from feature_prd_runner.agents.registry import AgentStatus

        agent = self.pool.spawn("implementer", task_id="t1")
        self.pool.mark_failed(agent.id, "test error")
        assert agent.status == AgentStatus.FAILED
        result = self.pool.auto_restart(agent.id)
        assert result is True
        assert agent.status == AgentStatus.IDLE
        assert agent.retries == 1

    def test_auto_restart_not_failed(self):
        agent = self.pool.spawn("implementer")
        result = self.pool.auto_restart(agent.id)
        assert result is False

    def test_auto_restart_exhausted_retries(self):
        agent = self.pool.spawn("implementer", task_id="t1")
        # Exhaust retries (default max_retries is 3)
        for _ in range(3):
            self.pool.mark_failed(agent.id, "error")
            self.pool.auto_restart(agent.id)
        # Now fail again — should not restart
        self.pool.mark_failed(agent.id, "error again")
        result = self.pool.auto_restart(agent.id)
        assert result is False

    def test_per_role_limit_enforced(self):
        from feature_prd_runner.agents.pool import AgentPool

        pool = AgentPool(
            registry=self.registry,
            max_agents=10,
            max_agents_per_role={"implementer": 2},
        )
        pool.spawn("implementer")
        pool.spawn("implementer")
        with pytest.raises(RuntimeError, match="Per-role limit"):
            pool.spawn("implementer")

    def test_per_role_limit_other_role_unaffected(self):
        from feature_prd_runner.agents.pool import AgentPool

        pool = AgentPool(
            registry=self.registry,
            max_agents=10,
            max_agents_per_role={"implementer": 1},
        )
        pool.spawn("implementer")
        # reviewer should still work
        agent = pool.spawn("reviewer")
        assert agent.agent_type == "reviewer"

    def test_reap_dead_agents(self):
        import datetime

        agent = self.pool.spawn("implementer", task_id="t1")
        # Fake an old heartbeat
        past = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).isoformat()
        agent.last_heartbeat = past

        reaped = self.pool.reap_dead_agents(timeout_seconds=1)
        assert agent.id in reaped

    def test_reap_dead_agents_no_stale(self):
        self.pool.spawn("implementer", task_id="t1")
        reaped = self.pool.reap_dead_agents(timeout_seconds=9999)
        assert reaped == []


# ---------------------------------------------------------------------------
# B4: Escalation handoff + auto-routing
# ---------------------------------------------------------------------------

class TestEscalationAndAutoRouting:
    def test_escalation_handoff_type_exists(self):
        from feature_prd_runner.agents.handoff import HandoffType

        assert HandoffType.ESCALATION == "escalation"

    def test_create_escalation(self):
        from feature_prd_runner.agents.handoff import create_escalation, HandoffType

        ho = create_escalation("agent-1", "task-1", "Cannot resolve conflict")
        assert ho.type == HandoffType.ESCALATION
        assert ho.from_agent_id == "agent-1"
        assert ho.task_id == "task-1"
        assert ho.to_agent_id is None  # always targets human
        assert "Cannot resolve conflict" in ho.summary

    def test_create_escalation_with_context(self):
        from feature_prd_runner.agents.handoff import create_escalation

        ctx = {"file": "main.py", "line": 42}
        ho = create_escalation("agent-1", "task-1", "Ambiguous requirement", context=ctx)
        assert "main.py" in ho.details

    def test_auto_route_review_feedback(self):
        from feature_prd_runner.agents.handoff import auto_route, Handoff, HandoffType

        ho = Handoff(type=HandoffType.REVIEW_FEEDBACK)
        role = auto_route(ho, None)
        assert role == "implementer"

    def test_auto_route_test_results(self):
        from feature_prd_runner.agents.handoff import auto_route, Handoff, HandoffType

        ho = Handoff(type=HandoffType.TEST_RESULTS)
        role = auto_route(ho, None)
        assert role == "reviewer"

    def test_auto_route_escalation_returns_none(self):
        from feature_prd_runner.agents.handoff import auto_route, Handoff, HandoffType

        ho = Handoff(type=HandoffType.ESCALATION)
        role = auto_route(ho, None)
        assert role is None  # requires human intervention

    def test_auto_route_bug_diagnosis(self):
        from feature_prd_runner.agents.handoff import auto_route, Handoff, HandoffType

        ho = Handoff(type=HandoffType.BUG_DIAGNOSIS)
        role = auto_route(ho, None)
        assert role == "implementer"

    def test_auto_route_task_split(self):
        from feature_prd_runner.agents.handoff import auto_route, Handoff, HandoffType

        ho = Handoff(type=HandoffType.TASK_SPLIT)
        role = auto_route(ho, None)
        assert role == "architect"

    def test_context_bus_send_and_receive_escalation(self):
        from feature_prd_runner.agents.handoff import ContextBus, create_escalation

        bus = ContextBus()
        ho = create_escalation("agent-1", "task-1", "Need help")
        bus.send_handoff(ho)
        received = bus.get_handoffs("task-1")
        assert len(received) == 1
        assert received[0].type.value == "escalation"


# ---------------------------------------------------------------------------
# C4: HITL mode SET endpoint + per-task override (collaboration_api)
# ---------------------------------------------------------------------------

class TestHITLModeEndpoints:
    """Test the HITL mode SET/GET endpoints via the FastAPI test client."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        return TestClient(app)

    def test_set_project_mode(self, client):
        resp = client.put("/api/v2/collaboration/modes", json={"mode": "supervised"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "supervised"
        assert "config" in data

    def test_set_invalid_mode_returns_400(self, client):
        resp = client.put("/api/v2/collaboration/modes", json={"mode": "invalid_mode"})
        assert resp.status_code == 400

    def test_set_and_get_task_mode(self, client):
        resp = client.put(
            "/api/v2/collaboration/modes/task/task-123",
            json={"mode": "collaborative"},
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "collaborative"

        resp = client.get("/api/v2/collaboration/modes/task/task-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_mode"] == "collaborative"
        assert data["effective_mode"] == "collaborative"

    def test_task_mode_falls_back_to_project(self, client):
        # Set project mode to supervised
        client.put("/api/v2/collaboration/modes", json={"mode": "supervised"})
        # Get task mode without setting one
        resp = client.get("/api/v2/collaboration/modes/task/task-no-override")
        data = resp.json()
        assert data["task_mode"] is None
        assert data["effective_mode"] == "supervised"

    def test_clear_task_mode(self, client):
        client.put(
            "/api/v2/collaboration/modes/task/task-456",
            json={"mode": "review_only"},
        )
        resp = client.delete("/api/v2/collaboration/modes/task/task-456")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

        resp = client.get("/api/v2/collaboration/modes/task/task-456")
        assert resp.json()["task_mode"] is None


# ---------------------------------------------------------------------------
# C6: StateChangeStore
# ---------------------------------------------------------------------------

class TestStateChangeStore:
    def test_record_state_change(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        event = store.record_state_change("task-1", "pending", "running")
        assert event.type == "status_change"
        assert "pending" in event.summary
        assert "running" in event.summary
        assert event.metadata["old_status"] == "pending"
        assert event.metadata["new_status"] == "running"

    def test_record_commit(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        event = store.record_commit("task-1", "abc123", "Fix auth bug")
        assert event.type == "commit"
        assert "Fix auth bug" in event.summary
        assert event.metadata["commit_hash"] == "abc123"

    def test_record_file_change(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        event = store.record_file_change("task-1", "src/main.py", "modified")
        assert event.type == "file_change"
        assert "src/main.py" in event.summary
        assert event.metadata["change_type"] == "modified"

    def test_get_events_empty(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        assert store.get_events("nonexistent") == []

    def test_get_events_returns_all(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        store.record_state_change("task-1", "pending", "running")
        store.record_commit("task-1", "abc", "msg")
        store.record_file_change("task-1", "a.py", "added")
        events = store.get_events("task-1")
        assert len(events) == 3

    def test_events_isolated_by_task(self):
        from feature_prd_runner.collaboration.timeline import StateChangeStore

        store = StateChangeStore()
        store.record_state_change("task-1", "pending", "running")
        store.record_state_change("task-2", "running", "done")
        assert len(store.get_events("task-1")) == 1
        assert len(store.get_events("task-2")) == 1

    def test_state_changes_in_timeline(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.collaboration.timeline import StateChangeStore, TimelineAggregator

        fb_store = FeedbackStore()
        sc_store = StateChangeStore()
        sc_store.record_state_change("task-1", "pending", "running")
        sc_store.record_commit("task-1", "abc", "commit msg")

        agg = TimelineAggregator(fb_store, state_change_store=sc_store)
        events = agg.get_timeline("task-1")
        assert len(events) == 2
        types = {e.type for e in events}
        assert "status_change" in types
        assert "commit" in types


# ---------------------------------------------------------------------------
# C7: WebNotificationService
# ---------------------------------------------------------------------------

class TestWebNotificationService:
    def test_task_completed_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.task_completed("t1", "Implement auth")
        hub.broadcast_sync.assert_called_once()
        args = hub.broadcast_sync.call_args
        assert args[0][0] == "notifications"
        assert args[0][1] == "new"
        assert "Task completed" in args[0][2]["title"]

    def test_task_failed_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.task_failed("t1", "OOM error")
        args = hub.broadcast_sync.call_args
        assert args[0][2]["severity"] == "error"

    def test_agent_error_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.agent_error("agent-1", "Crash")
        args = hub.broadcast_sync.call_args
        assert "Agent error" in args[0][2]["title"]

    def test_approval_needed_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.approval_needed("before_commit", task_id="t1")
        args = hub.broadcast_sync.call_args
        assert args[0][2]["severity"] == "warning"
        assert "Task t1" in args[0][2]["message"]

    def test_budget_warning_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.budget_warning("agent-1", 85.0)
        args = hub.broadcast_sync.call_args
        assert "85%" in args[0][2]["message"]

    def test_mode_changed_broadcast(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.mode_changed("supervised")
        args = hub.broadcast_sync.call_args
        assert "supervised" in args[0][2]["title"]

    def test_notification_ids_increment(self):
        from unittest.mock import MagicMock
        from feature_prd_runner.server.ws_hub import WebSocketHub, WebNotificationService

        hub = WebSocketHub()
        hub.broadcast_sync = MagicMock()
        svc = WebNotificationService(hub)

        svc.agent_spawned("a1", "implementer")
        id1 = hub.broadcast_sync.call_args[0][2]["id"]
        svc.review_requested("t1")
        id2 = hub.broadcast_sync.call_args[0][2]["id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# A1: Task JSON Schema + validate_dict
# ---------------------------------------------------------------------------

class TestTaskJSONSchema:
    def test_json_schema_has_required_keys(self):
        from feature_prd_runner.task_engine.model import Task

        schema = Task.json_schema()
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "Task"
        assert schema["type"] == "object"
        assert "id" in schema["required"]
        assert "title" in schema["required"]
        assert "properties" in schema

    def test_json_schema_enumerates_task_types(self):
        from feature_prd_runner.task_engine.model import Task, TaskType

        schema = Task.json_schema()
        schema_types = schema["properties"]["task_type"]["enum"]
        for tt in TaskType:
            assert tt.value in schema_types

    def test_json_schema_enumerates_priorities(self):
        from feature_prd_runner.task_engine.model import Task, TaskPriority

        schema = Task.json_schema()
        schema_prios = schema["properties"]["priority"]["enum"]
        for p in TaskPriority:
            assert p.value in schema_prios

    def test_json_schema_enumerates_statuses(self):
        from feature_prd_runner.task_engine.model import Task, TaskStatus

        schema = Task.json_schema()
        schema_statuses = schema["properties"]["status"]["enum"]
        for s in TaskStatus:
            assert s.value in schema_statuses

    def test_json_schema_has_id_pattern(self):
        from feature_prd_runner.task_engine.model import Task

        schema = Task.json_schema()
        assert "pattern" in schema["properties"]["id"]

    def test_validate_dict_valid(self):
        from feature_prd_runner.task_engine.model import Task

        data = {"id": "task-12345678", "title": "Valid task", "task_type": "feature", "priority": "P2"}
        errors = Task.validate_dict(data)
        assert errors == []

    def test_validate_dict_missing_title(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"id": "task-12345678"})
        assert any("title" in e for e in errors)

    def test_validate_dict_empty_title(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"id": "task-12345678", "title": ""})
        assert any("title" in e for e in errors)

    def test_validate_dict_invalid_task_type(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "task_type": "invalid_type"})
        assert any("task_type" in e for e in errors)

    def test_validate_dict_invalid_priority(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "priority": "PXXX"})
        assert any("priority" in e for e in errors)

    def test_validate_dict_invalid_status(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "status": "unknown_status"})
        assert any("status" in e for e in errors)

    def test_validate_dict_invalid_source(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "source": "alien"})
        assert any("source" in e for e in errors)

    def test_validate_dict_list_fields_wrong_type(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "labels": "not a list"})
        assert any("labels" in e for e in errors)

    def test_validate_dict_negative_retry_count(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict({"title": "Test", "retry_count": -1})
        assert any("retry_count" in e for e in errors)

    def test_validate_dict_not_a_dict(self):
        from feature_prd_runner.task_engine.model import Task

        errors = Task.validate_dict("not a dict")
        assert errors == ["Expected a dict"]

    def test_validate_dict_valid_with_all_fields(self):
        from feature_prd_runner.task_engine.model import Task

        t = Task(title="Full task")
        errors = Task.validate_dict(t.to_dict())
        assert errors == []

    def test_schema_matches_to_dict_keys(self):
        """All keys from to_dict should be in the JSON schema properties."""
        from feature_prd_runner.task_engine.model import Task

        schema = Task.json_schema()
        t = Task(title="Test")
        d = t.to_dict()
        for key in d:
            assert key in schema["properties"], f"Key '{key}' missing from schema"


# ---------------------------------------------------------------------------
# C5: Token-aware truncation + effectiveness tracking
# ---------------------------------------------------------------------------

class TestFeedbackEffectiveness:
    def test_effectiveness_empty_task(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore

        store = FeedbackStore()
        report = store.get_effectiveness_report("task-empty")
        assert report["total"] == 0
        assert report["addressed"] == 0
        assert report["active"] == 0
        assert report["dismissed"] == 0
        assert report["addressed_rate"] == 0.0
        assert report["unaddressed_items"] == []

    def test_effectiveness_all_addressed(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore

        store = FeedbackStore()
        fb1 = Feedback(task_id="t1", summary="Fix this")
        fb2 = Feedback(task_id="t1", summary="Fix that")
        store.add_feedback(fb1)
        store.add_feedback(fb2)
        store.address_feedback(fb1.id, "done")
        store.address_feedback(fb2.id, "done too")

        report = store.get_effectiveness_report("t1")
        assert report["total"] == 2
        assert report["addressed"] == 2
        assert report["active"] == 0
        assert report["addressed_rate"] == 1.0

    def test_effectiveness_mixed(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore

        store = FeedbackStore()
        store.add_feedback(Feedback(task_id="t1", summary="a"))
        fb2 = Feedback(task_id="t1", summary="b")
        store.add_feedback(fb2)
        fb3 = Feedback(task_id="t1", summary="c")
        store.add_feedback(fb3)
        store.address_feedback(fb2.id, "ok")
        store.dismiss_feedback(fb3.id)

        report = store.get_effectiveness_report("t1")
        assert report["total"] == 3
        assert report["addressed"] == 1
        assert report["dismissed"] == 1
        assert report["active"] == 1
        assert len(report["unaddressed_items"]) == 1

    def test_effectiveness_api_endpoint(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        store.add_feedback(Feedback(task_id="t1", summary="test"))
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/v2/collaboration/feedback/t1/effectiveness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["active"] == 1


class TestTokenAwareTruncation:
    def test_truncation_empty(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore

        store = FeedbackStore()
        result = store.get_prompt_instructions("nonexistent")
        assert result == ""

    def test_truncation_single_feedback(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore

        store = FeedbackStore()
        store.add_feedback(Feedback(task_id="t1", summary="Use pytest"))
        result = store.get_prompt_instructions("t1")
        assert "Human feedback to incorporate:" in result
        assert "Use pytest" in result

    def test_truncation_respects_max_chars(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore

        store = FeedbackStore()
        # Add many long feedback items
        for i in range(50):
            store.add_feedback(Feedback(
                task_id="t1",
                summary=f"Feedback item {i}: " + "x" * 100,
            ))
        result = store.get_prompt_instructions("t1", max_chars=500)
        assert len(result) <= 600  # some slack for the truncation message
        assert "[Earlier feedback truncated" in result

    def test_truncation_prioritizes_recent(self):
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore
        import time

        store = FeedbackStore()
        store.add_feedback(Feedback(task_id="t1", summary="OLD feedback"))
        time.sleep(0.01)
        store.add_feedback(Feedback(task_id="t1", summary="NEW feedback"))
        result = store.get_prompt_instructions("t1")
        # Newest should appear first (before OLD)
        new_pos = result.find("NEW feedback")
        old_pos = result.find("OLD feedback")
        assert new_pos < old_pos

    def test_prompt_instructions_api_endpoint(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import Feedback, FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        store.add_feedback(Feedback(task_id="t1", summary="Use hooks"))
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/v2/collaboration/feedback/t1/prompt")
        assert resp.status_code == 200
        data = resp.json()
        assert "Use hooks" in data["instructions"]


# ---------------------------------------------------------------------------
# Collaboration API: User + Presence endpoints
# ---------------------------------------------------------------------------

class TestUserPresenceEndpoints:
    @pytest.fixture
    def client_with_stores(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router
        from feature_prd_runner.server.users import UserStore, PresenceTracker

        app = FastAPI()
        fb_store = FeedbackStore()
        user_store = UserStore()
        presence = PresenceTracker()
        router = create_collaboration_router(
            get_feedback_store=lambda: fb_store,
            get_user_store=lambda: user_store,
            get_presence=lambda: presence,
        )
        app.include_router(router)
        return TestClient(app)

    def test_create_user(self, client_with_stores):
        resp = client_with_stores.post("/api/v2/collaboration/users", json={
            "username": "alice",
            "display_name": "Alice Smith",
            "role": "developer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["display_name"] == "Alice Smith"

    def test_get_user(self, client_with_stores):
        client_with_stores.post("/api/v2/collaboration/users", json={
            "username": "bob",
            "display_name": "Bob",
        })
        resp = client_with_stores.get("/api/v2/collaboration/users/bob")
        assert resp.status_code == 200
        assert resp.json()["username"] == "bob"

    def test_get_nonexistent_user(self, client_with_stores):
        resp = client_with_stores.get("/api/v2/collaboration/users/nobody")
        assert resp.status_code == 404

    def test_list_users(self, client_with_stores):
        initial = client_with_stores.get("/api/v2/collaboration/users").json()["users"]
        initial_count = len(initial)
        client_with_stores.post("/api/v2/collaboration/users", json={"username": "u1"})
        client_with_stores.post("/api/v2/collaboration/users", json={"username": "u2"})
        resp = client_with_stores.get("/api/v2/collaboration/users")
        assert resp.status_code == 200
        assert len(resp.json()["users"]) == initial_count + 2

    def test_update_presence(self, client_with_stores):
        client_with_stores.post("/api/v2/collaboration/users", json={"username": "dev1"})
        resp = client_with_stores.post("/api/v2/collaboration/presence", json={
            "username": "dev1",
            "viewing": "kanban",
            "task_id": "t1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_get_online_users(self, client_with_stores):
        client_with_stores.post("/api/v2/collaboration/presence", json={
            "username": "dev1",
        })
        resp = client_with_stores.get("/api/v2/collaboration/presence")
        assert resp.status_code == 200
        users = resp.json()["users"]
        assert len(users) >= 1


# ---------------------------------------------------------------------------
# Collaboration API: Feedback + Comments endpoints
# ---------------------------------------------------------------------------

class TestCollaborationAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        return TestClient(app)

    def test_add_and_get_feedback(self, client):
        resp = client.post("/api/v2/collaboration/feedback", json={
            "task_id": "t1",
            "summary": "Use better names",
            "feedback_type": "style_preference",
            "priority": "should",
        })
        assert resp.status_code == 200
        fb_id = resp.json()["id"]

        resp = client.get("/api/v2/collaboration/feedback/t1")
        data = resp.json()
        assert len(data["feedback"]) == 1
        assert data["feedback"][0]["id"] == fb_id

    def test_address_feedback(self, client):
        resp = client.post("/api/v2/collaboration/feedback", json={
            "task_id": "t1",
            "summary": "Fix bug",
        })
        fb_id = resp.json()["id"]

        resp = client.post(f"/api/v2/collaboration/feedback/{fb_id}/address")
        assert resp.status_code == 200
        assert resp.json()["status"] == "addressed"

    def test_dismiss_feedback(self, client):
        resp = client.post("/api/v2/collaboration/feedback", json={
            "task_id": "t1",
            "summary": "Not important",
        })
        fb_id = resp.json()["id"]

        resp = client.post(f"/api/v2/collaboration/feedback/{fb_id}/dismiss")
        assert resp.status_code == 200
        assert resp.json()["status"] == "dismissed"

    def test_address_nonexistent_feedback(self, client):
        resp = client.post("/api/v2/collaboration/feedback/nonexistent/address")
        assert resp.status_code == 404

    def test_add_and_get_comment(self, client):
        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "src/main.py",
            "line_number": 42,
            "body": "This looks wrong",
        })
        assert resp.status_code == 200
        comment_id = resp.json()["id"]

        resp = client.get("/api/v2/collaboration/comments/t1")
        data = resp.json()
        assert len(data["comments"]) == 1
        assert data["comments"][0]["id"] == comment_id

    def test_resolve_comment(self, client):
        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "src/main.py",
            "line_number": 10,
            "body": "Nit",
        })
        comment_id = resp.json()["id"]

        resp = client.post(f"/api/v2/collaboration/comments/{comment_id}/resolve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    def test_list_modes(self, client):
        resp = client.get("/api/v2/collaboration/modes")
        assert resp.status_code == 200
        modes = resp.json()["modes"]
        assert len(modes) >= 4  # autopilot, supervised, collaborative, review_only

    def test_get_timeline(self, client):
        # Add feedback so timeline has content
        client.post("/api/v2/collaboration/feedback", json={
            "task_id": "t1",
            "summary": "Timeline test",
        })
        resp = client.get("/api/v2/collaboration/timeline/t1")
        assert resp.status_code == 200
        events = resp.json()["events"]
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# C1: ReviewComment threading (parent_id)
# ---------------------------------------------------------------------------

class TestReviewCommentThreading:
    def test_review_comment_has_parent_id(self):
        from feature_prd_runner.collaboration.feedback import ReviewComment

        comment = ReviewComment(task_id="t1", body="Top-level")
        assert comment.parent_id is None

    def test_review_comment_with_parent_id(self):
        from feature_prd_runner.collaboration.feedback import ReviewComment

        parent = ReviewComment(task_id="t1", body="Top-level")
        reply = ReviewComment(task_id="t1", body="Reply", parent_id=parent.id)
        assert reply.parent_id == parent.id

    def test_review_comment_to_dict_includes_parent_id(self):
        from feature_prd_runner.collaboration.feedback import ReviewComment

        parent = ReviewComment(task_id="t1", body="Top")
        reply = ReviewComment(task_id="t1", body="Reply", parent_id=parent.id)
        d = reply.to_dict()
        assert "parent_id" in d
        assert d["parent_id"] == parent.id

    def test_get_replies(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore, ReviewComment

        store = FeedbackStore()
        parent = ReviewComment(task_id="t1", file_path="a.py", line_number=1, body="Issue here")
        store.add_comment(parent)
        reply1 = ReviewComment(task_id="t1", file_path="a.py", line_number=1, body="Agree", parent_id=parent.id)
        reply2 = ReviewComment(task_id="t1", file_path="a.py", line_number=1, body="Fixed", parent_id=parent.id)
        store.add_comment(reply1)
        store.add_comment(reply2)

        replies = store.get_replies(parent.id)
        assert len(replies) == 2
        assert replies[0].body == "Agree"
        assert replies[1].body == "Fixed"

    def test_get_replies_empty(self):
        from feature_prd_runner.collaboration.feedback import FeedbackStore, ReviewComment

        store = FeedbackStore()
        parent = ReviewComment(task_id="t1", body="No replies")
        store.add_comment(parent)
        assert store.get_replies(parent.id) == []

    def test_comment_threading_api(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        client = TestClient(app)

        # Create parent comment
        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "main.py",
            "line_number": 5,
            "body": "This needs work",
        })
        parent_id = resp.json()["id"]
        assert resp.json()["parent_id"] is None

        # Create reply
        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "main.py",
            "line_number": 5,
            "body": "I fixed it",
            "parent_id": parent_id,
        })
        assert resp.status_code == 200
        assert resp.json()["parent_id"] == parent_id

        # Get replies
        resp = client.get(f"/api/v2/collaboration/comments/{parent_id}/replies")
        assert resp.status_code == 200
        assert len(resp.json()["replies"]) == 1
        assert resp.json()["replies"][0]["body"] == "I fixed it"


# ---------------------------------------------------------------------------
# B1: ResourceLimits max_concurrent_files
# ---------------------------------------------------------------------------

class TestResourceLimitsMaxConcurrentFiles:
    def test_default_max_concurrent_files(self):
        from feature_prd_runner.agents.registry import ResourceLimits

        limits = ResourceLimits()
        assert limits.max_concurrent_files == 10

    def test_custom_max_concurrent_files(self):
        from feature_prd_runner.agents.registry import ResourceLimits

        limits = ResourceLimits(max_concurrent_files=25)
        assert limits.max_concurrent_files == 25

    def test_api_exposes_max_concurrent_files(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.server.agent_api import create_agent_router

        app = FastAPI()
        registry = AgentRegistry()
        pool = AgentPool(registry=registry)
        router = create_agent_router(
            get_pool=lambda: pool,
            get_registry=lambda: registry,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/v2/agents/types")
        assert resp.status_code == 200
        types = resp.json()["types"]
        assert len(types) > 0
        for t in types:
            assert "max_concurrent_files" in t["limits"]


# ---------------------------------------------------------------------------
# B2: Reassign + Message endpoints
# ---------------------------------------------------------------------------

class TestReassignMessageEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry
        from feature_prd_runner.server.agent_api import create_agent_router

        app = FastAPI()
        registry = AgentRegistry()
        pool = AgentPool(registry=registry)
        router = create_agent_router(
            get_pool=lambda: pool,
            get_registry=lambda: registry,
        )
        app.include_router(router)
        return TestClient(app), pool

    def test_reassign_agent(self, client):
        tc, pool = client
        agent = pool.spawn("implementer")
        resp = tc.post(f"/api/v2/agents/{agent.id}/reassign", json={
            "new_role": "reviewer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reassigned"
        assert data["old_agent_id"] == agent.id
        assert data["new_agent"]["agent_type"] == "reviewer"

    def test_reassign_nonexistent_agent(self, client):
        tc, pool = client
        resp = tc.post("/api/v2/agents/nonexistent/reassign", json={
            "new_role": "reviewer",
        })
        assert resp.status_code == 404

    def test_message_agent(self, client):
        tc, pool = client
        agent = pool.spawn("implementer")
        resp = tc.post(f"/api/v2/agents/{agent.id}/message", json={
            "content": "Please focus on tests",
            "sender": "alice",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "sent"
        # Verify message was appended to output_tail
        updated = pool.get(agent.id)
        assert any("Please focus on tests" in line for line in updated.output_tail)

    def test_message_nonexistent_agent(self, client):
        tc, pool = client
        resp = tc.post("/api/v2/agents/nonexistent/message", json={
            "content": "Hello",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# X4: Pipeline engine notification triggers
# ---------------------------------------------------------------------------

class TestPipelineNotificationTriggers:
    def test_engine_fires_task_completed(self):
        """Pipeline engine fires task_completed notification on successful completion."""
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef
        from feature_prd_runner.pipelines.steps.base import StepContext

        events = []
        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-notif-complete",
            display_name="Test",
            description="Test",
            task_types=("feature",),
            steps=(StepDef(name="plan"), StepDef(name="implement")),
        )
        reg.register(template)

        engine = PipelineEngine(
            pipeline_registry=reg,
            hitl_mode="autopilot",
            on_event=lambda evt, data: events.append((evt, data)),
        )
        exe = engine.create_execution("task-n1", template)
        ctx = StepContext(
            task_id="task-n1", task_type="feature", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"
        assert any(e[0] == "task_completed" for e in events)
        completed_event = [e for e in events if e[0] == "task_completed"][0]
        assert completed_event[1]["task_id"] == "task-n1"

    def test_engine_fires_approval_needed(self):
        """Pipeline engine fires approval_needed when HITL gate blocks."""
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.registry import PipelineRegistry, PipelineTemplate, StepDef
        from feature_prd_runner.pipelines.steps.base import StepContext

        events = []
        reg = PipelineRegistry()
        template = PipelineTemplate(
            id="test-notif-gate",
            display_name="Test",
            description="Test",
            task_types=("feature",),
            steps=(StepDef(name="implement"),),
        )
        reg.register(template)

        engine = PipelineEngine(
            pipeline_registry=reg,
            hitl_mode="supervised",
            on_event=lambda evt, data: events.append((evt, data)),
        )
        exe = engine.create_execution("task-n2", template)
        ctx = StepContext(
            task_id="task-n2", task_type="feature", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "failed"  # blocked by gate
        assert any(e[0] == "approval_needed" for e in events)
        gate_event = [e for e in events if e[0] == "approval_needed"][0]
        assert gate_event[1]["task_id"] == "task-n2"
        assert gate_event[1]["gate_type"] == "before_implement"

    def test_engine_without_on_event_works(self):
        """Pipeline engine works normally when no on_event callback is set."""
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.steps.base import StepContext

        engine = PipelineEngine()
        assert engine._on_event is None
        tmpl = engine.pipelines.get("research")
        exe = engine.create_execution("task-n3", tmpl)
        ctx = StepContext(
            task_id="task-n3", task_type="research", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"

    def test_engine_on_event_exception_doesnt_crash(self):
        """Errors in on_event callback don't crash the engine."""
        from feature_prd_runner.pipelines.engine import PipelineEngine
        from feature_prd_runner.pipelines.steps.base import StepContext

        def bad_callback(evt, data):
            raise RuntimeError("callback error")

        engine = PipelineEngine(on_event=bad_callback)
        tmpl = engine.pipelines.get("research")
        exe = engine.create_execution("task-n4", tmpl)
        ctx = StepContext(
            task_id="task-n4", task_type="research", task_title="T",
            task_description="", project_dir=Path("/tmp"),
            state_dir=Path("/tmp"), run_id="r1",
        )
        # Should not raise despite bad callback
        result = asyncio.run(engine.execute(exe, ctx))
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# X4: Agent pool notification wiring
# ---------------------------------------------------------------------------

class TestAgentPoolNotificationWiring:
    def test_spawned_event_fires(self):
        """Agent spawned event includes role in data."""
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry

        events = []
        pool = AgentPool(
            registry=AgentRegistry(),
            on_agent_event=lambda aid, evt, data: events.append((aid, evt, data)),
        )
        agent = pool.spawn("implementer", task_id="t1")
        spawned = [e for e in events if e[1] == "spawned"]
        assert len(spawned) == 1
        assert spawned[0][2]["role"] == "implementer"

    def test_failed_event_fires(self):
        """Agent failed event includes error in data."""
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry

        events = []
        pool = AgentPool(
            registry=AgentRegistry(),
            on_agent_event=lambda aid, evt, data: events.append((aid, evt, data)),
        )
        agent = pool.spawn("implementer", task_id="t1")
        pool.mark_failed(agent.id, "OOM error")
        failed = [e for e in events if e[1] == "failed"]
        assert len(failed) == 1
        assert failed[0][2]["error"] == "OOM error"

    def test_terminated_event_fires(self):
        """Agent terminated event is emitted."""
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry

        events = []
        pool = AgentPool(
            registry=AgentRegistry(),
            on_agent_event=lambda aid, evt, data: events.append((aid, evt, data)),
        )
        agent = pool.spawn("implementer")
        pool.terminate(agent.id)
        terminated = [e for e in events if e[1] == "terminated"]
        assert len(terminated) == 1

    def test_auto_restarted_event_fires(self):
        """Auto-restart event includes retry count."""
        from feature_prd_runner.agents.pool import AgentPool
        from feature_prd_runner.agents.registry import AgentRegistry

        events = []
        pool = AgentPool(
            registry=AgentRegistry(),
            on_agent_event=lambda aid, evt, data: events.append((aid, evt, data)),
        )
        agent = pool.spawn("implementer", task_id="t1")
        pool.mark_failed(agent.id, "crash")
        pool.auto_restart(agent.id)
        restarted = [e for e in events if e[1] == "auto_restarted"]
        assert len(restarted) == 1
        assert restarted[0][2]["retries"] == 1


# ---------------------------------------------------------------------------
# X4: Collaboration API notification triggers
# ---------------------------------------------------------------------------

class TestCollaborationNotificationTriggers:
    def test_mode_change_fires_notification(self):
        """Setting HITL mode calls web_notifications.mode_changed()."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        notifications = []

        class MockNotifications:
            def mode_changed(self, mode):
                notifications.append(("mode_changed", mode))

        app = FastAPI()
        store = FeedbackStore()
        mock_notif = MockNotifications()
        router = create_collaboration_router(
            get_feedback_store=lambda: store,
            get_web_notifications=lambda: mock_notif,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.put("/api/v2/collaboration/modes", json={"mode": "supervised"})
        assert resp.status_code == 200
        assert len(notifications) == 1
        assert notifications[0] == ("mode_changed", "supervised")

    def test_add_comment_fires_review_requested(self):
        """Adding a review comment calls web_notifications.review_requested()."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        notifications = []

        class MockNotifications:
            def review_requested(self, task_id):
                notifications.append(("review_requested", task_id))

        app = FastAPI()
        store = FeedbackStore()
        mock_notif = MockNotifications()
        router = create_collaboration_router(
            get_feedback_store=lambda: store,
            get_web_notifications=lambda: mock_notif,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "main.py",
            "line_number": 10,
            "body": "Fix this",
        })
        assert resp.status_code == 200
        assert len(notifications) == 1
        assert notifications[0] == ("review_requested", "t1")

    def test_no_notification_when_not_wired(self):
        """Without get_web_notifications, endpoints still work without errors."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from feature_prd_runner.collaboration.feedback import FeedbackStore
        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        app = FastAPI()
        store = FeedbackStore()
        router = create_collaboration_router(get_feedback_store=lambda: store)
        app.include_router(router)
        client = TestClient(app)

        # These should work without errors even without notifications wired
        resp = client.put("/api/v2/collaboration/modes", json={"mode": "supervised"})
        assert resp.status_code == 200

        resp = client.post("/api/v2/collaboration/comments", json={
            "task_id": "t1",
            "file_path": "main.py",
            "line_number": 10,
            "body": "Fix this",
        })
        assert resp.status_code == 200
