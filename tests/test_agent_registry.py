"""Tests for agent registry and agent types."""

import pytest
from feature_prd_runner.agents.registry import (
    AgentInstance,
    AgentRegistry,
    AgentRole,
    AgentStatus,
    AgentType,
    BUILTIN_AGENT_TYPES,
    ResourceLimits,
)


class TestAgentType:
    def test_builtin_types_exist(self):
        assert "implementer" in BUILTIN_AGENT_TYPES
        assert "reviewer" in BUILTIN_AGENT_TYPES
        assert "researcher" in BUILTIN_AGENT_TYPES
        assert "tester" in BUILTIN_AGENT_TYPES
        assert "architect" in BUILTIN_AGENT_TYPES

    def test_builtin_type_fields(self):
        impl = BUILTIN_AGENT_TYPES["implementer"]
        assert impl.role == AgentRole.IMPLEMENTER
        assert impl.display_name == "Implementer"
        assert "implement" in impl.allowed_steps
        assert "feature" in impl.task_type_affinity

    def test_resource_limits_defaults(self):
        limits = ResourceLimits()
        assert limits.max_tokens == 200_000
        assert limits.max_time_seconds == 3600
        assert limits.max_cost_usd == 5.0
        assert limits.max_retries == 3

    def test_architect_has_custom_limits(self):
        arch = BUILTIN_AGENT_TYPES["architect"]
        assert arch.limits.max_tokens == 300_000
        assert arch.limits.max_cost_usd == 3.0


class TestAgentInstance:
    def test_instance_creation(self):
        inst = AgentInstance(agent_type="implementer", display_name="Impl-1")
        assert inst.id.startswith("agent-")
        assert inst.status == AgentStatus.IDLE
        assert inst.tokens_used == 0
        assert inst.cost_usd == 0.0

    def test_to_dict(self):
        inst = AgentInstance(
            id="agent-test123",
            agent_type="reviewer",
            display_name="Rev-1",
            status=AgentStatus.RUNNING,
            task_id="task-abc",
            tokens_used=5000,
            cost_usd=0.15,
        )
        d = inst.to_dict()
        assert d["id"] == "agent-test123"
        assert d["agent_type"] == "reviewer"
        assert d["status"] == "running"
        assert d["task_id"] == "task-abc"
        assert d["tokens_used"] == 5000

    def test_output_tail_capped(self):
        inst = AgentInstance()
        inst.output_tail = [f"line-{i}" for i in range(100)]
        d = inst.to_dict()
        assert len(d["output_tail"]) == 50  # capped to last 50


class TestAgentRegistry:
    def test_list_types(self):
        reg = AgentRegistry()
        types = reg.list_types()
        assert len(types) == 6

    def test_get_type(self):
        reg = AgentRegistry()
        impl = reg.get_type("implementer")
        assert impl.role == AgentRole.IMPLEMENTER

    def test_get_type_unknown_raises(self):
        reg = AgentRegistry()
        with pytest.raises(KeyError, match="Unknown agent type"):
            reg.get_type("nonexistent")

    def test_has_type(self):
        reg = AgentRegistry()
        assert reg.has_type("reviewer")
        assert not reg.has_type("nonexistent")

    def test_register_custom_type(self):
        reg = AgentRegistry()
        custom = AgentType(
            role=AgentRole.IMPLEMENTER,  # override built-in
            display_name="Custom Implementer",
            description="Custom desc",
        )
        reg.register(custom)
        assert reg.get_type("implementer").display_name == "Custom Implementer"

    def test_unregister(self):
        reg = AgentRegistry()
        reg.unregister("tester")
        assert not reg.has_type("tester")

    def test_create_instance(self):
        reg = AgentRegistry()
        inst = reg.create_instance("implementer")
        assert inst.agent_type == "implementer"
        assert inst.display_name == "Implementer"

    def test_create_instance_with_overrides(self):
        reg = AgentRegistry()
        inst = reg.create_instance("reviewer", display_name="My Reviewer")
        assert inst.display_name == "My Reviewer"
        assert inst.agent_type == "reviewer"

    def test_best_role_for_task_type(self):
        reg = AgentRegistry()
        assert reg.best_role_for_task_type("feature") == "implementer"
        assert reg.best_role_for_task_type("review") == "reviewer"
        assert reg.best_role_for_task_type("research") == "researcher"
        assert reg.best_role_for_task_type("test") == "tester"
        assert reg.best_role_for_task_type("unknown") is None
