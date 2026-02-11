"""Tests for agent pool manager."""

import pytest
from feature_prd_runner.agents.registry import AgentRegistry, AgentStatus
from feature_prd_runner.agents.pool import AgentPool


class TestAgentPool:
    def setup_method(self):
        self.registry = AgentRegistry()
        self.events = []
        self.pool = AgentPool(
            registry=self.registry,
            max_agents=3,
            on_agent_event=lambda aid, evt, data: self.events.append((aid, evt, data)),
        )

    def test_spawn_agent(self):
        agent = self.pool.spawn("implementer")
        assert agent.id.startswith("agent-")
        assert agent.agent_type == "implementer"
        assert agent.status == AgentStatus.IDLE
        assert self.pool.idle_count == 1
        assert self.pool.active_count == 0

    def test_spawn_with_task(self):
        agent = self.pool.spawn("reviewer", task_id="task-123")
        assert agent.status == AgentStatus.RUNNING
        assert agent.task_id == "task-123"
        assert self.pool.active_count == 1

    def test_spawn_at_capacity_raises(self):
        self.pool.spawn("implementer", task_id="t1")
        self.pool.spawn("implementer", task_id="t2")
        self.pool.spawn("implementer", task_id="t3")
        with pytest.raises(RuntimeError, match="capacity"):
            self.pool.spawn("implementer", task_id="t4")

    def test_terminate_frees_slot(self):
        agent = self.pool.spawn("implementer", task_id="t1")
        self.pool.terminate(agent.id)
        assert agent.status == AgentStatus.TERMINATED
        assert self.pool.active_count == 0

    def test_pause_and_resume(self):
        agent = self.pool.spawn("implementer", task_id="t1")
        self.pool.pause(agent.id)
        assert agent.status == AgentStatus.PAUSED
        self.pool.resume(agent.id)
        assert agent.status == AgentStatus.RUNNING

    def test_assign_task(self):
        agent = self.pool.spawn("implementer")
        assert agent.status == AgentStatus.IDLE
        self.pool.assign_task(agent.id, "task-abc")
        assert agent.status == AgentStatus.RUNNING
        assert agent.task_id == "task-abc"

    def test_unassign_task(self):
        agent = self.pool.spawn("implementer", task_id="task-abc")
        old = self.pool.unassign_task(agent.id)
        assert old == "task-abc"
        assert agent.status == AgentStatus.IDLE
        assert agent.task_id is None

    def test_update_progress(self):
        agent = self.pool.spawn("implementer", task_id="t1")
        self.pool.update_progress(
            agent.id,
            current_step="implement",
            current_file="src/main.py",
            tokens_used=5000,
            cost_usd=0.15,
            output_line="Writing function...",
        )
        assert agent.current_step == "implement"
        assert agent.current_file == "src/main.py"
        assert agent.tokens_used == 5000
        assert agent.cost_usd == 0.15
        assert "Writing function..." in agent.output_tail

    def test_heartbeat(self):
        agent = self.pool.spawn("implementer")
        old_hb = agent.last_heartbeat
        self.pool.heartbeat(agent.id)
        assert agent.last_heartbeat >= old_hb

    def test_mark_failed(self):
        agent = self.pool.spawn("implementer", task_id="t1")
        self.pool.mark_failed(agent.id, "OOM error")
        assert agent.status == AgentStatus.FAILED
        assert "[ERROR] OOM error" in agent.output_tail

    def test_find_idle_agent(self):
        a1 = self.pool.spawn("implementer", task_id="t1")
        a2 = self.pool.spawn("reviewer")
        found = self.pool.find_idle_agent()
        assert found.id == a2.id

    def test_find_idle_agent_by_role(self):
        self.pool.spawn("implementer")
        a2 = self.pool.spawn("reviewer")
        found = self.pool.find_idle_agent(role="reviewer")
        assert found.id == a2.id

    def test_get_agent_for_task(self):
        agent = self.pool.spawn("implementer", task_id="task-xyz")
        found = self.pool.get_agent_for_task("task-xyz")
        assert found.id == agent.id

    def test_list_agents_filter(self):
        self.pool.spawn("implementer", task_id="t1")
        self.pool.spawn("reviewer")
        running = self.pool.list_agents(status=AgentStatus.RUNNING)
        idle = self.pool.list_agents(status=AgentStatus.IDLE)
        assert len(running) == 1
        assert len(idle) == 1

    def test_to_dict(self):
        self.pool.spawn("implementer")
        d = self.pool.to_dict()
        assert d["max_agents"] == 3
        assert d["idle_count"] == 1
        assert len(d["agents"]) == 1

    def test_events_emitted(self):
        agent = self.pool.spawn("implementer")
        self.pool.assign_task(agent.id, "task-1")
        self.pool.pause(agent.id)
        self.pool.resume(agent.id)
        self.pool.terminate(agent.id)
        event_types = [e[1] for e in self.events]
        assert "spawned" in event_types
        assert "task_assigned" in event_types
        assert "paused" in event_types
        assert "resumed" in event_types
        assert "terminated" in event_types

    def test_capacity(self):
        assert self.pool.capacity == 3
        self.pool.spawn("implementer", task_id="t1")
        assert self.pool.capacity == 2
        self.pool.spawn("implementer", task_id="t2")
        assert self.pool.capacity == 1

    def test_remove_agent(self):
        agent = self.pool.spawn("implementer")
        self.pool.terminate(agent.id)
        self.pool.remove(agent.id)
        assert self.pool.get(agent.id) is None
