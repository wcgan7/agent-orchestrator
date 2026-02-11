"""Tests for the smart scheduler."""

import pytest
from feature_prd_runner.agents.registry import AgentRegistry, AgentStatus
from feature_prd_runner.agents.pool import AgentPool
from feature_prd_runner.agents.scheduler import Assignment, Scheduler


class TestScheduler:
    def setup_method(self):
        self.registry = AgentRegistry()
        self.pool = AgentPool(registry=self.registry, max_agents=5)
        self.scheduler = Scheduler(pool=self.pool, registry=self.registry)

    # -- basic scheduling ---------------------------------------------------

    def test_schedule_empty(self):
        assert self.scheduler.schedule([]) == []

    def test_schedule_single_task(self):
        tasks = [{"id": "t1", "task_type": "feature", "priority": "P1"}]
        assignments = self.scheduler.schedule(tasks)
        assert len(assignments) == 1
        assert assignments[0].task_id == "t1"

    def test_schedule_priority_order(self):
        tasks = [
            {"id": "t-low", "task_type": "feature", "priority": "P3"},
            {"id": "t-high", "task_type": "feature", "priority": "P0"},
        ]
        assignments = self.scheduler.schedule(tasks)
        assert assignments[0].task_id == "t-high"

    def test_schedule_skips_already_assigned(self):
        self.pool.spawn("implementer", task_id="t1")
        tasks = [{"id": "t1", "task_type": "feature", "priority": "P1"}]
        assignments = self.scheduler.schedule(tasks)
        assert len(assignments) == 0

    def test_schedule_multiple_tasks(self):
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1"},
            {"id": "t2", "task_type": "bug", "priority": "P2"},
        ]
        assignments = self.scheduler.schedule(tasks)
        assert len(assignments) == 2

    # -- preemption ---------------------------------------------------------

    def test_preempt_for_p0(self):
        # Fill pool to capacity
        pool = AgentPool(registry=self.registry, max_agents=1)
        scheduler = Scheduler(pool=pool, registry=self.registry)
        pool.spawn("implementer", task_id="t-existing")
        p0 = {"id": "t-urgent", "task_type": "feature", "priority": "P0"}
        result = scheduler.preempt_for_p0(p0)
        assert result is not None
        assert result.task_id == "t-urgent"

    # -- file conflict detection --------------------------------------------

    def test_detect_file_conflicts_overlap(self):
        t1 = {"id": "a", "context_files": ["src/main.py", "src/util.py"]}
        t2 = {"id": "b", "context_files": ["src/util.py", "src/other.py"]}
        assert Scheduler._detect_file_conflicts(t1, t2) is True

    def test_detect_file_conflicts_no_overlap(self):
        t1 = {"id": "a", "context_files": ["src/main.py"]}
        t2 = {"id": "b", "context_files": ["src/other.py"]}
        assert Scheduler._detect_file_conflicts(t1, t2) is False

    def test_detect_file_conflicts_empty(self):
        t1 = {"id": "a", "context_files": []}
        t2 = {"id": "b", "context_files": ["src/main.py"]}
        assert Scheduler._detect_file_conflicts(t1, t2) is False

    def test_detect_file_conflicts_no_key(self):
        t1 = {"id": "a"}
        t2 = {"id": "b", "context_files": ["src/main.py"]}
        assert Scheduler._detect_file_conflicts(t1, t2) is False

    def test_schedule_skips_file_conflicts(self):
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1", "context_files": ["src/shared.py"]},
            {"id": "t2", "task_type": "feature", "priority": "P2", "context_files": ["src/shared.py"]},
        ]
        assignments = self.scheduler.schedule(tasks)
        # Only one should be assigned (t1 has higher priority)
        assert len(assignments) == 1
        assert assignments[0].task_id == "t1"

    def test_schedule_no_conflict_different_files(self):
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1", "context_files": ["src/a.py"]},
            {"id": "t2", "task_type": "feature", "priority": "P2", "context_files": ["src/b.py"]},
        ]
        assignments = self.scheduler.schedule(tasks)
        assert len(assignments) == 2

    # -- on_task_complete ---------------------------------------------------

    def test_on_task_complete_frees_agent(self):
        tasks = [{"id": "t1", "task_type": "feature", "priority": "P1"}]
        self.scheduler.schedule(tasks)
        agent = self.pool.get_agent_for_task("t1")
        assert agent is not None

        new_assignments = self.scheduler.on_task_complete("t1")
        freed_agent = self.pool.get(agent.id)
        assert freed_agent.status == AgentStatus.IDLE

    def test_on_task_complete_reschedules_pending(self):
        # Fill pool to 1 so second task is pending
        pool = AgentPool(registry=self.registry, max_agents=1)
        scheduler = Scheduler(pool=pool, registry=self.registry)
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1"},
            {"id": "t2", "task_type": "feature", "priority": "P2"},
        ]
        first_round = scheduler.schedule(tasks)
        assert len(first_round) == 1
        assert first_round[0].task_id == "t1"

        # Complete t1 — t2 should now be scheduled
        new_assignments = scheduler.on_task_complete("t1")
        assert len(new_assignments) == 1
        assert new_assignments[0].task_id == "t2"

    # -- on_agent_free ------------------------------------------------------

    def test_on_agent_free_assigns_pending(self):
        pool = AgentPool(registry=self.registry, max_agents=1)
        scheduler = Scheduler(pool=pool, registry=self.registry)
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1"},
            {"id": "t2", "task_type": "feature", "priority": "P2"},
        ]
        scheduler.schedule(tasks)
        agent = pool.get_agent_for_task("t1")

        # Free the agent manually
        pool.unassign_task(agent.id)
        result = scheduler.on_agent_free(agent.id)
        assert result is not None
        assert result.task_id == "t2"

    def test_on_agent_free_no_pending(self):
        agent = self.pool.spawn("implementer")
        result = self.scheduler.on_agent_free(agent.id)
        assert result is None

    # -- rebalance ----------------------------------------------------------

    def test_rebalance_reschedules(self):
        pool = AgentPool(registry=self.registry, max_agents=1)
        scheduler = Scheduler(pool=pool, registry=self.registry)
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1"},
            {"id": "t2", "task_type": "feature", "priority": "P2"},
        ]
        scheduler.schedule(tasks)
        # Complete t1 so agent is freed
        pool.unassign_task(pool.get_agent_for_task("t1").id)
        result = scheduler.rebalance()
        assert len(result) == 1
        assert result[0].task_id == "t2"

    def test_rebalance_empty_pending(self):
        assert self.scheduler.rebalance() == []

    # -- get_queue ----------------------------------------------------------

    def test_get_queue_empty(self):
        q = self.scheduler.get_queue()
        assert q["pending_tasks"] == []
        assert q["recent_assignments"] == []

    def test_get_queue_after_schedule(self):
        pool = AgentPool(registry=self.registry, max_agents=1)
        scheduler = Scheduler(pool=pool, registry=self.registry)
        tasks = [
            {"id": "t1", "task_type": "feature", "priority": "P1"},
            {"id": "t2", "task_type": "feature", "priority": "P2"},
        ]
        scheduler.schedule(tasks)
        q = scheduler.get_queue()
        assert len(q["recent_assignments"]) == 1
        assert q["recent_assignments"][0]["task_id"] == "t1"
        assert len(q["pending_tasks"]) == 1
        assert q["pending_tasks"][0]["id"] == "t2"
