"""Tests for the recommended-action feature: classifier, fallback, emit_task_blocked, and API surface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_orchestrator.runtime.domain.models import RunRecord, Task, now_iso
from agent_orchestrator.runtime.orchestrator.service import (
    OrchestratorService,
    _BLOCKED_RECOMMENDED_ACTIONS,
    _classify_block_reason,
)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestClassifyBlockReason:
    """Verify _classify_block_reason maps metadata and error patterns correctly."""

    def test_merge_conflict_metadata(self) -> None:
        """Merge-conflict metadata key maps to merge_conflict category."""
        task = Task(title="t", metadata={"merge_conflict": True})
        assert _classify_block_reason(task) == "merge_conflict"

    def test_scope_violation_metadata(self) -> None:
        """Scope-violation metadata key maps to scope_violation category."""
        task = Task(title="t", metadata={"scope_violation": ["file.py"]})
        assert _classify_block_reason(task) == "scope_violation"

    def test_missing_workdoc_metadata(self) -> None:
        """Missing-workdoc-path metadata key maps to missing_workdoc category."""
        task = Task(title="t", metadata={"missing_workdoc_path": "/some/path"})
        assert _classify_block_reason(task) == "missing_workdoc"

    def test_invalid_workdoc_metadata(self) -> None:
        """Invalid-workdoc-path metadata key maps to invalid_workdoc category."""
        task = Task(title="t", metadata={"invalid_workdoc_path": "/some/path"})
        assert _classify_block_reason(task) == "invalid_workdoc"

    def test_human_intervention_metadata(self) -> None:
        """Non-empty human_blocking_issues list maps to human_intervention."""
        task = Task(title="t", metadata={"human_blocking_issues": [{"summary": "fix this"}]})
        assert _classify_block_reason(task) == "human_intervention"

    def test_human_intervention_empty_list(self) -> None:
        """Empty list should NOT classify as human_intervention."""
        task = Task(title="t", metadata={"human_blocking_issues": []})
        task.error = "something else"
        assert _classify_block_reason(task) != "human_intervention"

    def test_dirty_overlapping_metadata(self) -> None:
        """Merge-failure reason code dirty_overlapping maps correctly."""
        task = Task(title="t", metadata={"merge_failure_reason_code": "dirty_overlapping"})
        assert _classify_block_reason(task) == "dirty_overlapping"

    def test_git_error_metadata(self) -> None:
        """Merge-failure reason code git_error maps correctly."""
        task = Task(title="t", metadata={"merge_failure_reason_code": "git_error"})
        assert _classify_block_reason(task) == "git_error"

    def test_scope_violation_error_text(self) -> None:
        """Scope-violation error text maps to scope_violation category."""
        task = Task(title="t", error="Scope violation: modified out-of-scope files: foo.py")
        assert _classify_block_reason(task) == "scope_violation"

    def test_merge_conflict_error_text(self) -> None:
        """Merge-conflict error text maps to merge_conflict category."""
        task = Task(title="t", error="Merge conflict could not be resolved automatically")
        assert _classify_block_reason(task) == "merge_conflict"

    def test_missing_workdoc_error_text(self) -> None:
        """Missing-workdoc error text maps to missing_workdoc category."""
        task = Task(title="t", error="Missing required workdoc: task_abc.workdoc.md")
        assert _classify_block_reason(task) == "missing_workdoc"

    def test_invalid_workdoc_error_text(self) -> None:
        """Invalid-workdoc error text maps to invalid_workdoc category."""
        task = Task(title="t", error="Invalid workdoc encoding (expected UTF-8): /path/to/file")
        assert _classify_block_reason(task) == "invalid_workdoc"

    def test_verify_fix_exhausted_error_text(self) -> None:
        """Verify-fix-exhausted error text maps correctly."""
        task = Task(title="t", error="Could not fix verify after 3 attempts")
        assert _classify_block_reason(task) == "verify_fix_exhausted"

    def test_review_cap_exceeded_error_text(self) -> None:
        """Review-cap-exceeded error text maps correctly."""
        task = Task(title="t", error="Review attempt cap exceeded")
        assert _classify_block_reason(task) == "review_cap_exceeded"

    def test_unresolved_review_findings_error_text(self) -> None:
        """Unresolved review findings error text maps to review_cap_exceeded."""
        task = Task(title="t", error="Blocked due to unresolved review findings")
        assert _classify_block_reason(task) == "review_cap_exceeded"

    def test_no_changes_error_text(self) -> None:
        """No-changes error text maps to no_changes category."""
        task = Task(title="t", error="No file changes detected after implementation")
        assert _classify_block_reason(task) == "no_changes"

    def test_human_intervention_error_text(self) -> None:
        """Human-intervention error text maps correctly."""
        task = Task(title="t", error="human intervention required for this step")
        assert _classify_block_reason(task) == "human_intervention"

    def test_gate_timeout_error_text(self) -> None:
        """Gate-timeout error text maps correctly."""
        task = Task(title="t", error="Gate before_implement timed out after 300s")
        assert _classify_block_reason(task) == "gate_timeout"

    def test_precommit_context_missing_error_text(self) -> None:
        """Pre-commit context missing error text maps correctly."""
        task = Task(title="t", error="Pre-commit context missing; request changes")
        assert _classify_block_reason(task) == "precommit_context_missing"

    def test_dirty_overlapping_error_text(self) -> None:
        """Dirty-overlapping error text maps correctly."""
        task = Task(title="t", error="Integration branch has overlapping local changes")
        assert _classify_block_reason(task) == "dirty_overlapping"

    def test_git_merge_failed_error_text(self) -> None:
        """Git merge failed error text maps to git_error category."""
        task = Task(title="t", error="Git merge failed before conflict resolution")
        assert _classify_block_reason(task) == "git_error"

    def test_internal_error_fallback(self) -> None:
        """Unknown error text falls back to internal_error."""
        task = Task(title="t", error="Something totally unexpected happened")
        assert _classify_block_reason(task) == "internal_error"

    def test_no_error_defaults_to_internal(self) -> None:
        """Task with no error defaults to internal_error."""
        task = Task(title="t")
        assert _classify_block_reason(task) == "internal_error"

    def test_metadata_takes_precedence_over_error_text(self) -> None:
        """When metadata and error text both match, metadata wins."""
        task = Task(
            title="t",
            error="Could not fix verify after 3 attempts",
            metadata={"merge_conflict": True},
        )
        assert _classify_block_reason(task) == "merge_conflict"


# ---------------------------------------------------------------------------
# Fallback mapping completeness
# ---------------------------------------------------------------------------

class TestFallbackMapping:
    """Ensure every category in the classifier has a non-empty fallback string."""

    EXPECTED_CATEGORIES = {
        "merge_conflict",
        "scope_violation",
        "missing_workdoc",
        "invalid_workdoc",
        "verify_fix_exhausted",
        "review_cap_exceeded",
        "no_changes",
        "human_intervention",
        "gate_timeout",
        "precommit_context_missing",
        "dirty_overlapping",
        "git_error",
        "internal_error",
    }

    def test_all_categories_have_fallback(self) -> None:
        """Every block-reason category has a non-empty fallback action string."""
        for cat in self.EXPECTED_CATEGORIES:
            action = _BLOCKED_RECOMMENDED_ACTIONS.get(cat)
            assert action, f"Category '{cat}' has no fallback action"
            assert len(action.strip()) > 5, f"Category '{cat}' fallback is too short"

    def test_no_extra_categories(self) -> None:
        """No stale entries in the mapping that don't correspond to a valid category."""
        extra = set(_BLOCKED_RECOMMENDED_ACTIONS.keys()) - self.EXPECTED_CATEGORIES
        assert not extra, f"Unknown categories in mapping: {extra}"


# ---------------------------------------------------------------------------
# _generate_recommended_action tests
# ---------------------------------------------------------------------------

class TestGenerateRecommendedAction:
    """Test LLM-first with hardcoded fallback behavior."""

    def _make_service(self, *, llm_result: str = "") -> OrchestratorService:
        """Create a minimal mock OrchestratorService for testing."""
        svc = MagicMock(spec=OrchestratorService)
        svc._generate_recommended_action = OrchestratorService._generate_recommended_action.__get__(svc)
        adapter = MagicMock()
        adapter.generate_recommended_action.return_value = llm_result
        svc.worker_adapter = adapter
        return svc

    def test_returns_llm_result_when_available(self) -> None:
        """LLM result is returned when available and non-empty."""
        svc = self._make_service(llm_result="Run 'npm test' to check.")
        task = Task(title="t", error="verify failed", metadata={"merge_conflict": True})
        result = svc._generate_recommended_action(task)
        assert result == "Run 'npm test' to check."

    def test_falls_back_when_llm_returns_empty(self) -> None:
        """Falls back to hardcoded action when LLM returns empty string."""
        svc = self._make_service(llm_result="")
        task = Task(title="t", error="Merge conflict could not be resolved automatically", metadata={"merge_conflict": True})
        result = svc._generate_recommended_action(task)
        assert result == _BLOCKED_RECOMMENDED_ACTIONS["merge_conflict"]

    def test_falls_back_when_llm_raises(self) -> None:
        """Falls back to hardcoded action when LLM raises an exception."""
        svc = self._make_service()
        svc.worker_adapter.generate_recommended_action.side_effect = RuntimeError("boom")
        task = Task(title="t", error="Missing required workdoc: x.md", metadata={"missing_workdoc_path": "/p"})
        result = svc._generate_recommended_action(task)
        assert result == _BLOCKED_RECOMMENDED_ACTIONS["missing_workdoc"]

    def test_internal_error_fallback_for_unknown_error(self) -> None:
        """Unknown error with empty LLM result falls back to internal_error action."""
        svc = self._make_service(llm_result="")
        task = Task(title="t", error="Something weird")
        result = svc._generate_recommended_action(task)
        assert result == _BLOCKED_RECOMMENDED_ACTIONS["internal_error"]


# ---------------------------------------------------------------------------
# _emit_task_blocked tests
# ---------------------------------------------------------------------------

class TestEmitTaskBlocked:
    """Verify centralized _emit_task_blocked stores action and emits event."""

    def _make_service(self, *, llm_result: str = "") -> OrchestratorService:
        """Create a minimal mock OrchestratorService for _emit_task_blocked testing."""
        svc = MagicMock(spec=OrchestratorService)
        svc._emit_task_blocked = OrchestratorService._emit_task_blocked.__get__(svc)
        svc._generate_recommended_action = MagicMock(return_value=llm_result or "Fallback action.")
        svc.bus = MagicMock()
        svc.container = MagicMock()
        return svc

    def test_stores_recommended_action_in_metadata(self) -> None:
        """Generated action is stored in task.metadata and task is persisted."""
        svc = self._make_service(llm_result="Do X.")
        svc._generate_recommended_action.return_value = "Do X."
        task = Task(title="t", error="err", metadata={})
        svc._emit_task_blocked(task)
        assert task.metadata["recommended_action"] == "Do X."
        svc.container.tasks.upsert.assert_called_once_with(task)

    def test_emits_task_blocked_event(self) -> None:
        """Emits a task.blocked event with default error payload."""
        svc = self._make_service()
        task = Task(title="t", error="err", metadata={})
        svc._emit_task_blocked(task)
        svc.bus.emit.assert_called_once_with(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": "err"},
        )

    def test_passes_custom_payload(self) -> None:
        """Custom payload dict is forwarded to the bus.emit call."""
        svc = self._make_service()
        task = Task(title="t", error="err", metadata={})
        custom = {"error": "err", "gate": "before_implement", "policy": "gate_max_wait"}
        svc._emit_task_blocked(task, payload=custom)
        svc.bus.emit.assert_called_once_with(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload=custom,
        )

    def test_idempotent_does_not_overwrite_existing_action(self) -> None:
        """Existing recommended_action is preserved; no regeneration or re-persist."""
        svc = self._make_service()
        task = Task(title="t", error="err", metadata={"recommended_action": "Already set."})
        svc._emit_task_blocked(task)
        assert task.metadata["recommended_action"] == "Already set."
        svc._generate_recommended_action.assert_not_called()
        # Should NOT re-persist since metadata wasn't changed by emit
        svc.container.tasks.upsert.assert_not_called()

    def test_emits_even_when_action_already_set(self) -> None:
        """Event is emitted even when recommended_action was already populated."""
        svc = self._make_service()
        task = Task(title="t", error="err", metadata={"recommended_action": "Already set."})
        svc._emit_task_blocked(task)
        svc.bus.emit.assert_called_once()


# ---------------------------------------------------------------------------
# API payload surface test
# ---------------------------------------------------------------------------

class TestTaskPayloadRecommendedAction:
    """Verify _task_payload includes recommended_action from metadata."""

    def test_includes_recommended_action_from_metadata(self) -> None:
        """Non-empty recommended_action is included in the API payload."""
        from agent_orchestrator.runtime.api.router_impl import _task_payload

        task = Task(title="t", metadata={"recommended_action": "Click Retry."})
        payload = _task_payload(task)
        assert payload["recommended_action"] == "Click Retry."

    def test_returns_none_when_absent(self) -> None:
        """Missing recommended_action returns None in the API payload."""
        from agent_orchestrator.runtime.api.router_impl import _task_payload

        task = Task(title="t", metadata={})
        payload = _task_payload(task)
        assert payload["recommended_action"] is None

    def test_returns_none_for_empty_string(self) -> None:
        """Empty-string recommended_action returns None in the API payload."""
        from agent_orchestrator.runtime.api.router_impl import _task_payload

        task = Task(title="t", metadata={"recommended_action": ""})
        payload = _task_payload(task)
        assert payload["recommended_action"] is None

    def test_returns_none_for_whitespace(self) -> None:
        """Whitespace-only recommended_action returns None in the API payload."""
        from agent_orchestrator.runtime.api.router_impl import _task_payload

        task = Task(title="t", metadata={"recommended_action": "   "})
        payload = _task_payload(task)
        assert payload["recommended_action"] is None


# ---------------------------------------------------------------------------
# Stale recommended_action clearing on status transitions
# ---------------------------------------------------------------------------

class TestRecommendedActionClearing:
    """Verify recommended_action is cleared when tasks leave 'blocked' status."""

    def test_emit_task_blocked_generates_fresh_action_after_clear(self) -> None:
        """After clearing and re-blocking, a new action is generated."""
        svc = MagicMock(spec=OrchestratorService)
        svc._emit_task_blocked = OrchestratorService._emit_task_blocked.__get__(svc)
        svc._generate_recommended_action = MagicMock(return_value="New action.")
        svc.bus = MagicMock()
        svc.container = MagicMock()

        task = Task(title="t", error="new error", metadata={})
        svc._emit_task_blocked(task)
        assert task.metadata["recommended_action"] == "New action."

        # Simulate retry clearing the action
        task.metadata.pop("recommended_action", None)
        task.status = "queued"
        task.error = None

        # Re-block with a different error
        task.status = "blocked"
        task.error = "different error"
        svc._generate_recommended_action.return_value = "Different action."
        svc._emit_task_blocked(task)
        assert task.metadata["recommended_action"] == "Different action."

    def test_stale_action_not_returned_after_clearing_metadata(self) -> None:
        """After clearing recommended_action, API payload returns None."""
        from agent_orchestrator.runtime.api.router_impl import _task_payload

        task = Task(title="t", metadata={"recommended_action": "Old stale action."})
        task.metadata.pop("recommended_action", None)
        payload = _task_payload(task)
        assert payload["recommended_action"] is None
