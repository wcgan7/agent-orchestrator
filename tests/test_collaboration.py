"""Tests for the collaboration package — feedback, review comments, HITL modes, and API."""

import pytest

from feature_prd_runner.collaboration.feedback import (
    Feedback,
    FeedbackPriority,
    FeedbackStatus,
    FeedbackStore,
    FeedbackType,
    ReviewComment,
)
from feature_prd_runner.collaboration.modes import (
    HITLMode,
    MODE_CONFIGS,
    get_mode_config,
    should_gate,
)


# ============================================================================
# Feedback model
# ============================================================================


class TestFeedbackModel:
    def test_defaults(self):
        fb = Feedback()
        assert fb.feedback_type == FeedbackType.GENERAL
        assert fb.priority == FeedbackPriority.SHOULD
        assert fb.status == FeedbackStatus.ACTIVE
        assert fb.id.startswith("fb-")
        assert fb.created_at  # non-empty

    def test_to_dict_roundtrip(self):
        fb = Feedback(
            task_id="t-1",
            feedback_type=FeedbackType.LIBRARY_SWAP,
            priority=FeedbackPriority.MUST,
            summary="Use pandas instead of csv",
            original_value="csv",
            replacement_value="pandas",
        )
        d = fb.to_dict()
        assert d["feedback_type"] == "library_swap"
        assert d["priority"] == "must"
        assert d["summary"] == "Use pandas instead of csv"

        fb2 = Feedback.from_dict(d)
        assert fb2.feedback_type == FeedbackType.LIBRARY_SWAP
        assert fb2.priority == FeedbackPriority.MUST
        assert fb2.original_value == "csv"
        assert fb2.replacement_value == "pandas"

    def test_from_dict_unknown_fields_ignored(self):
        fb = Feedback.from_dict({"task_id": "t-1", "unknown_field": "val"})
        assert fb.task_id == "t-1"

    def test_prompt_instruction_approach_change(self):
        fb = Feedback(
            feedback_type=FeedbackType.APPROACH_CHANGE,
            priority=FeedbackPriority.MUST,
            summary="Use async/await",
            action="Refactor to async",
        )
        result = fb.to_prompt_instruction()
        assert "[MUST]" in result
        assert "Change approach" in result
        assert "Refactor to async" in result

    def test_prompt_instruction_library_swap(self):
        fb = Feedback(
            feedback_type=FeedbackType.LIBRARY_SWAP,
            priority=FeedbackPriority.SHOULD,
            original_value="requests",
            replacement_value="httpx",
        )
        result = fb.to_prompt_instruction()
        assert "[SHOULD]" in result
        assert "'httpx'" in result
        assert "'requests'" in result

    def test_prompt_instruction_file_restriction(self):
        fb = Feedback(
            feedback_type=FeedbackType.FILE_RESTRICTION,
            priority=FeedbackPriority.MUST,
            target_file="config.py",
        )
        result = fb.to_prompt_instruction()
        assert "Do NOT modify" in result
        assert "config.py" in result

    def test_prompt_instruction_style_preference(self):
        fb = Feedback(
            feedback_type=FeedbackType.STYLE_PREFERENCE,
            priority=FeedbackPriority.SUGGESTION,
            summary="Use snake_case for all functions",
        )
        result = fb.to_prompt_instruction()
        assert "[SUGGESTION]" in result
        assert "Style:" in result

    def test_prompt_instruction_bug_report(self):
        fb = Feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            priority=FeedbackPriority.MUST,
            summary="Off-by-one in loop",
            target_file="main.py",
            target_lines="10-15",
        )
        result = fb.to_prompt_instruction()
        assert "Fix bug" in result
        assert "main.py" in result
        assert "lines 10-15" in result

    def test_prompt_instruction_general(self):
        fb = Feedback(
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.SHOULD,
            summary="Be more concise",
            details="Reduce function length to under 30 lines",
        )
        result = fb.to_prompt_instruction()
        assert "Be more concise" in result
        assert "Details:" in result


# ============================================================================
# Review comments
# ============================================================================


class TestReviewComment:
    def test_defaults(self):
        rc = ReviewComment()
        assert rc.id.startswith("rc-")
        assert rc.resolved is False
        assert rc.author_type == "human"

    def test_to_dict(self):
        rc = ReviewComment(
            task_id="t-1",
            file_path="src/app.py",
            line_number=42,
            body="This should be extracted to a helper",
            author="alice",
        )
        d = rc.to_dict()
        assert d["file_path"] == "src/app.py"
        assert d["line_number"] == 42
        assert d["body"] == "This should be extracted to a helper"
        assert d["resolved"] is False


# ============================================================================
# Feedback store
# ============================================================================


class TestFeedbackStore:
    def setup_method(self):
        self.store = FeedbackStore()

    def test_add_and_get_feedback(self):
        fb = Feedback(task_id="t-1", summary="Test feedback")
        self.store.add_feedback(fb)

        items = self.store.get_feedback("t-1")
        assert len(items) == 1
        assert items[0].summary == "Test feedback"

    def test_get_feedback_empty(self):
        items = self.store.get_feedback("nonexistent")
        assert items == []

    def test_active_only_filter(self):
        self.store.add_feedback(Feedback(task_id="t-1", summary="Active"))
        fb2 = Feedback(task_id="t-1", summary="Dismissed")
        fb2.status = FeedbackStatus.DISMISSED
        self.store.add_feedback(fb2)

        all_items = self.store.get_feedback("t-1")
        assert len(all_items) == 2

        active = self.store.get_feedback("t-1", active_only=True)
        assert len(active) == 1
        assert active[0].summary == "Active"

    def test_type_filter(self):
        self.store.add_feedback(Feedback(task_id="t-1", feedback_type=FeedbackType.BUG_REPORT, summary="Bug"))
        self.store.add_feedback(Feedback(task_id="t-1", feedback_type=FeedbackType.GENERAL, summary="General"))

        bugs = self.store.get_feedback("t-1", feedback_type=FeedbackType.BUG_REPORT)
        assert len(bugs) == 1
        assert bugs[0].summary == "Bug"

    def test_address_feedback(self):
        fb = Feedback(task_id="t-1", summary="Fix this")
        self.store.add_feedback(fb)

        result = self.store.address_feedback(fb.id, "Fixed in commit abc")
        assert result is True

        items = self.store.get_feedback("t-1")
        assert items[0].status == FeedbackStatus.ADDRESSED
        assert items[0].agent_response == "Fixed in commit abc"
        assert items[0].addressed_at is not None

    def test_address_nonexistent(self):
        assert self.store.address_feedback("no-such-id") is False

    def test_dismiss_feedback(self):
        fb = Feedback(task_id="t-1", summary="Dismiss me")
        self.store.add_feedback(fb)

        result = self.store.dismiss_feedback(fb.id)
        assert result is True
        assert self.store.get_feedback("t-1")[0].status == FeedbackStatus.DISMISSED

    def test_dismiss_nonexistent(self):
        assert self.store.dismiss_feedback("no-such-id") is False

    def test_get_prompt_instructions(self):
        self.store.add_feedback(Feedback(
            task_id="t-1",
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.MUST,
            summary="Be explicit",
        ))
        self.store.add_feedback(Feedback(
            task_id="t-1",
            feedback_type=FeedbackType.FILE_RESTRICTION,
            priority=FeedbackPriority.MUST,
            target_file="secret.py",
        ))

        instructions = self.store.get_prompt_instructions("t-1")
        assert "Human feedback to incorporate:" in instructions
        assert "Be explicit" in instructions
        assert "Do NOT modify" in instructions
        assert "secret.py" in instructions

    def test_get_prompt_instructions_empty(self):
        assert self.store.get_prompt_instructions("t-1") == ""

    # -- Comments CRUD -------------------------------------------------------

    def test_add_and_get_comments(self):
        c = ReviewComment(task_id="t-1", file_path="a.py", line_number=10, body="Good")
        self.store.add_comment(c)

        items = self.store.get_comments("t-1")
        assert len(items) == 1
        assert items[0].body == "Good"

    def test_filter_comments_by_file(self):
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="a.py", line_number=1, body="A"))
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="b.py", line_number=1, body="B"))

        items = self.store.get_comments("t-1", file_path="a.py")
        assert len(items) == 1
        assert items[0].body == "A"

    def test_filter_unresolved_comments(self):
        c1 = ReviewComment(task_id="t-1", file_path="a.py", body="Open")
        c2 = ReviewComment(task_id="t-1", file_path="a.py", body="Resolved")
        c2.resolved = True
        self.store.add_comment(c1)
        self.store.add_comment(c2)

        items = self.store.get_comments("t-1", unresolved_only=True)
        assert len(items) == 1
        assert items[0].body == "Open"

    def test_resolve_comment(self):
        c = ReviewComment(task_id="t-1", file_path="a.py", body="Resolve me")
        self.store.add_comment(c)

        result = self.store.resolve_comment(c.id)
        assert result is True
        assert self.store.get_comments("t-1")[0].resolved is True
        assert self.store.get_comments("t-1")[0].resolved_at is not None

    def test_resolve_nonexistent_comment(self):
        assert self.store.resolve_comment("no-such-id") is False

    def test_clear_task(self):
        self.store.add_feedback(Feedback(task_id="t-1", summary="FB"))
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="a.py", body="C"))

        self.store.clear_task("t-1")
        assert self.store.get_feedback("t-1") == []
        assert self.store.get_comments("t-1") == []


# ============================================================================
# HITL modes
# ============================================================================


class TestHITLModes:
    def test_all_modes_defined(self):
        for mode in HITLMode:
            assert mode.value in MODE_CONFIGS

    def test_autopilot_config(self):
        config = MODE_CONFIGS["autopilot"]
        assert config.allow_unattended is True
        assert config.approve_before_plan is False
        assert config.approve_before_implement is False
        assert config.approve_before_commit is False

    def test_supervised_config(self):
        config = MODE_CONFIGS["supervised"]
        assert config.allow_unattended is False
        assert config.approve_before_plan is True
        assert config.approve_before_implement is True
        assert config.approve_before_commit is True
        assert config.require_reasoning is True

    def test_collaborative_config(self):
        config = MODE_CONFIGS["collaborative"]
        assert config.approve_after_implement is True
        assert config.approve_before_commit is True

    def test_review_only_config(self):
        config = MODE_CONFIGS["review_only"]
        assert config.allow_unattended is True
        assert config.approve_after_implement is True
        assert config.approve_before_commit is True

    def test_get_mode_config_valid(self):
        config = get_mode_config("supervised")
        assert config.mode == HITLMode.SUPERVISED

    def test_get_mode_config_invalid_falls_back(self):
        config = get_mode_config("nonexistent")
        assert config.mode == HITLMode.AUTOPILOT

    def test_mode_config_to_dict(self):
        config = MODE_CONFIGS["supervised"]
        d = config.to_dict()
        assert d["mode"] == "supervised"
        assert d["approve_before_plan"] is True

    def test_should_gate(self):
        assert should_gate("supervised", "before_plan") is True
        assert should_gate("supervised", "before_implement") is True
        assert should_gate("autopilot", "before_plan") is False
        assert should_gate("autopilot", "before_commit") is False

    def test_should_gate_unknown(self):
        assert should_gate("supervised", "unknown_gate") is False

    def test_should_gate_review_only(self):
        assert should_gate("review_only", "after_implement") is True
        assert should_gate("review_only", "before_commit") is True
        assert should_gate("review_only", "before_plan") is False


# ============================================================================
# Collaboration API
# ============================================================================


class TestCollaborationAPI:
    """Test the API router via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from feature_prd_runner.server.collaboration_api import create_collaboration_router

        self.store = FeedbackStore()
        router = create_collaboration_router(lambda: self.store)
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_add_feedback(self):
        resp = self.client.post("/api/v2/collaboration/feedback", json={
            "task_id": "t-1",
            "feedback_type": "bug_report",
            "priority": "must",
            "summary": "Off-by-one",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "t-1"
        assert data["feedback_type"] == "bug_report"
        assert data["priority"] == "must"

    def test_get_feedback(self):
        self.store.add_feedback(Feedback(task_id="t-1", summary="A"))
        self.store.add_feedback(Feedback(task_id="t-1", summary="B"))

        resp = self.client.get("/api/v2/collaboration/feedback/t-1")
        assert resp.status_code == 200
        items = resp.json()["feedback"]
        assert len(items) == 2

    def test_get_feedback_active_only(self):
        fb = Feedback(task_id="t-1", summary="Active")
        self.store.add_feedback(fb)
        fb2 = Feedback(task_id="t-1", summary="Dismissed")
        fb2.status = FeedbackStatus.DISMISSED
        self.store.add_feedback(fb2)

        resp = self.client.get("/api/v2/collaboration/feedback/t-1?active_only=true")
        items = resp.json()["feedback"]
        assert len(items) == 1
        assert items[0]["summary"] == "Active"

    def test_address_feedback(self):
        fb = Feedback(task_id="t-1", summary="Fix me")
        self.store.add_feedback(fb)

        resp = self.client.post(f"/api/v2/collaboration/feedback/{fb.id}/address?agent_response=Done")
        assert resp.status_code == 200
        assert self.store.get_feedback("t-1")[0].status == FeedbackStatus.ADDRESSED

    def test_address_not_found(self):
        resp = self.client.post("/api/v2/collaboration/feedback/no-such/address")
        assert resp.status_code == 404

    def test_dismiss_feedback(self):
        fb = Feedback(task_id="t-1", summary="Not useful")
        self.store.add_feedback(fb)

        resp = self.client.post(f"/api/v2/collaboration/feedback/{fb.id}/dismiss")
        assert resp.status_code == 200
        assert self.store.get_feedback("t-1")[0].status == FeedbackStatus.DISMISSED

    def test_dismiss_not_found(self):
        resp = self.client.post("/api/v2/collaboration/feedback/no-such/dismiss")
        assert resp.status_code == 404

    def test_get_feedback_prompt(self):
        self.store.add_feedback(Feedback(
            task_id="t-1",
            feedback_type=FeedbackType.GENERAL,
            priority=FeedbackPriority.MUST,
            summary="Be verbose",
        ))

        resp = self.client.get("/api/v2/collaboration/feedback/t-1/prompt")
        assert resp.status_code == 200
        assert "Be verbose" in resp.json()["instructions"]

    def test_add_comment(self):
        resp = self.client.post("/api/v2/collaboration/comments", json={
            "task_id": "t-1",
            "file_path": "src/app.py",
            "line_number": 42,
            "body": "Extract this",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_path"] == "src/app.py"
        assert data["line_number"] == 42

    def test_get_comments(self):
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="a.py", body="X"))

        resp = self.client.get("/api/v2/collaboration/comments/t-1")
        assert resp.status_code == 200
        assert len(resp.json()["comments"]) == 1

    def test_get_comments_filtered(self):
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="a.py", body="A"))
        self.store.add_comment(ReviewComment(task_id="t-1", file_path="b.py", body="B"))

        resp = self.client.get("/api/v2/collaboration/comments/t-1?file_path=a.py")
        items = resp.json()["comments"]
        assert len(items) == 1
        assert items[0]["body"] == "A"

    def test_resolve_comment(self):
        c = ReviewComment(task_id="t-1", file_path="a.py", body="Resolve me")
        self.store.add_comment(c)

        resp = self.client.post(f"/api/v2/collaboration/comments/{c.id}/resolve")
        assert resp.status_code == 200
        assert self.store.get_comments("t-1")[0].resolved is True

    def test_resolve_not_found(self):
        resp = self.client.post("/api/v2/collaboration/comments/no-such/resolve")
        assert resp.status_code == 404

    def test_list_modes(self):
        resp = self.client.get("/api/v2/collaboration/modes")
        assert resp.status_code == 200
        modes = resp.json()["modes"]
        assert len(modes) == 4
        mode_names = [m["mode"] for m in modes]
        assert "autopilot" in mode_names
        assert "supervised" in mode_names

    def test_get_mode(self):
        resp = self.client.get("/api/v2/collaboration/modes/supervised")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "supervised"
        assert data["approve_before_plan"] is True
