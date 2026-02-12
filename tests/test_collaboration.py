"""Tests for the collaboration package — HITL modes."""

from feature_prd_runner.collaboration.modes import (
    HITLMode,
    MODE_CONFIGS,
    get_mode_config,
    should_gate,
)


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
