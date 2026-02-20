"""Tests for the collaboration package — HITL modes."""

from agent_orchestrator.collaboration.modes import (
    HITLMode,
    MODE_CONFIGS,
    get_mode_config,
    should_gate,
)


class TestHITLModes:
    """Represents TestHITLModes."""
    def test_all_modes_defined(self):
        """Test that all modes defined."""
        for mode in HITLMode:
            assert mode.value in MODE_CONFIGS

    def test_autopilot_config(self):
        """Test that autopilot config."""
        config = MODE_CONFIGS["autopilot"]
        assert config.allow_unattended is True
        assert config.approve_before_plan is False
        assert config.approve_before_implement is False
        assert config.approve_before_commit is False

    def test_supervised_config(self):
        """Test that supervised config."""
        config = MODE_CONFIGS["supervised"]
        assert config.allow_unattended is False
        assert config.approve_before_plan is False
        assert config.approve_before_implement is True
        assert config.approve_after_implement is True
        assert config.approve_before_commit is True
        assert config.require_reasoning is True

    def test_collaborative_config(self):
        """Test that collaborative config."""
        config = MODE_CONFIGS["collaborative"]
        assert config.approve_after_implement is True
        assert config.approve_before_commit is True

    def test_review_only_config(self):
        """Test that review only config."""
        config = MODE_CONFIGS["review_only"]
        assert config.allow_unattended is True
        assert config.approve_after_implement is True
        assert config.approve_before_commit is True

    def test_get_mode_config_valid(self):
        """Test that get mode config valid."""
        config = get_mode_config("supervised")
        assert config.mode == HITLMode.SUPERVISED

    def test_get_mode_config_invalid_falls_back(self):
        """Test that get mode config invalid falls back."""
        config = get_mode_config("nonexistent")
        assert config.mode == HITLMode.AUTOPILOT

    def test_mode_config_to_dict(self):
        """Test that mode config to dict."""
        config = MODE_CONFIGS["supervised"]
        d = config.to_dict()
        assert d["mode"] == "supervised"
        assert d["approve_before_plan"] is False
        assert d["approve_after_implement"] is True

    def test_should_gate(self):
        """Test that should gate."""
        assert should_gate("supervised", "before_plan") is False
        assert should_gate("supervised", "before_implement") is True
        assert should_gate("supervised", "after_implement") is True
        assert should_gate("autopilot", "before_plan") is False
        assert should_gate("autopilot", "before_commit") is False

    def test_should_gate_unknown(self):
        """Test that should gate unknown."""
        assert should_gate("supervised", "unknown_gate") is False

    def test_should_gate_review_only(self):
        """Test that should gate review only."""
        assert should_gate("review_only", "after_implement") is True
        assert should_gate("review_only", "before_commit") is True
        assert should_gate("review_only", "before_plan") is False
