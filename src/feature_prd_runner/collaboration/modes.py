"""Human-in-the-loop mode definitions and enforcement.

Each task can be configured with a HITL mode that controls how agents interact
with humans during execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HITLMode(str, Enum):
    AUTOPILOT = "autopilot"         # agents run freely, humans review after
    SUPERVISED = "supervised"       # agents propose, humans approve each step
    COLLABORATIVE = "collaborative" # humans and agents take turns
    REVIEW_ONLY = "review_only"     # agents implement, humans do all code review


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a HITL mode — which gates are active."""
    mode: HITLMode
    display_name: str
    description: str

    # Which approval gates are required
    approve_before_plan: bool = False
    approve_before_implement: bool = False
    approve_before_commit: bool = False
    approve_after_implement: bool = False

    # Whether agent can proceed without human presence
    allow_unattended: bool = True

    # Whether the agent should explain its reasoning at each step
    require_reasoning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "display_name": self.display_name,
            "description": self.description,
            "approve_before_plan": self.approve_before_plan,
            "approve_before_implement": self.approve_before_implement,
            "approve_before_commit": self.approve_before_commit,
            "approve_after_implement": self.approve_after_implement,
            "allow_unattended": self.allow_unattended,
            "require_reasoning": self.require_reasoning,
        }


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------

MODE_CONFIGS: dict[str, ModeConfig] = {
    HITLMode.AUTOPILOT.value: ModeConfig(
        mode=HITLMode.AUTOPILOT,
        display_name="Autopilot",
        description="Agents run freely. Review results when they finish.",
        allow_unattended=True,
        require_reasoning=False,
    ),

    HITLMode.SUPERVISED.value: ModeConfig(
        mode=HITLMode.SUPERVISED,
        display_name="Supervised",
        description="Agents propose at each step. You approve before they continue.",
        approve_before_plan=True,
        approve_before_implement=True,
        approve_before_commit=True,
        allow_unattended=False,
        require_reasoning=True,
    ),

    HITLMode.COLLABORATIVE.value: ModeConfig(
        mode=HITLMode.COLLABORATIVE,
        display_name="Collaborative",
        description="You and agents work together. Review implementation before commit.",
        approve_after_implement=True,
        approve_before_commit=True,
        allow_unattended=False,
        require_reasoning=True,
    ),

    HITLMode.REVIEW_ONLY.value: ModeConfig(
        mode=HITLMode.REVIEW_ONLY,
        display_name="Review Only",
        description="Agents implement. You review all changes before commit.",
        approve_after_implement=True,
        approve_before_commit=True,
        allow_unattended=True,
        require_reasoning=False,
    ),
}


def get_mode_config(mode: str) -> ModeConfig:
    """Get the configuration for a HITL mode."""
    if mode not in MODE_CONFIGS:
        return MODE_CONFIGS[HITLMode.AUTOPILOT.value]
    return MODE_CONFIGS[mode]


def should_gate(mode: str, gate_name: str) -> bool:
    """Check if a given approval gate should be active for a mode.

    gate_name should be one of:
    - 'before_plan'
    - 'before_implement'
    - 'before_commit'
    - 'after_implement'
    """
    config = get_mode_config(mode)
    mapping = {
        "before_plan": config.approve_before_plan,
        "before_implement": config.approve_before_implement,
        "before_commit": config.approve_before_commit,
        "after_implement": config.approve_after_implement,
    }
    return mapping.get(gate_name, False)
