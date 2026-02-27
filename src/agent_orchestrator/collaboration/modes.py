"""Human-in-the-loop mode definitions and enforcement.

Each task can be configured with a HITL mode that controls how agents interact
with humans during execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HITLMode(str, Enum):
    """Supported human-in-the-loop execution modes."""
    AUTOPILOT = "autopilot"         # fully autonomous execution
    SUPERVISED = "supervised"       # approve plan, then approve before commit
    REVIEW_ONLY = "review_only"     # skip plan gate, review before commit


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a HITL mode — which gates are active."""
    mode: HITLMode
    display_name: str
    description: str

    # Which approval gates are required
    approve_before_plan: bool = False
    approve_before_implement: bool = False
    approve_before_generate_tasks: bool = False
    approve_before_commit: bool = False
    approve_before_done: bool = False
    approve_after_implement: bool = False

    # Whether agent can proceed without human presence
    allow_unattended: bool = True

    # Whether the agent should explain its reasoning at each step
    require_reasoning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the mode configuration for API and storage payloads.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        return {
            "mode": self.mode.value,
            "display_name": self.display_name,
            "description": self.description,
            "approve_before_plan": self.approve_before_plan,
            "approve_before_implement": self.approve_before_implement,
            "approve_before_generate_tasks": self.approve_before_generate_tasks,
            "approve_before_commit": self.approve_before_commit,
            "approve_before_done": self.approve_before_done,
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
        description="No approvals. Agents run end-to-end automatically.",
        allow_unattended=True,
        require_reasoning=False,
    ),

    HITLMode.SUPERVISED.value: ModeConfig(
        mode=HITLMode.SUPERVISED,
        display_name="Supervised",
        description="Approve the plan, then review implementation before commit.",
        approve_before_plan=False,
        approve_before_implement=True,
        approve_before_generate_tasks=True,
        approve_after_implement=False,
        approve_before_commit=True,
        approve_before_done=True,
        allow_unattended=False,
        require_reasoning=True,
    ),

    HITLMode.REVIEW_ONLY.value: ModeConfig(
        mode=HITLMode.REVIEW_ONLY,
        display_name="Review Only",
        description="Skip plan approval. Review implementation before commit.",
        approve_after_implement=False,
        approve_before_commit=True,
        approve_before_done=True,
        allow_unattended=True,
        require_reasoning=False,
    ),
}


def normalize_hitl_mode(mode: str | None, *, default: str = HITLMode.AUTOPILOT.value) -> str:
    """Normalize persisted/user mode values, including legacy aliases."""
    raw = str(mode or "").strip().lower()
    if raw == "collaborative":
        raw = HITLMode.SUPERVISED.value
    if raw in MODE_CONFIGS:
        return raw
    return default


def get_mode_config(mode: str) -> ModeConfig:
    """Get the configuration for a HITL mode.

    Args:
        mode (str): Requested mode identifier. Supported values are
            ``autopilot``, ``supervised``, and ``review_only``.

    Returns:
        ModeConfig: Configuration that controls approval gates and
            unattended execution rules for the mode. Unknown values fall
            back to the ``autopilot`` configuration.
    """
    return MODE_CONFIGS[normalize_hitl_mode(mode)]


def should_gate(mode: str, gate_name: str) -> bool:
    """Check whether a mode enables a specific approval gate.

    Args:
        mode (str): Requested mode identifier. Unknown values use the
            ``autopilot`` defaults via :func:`get_mode_config`.
        gate_name (str): Gate name to evaluate. Supported values are
            ``before_plan``, ``before_implement``, ``before_generate_tasks``,
            ``before_commit``, ``before_done``, and ``after_implement``.

    Returns:
        bool: ``True`` when the mode requires the named gate, otherwise
            ``False``. Unsupported gate names return ``False``.
    """
    config = get_mode_config(mode)
    mapping = {
        "before_plan": config.approve_before_plan,
        "before_implement": config.approve_before_implement,
        "before_generate_tasks": config.approve_before_generate_tasks,
        "before_commit": config.approve_before_commit,
        "before_done": config.approve_before_done,
        "after_implement": config.approve_after_implement,
    }
    return mapping.get(gate_name, False)
