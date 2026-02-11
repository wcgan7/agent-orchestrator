"""Base class and registry for pluggable pipeline steps.

Each step is a class that knows how to execute one stage of a pipeline.  Steps
receive a context object with the task, project directory, worker config, and
any step-specific configuration from the pipeline template.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"      # needs human intervention


@dataclass
class StepResult:
    """Outcome of executing a single pipeline step."""
    outcome: StepOutcome
    message: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.outcome == StepOutcome.SUCCESS

    @property
    def failed(self) -> bool:
        return self.outcome == StepOutcome.FAILED


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------

@dataclass
class StepContext:
    """Everything a step needs to execute."""
    task_id: str
    task_type: str
    task_title: str
    task_description: str
    project_dir: Path
    state_dir: Path
    run_id: str
    step_config: dict[str, Any] = field(default_factory=dict)

    # Populated by the pipeline engine before execution
    previous_results: dict[str, StepResult] = field(default_factory=dict)
    agent_id: Optional[str] = None

    # Mutable accumulator for artifacts produced during execution
    artifacts: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base step class
# ---------------------------------------------------------------------------

class PipelineStep(ABC):
    """Abstract base for pipeline step implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique step identifier (matches StepDef.name in templates)."""
        ...

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ").title()

    @abstractmethod
    async def execute(self, ctx: StepContext) -> StepResult:
        """Run the step. Returns a StepResult."""
        ...

    def can_skip(self, ctx: StepContext) -> bool:
        """Override to define conditions under which this step should be skipped."""
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class StepRegistry:
    """Global registry of step implementations.

    Steps register themselves on import via ``register()``.  The pipeline engine
    looks up steps by name when executing a template.
    """

    def __init__(self) -> None:
        self._steps: dict[str, type[PipelineStep]] = {}

    def register(self, step_cls: type[PipelineStep]) -> type[PipelineStep]:
        """Register a step class. Can be used as a decorator."""
        instance = step_cls()
        self._steps[instance.name] = step_cls
        return step_cls

    def get(self, name: str) -> PipelineStep:
        if name not in self._steps:
            available = ", ".join(sorted(self._steps.keys()))
            raise KeyError(f"Unknown step '{name}' (registered: {available})")
        return self._steps[name]()

    def has(self, name: str) -> bool:
        return name in self._steps

    def list_steps(self) -> list[str]:
        return sorted(self._steps.keys())


# Singleton registry — steps register here on import
step_registry = StepRegistry()
