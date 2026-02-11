"""Pipeline execution engine — runs a task through its assigned template steps.

The engine resolves the correct template for a task, evaluates conditions, and
drives each step to completion. It integrates with the agent pool for step
execution and with the task engine for status updates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .registry import PipelineRegistry, PipelineTemplate, StepDef
from .steps.base import (
    PipelineStep,
    StepContext,
    StepOutcome,
    StepResult,
    StepRegistry,
    step_registry,
)
from ..collaboration.modes import should_gate
from ..collaboration.reasoning import ReasoningStore

# Ensure standard steps are registered on import
from .steps import standard as _standard  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline execution state
# ---------------------------------------------------------------------------

@dataclass
class StepState:
    """Tracks execution state for one step within a running pipeline."""
    name: str
    display_name: str
    status: str = "pending"    # pending | running | completed | failed | skipped
    result: Optional[StepResult] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
            "attempts": self.attempts,
            "result_message": self.result.message if self.result else None,
            "error": self.result.error if self.result else None,
        }


@dataclass
class PipelineExecution:
    """Full execution state for a task's pipeline run."""
    task_id: str
    template_id: str
    steps: list[StepState] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "pending"    # pending | running | completed | failed | cancelled
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def current_step(self) -> Optional[StepState]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status in ("completed", "skipped"))
        return done / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "template_id": self.template_id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "current_step": self.current_step.to_dict() if self.current_step else None,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PipelineEngine:
    """Executes pipeline templates for tasks.

    The engine:
    1. Resolves the template for a task type (or uses an explicit override)
    2. Creates execution state for each step
    3. Iterates through steps, skipping conditional steps as needed
    4. Invokes each step implementation from the step registry
    5. Tracks results and handles retries

    The ``execute()`` method is async so steps can do IO-bound work.
    """

    def __init__(
        self,
        pipeline_registry: Optional[PipelineRegistry] = None,
        steps: Optional[StepRegistry] = None,
        hitl_mode: str = "autopilot",
        reasoning_store: Optional[ReasoningStore] = None,
        on_event: Optional[Any] = None,
    ) -> None:
        self.pipelines = pipeline_registry or PipelineRegistry()
        self.steps = steps or step_registry
        self.hitl_mode = hitl_mode
        self.reasoning_store = reasoning_store
        self._on_event = on_event  # callback(event_type: str, data: dict)

    def resolve_template(self, task_type: str, template_override: Optional[str] = None) -> PipelineTemplate:
        if template_override:
            return self.pipelines.get(template_override)
        return self.pipelines.resolve_for_task_type(task_type)

    def create_execution(self, task_id: str, template: PipelineTemplate) -> PipelineExecution:
        step_states = [
            StepState(name=sd.name, display_name=sd.display_name or sd.name.replace("_", " ").title())
            for sd in template.steps
        ]
        return PipelineExecution(
            task_id=task_id,
            template_id=template.id,
            steps=step_states,
        )

    async def execute(self, execution: PipelineExecution, ctx: StepContext) -> PipelineExecution:
        """Run all steps in the pipeline from the current index onward.

        Returns the updated execution state.
        """
        execution.status = "running"
        execution.started_at = execution.started_at or time.time()

        template = self.pipelines.get(execution.template_id)

        while execution.current_step_index < len(execution.steps):
            step_state = execution.steps[execution.current_step_index]
            step_def = template.steps[execution.current_step_index]

            # Check if step should be skipped
            if self._should_skip(step_def, ctx):
                step_state.status = "skipped"
                step_state.result = StepResult(outcome=StepOutcome.SKIPPED, message="Condition not met")
                execution.current_step_index += 1
                continue

            # Resolve step implementation
            if not self.steps.has(step_state.name):
                logger.warning("Step '%s' not registered, skipping", step_state.name)
                step_state.status = "skipped"
                step_state.result = StepResult(outcome=StepOutcome.SKIPPED, message="Step not registered")
                execution.current_step_index += 1
                continue

            step_impl = self.steps.get(step_state.name)

            # Check step's own skip condition
            if step_impl.can_skip(ctx):
                step_state.status = "skipped"
                step_state.result = StepResult(outcome=StepOutcome.SKIPPED, message="Step self-skipped")
                execution.current_step_index += 1
                continue

            # Check HITL approval gates
            gate_name = self._step_gate(step_state.name)
            if gate_name and should_gate(self.hitl_mode, gate_name):
                step_state.status = "failed"
                step_state.result = StepResult(
                    outcome=StepOutcome.BLOCKED,
                    message=f"Approval required ({gate_name}) — HITL mode: {self.hitl_mode}",
                )
                execution.status = "failed"
                self._notify("approval_needed", {
                    "gate_type": gate_name,
                    "task_id": execution.task_id,
                    "step": step_state.name,
                })
                break

            # Execute with retries
            step_state.status = "running"
            step_state.started_at = time.time()
            ctx.step_config = dict(step_def.config)

            # Record reasoning: step started
            agent_id = ctx.agent_id or "pipeline"
            if self.reasoning_store is not None:
                self.reasoning_store.get_or_create(
                    task_id=execution.task_id,
                    agent_id=agent_id,
                    agent_role=agent_id,
                )
                self.reasoning_store.start_step(
                    task_id=execution.task_id,
                    agent_id=agent_id,
                    agent_role=agent_id,
                    step_name=step_state.name,
                )

            result = await self._run_with_retries(step_impl, ctx, step_def.retry_limit)

            step_state.result = result
            step_state.finished_at = time.time()
            step_state.attempts += 1

            # Record reasoning: step completed
            if self.reasoning_store is not None:
                self.reasoning_store.complete_step(
                    task_id=execution.task_id,
                    agent_id=agent_id,
                    step_name=step_state.name,
                    status=result.outcome.value,
                    output=result.message,
                )

            # Store result for downstream steps
            ctx.previous_results[step_state.name] = result

            if result.outcome == StepOutcome.SUCCESS:
                step_state.status = "completed"
                execution.current_step_index += 1
            elif result.outcome == StepOutcome.SKIPPED:
                step_state.status = "skipped"
                execution.current_step_index += 1
            elif result.outcome == StepOutcome.BLOCKED:
                step_state.status = "failed"
                execution.status = "failed"
                self._notify("task_blocked", {
                    "task_id": execution.task_id,
                    "step": step_state.name,
                    "error": result.message or "Step blocked",
                })
                break
            else:
                step_state.status = "failed"
                execution.status = "failed"
                self._notify("task_failed", {
                    "task_id": execution.task_id,
                    "step": step_state.name,
                    "error": result.error or result.message or "Step failed",
                })
                break

        # Check if all steps completed
        if execution.current_step_index >= len(execution.steps):
            execution.status = "completed"
            self._notify("task_completed", {
                "task_id": execution.task_id,
                "template_id": execution.template_id,
            })

        execution.finished_at = time.time()
        return execution

    async def _run_with_retries(
        self, step: PipelineStep, ctx: StepContext, max_retries: int
    ) -> StepResult:
        last_result: Optional[StepResult] = None
        for attempt in range(max_retries + 1):
            try:
                result = await step.execute(ctx)
                if result.succeeded or result.outcome == StepOutcome.BLOCKED:
                    return result
                last_result = result
            except Exception as exc:
                logger.exception("Step %s failed (attempt %d)", step.name, attempt + 1)
                last_result = StepResult(
                    outcome=StepOutcome.FAILED,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        return last_result or StepResult(outcome=StepOutcome.FAILED, error="Max retries exhausted")

    def _should_skip(self, step_def: StepDef, ctx: StepContext) -> bool:
        """Evaluate the step's condition expression.

        Supports:
        - Named shortcuts: "skip_if_docs_only", "skip_if_small_change"
        - Simple equality:   ``task_type == 'docs'``
        - Simple inequality: ``task_type != 'feature'``
        - Numeric comparisons: ``change_lines < 10``, ``code_file_count > 100``
        - The condition string describes when the step *should* run.
          If it evaluates to False the step is skipped.
        """
        if not step_def.condition:
            return False

        # Simple condition evaluation — supports basic expressions
        cond = step_def.condition.strip()

        # Named shortcuts (preserve existing behaviour)
        if cond == "skip_if_docs_only" and ctx.task_type == "docs":
            return True

        if cond == "skip_if_small_change":
            prev = ctx.previous_results.get("implement")
            if prev and prev.artifacts.get("lines_changed", 0) < 10:
                return True

        # -------------------------------------------------------------------
        # Expression-based evaluation
        # Build a namespace from context fields and previous-step artifacts.
        # -------------------------------------------------------------------
        ns = self._build_condition_namespace(ctx)

        result = self._eval_condition(cond, ns)
        if result is not None:
            # The condition describes when to RUN the step.
            # If it evaluated to False, we skip.
            return not result

        return False

    @staticmethod
    def _build_condition_namespace(ctx: StepContext) -> dict[str, Any]:
        """Build the variable namespace available to condition expressions."""
        ns: dict[str, Any] = {
            "task_type": ctx.task_type,
            "task_id": ctx.task_id,
            "task_title": ctx.task_title,
            "task_description": ctx.task_description,
        }
        # Merge in artifacts from all previous steps (later steps win on key collision)
        for _step_name, result in ctx.previous_results.items():
            if result.artifacts:
                for k, v in result.artifacts.items():
                    if isinstance(v, (str, int, float, bool)):
                        ns[k] = v
        # Also expose step_config
        for k, v in ctx.step_config.items():
            if isinstance(v, (str, int, float, bool)):
                ns[k] = v
        return ns

    @staticmethod
    def _eval_condition(cond: str, ns: dict[str, Any]) -> bool | None:
        """Evaluate a simple condition expression against a namespace.

        Returns True/False if the expression could be parsed, or None if
        the expression format is not recognised (caller falls through to
        the default behaviour).

        Supported forms:
            var == 'literal'    /  var == literal
            var != 'literal'
            var < number        /  var > number
            var <= number       /  var >= number
        """
        import re

        # --- equality / inequality ----------------------------------------
        m = re.match(r"^(\w+)\s*(==|!=)\s*['\"]?([^'\"]*)['\"]?$", cond)
        if m:
            var, op, val = m.group(1), m.group(2), m.group(3)
            actual = ns.get(var)
            if actual is None:
                return None
            # Attempt numeric coercion
            try:
                val_cmp: Any = type(actual)(val)
            except (ValueError, TypeError):
                val_cmp = val
            if op == "==":
                return actual == val_cmp
            return actual != val_cmp

        # --- numeric comparisons ------------------------------------------
        m = re.match(r"^(\w+)\s*(<|>|<=|>=)\s*(-?\d+(?:\.\d+)?)$", cond)
        if m:
            var, op, val_str = m.group(1), m.group(2), m.group(3)
            actual = ns.get(var)
            if actual is None:
                return None
            try:
                actual_num = float(actual)
                val_num = float(val_str)
            except (ValueError, TypeError):
                return None
            if op == "<":
                return actual_num < val_num
            if op == ">":
                return actual_num > val_num
            if op == "<=":
                return actual_num <= val_num
            return actual_num >= val_num

        return None

    _GATE_MAPPING: dict[str, str] = {
        "plan": "before_plan",
        "plan_impl": "before_plan",
        "implement": "before_implement",
        "commit": "before_commit",
        "review": "after_implement",
    }

    def _step_gate(self, step_name: str) -> Optional[str]:
        """Map a step name to its HITL approval gate, if any."""
        return self._GATE_MAPPING.get(step_name)

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        """Fire an event notification if a callback is registered."""
        if self._on_event:
            try:
                self._on_event(event_type, data)
            except Exception:
                logger.exception("Error in pipeline event callback")

    # -- Pipeline modification at runtime ------------------------------------

    def skip_step(self, execution: PipelineExecution, step_index: int) -> None:
        """Skip a step that hasn't run yet."""
        if 0 <= step_index < len(execution.steps):
            state = execution.steps[step_index]
            if state.status == "pending":
                state.status = "skipped"
                state.result = StepResult(outcome=StepOutcome.SKIPPED, message="Manually skipped")

    def insert_step(
        self,
        execution: PipelineExecution,
        step_name: str,
        after_index: int,
    ) -> None:
        """Insert a new step into the pipeline at runtime."""
        new_state = StepState(name=step_name, display_name=step_name.replace("_", " ").title())
        insert_at = min(after_index + 1, len(execution.steps))
        execution.steps.insert(insert_at, new_state)
        # Adjust current index if insertion is before it
        if insert_at <= execution.current_step_index:
            execution.current_step_index += 1
