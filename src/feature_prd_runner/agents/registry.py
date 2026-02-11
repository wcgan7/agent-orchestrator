"""Agent type registry — defines agent roles, capabilities, and resource limits.

Each *AgentType* is a blueprint describing what an agent can do (model, prompt,
tools, budget).  An *AgentInstance* is a running incarnation of a type that has
been assigned a task and is tracked by the pool manager.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    TESTER = "tester"
    ARCHITECT = "architect"
    DEBUGGER = "debugger"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    TERMINATED = "terminated"


# ---------------------------------------------------------------------------
# Agent Type (blueprint)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceLimits:
    """Per-agent resource caps."""
    max_tokens: int = 200_000
    max_time_seconds: int = 3600
    max_cost_usd: float = 5.0
    max_retries: int = 3
    max_concurrent_files: int = 10


@dataclass(frozen=True)
class AgentType:
    """Immutable blueprint for creating agent instances."""
    role: AgentRole
    display_name: str
    description: str

    # Model / provider configuration
    worker_provider: str = "codex"          # key in WorkersRuntimeConfig.providers
    model_override: Optional[str] = None    # override model on the provider

    # Prompt template injected as system instruction
    system_prompt: str = ""

    # Tool/step access
    allowed_steps: tuple[str, ...] = ()     # pipeline steps this agent may run
    tool_access: tuple[str, ...] = ()       # additional tool names

    # Resource limits
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Affinity — task types this agent prefers (used by scheduler)
    task_type_affinity: tuple[str, ...] = ()

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Instance (runtime)
# ---------------------------------------------------------------------------

def _gen_agent_id() -> str:
    return f"agent-{uuid.uuid4().hex[:8]}"


@dataclass
class AgentInstance:
    """Mutable runtime state of a running agent."""
    id: str = field(default_factory=_gen_agent_id)
    agent_type: str = ""                  # AgentRole.value
    display_name: str = ""
    status: AgentStatus = AgentStatus.IDLE

    # Current assignment
    task_id: Optional[str] = None
    current_step: Optional[str] = None
    current_file: Optional[str] = None

    # Resource tracking
    tokens_used: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    retries: int = 0

    # Lifecycle
    started_at: Optional[str] = None
    last_heartbeat: Optional[str] = None

    # Streaming output (last N lines kept for UI)
    output_tail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "display_name": self.display_name,
            "status": self.status.value,
            "task_id": self.task_id,
            "current_step": self.current_step,
            "current_file": self.current_file,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "retries": self.retries,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "output_tail": self.output_tail[-50:],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Built-in agent types (can be extended via config)
BUILTIN_AGENT_TYPES: dict[str, AgentType] = {
    AgentRole.IMPLEMENTER.value: AgentType(
        role=AgentRole.IMPLEMENTER,
        display_name="Implementer",
        description="Writes and modifies code according to task specifications.",
        system_prompt=(
            "You are an expert software engineer. Implement the task precisely, "
            "following existing code conventions. Write clean, tested, production-ready code."
        ),
        allowed_steps=("plan", "plan_impl", "implement", "commit"),
        task_type_affinity=("feature", "bug", "refactor"),
    ),

    AgentRole.REVIEWER.value: AgentType(
        role=AgentRole.REVIEWER,
        display_name="Reviewer",
        description="Reviews code changes for correctness, style, and security.",
        system_prompt=(
            "You are a senior code reviewer. Examine changes for bugs, security issues, "
            "style violations, and adherence to the acceptance criteria. Be thorough but constructive."
        ),
        allowed_steps=("review",),
        task_type_affinity=("review",),
    ),

    AgentRole.RESEARCHER.value: AgentType(
        role=AgentRole.RESEARCHER,
        display_name="Researcher",
        description="Gathers context, analyzes codebases, and answers technical questions.",
        system_prompt=(
            "You are a technical researcher. Analyze codebases, gather relevant context, "
            "and provide clear, actionable findings. Cite specific files and line numbers."
        ),
        allowed_steps=("gather", "analyze", "summarize", "report"),
        task_type_affinity=("research",),
    ),

    AgentRole.TESTER.value: AgentType(
        role=AgentRole.TESTER,
        display_name="Tester",
        description="Runs tests, writes test cases, and verifies correctness.",
        system_prompt=(
            "You are a QA engineer. Run existing tests, identify gaps in test coverage, "
            "and write new test cases. Verify that code changes work correctly."
        ),
        allowed_steps=("verify", "implement"),
        task_type_affinity=("test",),
    ),

    AgentRole.ARCHITECT.value: AgentType(
        role=AgentRole.ARCHITECT,
        display_name="Architect",
        description="Plans high-level architecture, decomposes features into tasks.",
        system_prompt=(
            "You are a software architect. Break down complex requirements into clear, "
            "implementable tasks. Consider dependencies, risks, and design trade-offs. "
            "Produce detailed plans that implementer agents can execute directly."
        ),
        allowed_steps=("plan", "plan_impl", "analyze"),
        task_type_affinity=("feature", "refactor"),
        limits=ResourceLimits(max_tokens=300_000, max_time_seconds=1800, max_cost_usd=3.0),
    ),

    AgentRole.DEBUGGER.value: AgentType(
        role=AgentRole.DEBUGGER,
        display_name="Debugger",
        description="Diagnoses failures, reads logs, identifies root causes.",
        system_prompt=(
            "You are an expert debugger. Analyze error messages, stack traces, and logs "
            "to identify root causes. Propose minimal, targeted fixes. Focus on reproducing "
            "the issue first, then diagnosing, then fixing."
        ),
        allowed_steps=("reproduce", "diagnose", "implement", "verify"),
        task_type_affinity=("bug",),
        limits=ResourceLimits(max_tokens=200_000, max_time_seconds=1800, max_cost_usd=3.0),
    ),
}


class _CustomRole:
    """Lightweight stand-in for AgentRole when a YAML-defined role is not in the enum.

    Quacks enough like an ``AgentRole`` member that ``AgentType`` works
    correctly (frozen dataclass with a ``.value`` attribute).
    """

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"_CustomRole({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CustomRole):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


class AgentRegistry:
    """Registry of known agent types.

    Starts with built-in types and allows runtime registration of custom types.
    """

    def __init__(self) -> None:
        self._types: dict[str, AgentType] = dict(BUILTIN_AGENT_TYPES)

    # -- query ---------------------------------------------------------------

    def get_type(self, role: str) -> AgentType:
        if role not in self._types:
            available = ", ".join(sorted(self._types.keys()))
            raise KeyError(f"Unknown agent type '{role}' (available: {available})")
        return self._types[role]

    def list_types(self) -> list[AgentType]:
        return list(self._types.values())

    def has_type(self, role: str) -> bool:
        return role in self._types

    # -- mutation ------------------------------------------------------------

    def register(self, agent_type: AgentType) -> None:
        self._types[agent_type.role.value] = agent_type

    def unregister(self, role: str) -> None:
        self._types.pop(role, None)

    # -- factory -------------------------------------------------------------

    def create_instance(self, role: str, **overrides: Any) -> AgentInstance:
        """Create a new agent instance from a registered type."""
        atype = self.get_type(role)
        return AgentInstance(
            agent_type=atype.role.value,
            display_name=overrides.get("display_name", atype.display_name),
            **{k: v for k, v in overrides.items() if k != "display_name"},
        )

    # -- affinity matching ---------------------------------------------------

    def best_role_for_task_type(self, task_type: str) -> Optional[str]:
        """Return the agent role best suited for a given task type, or None."""
        for role, atype in self._types.items():
            if task_type in atype.task_type_affinity:
                return role
        return None

    # -- YAML loading --------------------------------------------------------

    def load_from_yaml(self, path: Path) -> None:
        """Load agent type definitions/overrides from a YAML file.

        The file should contain an ``agents`` key with a list of agent
        definitions.  Each entry **must** include a ``role`` string.  If the
        role matches an existing built-in agent type the YAML values are
        merged on top of it; otherwise a brand-new ``AgentType`` is created.

        Unrecognised keys are silently placed in ``metadata``.
        """
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load agent YAML files. Install pyyaml."
            )

        if not path.exists():
            logger.debug("Agent YAML file does not exist: %s", path)
            return

        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            logger.warning("Agent YAML root is not a mapping: %s", path)
            return

        agents_list = data.get("agents")
        if not isinstance(agents_list, list):
            logger.warning("Agent YAML missing 'agents' list: %s", path)
            return

        for entry in agents_list:
            if not isinstance(entry, dict) or "role" not in entry:
                logger.warning("Skipping agent entry without 'role': %s", entry)
                continue
            self._load_agent_entry(entry)

    def _load_agent_entry(self, entry: dict[str, Any]) -> None:
        """Create or merge a single agent definition from a YAML dict."""
        role_str: str = entry["role"]

        # Determine the AgentRole enum value
        try:
            role_enum = AgentRole(role_str)
        except ValueError:
            # Custom role not in the enum — create a synthetic one using the
            # string value directly.  We create the AgentType with a role that
            # matches the string.
            role_enum = None  # type: ignore[assignment]

        # Known AgentType field names (excluding 'role' itself)
        _KNOWN_FIELDS = {
            "display_name", "description", "worker_provider", "model_override",
            "system_prompt", "allowed_steps", "tool_access", "task_type_affinity",
            "max_tokens", "max_time_seconds", "max_cost_usd", "max_retries",
            "max_concurrent_files",
        }

        # Separate known fields from metadata
        overrides: dict[str, Any] = {}
        limit_overrides: dict[str, Any] = {}
        extra_metadata: dict[str, Any] = {}

        _LIMIT_KEYS = {"max_tokens", "max_time_seconds", "max_cost_usd", "max_retries", "max_concurrent_files"}

        for key, value in entry.items():
            if key == "role":
                continue
            elif key in _LIMIT_KEYS:
                limit_overrides[key] = value
            elif key in _KNOWN_FIELDS:
                overrides[key] = value
            else:
                extra_metadata[key] = value

        # Convert list values to tuples where needed
        for tuple_field in ("allowed_steps", "tool_access", "task_type_affinity"):
            if tuple_field in overrides and isinstance(overrides[tuple_field], list):
                overrides[tuple_field] = tuple(overrides[tuple_field])

        # If the role already exists, merge overrides onto the existing type
        existing: Optional[AgentType] = None
        if self.has_type(role_str):
            existing = self.get_type(role_str)

        if existing is not None:
            # Merge: start from existing values, override with YAML values
            limits_kwargs = {
                "max_tokens": limit_overrides.get("max_tokens", existing.limits.max_tokens),
                "max_time_seconds": limit_overrides.get("max_time_seconds", existing.limits.max_time_seconds),
                "max_cost_usd": limit_overrides.get("max_cost_usd", existing.limits.max_cost_usd),
                "max_retries": limit_overrides.get("max_retries", existing.limits.max_retries),
                "max_concurrent_files": limit_overrides.get("max_concurrent_files", existing.limits.max_concurrent_files),
            }
            merged_metadata = {**existing.metadata, **extra_metadata}
            agent_type = AgentType(
                role=existing.role,
                display_name=overrides.get("display_name", existing.display_name),
                description=overrides.get("description", existing.description),
                worker_provider=overrides.get("worker_provider", existing.worker_provider),
                model_override=overrides.get("model_override", existing.model_override),
                system_prompt=overrides.get("system_prompt", existing.system_prompt),
                allowed_steps=overrides.get("allowed_steps", existing.allowed_steps),
                tool_access=overrides.get("tool_access", existing.tool_access),
                limits=ResourceLimits(**limits_kwargs),
                task_type_affinity=overrides.get("task_type_affinity", existing.task_type_affinity),
                metadata=merged_metadata,
            )
        else:
            # Brand-new custom agent type
            if role_enum is None:
                # For custom roles not in the AgentRole enum, we need a
                # lightweight wrapper so AgentType.role.value still works.
                # We use a simple approach: create a tiny enum member-like
                # string wrapper.
                role_enum = _CustomRole(role_str)  # type: ignore[assignment]

            limits_kwargs = {}
            for k in ("max_tokens", "max_time_seconds", "max_cost_usd", "max_retries"):
                if k in limit_overrides:
                    limits_kwargs[k] = limit_overrides[k]

            agent_type = AgentType(
                role=role_enum,
                display_name=overrides.get("display_name", role_str.replace("_", " ").title()),
                description=overrides.get("description", ""),
                worker_provider=overrides.get("worker_provider", "codex"),
                model_override=overrides.get("model_override"),
                system_prompt=overrides.get("system_prompt", ""),
                allowed_steps=overrides.get("allowed_steps", ()),
                tool_access=overrides.get("tool_access", ()),
                limits=ResourceLimits(**limits_kwargs) if limits_kwargs else ResourceLimits(),
                task_type_affinity=overrides.get("task_type_affinity", ()),
                metadata=extra_metadata,
            )

        # Register — for custom roles we register under the string key directly
        if isinstance(agent_type.role, _CustomRole):
            self._types[role_str] = agent_type
        else:
            self.register(agent_type)
