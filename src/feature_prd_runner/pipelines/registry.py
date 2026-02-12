"""Pipeline template registry — defines step sequences for different task types.

A *PipelineTemplate* describes the ordered steps a task goes through, along with
optional conditions, parallel groups, and per-step configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step definition within a template
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepDef:
    """One step in a pipeline template."""
    name: str                                      # references StepRegistry key
    display_name: str = ""                         # human-readable label
    required: bool = True                          # can be skipped?
    condition: Optional[str] = None                # skip-rule expression (evaluated at runtime)
    timeout_seconds: int = 600                     # max time for this step
    retry_limit: int = 3                           # retries before escalating
    agent_role: Optional[str] = None               # preferred agent role (None = auto)
    config: dict[str, Any] = field(default_factory=dict)  # step-specific config


# ---------------------------------------------------------------------------
# Pipeline Template
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineTemplate:
    """Immutable template defining how a type of task is executed."""
    id: str
    display_name: str
    description: str
    steps: tuple[StepDef, ...]
    task_types: tuple[str, ...] = ()     # which task types use this by default
    allow_skip: bool = True              # can user skip individual steps?
    allow_reorder: bool = False          # can steps be reordered at runtime?
    metadata: dict[str, Any] = field(default_factory=dict)

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

FEATURE_PIPELINE = PipelineTemplate(
    id="feature",
    display_name="Feature Implementation",
    description="Full feature lifecycle: plan, implement, verify, review, commit.",
    task_types=("feature",),
    steps=(
        StepDef(name="plan", display_name="Plan"),
        StepDef(name="plan_impl", display_name="Plan Implementation"),
        StepDef(name="implement", display_name="Implement"),
        StepDef(name="verify", display_name="Verify"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

BUG_FIX_PIPELINE = PipelineTemplate(
    id="bug_fix",
    display_name="Bug Fix",
    description="Reproduce, diagnose, fix, verify, review, commit.",
    task_types=("bug",),
    steps=(
        StepDef(name="reproduce", display_name="Reproduce", timeout_seconds=300),
        StepDef(name="diagnose", display_name="Diagnose"),
        StepDef(name="implement", display_name="Fix"),
        StepDef(name="verify", display_name="Verify"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

REFACTOR_PIPELINE = PipelineTemplate(
    id="refactor",
    display_name="Refactoring",
    description="Analyze current code, plan refactor, implement, verify, review.",
    task_types=("refactor",),
    steps=(
        StepDef(name="analyze", display_name="Analyze"),
        StepDef(name="plan", display_name="Plan"),
        StepDef(name="implement", display_name="Implement"),
        StepDef(name="verify", display_name="Verify"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

RESEARCH_PIPELINE = PipelineTemplate(
    id="research",
    display_name="Research",
    description="Gather information, analyze, summarize findings.",
    task_types=("research",),
    steps=(
        StepDef(name="gather", display_name="Gather"),
        StepDef(name="analyze", display_name="Analyze"),
        StepDef(name="summarize", display_name="Summarize"),
        StepDef(name="report", display_name="Report", required=False),
    ),
)

DOCS_PIPELINE = PipelineTemplate(
    id="docs",
    display_name="Documentation",
    description="Analyze code, write documentation, review, commit.",
    task_types=("docs",),
    steps=(
        StepDef(name="analyze", display_name="Analyze Code"),
        StepDef(name="implement", display_name="Write Docs"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

TEST_PIPELINE = PipelineTemplate(
    id="test",
    display_name="Testing",
    description="Analyze coverage, write tests, verify, commit.",
    task_types=("test",),
    steps=(
        StepDef(name="analyze", display_name="Analyze Coverage"),
        StepDef(name="implement", display_name="Write Tests"),
        StepDef(name="verify", display_name="Run Tests"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

REPO_REVIEW_PIPELINE = PipelineTemplate(
    id="repo_review",
    display_name="Repository Review",
    description="Scan codebase, analyze findings, generate improvement tasks.",
    task_types=("repo_review",),
    steps=(
        StepDef(name="scan", display_name="Scan"),
        StepDef(name="analyze", display_name="Analyze"),
        StepDef(name="generate_tasks", display_name="Generate Tasks"),
    ),
)

SECURITY_AUDIT_PIPELINE = PipelineTemplate(
    id="security_audit",
    display_name="Security Audit",
    description="Scan dependencies and code for security issues.",
    task_types=("security", "security_audit"),
    steps=(
        StepDef(name="scan_deps", display_name="Scan Dependencies"),
        StepDef(name="scan_code", display_name="Scan Code"),
        StepDef(name="report", display_name="Generate Report"),
        StepDef(name="generate_tasks", display_name="Generate Fix Tasks"),
    ),
)

REVIEW_PIPELINE = PipelineTemplate(
    id="review",
    display_name="Code Review",
    description="Analyze existing work, review changes, and produce a report.",
    task_types=("review",),
    steps=(
        StepDef(name="analyze", display_name="Analyze"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="report", display_name="Report", required=False),
    ),
)

PERFORMANCE_PIPELINE = PipelineTemplate(
    id="performance",
    display_name="Performance Optimization",
    description="Profile baseline, plan optimization, implement, benchmark to verify improvement.",
    task_types=("performance",),
    steps=(
        StepDef(name="profile", display_name="Profile Baseline"),
        StepDef(name="plan", display_name="Plan Optimization"),
        StepDef(name="implement", display_name="Implement"),
        StepDef(name="benchmark", display_name="Benchmark"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)


HOTFIX_PIPELINE = PipelineTemplate(
    id="hotfix",
    display_name="Hotfix",
    description="Abbreviated bug fix: skip diagnosis, go straight to fix, verify, review, commit.",
    task_types=("hotfix",),
    steps=(
        StepDef(name="implement", display_name="Fix"),
        StepDef(name="verify", display_name="Verify"),
        StepDef(name="review", display_name="Review"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

SPIKE_PIPELINE = PipelineTemplate(
    id="spike",
    display_name="Spike",
    description="Timeboxed exploration with throwaway prototyping. No commit.",
    task_types=("spike",),
    steps=(
        StepDef(name="gather", display_name="Gather Context"),
        StepDef(name="prototype", display_name="Prototype"),
        StepDef(name="summarize", display_name="Summarize Findings"),
        StepDef(name="report", display_name="Report"),
    ),
)

CHORE_PIPELINE = PipelineTemplate(
    id="chore",
    display_name="Chore",
    description="Mechanical code change: implement, verify, commit. No plan or review.",
    task_types=("chore",),
    steps=(
        StepDef(name="implement", display_name="Implement"),
        StepDef(name="verify", display_name="Verify"),
        StepDef(name="commit", display_name="Commit"),
    ),
)

PLAN_ONLY_PIPELINE = PipelineTemplate(
    id="plan_only",
    display_name="Plan Only",
    description="Analyze and produce a plan or spec without implementing.",
    task_types=("plan_only", "plan"),
    steps=(
        StepDef(name="analyze", display_name="Analyze"),
        StepDef(name="plan", display_name="Plan"),
        StepDef(name="report", display_name="Report"),
    ),
)

DECOMPOSE_PIPELINE = PipelineTemplate(
    id="decompose",
    display_name="Decompose",
    description="Break a large task into implementable subtasks.",
    task_types=("decompose",),
    steps=(
        StepDef(name="analyze", display_name="Analyze Scope"),
        StepDef(name="plan", display_name="Plan Breakdown"),
        StepDef(name="generate_tasks", display_name="Generate Subtasks"),
    ),
)

VERIFY_ONLY_PIPELINE = PipelineTemplate(
    id="verify_only",
    display_name="Verify Only",
    description="Run tests and checks on current state without making changes.",
    task_types=("verify_only", "verify"),
    steps=(
        StepDef(name="verify", display_name="Run Checks"),
        StepDef(name="report", display_name="Report Results"),
    ),
)


BUILTIN_TEMPLATES: dict[str, PipelineTemplate] = {
    t.id: t
    for t in [
        FEATURE_PIPELINE,
        BUG_FIX_PIPELINE,
        REFACTOR_PIPELINE,
        RESEARCH_PIPELINE,
        DOCS_PIPELINE,
        TEST_PIPELINE,
        REPO_REVIEW_PIPELINE,
        SECURITY_AUDIT_PIPELINE,
        REVIEW_PIPELINE,
        PERFORMANCE_PIPELINE,
        HOTFIX_PIPELINE,
        SPIKE_PIPELINE,
        CHORE_PIPELINE,
        PLAN_ONLY_PIPELINE,
        DECOMPOSE_PIPELINE,
        VERIFY_ONLY_PIPELINE,
    ]
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PipelineRegistry:
    """Registry of pipeline templates.

    Starts with built-in templates and allows registration of custom templates
    (e.g. loaded from user .prd_runner/pipelines/ YAML files).
    """

    def __init__(self) -> None:
        self._templates: dict[str, PipelineTemplate] = dict(BUILTIN_TEMPLATES)
        self._type_mapping: dict[str, str] = {}
        self._rebuild_type_mapping()

    def _rebuild_type_mapping(self) -> None:
        self._type_mapping = {}
        for tmpl in self._templates.values():
            for tt in tmpl.task_types:
                self._type_mapping[tt] = tmpl.id

    # -- query ---------------------------------------------------------------

    def get(self, template_id: str) -> PipelineTemplate:
        if template_id not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(f"Unknown pipeline '{template_id}' (available: {available})")
        return self._templates[template_id]

    def list_templates(self) -> list[PipelineTemplate]:
        return list(self._templates.values())

    def resolve_for_task_type(self, task_type: str) -> PipelineTemplate:
        """Return the best pipeline template for a given task type."""
        tmpl_id = self._type_mapping.get(task_type)
        if tmpl_id:
            return self._templates[tmpl_id]
        # Default to feature pipeline for unknown types
        return self._templates["feature"]

    # -- mutation ------------------------------------------------------------

    def register(self, template: PipelineTemplate) -> None:
        self._templates[template.id] = template
        self._rebuild_type_mapping()

    def unregister(self, template_id: str) -> None:
        self._templates.pop(template_id, None)
        self._rebuild_type_mapping()

    # -- YAML loading --------------------------------------------------------

    def load_from_yaml(self, path: Path) -> None:
        """Load pipeline templates from YAML files.

        *path* may be either:
        - A single ``.yaml`` / ``.yml`` file defining one template, or
        - A directory containing multiple YAML files (one template each).

        Each YAML file must be a mapping with at least ``id``, ``display_name``,
        ``description``, and ``steps`` (a list of step definitions).
        """
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load pipeline YAML files. Install pyyaml."
            )

        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix in (".yaml", ".yml") and child.is_file():
                    self._load_single_yaml(child)
        elif path.is_file():
            self._load_single_yaml(path)
        else:
            logger.debug("Pipeline YAML path does not exist: %s", path)

    def _load_single_yaml(self, path: Path) -> None:
        """Parse one YAML file into a ``PipelineTemplate`` and register it."""
        try:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
        except Exception:
            logger.warning("Failed to parse pipeline YAML: %s", path, exc_info=True)
            return

        if not isinstance(data, dict):
            logger.warning("Pipeline YAML root is not a mapping: %s", path)
            return

        template_id = data.get("id")
        if not template_id:
            logger.warning("Pipeline YAML missing 'id': %s", path)
            return

        # Parse steps
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            logger.warning("Pipeline YAML 'steps' is not a list: %s", path)
            return

        step_defs: list[StepDef] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict) or "name" not in raw_step:
                logger.warning("Skipping step entry without 'name' in %s", path)
                continue

            step_kwargs: dict[str, Any] = {"name": raw_step["name"]}
            # Copy recognized StepDef fields
            for field_name in (
                "display_name", "required", "condition", "timeout_seconds",
                "retry_limit", "agent_role",
            ):
                if field_name in raw_step:
                    step_kwargs[field_name] = raw_step[field_name]
            if "config" in raw_step and isinstance(raw_step["config"], dict):
                step_kwargs["config"] = raw_step["config"]

            step_defs.append(StepDef(**step_kwargs))

        # Parse task_types
        task_types = data.get("task_types", ())
        if isinstance(task_types, list):
            task_types = tuple(task_types)

        # Build metadata from unknown keys
        _KNOWN = {"id", "display_name", "description", "steps", "task_types",
                   "allow_skip", "allow_reorder"}
        metadata = {k: v for k, v in data.items() if k not in _KNOWN}

        template = PipelineTemplate(
            id=template_id,
            display_name=data.get("display_name", template_id.replace("_", " ").title()),
            description=data.get("description", ""),
            steps=tuple(step_defs),
            task_types=task_types,
            allow_skip=data.get("allow_skip", True),
            allow_reorder=data.get("allow_reorder", False),
            metadata=metadata,
        )

        self.register(template)
