"""Worker adapter that dispatches pipeline steps to real Codex/Ollama providers."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...pipelines.registry import PipelineRegistry
from ...prompts import load as load_prompt
from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...worker import WorkerCancelledError
from ...workers.run import WorkerRunResult, run_worker
from ..domain.models import RunRecord, Task, now_iso
from ..storage.container import Container
from .worker_adapter import StepResult

logger = logging.getLogger(__name__)

# Step category mapping
_PLANNING_STEPS = {"plan", "analyze", "plan_refine", "initiative_plan", "initiative_plan_refine"}
_IMPL_STEPS = {"implement", "prototype"}
_FIX_STEPS = {"implement_fix"}
_VERIFY_STEPS = {"verify", "benchmark", "reproduce"}
_REVIEW_STEPS = {"review"}
_REPORT_STEPS = {"report", "summarize"}
_SCAN_STEPS = {"scan_deps", "scan_code"}
_TASK_GEN_STEPS = {"generate_tasks"}
_DIAGNOSIS_STEPS = {"diagnose"}
_MERGE_RESOLVE_STEPS = {"resolve_merge"}
_DEP_ANALYSIS_STEPS = {"analyze_deps"}
_PIPELINE_CLASSIFICATION_STEPS = {"pipeline_classify"}

# Which prior step outputs each category should receive in its prompt.
# None = inject all available outputs (for reporting/summarize steps).
_STEP_OUTPUT_INJECTION: dict[str, tuple[str, ...] | None] = {
    "implementation": ("plan", "analyze", "reproduce", "diagnose", "profile"),
    "diagnosis": ("reproduce",),
    "planning": (),
    "review": ("verify", "benchmark"),
    "reporting": None,
    "task_generation": ("initiative_plan", "plan", "analyze", "scan_deps", "scan_code"),
    "fix": ("plan",),
    "scanning": ("scan_deps",),
}

_STEP_TIMEOUT_ALIASES = {"implement_fix": "implement"}
_DEFAULT_STEP_TIMEOUT_SECONDS = 600
_DEFAULT_HEARTBEAT_SECONDS = 60
_DEFAULT_HEARTBEAT_GRACE_SECONDS = 240

# ---------------------------------------------------------------------------
# Prompt layers
# ---------------------------------------------------------------------------

_STEP_PROMPT_FILES: dict[str, str] = {
    "planning": "steps/plan.md",
    "implementation": "steps/implement.md",
    "fix": "steps/fix.md",
    "verification": "steps/verify.md",
    "review": "steps/review.md",
    "reporting": "steps/report.md",
    "scanning": "steps/scan_code.md",
    "task_generation": "steps/task_generation.md",
    "diagnosis": "steps/diagnose.md",
    "merge_resolution": "steps/merge_resolution.md",
    "dependency_analysis": "steps/dependency_analysis.md",
    "pipeline_classification": "steps/pipeline_classify.md",
    "general": "steps/general.md",
}

_LANGUAGE_DEFAULTS: dict[str, str] = {
    "python": (
        "### Python defaults\n"
        "- Google-style docstrings.\n"
        "- Type hints (Python 3.10+ syntax) on public APIs.\n"
        "- Prefer precise typing; avoid untyped public interfaces."
    ),
    "typescript": (
        "### TypeScript defaults\n"
        "- JSDoc on exported symbols where helpful.\n"
        "- Strict TypeScript; avoid `any` (prefer `unknown` + narrowing).\n"
        "- Preserve null/undefined safety."
    ),
    "javascript": (
        "### JavaScript defaults\n"
        "- JSDoc on exported symbols and non-trivial params/returns.\n"
        "- Keep runtime validation clear at module boundaries."
    ),
    "go": (
        "### Go defaults\n"
        "- Godoc conventions on exported symbols.\n"
        "- Return errors with context; avoid panics in library paths."
    ),
    "rust": (
        "### Rust defaults\n"
        "- `///` doc comments on public items.\n"
        "- Prefer `Result` propagation over `unwrap`/`expect` outside tests/tools."
    ),
}

_LANGUAGE_MARKERS: list[tuple[str, str]] = [
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("tsconfig.json", "typescript"),
    ("package.json", "javascript"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
]


_DEFAULT_PROJECT_COMMANDS: dict[str, dict[str, str]] = {
    "python": {
        "test": "pytest",
        "lint": "ruff check .",
        "typecheck": "mypy .",
    },
    "typescript": {
        "test": "npx vitest run 2>/dev/null || npm test",
        "lint": "npx eslint . 2>/dev/null || true",
        "typecheck": "npx tsc --noEmit",
    },
    "javascript": {
        "test": "npm test",
        "lint": "npx eslint . 2>/dev/null || true",
    },
    "go": {
        "test": "go test ./...",
        "lint": "golangci-lint run 2>/dev/null || true",
    },
    "rust": {
        "test": "cargo test",
        "lint": "cargo clippy -- -D warnings 2>/dev/null || true",
    },
}


def _resolve_command_paths(
    commands: dict[str, dict[str, str]],
    project_dir: Path,
) -> dict[str, dict[str, str]]:
    """Resolve relative executable paths in project commands to absolute paths.

    If the first token of a command starts with '.' and contains '/'
    (e.g. ``.venv/bin/ruff``), resolve it against *project_dir* so that
    workers running in git worktrees (which lack gitignored dirs like
    ``.venv``) can still find the binary.
    """
    resolved: dict[str, dict[str, str]] = {}
    for lang, cmds in commands.items():
        resolved_cmds: dict[str, str] = {}
        for key, cmd in cmds.items():
            parts = cmd.split(None, 1)  # split on first whitespace
            if parts and parts[0].startswith(".") and "/" in parts[0]:
                abs_exe = str((project_dir / parts[0]).resolve())
                resolved_cmds[key] = abs_exe if len(parts) == 1 else f"{abs_exe} {parts[1]}"
            else:
                resolved_cmds[key] = cmd
        resolved[lang] = resolved_cmds
    return resolved


def detect_project_languages(project_dir: Path) -> list[str]:
    """Return all detected project languages based on marker files.

    Multiple markers for the same language are deduplicated (e.g. pyproject.toml
    and setup.py both map to "python").  If a tsconfig.json is found alongside
    package.json, only "typescript" is returned (it subsumes "javascript").
    """
    seen: dict[str, None] = {}  # ordered set
    for marker, lang in _LANGUAGE_MARKERS:
        if (project_dir / marker).exists() and lang not in seen:
            seen[lang] = None
    langs = list(seen)
    # TypeScript subsumes JavaScript — drop the weaker signal
    if "typescript" in seen and "javascript" in seen:
        langs.remove("javascript")
    return langs


_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "python": "Python",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "go": "Go",
    "rust": "Rust",
}


def _format_project_commands(
    project_commands: dict[str, dict[str, str]],
    project_languages: list[str],
) -> str:
    """Format project commands for detected languages into a prompt section."""
    _COMMAND_LABELS = {"test": "Test", "lint": "Lint", "typecheck": "Typecheck", "format": "Format"}
    blocks: list[tuple[str, list[str]]] = []
    for lang in project_languages:
        cmds = project_commands.get(lang)
        if not isinstance(cmds, dict):
            continue
        lines = []
        for key in ("test", "lint", "typecheck", "format"):
            val = cmds.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(f"- {_COMMAND_LABELS[key]}: `{val.strip()}`")
        if lines:
            blocks.append((lang, lines))
    if not blocks:
        return ""
    parts = ["## Project commands"]
    if len(blocks) == 1:
        parts.extend(blocks[0][1])
    else:
        for lang, lines in blocks:
            display = _LANGUAGE_DISPLAY_NAMES.get(lang, lang.title())
            parts.append(f"### {display}")
            parts.extend(lines)
    return "\n".join(parts)


def _step_category(step: str) -> str:
    if step in _PLANNING_STEPS:
        return "planning"
    if step in _IMPL_STEPS:
        return "implementation"
    if step in _FIX_STEPS:
        return "fix"
    if step in _VERIFY_STEPS:
        return "verification"
    if step in _REVIEW_STEPS:
        return "review"
    if step in _REPORT_STEPS:
        return "reporting"
    if step in _SCAN_STEPS:
        return "scanning"
    if step in _TASK_GEN_STEPS:
        return "task_generation"
    if step in _DIAGNOSIS_STEPS:
        return "diagnosis"
    if step in _MERGE_RESOLVE_STEPS:
        return "merge_resolution"
    if step in _DEP_ANALYSIS_STEPS:
        return "dependency_analysis"
    if step in _PIPELINE_CLASSIFICATION_STEPS:
        return "pipeline_classification"
    return "general"



_WORKDOC_STEP_INSTRUCTIONS: dict[str, str] = {
    "planning": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it to understand the task context, then **replace** the `## Plan` section's\n"
        "placeholder with your full plan. Write the file back when done."
    ),
    "implementation": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read the `## Plan` section for the implementation guide. As you work,\n"
        "update the `## Implementation Log` section with completed items,\n"
        "key decisions, and any deviations from the plan. Write the file back when done."
    ),
    "fix": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read the full document for context on prior work.\n"
        "Do NOT modify `.workdoc.md` in this step — the orchestrator appends\n"
        "structured fix-cycle entries to `## Fix Log`."
    ),
    "verification": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read the implementation context for what was changed.\n"
        "Do NOT modify `.workdoc.md` in this step — the orchestrator writes\n"
        "normalized verification results after classification."
    ),
    "review": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it for full context on the plan, implementation, and verification.\n"
        "Do NOT modify the file — the orchestrator will append review findings."
    ),
}

_WORKDOC_STEP_INSTRUCTIONS_BY_STEP: dict[str, str] = {
    "analyze": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it to understand the task context, then **replace** the `## Analysis` section's\n"
        "placeholder with your analysis. Write the file back when done."
    ),
    "prototype": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read the `## Plan` section for context when present. As you work,\n"
        "update the `## Implementation Log` section with prototype notes:\n"
        "- hypotheses tested,\n"
        "- experiments run,\n"
        "- observed results,\n"
        "- decision recommendation.\n"
        "Write the file back when done."
    ),
    "diagnose": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it to understand the task context, then **replace** the `## Analysis` section's\n"
        "placeholder with your diagnosis. Write the file back when done."
    ),
    "initiative_plan": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it to understand the task context, then **replace** the `## Plan` section's\n"
        "placeholder with your initiative plan. Write the file back when done."
    ),
    "profile": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it for context before profiling.\n"
        "Do NOT modify `.workdoc.md` in this step — the orchestrator writes\n"
        "profiling output into `## Profiling Baseline`."
    ),
    "report": (
        "## Working Document\n"
        "A working document is available at `.workdoc.md` in the project root.\n"
        "Read it for final context (plan, implementation, verification, review).\n"
        "Do NOT modify `.workdoc.md` in this step — the orchestrator writes\n"
        "the report output into `## Final Report`."
    ),
}


_WORKDOC_SKIP_STEPS = {"plan_refine", "initiative_plan_refine"}  # These steps don't go through the sync path.


def _workdoc_prompt_section(step: str) -> str:
    """Return the workdoc instruction block for a step, or empty string."""
    if step in _WORKDOC_SKIP_STEPS:
        return ""
    step_specific = _WORKDOC_STEP_INSTRUCTIONS_BY_STEP.get(step)
    if step_specific:
        return step_specific
    category = _step_category(step)
    return _WORKDOC_STEP_INSTRUCTIONS.get(category, "")


_DEPENDENCY_POLICY_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "implementation": {
        "permissive": (
            "## Dependency policy — permissive\n"
            "Prefer using well-maintained libraries over manual implementation.\n"
            "Install what you need — favor proven packages for non-trivial functionality."
        ),
        "prudent": (
            "## Dependency policy — prudent\n"
            "Prefer using what is already available in the project.\n"
            "Only install a new dependency if implementing it manually would be\n"
            "unreliable or disproportionately complex. Justify any new addition."
        ),
        "strict": (
            "## Dependency policy — strict\n"
            "Do NOT install new dependencies or add entries to package.json,\n"
            "requirements.txt, pyproject.toml, or any other manifest.\n"
            "Work only with what is already installed in the project."
        ),
    },
    "planning": {
        "permissive": (
            "## Dependency policy — permissive\n"
            "The plan should recommend libraries where appropriate.\n"
            "Prefer well-maintained packages for non-trivial functionality."
        ),
        "prudent": (
            "## Dependency policy — prudent\n"
            "The plan should prefer existing dependencies but may suggest new\n"
            "ones with clear justification for why manual implementation is\n"
            "unreliable or disproportionately complex."
        ),
        "strict": (
            "## Dependency policy — strict\n"
            "The plan must NOT include any new dependencies.\n"
            "All solutions must use only what is already installed in the project."
        ),
    },
    "review": {
        "permissive": (
            "## Dependency policy — permissive\n"
            "Do not raise findings for adding new dependencies.\n"
            "New well-maintained libraries are acceptable."
        ),
        "prudent": (
            "## Dependency policy — prudent\n"
            "Flag any newly added dependency as a low-severity finding for awareness.\n"
            "Verify the addition is justified and not easily replaceable with existing code."
        ),
        "strict": (
            "## Dependency policy — strict\n"
            "Flag any change to dependency manifests (package.json, requirements.txt,\n"
            "pyproject.toml, etc.) as a high-severity finding.\n"
            "No new dependencies are allowed under this policy."
        ),
    },
}

_REVIEW_ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
_REVIEW_ALLOWED_CATEGORIES = {
    "correctness",
    "reliability",
    "security",
    "performance",
    "api_contract",
    "documentation",
    "maintainability",
    "test_coverage",
    "other",
}
_VERIFY_ALLOWED_STATUSES = {"pass", "fail", "skip", "environment"}
_VERIFY_ALLOWED_REASON_CODES = {
    "assertion_failure",
    "type_error",
    "lint_violation",
    "tool_missing",
    "config_missing",
    "no_tests",
    "permission_denied",
    "resource_limit",
    "os_incompatibility",
    "infrastructure",
    "unknown",
}

_PLAN_PREAMBLE_PREFIXES: tuple[str, ...] = (
    "the plan has been prepared",
    "here's a summary",
    "here is a summary",
    "summary of the key",
)
_PLAN_TRAILING_CHAT_PREFIXES: tuple[str, ...] = (
    "should i ",
    "would you like",
    "want me to ",
    "do you want me to ",
)


def _normalize_planning_text(text: str) -> str:
    """Normalize planning/refine outputs by stripping chatty wrappers.

    Planning revisions should contain only the plan body. Some worker replies
    add conversational prefaces or trailing confirmation questions; remove those
    when they can be identified deterministically.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # Remove markdown fences when the whole payload is fenced text.
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            raw = "\n".join(lines[1:-1]).strip()

    lines = raw.splitlines()
    heading_idx = next((idx for idx, line in enumerate(lines) if re.match(r"^\s*#{1,6}\s+\S", line)), None)

    # Drop meta-preface if the message starts with a known wrapper phrase.
    if lines:
        first = lines[0].strip().lower()
        if any(first.startswith(prefix) for prefix in _PLAN_PREAMBLE_PREFIXES) and heading_idx is not None:
            lines = lines[heading_idx:]

    # Drop trailing "should I..." style follow-up questions and any divider right above.
    cut_idx: int | None = None
    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if any(lowered.startswith(prefix) for prefix in _PLAN_TRAILING_CHAT_PREFIXES):
            cut_idx = idx
            break
    if cut_idx is not None:
        lines = lines[:cut_idx]
        while lines and lines[-1].strip() in {"", "---"}:
            lines.pop()

    return "\n".join(lines).strip()


def _load_workdoc_snapshot(task: Task, project_dir: Path) -> str:
    """Load canonical/worktree workdoc text for run-end summary context."""
    candidates: list[Path] = []
    if isinstance(task.metadata, dict):
        workdoc_path = task.metadata.get("workdoc_path")
        if isinstance(workdoc_path, str) and workdoc_path.strip():
            candidates.append(Path(workdoc_path.strip()))
    candidates.append(project_dir / ".workdoc.md")

    for path in candidates:
        try:
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text[:30000]
        except Exception:
            continue
    return "(workdoc unavailable)"


def build_step_prompt(
    *,
    task: Task,
    step: str,
    attempt: int,
    project_languages: list[str] | None = None,
    project_commands: dict[str, dict[str, str]] | None = None,
) -> str:
    """Build a prompt from Task fields with step-specific instructions."""
    category = _step_category(step)
    pipeline_id = PipelineRegistry().resolve_for_task_type(task.task_type).id
    if step == "plan_refine":
        instruction_name = "steps/plan_refine.md"
    elif step == "initiative_plan_refine":
        instruction_name = "steps/initiative_plan_refine.md"
    elif step == "implement" and pipeline_id == "docs":
        instruction_name = "steps/implement_docs.md"
    elif step == "analyze":
        instruction_name = "steps/analyze.md"
    elif step == "initiative_plan":
        instruction_name = "steps/initiative_plan.md"
    elif step == "diagnose":
        instruction_name = "steps/diagnose.md"
    elif step == "prototype":
        instruction_name = "steps/prototype.md"
    elif step == "profile":
        instruction_name = "steps/profile.md"
    elif step == "scan_deps":
        instruction_name = "steps/scan_deps.md"
    elif step == "scan_code":
        instruction_name = "steps/scan_code.md"
    elif step == "summarize":
        instruction_name = "steps/summarize.md"
    else:
        instruction_name = _STEP_PROMPT_FILES[category]
    instruction = load_prompt(instruction_name)
    preamble = load_prompt("preamble.md")
    guardrails = load_prompt("guardrails.md")

    # Special prompt for dependency analysis
    if category == "dependency_analysis" and isinstance(task.metadata, dict):
        parts = [preamble, "", instruction, ""]

        candidate_tasks = task.metadata.get("candidate_tasks")
        if isinstance(candidate_tasks, list) and candidate_tasks:
            parts.append("## Tasks to analyze")
            parts.append("")
            for ct in candidate_tasks:
                if not isinstance(ct, dict):
                    continue
                parts.append(f"- ID: {ct.get('id', '?')}")
                parts.append(f"  Title: {ct.get('title', '?')}")
                desc = str(ct.get("description") or "")[:200]
                if desc:
                    parts.append(f"  Description: {desc}")
                parts.append(f"  Type: {ct.get('task_type', 'feature')}")
                labels = ct.get("labels")
                if isinstance(labels, list) and labels:
                    parts.append(f"  Labels: {', '.join(str(label) for label in labels)}")
                parts.append("")

        existing_tasks = task.metadata.get("existing_tasks")
        if isinstance(existing_tasks, list) and existing_tasks:
            parts.append("## Already-scheduled tasks (may be blockers)")
            parts.append("")
            for et in existing_tasks:
                if not isinstance(et, dict):
                    continue
                parts.append(f"- ID: {et.get('id', '?')}")
                parts.append(f"  Title: {et.get('title', '?')}")
                parts.append(f"  Status: {et.get('status', '?')}")
                parts.append("")

        parts.append("## Rules")
        parts.append("- Only output edges where one task MUST complete before another can start.")
        parts.append("- If uncertain, omit the edge (prefer false negatives over false positives).")
        parts.append("- Use the exact task IDs from above.")
        parts.append("- Add concise, concrete reason per edge.")
        parts.append("- Include confidence (`high` or `medium`) and short evidence reference per edge.")
        parts.append("- If all tasks are independent, return an empty edges array.")
        parts.append("- Do not create circular dependencies.")

        parts.append("")
        parts.append(guardrails)

        return "\n".join(parts)

    if category == "pipeline_classification":
        parts = [preamble, "", instruction, ""]
        parts.append(f"Task: {task.title}")
        if task.description:
            parts.append(f"Description: {task.description}")
        allowed = task.metadata.get("classification_allowed_pipelines") if isinstance(task.metadata, dict) else None
        if isinstance(allowed, list) and allowed:
            cleaned = [str(item).strip() for item in allowed if str(item).strip()]
            if cleaned:
                parts.append("")
                parts.append("## Allowed pipeline IDs")
                for pipeline_id in cleaned:
                    parts.append(f"- {pipeline_id}")
        parts.append("")
        parts.append(guardrails)
        return "\n".join(parts)

    parts = [preamble, "", instruction, ""]

    parts.append(f"Task: {task.title}")
    if task.description:
        parts.append(f"Description: {task.description}")
    parts.append(f"Type: {task.task_type}")
    if attempt > 1 and step != "implement":
        parts.append(f"Attempt: {attempt}")

    # Inject workdoc instructions for steps that use the working document.
    workdoc_block = _workdoc_prompt_section(step)
    if workdoc_block:
        parts.append("")
        parts.append(workdoc_block)

    # Inject outputs from prior pipeline steps.
    # For implement/implement_fix, rely on the workdoc as the single source of truth.
    step_outputs = task.metadata.get("step_outputs") if isinstance(task.metadata, dict) else None
    if (
        isinstance(step_outputs, dict)
        and step_outputs
        and category in _STEP_OUTPUT_INJECTION
        and step not in {"implement", "implement_fix", "initiative_plan"}
    ):
        inject_keys = _STEP_OUTPUT_INJECTION[category]
        if step == "report" and task.task_type in {"security", "security_audit"}:
            # Security audit report should synthesize only scan evidence.
            inject_keys = ("scan_deps", "scan_code")
        if step == "generate_tasks" and task.task_type == "initiative_plan":
            # Initiative planning flow uses initiative_plan output as the only
            # decomposition source to avoid legacy fallback drift.
            inject_keys = ("initiative_plan",)
        elif step == "generate_tasks" and task.task_type in {"security", "security_audit"}:
            # Security remediation tasks should be derived from scan evidence and
            # report synthesis, not generic planning outputs.
            inject_keys = ("report", "scan_deps", "scan_code")
        if inject_keys is None:
            inject_keys = tuple(step_outputs.keys())
        for key in inject_keys:
            val = step_outputs.get(key)
            if isinstance(val, str) and val.strip():
                parts.append("")
                parts.append(f"## Output from prior '{key}' step")
                parts.append(val.strip())

    is_fix_step = step == "implement_fix"
    review_findings = task.metadata.get("review_findings") if isinstance(task.metadata, dict) else None

    # Insert a dedicated remediation payload section for implement_fix.
    has_review_findings = bool(review_findings and isinstance(review_findings, list))
    has_verify_failure = (
        isinstance(task.metadata, dict)
        and isinstance(task.metadata.get("verify_failure"), str)
        and task.metadata.get("verify_failure", "").strip()
    )
    has_verify_output = (
        isinstance(task.metadata, dict)
        and isinstance(task.metadata.get("verify_output"), str)
        and task.metadata.get("verify_output", "").strip()
    )
    has_verify_reason_code = (
        isinstance(task.metadata, dict)
        and isinstance(task.metadata.get("verify_reason_code"), str)
        and task.metadata.get("verify_reason_code", "").strip()
    )
    retry_guidance = task.metadata.get("retry_guidance") if isinstance(task.metadata, dict) else None
    has_previous_error = False
    if isinstance(retry_guidance, dict):
        prev_error = str(retry_guidance.get("previous_error") or "").strip()
        has_previous_error = bool(prev_error)

    if is_fix_step and (
        has_review_findings or has_verify_failure or has_verify_output or has_verify_reason_code or has_previous_error
    ):
        parts.append("")
        parts.append("## Issues to fix")

    # Include review findings for fix steps.
    if review_findings and isinstance(review_findings, list) and is_fix_step:
        parts.append("")
        parts.append("### Review findings to address")
        for finding in review_findings:
            if isinstance(finding, dict):
                sev = finding.get("severity", "medium")
                summary = finding.get("summary", "")
                file_ = finding.get("file", "")
                line_ = finding.get("line", "")
                loc = f" ({file_}:{line_})" if file_ else ""
                parts.append(f"  - [{sev}] {summary}{loc}")

    # Include review history for review and fix steps
    review_history = task.metadata.get("review_history") if isinstance(task.metadata, dict) else None
    if review_history and isinstance(review_history, list) and step in {"review", "implement_fix"}:
        parts.append("")
        parts.append("## Previous review cycle history")
        parts.append(
            "IMPORTANT: Do NOT contradict decisions from prior cycles. "
            "Do NOT re-raise findings that were already resolved. "
            "Do NOT undo fixes applied in response to earlier findings."
        )
        for cycle_entry in review_history:
            if not isinstance(cycle_entry, dict):
                continue
            attempt_num = cycle_entry.get("attempt", "?")
            decision = cycle_entry.get("decision", "?")
            parts.append(f"  Cycle {attempt_num} — decision: {decision}")
            for cf in cycle_entry.get("findings", []):
                if not isinstance(cf, dict):
                    continue
                sev = cf.get("severity", "?")
                summ = cf.get("summary", "")
                st = cf.get("status", "open")
                parts.append(f"    - [{sev}] {summ} (status: {st})")

    # Surface verify environment notes to the reviewer
    if category == "review" and isinstance(task.metadata, dict):
        env_note = task.metadata.get("verify_environment_note")
        if isinstance(env_note, str) and env_note.strip():
            parts.append("")
            parts.append("## Verification environment note")
            parts.append(
                "The verify step encountered failures caused by environment/infrastructure "
                "constraints, not code defects. The pipeline proceeded without fix attempts. "
                "Please assess whether this is acceptable for this task."
            )
            parts.append(f"  {env_note.strip()}")

    # Include plan context for task generation
    plan_for_generation = task.metadata.get("plan_for_generation") if isinstance(task.metadata, dict) else None
    if plan_for_generation and category == "task_generation" and task.task_type != "initiative_plan":
        parts.append("")
        parts.append("## Plan to decompose into subtasks")
        parts.append(str(plan_for_generation))
        parts.append("")
        parts.append(
            "Decompose this plan into ordered subtasks. Use the depends_on field "
            "(array of task IDs) to specify execution order where needed."
        )

    if step == "generate_tasks" and task.task_type in {"security", "security_audit"}:
        parts.append("")
        parts.append("## Remediation task generation focus")
        parts.append(
            "Generate concrete remediation tasks from security findings only. "
            "Each task should map to one or more specific findings with clear "
            "scope and expected risk reduction. Use depends_on IDs where order "
            "is required (e.g., dependency upgrade before follow-up code hardening)."
        )

    # Include context for iterative plan refinement.
    if category == "planning" and step in {"plan_refine", "initiative_plan_refine"} and isinstance(task.metadata, dict):
        refine_key = "initiative_plan_refine" if step == "initiative_plan_refine" else "plan_refine"
        base_plan = str(task.metadata.get(f"{refine_key}_base") or "").strip()
        feedback = str(task.metadata.get(f"{refine_key}_feedback") or "").strip()
        instructions = str(task.metadata.get(f"{refine_key}_instructions") or "").strip()
        if base_plan:
            parts.append("")
            parts.append("## Base initiative plan" if step == "initiative_plan_refine" else "## Base plan")
            parts.append(base_plan)
        if feedback:
            parts.append("")
            parts.append("## Feedback to address")
            parts.append(feedback)
        if instructions:
            parts.append("")
            parts.append("## Additional instructions")
            parts.append(instructions)
        parts.append("")
        if step == "initiative_plan_refine":
            parts.append("Return a full rewritten initiative plan in the `initiative_plan` field.")
        else:
            parts.append("Return a full rewritten plan in the `plan` field.")
    if step in {"plan", "plan_refine"}:
        parts.append("")
        parts.append(load_prompt("partials/plan_output_format.md"))

    # Include merge conflict context for resolve_merge step
    if category == "merge_resolution" and isinstance(task.metadata, dict):
        conflict_files = task.metadata.get("merge_conflict_files")
        if isinstance(conflict_files, dict):
            parts.append("")
            parts.append("Conflicted files (with <<<<<<< / ======= / >>>>>>> markers):")
            for fpath, content in conflict_files.items():
                parts.append(f"\n--- {fpath} ---")
                parts.append(content)

        other_tasks = task.metadata.get("merge_other_tasks")
        if isinstance(other_tasks, list) and other_tasks:
            parts.append("")
            parts.append("Other task(s) whose changes conflict with this task:")
            for info in other_tasks:
                parts.append(str(info))

        current_objective = task.metadata.get("merge_current_objective")
        if isinstance(current_objective, str) and current_objective.strip():
            parts.append("")
            parts.append("Current task objective context:")
            parts.append(current_objective.strip())

        other_objectives = task.metadata.get("merge_other_objectives")
        if isinstance(other_objectives, list) and other_objectives:
            parts.append("")
            parts.append("Other task objective context (must also be preserved):")
            for info in other_objectives:
                if isinstance(info, str) and info.strip():
                    parts.append(info.strip())

        parts.append("")
        parts.append("Edit the conflicted files to resolve all conflicts. "
                      "Ensure BOTH this task's and the other task(s)' objectives are preserved.")

    # Include verify failure context for fix steps.
    if isinstance(task.metadata, dict) and is_fix_step:
        verify_failure = task.metadata.get("verify_failure")
        if verify_failure:
            parts.append("")
            parts.append("### Verification failure")
            parts.append(f"  {verify_failure}")
        verify_reason_code = task.metadata.get("verify_reason_code")
        if isinstance(verify_reason_code, str) and verify_reason_code.strip():
            parts.append("")
            parts.append("### Verification reason code")
            parts.append(f"  {verify_reason_code.strip()}")
        verify_output = task.metadata.get("verify_output")
        if isinstance(verify_output, str) and verify_output.strip():
            parts.append("")
            parts.append("### Verification output (test/lint/typecheck logs)")
            parts.append(verify_output.strip())

    # Inject previous attempt error as a concrete issue to fix.
    if isinstance(task.metadata, dict) and is_fix_step:
        retry_guidance = task.metadata.get("retry_guidance")
        if isinstance(retry_guidance, dict):
            prev_error = str(retry_guidance.get("previous_error") or "").strip()
            if prev_error:
                parts.append("")
                parts.append("### Previous attempt error")
                parts.append(prev_error)

    # Inject guidance/adjustments from previous review or retry.
    if isinstance(task.metadata, dict):
        retry_guidance = task.metadata.get("retry_guidance")
        requested_changes = task.metadata.get("requested_changes")
        if is_fix_step and (
            (isinstance(retry_guidance, dict) and str(retry_guidance.get("guidance") or "").strip())
            or (isinstance(requested_changes, dict) and str(requested_changes.get("guidance") or "").strip())
        ):
            parts.append("")
            parts.append("## Requested adjustments")
        if isinstance(retry_guidance, dict) and is_fix_step:
            text = str(retry_guidance.get("guidance") or "").strip()
            if text:
                parts.append("")
                parts.append("### Feedback from previous attempt")
                parts.append(text)
        if isinstance(requested_changes, dict) and is_fix_step:
            text = str(requested_changes.get("guidance") or "").strip()
            if text:
                parts.append("")
                parts.append("### Requested changes from reviewer")
                parts.append(text)

    # Inject style guidelines and language defaults for implementation and review steps
    if category in ("implementation", "fix", "review"):
        parts.append("")
        parts.append(load_prompt("style.md"))
        if project_languages:
            for lang in project_languages:
                lang_block = _LANGUAGE_DEFAULTS.get(lang)
                if lang_block:
                    parts.append(lang_block)

    # Inject project commands for implementation and verification steps
    if project_commands and project_languages and category in ("implementation", "fix", "verification"):
        cmds_block = _format_project_commands(project_commands, project_languages)
        if cmds_block:
            parts.append("")
            parts.append(cmds_block)

    # Inject dependency policy instruction
    dep_policy = getattr(task, "dependency_policy", "prudent") or "prudent"
    dep_instruction = _DEPENDENCY_POLICY_INSTRUCTIONS.get(category, {}).get(dep_policy)
    if dep_instruction:
        parts.append("")
        parts.append(dep_instruction)

    parts.append("")
    parts.append(guardrails)

    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text, handling markdown fences."""
    text = text.strip()
    # Try stripping markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner_lines = []
        started = False
        for line in lines:
            if not started:
                if line.strip().startswith("```"):
                    started = True
                    continue
            elif line.strip() == "```":
                break
            else:
                inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_value(text: str) -> Any | None:
    """Try to extract a top-level JSON value (object or array) from text."""
    text = text.strip()
    if not text:
        return None

    # Try stripping markdown code fences first.
    if text.startswith("```"):
        lines = text.split("\n")
        inner_lines = []
        started = False
        for line in lines:
            if not started:
                if line.strip().startswith("```"):
                    started = True
                    continue
            elif line.strip() == "```":
                break
            else:
                inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    # If the entire remaining payload is JSON, parse directly.
    try:
        value = json.loads(text)
        if isinstance(value, (dict, list)):
            return value
    except json.JSONDecodeError:
        pass

    # Fallback object extraction.
    obj = _extract_json(text)
    if obj is not None:
        return obj

    # Fallback array extraction.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _normalize_review_findings(raw_findings: Any) -> list[dict[str, Any]]:
    """Normalize raw findings to a stable actionable schema."""
    if not isinstance(raw_findings, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue

        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue

        severity = str(item.get("severity") or "medium").strip().lower()
        if severity == "info":
            continue
        if severity not in _REVIEW_ALLOWED_SEVERITIES:
            severity = "medium"

        category = str(item.get("category") or "other").strip().lower().replace(" ", "_")
        if category not in _REVIEW_ALLOWED_CATEGORIES:
            category = "other"

        file_raw = item.get("file")
        file_path = str(file_raw).strip() if file_raw is not None else ""

        line_raw = item.get("line")
        line_num: int | None
        try:
            line_num = int(line_raw) if line_raw is not None else None
            if line_num is not None and line_num <= 0:
                line_num = None
        except (TypeError, ValueError):
            line_num = None

        suggested_fix = str(item.get("suggested_fix") or "").strip()

        status = str(item.get("status") or "open").strip().lower()
        if status in {"fixed", "closed", "done"}:
            status = "resolved"
        elif status not in {"open", "resolved"}:
            status = "open"

        normalized.append(
            {
                "severity": severity,
                "category": category,
                "summary": summary,
                "file": file_path,
                "line": line_num,
                "suggested_fix": suggested_fix,
                "status": status,
            }
        )
    return normalized


def _normalize_verify_reason_code(value: Any) -> str:
    reason = str(value or "").strip().lower()
    return reason if reason in _VERIFY_ALLOWED_REASON_CODES else "unknown"


def _format_verify_summary(summary: Any, reason_code: str) -> str | None:
    text = str(summary).strip() if summary is not None else ""
    if not text and reason_code == "unknown":
        return None
    if not text:
        return f"[reason_code={reason_code}]"
    if reason_code == "unknown":
        return text
    return f"{text} [reason_code={reason_code}]"


class LiveWorkerAdapter:
    """Worker adapter that dispatches to real Codex/Ollama providers."""

    def __init__(self, container: Container) -> None:
        self._container = container

    @staticmethod
    def _coerce_timeout(value: Any, default: int = _DEFAULT_STEP_TIMEOUT_SECONDS) -> int:
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            return default
        return timeout if timeout > 0 else default

    def _timeout_for_step(self, task: Task, step: str) -> int:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        overrides = metadata.get("step_timeouts")
        if isinstance(overrides, dict):
            for key in (step, _STEP_TIMEOUT_ALIASES.get(step)):
                if not key:
                    continue
                if key in overrides:
                    return self._coerce_timeout(overrides.get(key))

        try:
            template = PipelineRegistry().resolve_for_task_type(task.task_type)
        except Exception:
            return _DEFAULT_STEP_TIMEOUT_SECONDS

        step_timeouts = {sd.name: self._coerce_timeout(sd.timeout_seconds) for sd in template.steps}
        for key in (step, _STEP_TIMEOUT_ALIASES.get(step)):
            if key and key in step_timeouts:
                return step_timeouts[key]
        return _DEFAULT_STEP_TIMEOUT_SECONDS

    @staticmethod
    def _heartbeat_settings(cfg: dict[str, Any]) -> tuple[int, int]:
        workers_cfg = cfg.get("workers") if isinstance(cfg, dict) else {}
        workers_cfg = workers_cfg if isinstance(workers_cfg, dict) else {}
        heartbeat_seconds = LiveWorkerAdapter._coerce_timeout(
            workers_cfg.get("heartbeat_seconds"), _DEFAULT_HEARTBEAT_SECONDS
        )
        heartbeat_grace_seconds = LiveWorkerAdapter._coerce_timeout(
            workers_cfg.get("heartbeat_grace_seconds"), _DEFAULT_HEARTBEAT_GRACE_SECONDS
        )
        if heartbeat_grace_seconds < heartbeat_seconds:
            heartbeat_grace_seconds = heartbeat_seconds
        return heartbeat_seconds, heartbeat_grace_seconds

    @staticmethod
    def _human_blocker_summary(issues: list[dict[str, str]]) -> str:
        count = len(issues)
        first = issues[0].get("summary", "").strip() if issues else ""
        if not first:
            return f"Human intervention required ({count} blocking issue{'s' if count != 1 else ''})."
        suffix = "issue" if count == 1 else "issues"
        return f"Human intervention required ({count} {suffix}): {first}"

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Return run step."""
        return self._execute_step(task=task, step=step, attempt=attempt, persist=True)

    def run_step_ephemeral(self, *, task: Task, step: str, attempt: int) -> StepResult:
        """Like run_step but does not persist the task to storage.

        Use for synthetic/throwaway tasks (e.g. pipeline classification,
        dependency analysis) where the task object is only a prompt carrier.
        """
        return self._execute_step(task=task, step=step, attempt=attempt, persist=False)

    def _execute_step(self, *, task: Task, step: str, attempt: int, persist: bool) -> StepResult:
        # 1. Resolve worker
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, step)
            if spec.type in {"codex", "claude"}:
                task_model = str(getattr(task, "worker_model", "") or "").strip()
                if not task_model and isinstance(task.metadata, dict):
                    task_model = str(task.metadata.get("worker_model") or "").strip()
                default_model = str(getattr(runtime, "default_model", "") or "").strip() if spec.type == "codex" else ""
                effective_model = task_model or default_model or str(spec.model or "").strip()
                if effective_model and effective_model != (spec.model or ""):
                    spec = replace(spec, model=effective_model)
            available, reason = test_worker(spec)
            if not available:
                return StepResult(status="error", summary=f"Worker not available: {reason}")
        except (ValueError, KeyError) as exc:
            return StepResult(status="error", summary=f"Cannot resolve worker: {exc}")

        # 2. Build prompt
        worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
        project_dir = Path(worktree_path) if worktree_path else self._container.project_dir
        langs = detect_project_languages(project_dir)
        raw_commands = (cfg.get("project") or {}).get("commands") or {}
        project_commands = {
            lang: cmds for lang, cmds in raw_commands.items()
            if isinstance(cmds, dict)
        } or None
        # Overlay per-task project commands (language-level replace)
        task_pc = task.project_commands
        if isinstance(task_pc, dict) and task_pc:
            if project_commands is None:
                project_commands = {}
            for lang, cmds in task_pc.items():
                if isinstance(cmds, dict) and cmds:
                    project_commands[lang] = dict(cmds)
        # Fill in defaults for detected languages with no configured commands
        if langs:
            if project_commands is None:
                project_commands = {}
            for lang in langs:
                if lang not in project_commands:
                    defaults = _DEFAULT_PROJECT_COMMANDS.get(lang)
                    if defaults:
                        project_commands[lang] = dict(defaults)
            if not project_commands:
                project_commands = None
        # Resolve relative executable paths against the *original* project dir
        # so workers in git worktrees can find binaries in gitignored dirs.
        if project_commands:
            project_commands = _resolve_command_paths(
                project_commands, self._container.project_dir,
            )
        prompt = build_step_prompt(
            task=task, step=step, attempt=attempt,
            project_languages=langs or None,
            project_commands=project_commands,
        )

        # 3. Execute
        run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = run_dir / "progress.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        timeout_seconds = self._timeout_for_step(task, step)
        heartbeat_seconds, heartbeat_grace_seconds = self._heartbeat_settings(cfg)
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        log_meta = {
            "step": step,
            "run_dir": str(run_dir),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "progress_path": str(progress_path),
            "started_at": now_iso(),
        }
        task.metadata["active_logs"] = log_meta
        if persist:
            self._container.tasks.upsert(task)

        def _check_task_cancelled() -> bool:
            if not persist:
                return False
            fresh = self._container.tasks.get(task.id)
            return fresh is not None and fresh.status == "cancelled"

        try:
            result = run_worker(
                spec=spec,
                prompt=prompt,
                project_dir=project_dir,
                run_dir=run_dir,
                timeout_seconds=timeout_seconds,
                heartbeat_seconds=heartbeat_seconds,
                heartbeat_grace_seconds=heartbeat_grace_seconds,
                progress_path=progress_path,
                is_cancelled=_check_task_cancelled,
            )
        except WorkerCancelledError:
            raise  # Let the orchestrator handle cancellation
        except Exception as exc:
            return StepResult(status="error", summary=f"Worker execution failed: {exc}")
        finally:
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata.pop("active_logs", None)
            task.metadata["last_logs"] = {**log_meta, "finished_at": now_iso()}
            if persist:
                self._container.tasks.upsert(task)

        # 4. Map result
        step_result = self._map_result(result, spec, step, task)
        raw_json = _extract_json(result.response_text) if result.response_text else None
        raw_is_json = isinstance(raw_json, dict)

        # 5. For verification steps that fell through to default "ok"
        #    (no structured summary), run a lightweight LLM formatter to parse the
        #    freeform output into pass/fail.
        category = _step_category(step)
        if (
            category == "verification"
            and result.response_text
            and step_result.status == "ok"
            and not raw_is_json
        ):
            step_result = self._parse_verify_output(
                spec=spec,
                response_text=result.response_text,
                project_dir=project_dir,
                task=task,
            )

        # 6. For review steps that fell through with no findings,
        #    run a lightweight LLM formatter to extract structured findings.
        if (
            category == "review"
            and result.response_text
            and step_result.status == "ok"
            and step_result.findings is None
            and not raw_is_json
        ):
            step_result = self._parse_review_output(
                spec=spec,
                response_text=result.response_text,
                project_dir=project_dir,
            )

        # 7. For task generation steps that fell through with no
        #    generated_tasks, run a lightweight LLM formatter to extract them.
        if (
            category == "task_generation"
            and result.response_text
            and step_result.status == "ok"
            and step_result.generated_tasks is None
            and not raw_is_json
        ):
            step_result = self._parse_task_generation_output(
                spec=spec,
                response_text=result.response_text,
                project_dir=project_dir,
            )

        return step_result

    # ------------------------------------------------------------------
    # Verify-output formatter (codex / claude)
    # ------------------------------------------------------------------

    def _parse_verify_output(
        self,
        *,
        spec: Any,
        response_text: str,
        project_dir: Path,
        task: Task,
    ) -> StepResult:
        """Use a short LLM call to classify freeform verification output."""
        formatter_prompt = load_prompt("formatters/verify_output.md").format(
            output=response_text[:8000],
        )

        fmt_run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = fmt_run_dir / "progress.json"
        try:
            fmt_result = run_worker(
                spec=spec,
                prompt=formatter_prompt,
                project_dir=project_dir,
                run_dir=fmt_run_dir,
                timeout_seconds=60,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=60,
                progress_path=progress_path,
            )
        except Exception:
            logger.debug("Verify formatter call failed; returning error")
            return StepResult(status="error", summary="Verification output formatter failed")

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Verify formatter returned unparseable output")
            return StepResult(status="error", summary="Verification output formatter returned invalid JSON")

        status_val = str(parsed.get("status", "")).lower().strip()
        if status_val not in _VERIFY_ALLOWED_STATUSES:
            status_val = "fail"
        reason_code = _normalize_verify_reason_code(parsed.get("reason_code"))
        summary = _format_verify_summary(parsed.get("summary"), reason_code)
        if isinstance(task.metadata, dict):
            task.metadata["verify_reason_code"] = reason_code
        if status_val == "fail":
            return StepResult(status="error", summary=str(summary) if summary else response_text[:500])
        if status_val == "environment":
            note = str(summary) if summary else response_text[:500]
            if isinstance(task.metadata, dict):
                task.metadata["verify_environment_note"] = note
            return StepResult(status="ok", summary=note)
        return StepResult(status="ok", summary=str(summary) if summary else None)

    # ------------------------------------------------------------------
    # Review-output formatter (codex / claude)
    # ------------------------------------------------------------------

    def _parse_review_output(
        self,
        *,
        spec: Any,
        response_text: str,
        project_dir: Path,
    ) -> StepResult:
        """Use a short LLM call to extract structured findings from freeform review output."""
        formatter_prompt = load_prompt("formatters/review_output.md").format(
            output=response_text[:8000],
        )

        fmt_run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = fmt_run_dir / "progress.json"
        try:
            fmt_result = run_worker(
                spec=spec,
                prompt=formatter_prompt,
                project_dir=project_dir,
                run_dir=fmt_run_dir,
                timeout_seconds=60,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=60,
                progress_path=progress_path,
            )
        except Exception:
            logger.debug("Review formatter call failed; returning error")
            return StepResult(status="error", summary="Review output formatter failed")

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Review formatter returned unparseable output")
            return StepResult(status="error", summary="Review output formatter returned invalid JSON")

        findings = parsed.get("findings")
        if isinstance(findings, list):
            return StepResult(status="ok", findings=_normalize_review_findings(findings))

        return StepResult(status="error", summary="Review output formatter returned no findings field")

    # ------------------------------------------------------------------
    # Task-generation-output formatter (codex / claude)
    # ------------------------------------------------------------------

    def _parse_task_generation_output(
        self,
        *,
        spec: Any,
        response_text: str,
        project_dir: Path,
    ) -> StepResult:
        """Use a short LLM call to extract structured tasks from freeform generation output."""
        formatter_prompt = load_prompt("formatters/task_generation_output.md").format(
            output=response_text[:12000],
        )

        fmt_run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = fmt_run_dir / "progress.json"
        try:
            fmt_result = run_worker(
                spec=spec,
                prompt=formatter_prompt,
                project_dir=project_dir,
                run_dir=fmt_run_dir,
                timeout_seconds=90,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=60,
                progress_path=progress_path,
            )
        except Exception:
            logger.debug("Task generation formatter call failed; returning error")
            return StepResult(status="error", summary="Task generation output formatter failed")

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Task generation formatter returned unparseable output")
            return StepResult(status="error", summary="Task generation output formatter returned invalid JSON")

        tasks = (
            parsed.get("tasks")
            or parsed.get("subtasks")
            or parsed.get("items")
        )
        if isinstance(tasks, list):
            return StepResult(status="ok", generated_tasks=tasks)

        return StepResult(status="error", summary="Task generation output formatter returned no tasks list")

    def _map_result(self, result: WorkerRunResult, spec: Any, step: str, task: Task) -> StepResult:
        if result.human_blocking_issues:
            return StepResult(
                status="human_blocked",
                summary=self._human_blocker_summary(result.human_blocking_issues),
                human_blocking_issues=result.human_blocking_issues,
            )
        if result.no_heartbeat:
            summary = "Worker stalled (no heartbeat or output activity)."
            if result.stderr_path:
                try:
                    err_text = Path(result.stderr_path).read_text(errors="replace").strip()
                    if err_text:
                        tail = err_text[-500:]
                        summary = f"{summary}\n{tail}"
                except Exception:
                    logger.debug("Failed reading stderr for no-heartbeat worker result", exc_info=True)
            return StepResult(status="error", summary=summary)
        if result.timed_out:
            summary = "Worker timed out"
            if result.stderr_path:
                try:
                    err_text = Path(result.stderr_path).read_text(errors="replace").strip()
                    if err_text:
                        tail = err_text[-500:]
                        summary = f"{summary}\n{tail}"
                except Exception:
                    logger.debug("Failed reading stderr for timed-out worker result", exc_info=True)
            return StepResult(status="error", summary=summary)
        if result.exit_code != 0:
            summary = f"Worker exited with code {result.exit_code}"
            # Try to include stderr info
            if result.stderr_path:
                try:
                    err_text = Path(result.stderr_path).read_text(errors="replace").strip()
                    if err_text:
                        summary = err_text[:500]
                except Exception:
                    logger.debug("Failed reading stderr for non-zero worker exit", exc_info=True)
            return StepResult(status="error", summary=summary)

        # Dependency analysis: always parse response text.
        category = _step_category(step)
        if category == "dependency_analysis" and result.response_text:
            return self._parse_dep_analysis_output(result.response_text)

        if category == "verification" and result.response_text:
            parsed = _extract_json(result.response_text)
            if isinstance(parsed, dict):
                status_val = str(parsed.get("status", "")).lower().strip()
                if status_val not in _VERIFY_ALLOWED_STATUSES:
                    return StepResult(status="error", summary="Verification output JSON has invalid status")
                reason_code = _normalize_verify_reason_code(parsed.get("reason_code"))
                verify_summary = _format_verify_summary(parsed.get("summary"), reason_code)
                if isinstance(task.metadata, dict):
                    task.metadata["verify_reason_code"] = reason_code
                if status_val == "fail":
                    return StepResult(status="error", summary=str(verify_summary) if verify_summary else "Verification failed")
                if status_val == "environment":
                    note = str(verify_summary) if verify_summary else "Verification blocked by environment constraints"
                    if isinstance(task.metadata, dict):
                        task.metadata["verify_environment_note"] = note
                    return StepResult(status="ok", summary=note)
                return StepResult(status="ok", summary=str(verify_summary) if verify_summary else "")

        if category == "review" and result.response_text:
            parsed = _extract_json(result.response_text)
            if isinstance(parsed, dict):
                findings = parsed.get("findings")
                if isinstance(findings, list):
                    return StepResult(status="ok", findings=_normalize_review_findings(findings))
                return StepResult(status="error", summary="Review output JSON missing findings array")

        if category == "pipeline_classification" and result.response_text:
            parsed = _extract_json_value(result.response_text)
            if isinstance(parsed, dict):
                return StepResult(status="ok", summary=json.dumps(parsed))
            return StepResult(status="error", summary="Pipeline classification output must be valid JSON")

        if category == "planning" and result.response_text:
            parsed = _extract_json(result.response_text)
            if isinstance(parsed, dict):
                planning_summary: Any | None = parsed.get("analysis") if step == "analyze" else None
                planning_summary = planning_summary or (
                    parsed.get("initiative_plan") if step in {"initiative_plan", "initiative_plan_refine"} else None
                )
                planning_summary = planning_summary or parsed.get("plan") or parsed.get("summary")
                if planning_summary:
                    return StepResult(status="ok", summary=_normalize_planning_text(str(planning_summary))[:20000])
            return StepResult(status="ok", summary=_normalize_planning_text(result.response_text)[:20000])

        if category == "diagnosis" and result.response_text:
            parsed = _extract_json(result.response_text)
            if isinstance(parsed, dict):
                diagnosis_summary = parsed.get("diagnosis") or parsed.get("summary")
                if diagnosis_summary:
                    return StepResult(status="ok", summary=str(diagnosis_summary).strip()[:20000])
            return StepResult(status="ok", summary=result.response_text.strip()[:20000])

        if category == "fix" and result.response_text:
            return StepResult(status="ok", summary=result.response_text.strip()[:20000])

        if category == "task_generation" and result.response_text:
            parsed_value = _extract_json_value(result.response_text)
            if isinstance(parsed_value, list):
                return StepResult(status="ok", generated_tasks=parsed_value)
            if isinstance(parsed_value, dict):
                tasks = (
                    parsed_value.get("tasks")
                    or parsed_value.get("subtasks")
                    or parsed_value.get("items")
                )
                if isinstance(tasks, list):
                    return StepResult(status="ok", generated_tasks=tasks)

        # Parse remaining structured output for ollama-specific categories.
        if spec.type == "ollama" and result.response_text:
            return self._parse_ollama_output(result.response_text, step)

        return StepResult(status="ok")

    def _parse_dep_analysis_output(self, text: str) -> StepResult:
        parsed = _extract_json(text)
        if parsed is None:
            return StepResult(status="ok", dependency_edges=[])
        edges = parsed.get("edges")
        if isinstance(edges, list):
            return StepResult(status="ok", dependency_edges=edges)
        return StepResult(status="ok", dependency_edges=[])

    def _parse_ollama_output(self, text: str, step: str) -> StepResult:
        parsed = _extract_json(text)
        if parsed is None:
            return StepResult(status="ok", summary=text[:500] if text else None)

        category = _step_category(step)

        if category == "review" or category == "scanning":
            findings = parsed.get("findings")
            if isinstance(findings, list):
                if category == "review":
                    return StepResult(status="ok", findings=_normalize_review_findings(findings))
                return StepResult(status="ok", findings=findings)

        if category == "task_generation":
            tasks = parsed.get("tasks")
            if isinstance(tasks, list):
                return StepResult(status="ok", generated_tasks=tasks)

        if category == "verification":
            status = parsed.get("status", "ok")
            summary = parsed.get("summary")
            mapped_status = "ok" if status in ("ok", "pass") else "error"
            return StepResult(status=mapped_status, summary=summary)

        # For planning, implementation, reporting, diagnosis — extract summary
        summary = parsed.get("summary") or parsed.get("plan") or parsed.get("initiative_plan") or parsed.get("diagnosis")
        return StepResult(status="ok", summary=str(summary) if summary else None)

    # ------------------------------------------------------------------
    # Run summary generator
    # ------------------------------------------------------------------

    def _run_summarize_call(
        self,
        *,
        spec: Any,
        task: Task,
        run: RunRecord,
        project_dir: Path,
    ) -> str:
        """Use a short LLM call to produce a human-readable execution summary."""
        # Build step log section
        step_lines: list[str] = []
        for entry in run.steps:
            if not isinstance(entry, dict):
                continue
            name = entry.get("step", "?")
            status = entry.get("status", "?")
            line = f"- {name}: {status}"
            summary = entry.get("summary")
            if summary:
                line += f" — {str(summary)[:200]}"
            open_counts = entry.get("open_counts")
            if isinstance(open_counts, dict):
                line += f" (findings: {open_counts})"
            step_lines.append(line)

        # Get git diff stat
        diff_stat = ""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                diff_stat = result.stdout.strip()[:4000]
        except Exception:
            logger.debug("Failed collecting git diff stat for run summary", exc_info=True)

        description_section = f"Description: {task.description[:500]}\n" if task.description else ""
        execution_log = "\n".join(step_lines) if step_lines else "(no steps recorded)"
        diff_section = f"\n## Git diff stat\n{diff_stat}" if diff_stat else ""
        error_section = f"\n## Error: {task.error}" if task.error else ""
        workdoc_snapshot = _load_workdoc_snapshot(task, project_dir)
        run_status = str(run.status or "unknown")

        prompt = load_prompt("formatters/summarize.md").format(
            task_title=task.title,
            description_section=description_section,
            task_type=task.task_type,
            run_status=run_status,
            workdoc_snapshot=workdoc_snapshot,
            execution_log=execution_log,
            diff_section=diff_section,
            error_section=error_section,
        )

        fmt_run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = fmt_run_dir / "progress.json"
        try:
            fmt_result = run_worker(
                spec=spec,
                prompt=prompt,
                project_dir=project_dir,
                run_dir=fmt_run_dir,
                timeout_seconds=120,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=90,
                progress_path=progress_path,
            )
        except Exception:
            logger.debug("Summarize call failed; returning fallback")
            return "Summary generation failed"

        parsed = _extract_json(fmt_result.response_text or "")
        if isinstance(parsed, dict):
            summary = parsed.get("summary")
            if summary:
                return str(summary)

        # Fall back to raw text if JSON parsing failed
        raw = (fmt_result.response_text or "").strip()
        return raw[:2000] if raw else "Summary generation failed"

    def generate_run_summary(self, *, task: Task, run: RunRecord, project_dir: Path) -> str:
        """Produce a worker-generated summary for a completed run."""
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, "summarize")
            available, reason = test_worker(spec)
            if not available:
                logger.debug("Summary worker not available: %s", reason)
                return ""
        except Exception:
            logger.debug("Cannot resolve worker for summarize step")
            return ""

        return self._run_summarize_call(
            spec=spec,
            task=task,
            run=run,
            project_dir=project_dir,
        )
