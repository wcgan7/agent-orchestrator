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
from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...worker import WorkerCancelledError
from ...workers.run import WorkerRunResult, run_worker
from ..domain.models import RunRecord, Task, now_iso
from ..storage.container import Container
from .worker_adapter import StepResult

logger = logging.getLogger(__name__)

# Step category mapping
_PLANNING_STEPS = {"plan", "analyze", "plan_refine"}
_IMPL_STEPS = {"implement", "prototype"}
_FIX_STEPS = {"implement_fix"}
_VERIFY_STEPS = {"verify", "benchmark", "reproduce"}
_REVIEW_STEPS = {"review"}
_REPORT_STEPS = {"report", "summarize"}
_SCAN_STEPS = {"scan", "scan_deps", "scan_code", "gather"}
_TASK_GEN_STEPS = {"generate_tasks", "diagnose"}
_MERGE_RESOLVE_STEPS = {"resolve_merge"}
_DEP_ANALYSIS_STEPS = {"analyze_deps"}

# Which prior step outputs each category should receive in its prompt.
# None = inject all available outputs (for reporting/summarize steps).
_STEP_OUTPUT_INJECTION: dict[str, tuple[str, ...] | None] = {
    "implementation": ("plan", "analyze", "reproduce", "diagnose", "profile", "gather"),
    "review": ("verify", "benchmark"),
    "reporting": None,
    "task_generation": ("plan", "analyze", "scan", "scan_deps", "scan_code"),
    "fix": ("plan",),
    "scanning": ("scan_deps", "gather"),
}

_STEP_TIMEOUT_ALIASES = {"implement_fix": "implement"}
_DEFAULT_STEP_TIMEOUT_SECONDS = 600
_DEFAULT_HEARTBEAT_SECONDS = 60
_DEFAULT_HEARTBEAT_GRACE_SECONDS = 240

# ---------------------------------------------------------------------------
# Prompt layers
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "You are an autonomous coding agent managed by a coordinator process.\n"
    "The coordinator is the final authority on task state — it assigns steps,\n"
    "tracks progress, and handles all git commits.\n\n"
    "## Human-blocking issues\n"
    "If you encounter a problem that genuinely cannot be resolved without human\n"
    "intervention, report it as a human-blocking issue. Valid reasons:\n"
    "specification is missing or contradictory, required credentials or access\n"
    "are unavailable. Do NOT escalate code-quality concerns, design preferences,\n"
    "refactoring suggestions, or review feedback — handle those within your\n"
    "step output."
)

_GUARDRAILS = (
    "## Guardrails\n"
    "- Do NOT commit, push, or rebase — the coordinator handles all commits.\n"
    "- Do NOT modify files under `.agent_orchestrator/` — those are coordinator state.\n"
    "- Do NOT suppress or down-rank review findings.\n"
    "- Prefer fixing issues over escalating; escalate only when truly stuck.\n"
    "- Be explicit about risks, uncertainty, and assumptions."
)

_LANGUAGE_STANDARDS: dict[str, str] = {
    "python": (
        "## Language standards — Python\n"
        "- Google-style docstrings; module-level docstring in every file.\n"
        "- Type hints (Python 3.10+ syntax). Aim for mypy strict compliance.\n"
        "- Format with ruff; lint with ruff check."
    ),
    "typescript": (
        "## Language standards — TypeScript\n"
        "- JSDoc on exported symbols. Strict tsconfig (no `any`).\n"
        "- Compile-check with tsc --noEmit. Lint with ESLint."
    ),
    "javascript": (
        "## Language standards — JavaScript\n"
        "- JSDoc on exported symbols.\n"
        "- Lint with ESLint; format with Prettier."
    ),
    "go": (
        "## Language standards — Go\n"
        "- Godoc conventions on exported symbols.\n"
        "- Format with gofmt; lint with golangci-lint."
    ),
    "rust": (
        "## Language standards — Rust\n"
        "- `///` doc comments on public items.\n"
        "- Format with cargo fmt; lint with cargo clippy."
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
    if step in _MERGE_RESOLVE_STEPS:
        return "merge_resolution"
    if step in _DEP_ANALYSIS_STEPS:
        return "dependency_analysis"
    return "general"


_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "planning": (
        "Create a scoped, independently testable plan for the following task.\n"
        "Describe a coherent technical approach. Do not assume infrastructure or\n"
        "services that are not already present. Planning does not modify\n"
        "repository code."
    ),
    "implementation": (
        "Implement the changes described in the following task.\n"
        "Complete the entire step fully — partial work leaves the repository in\n"
        "an inconsistent state.\n"
        "IMPORTANT: If this change affects user-facing behavior, CLI usage,\n"
        "configuration, setup instructions, or API surface, you MUST update\n"
        "README.md and any relevant documentation files to reflect the changes.\n"
        "This is an explicit instruction, not a suggestion.\n"
        "If previous review cycle history is shown below, preserve all fixes\n"
        "applied in prior cycles. Do not revert or reimplement from scratch —\n"
        "make targeted, incremental fixes only."
    ),
    "fix": (
        "Fix the issues identified in the review findings and/or verification\n"
        "failures listed below. Do NOT reimplement the task from scratch.\n"
        "Make only the minimal, targeted changes needed to address each finding.\n"
        "Preserve all existing work that is correct — do not refactor, reorganize,\n"
        "or rewrite code that is not directly related to a finding.\n"
        "If review cycle history is shown, ensure your fixes do not regress issues\n"
        "resolved in prior cycles."
    ),
    "verification": (
        "Run the project's test, lint, and type-check commands for the following\n"
        "task. Do not bypass or skip tests. Report results accurately — do not\n"
        "mask failures. If you can identify the root cause of a failure, note it\n"
        "clearly so the next step can address it."
    ),
    "review": (
        "Review the implementation and list findings.\n"
        "Each finding must include a severity (critical / high / medium / low).\n"
        "Evaluate every acceptance criterion explicitly. Provide concrete\n"
        "evidence tied to files and diffs — do not speculate. Do not down-rank\n"
        "findings.\n"
        "SCOPE: Only flag issues introduced or directly affected by THIS task's\n"
        "changes. Pre-existing issues in unchanged code are out of scope.\n"
        "Missing project-wide tooling (e.g. linter configs not present before\n"
        "this task) is out of scope unless the task description requires it.\n"
        "CONSISTENCY: If previous review cycles are shown below, do not\n"
        "contradict earlier decisions. Do not reverse an approach you\n"
        "previously accepted unless new evidence of a critical defect emerged.\n"
        "Focus on remaining open issues.\n"
        "CONVERGENCE: Each cycle must move closer to approval. Do not raise\n"
        "new low-severity or cosmetic findings after the first cycle unless\n"
        "they concern code changed since the last review.\n"
        "If the change affects user-facing behavior, CLI usage, configuration,\n"
        "or API surface, verify that README.md and relevant documentation were\n"
        "updated. Raise a medium-severity finding if documentation is stale or\n"
        "missing."
    ),
    "reporting": (
        "Produce a summary report for the following task.\n"
        "Tie conclusions to concrete evidence. Be explicit about risks and\n"
        "remaining uncertainty."
    ),
    "scanning": (
        "Scan and gather information for the following task.\n"
        "Report findings with severity and file locations. Provide concrete\n"
        "evidence only."
    ),
    "task_generation": (
        "Generate subtasks for the following task.\n"
        "Each subtask must be independently implementable. Include title,\n"
        "description, task_type, and priority. Cover the full scope without\n"
        "overlap.\n"
        "If a plan is provided, decompose it into ordered subtasks and specify\n"
        "depends_on indices for tasks that must complete before others can start."
    ),
    "merge_resolution": "Resolve the merge conflicts in the following files. Both tasks' objectives must be fulfilled in the resolution.",
    "dependency_analysis": (
        "Analyze task dependencies for this codebase.\n\n"
        "First, examine the project structure to understand what already exists:\n"
        "- Look at the directory layout and key files\n"
        "- Check existing modules, APIs, and shared code\n"
        "- Identify what infrastructure is already in place\n\n"
        "Then, given the pending tasks below, determine which tasks depend on others.\n"
        "A task B depends on task A if:\n"
        "- B requires code, APIs, schemas, or artifacts that task A will CREATE (not already existing)\n"
        "- B imports or builds on modules that task A will introduce\n"
        "- B cannot produce correct results without task A's changes being present\n\n"
        "Do NOT create a dependency if:\n"
        "- Both tasks touch the same area but don't actually need each other's output\n"
        "- The dependency is based on vague thematic similarity\n"
        "- The required code/API already exists in the codebase\n\n"
        "If tasks can safely run in parallel, leave them independent."
    ),
    "general": "Follow the task description and report results clearly.",
}

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

_CATEGORY_JSON_SCHEMAS: dict[str, str] = {
    "planning": '{"plan": "string describing the plan"}',
    "implementation": '{"patch": "unified diff of changes", "summary": "description of changes"}',
    "verification": '{"status": "pass|fail", "summary": "test results summary"}',
    "review": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path", "line": 0, "suggested_fix": "string"}]}',
    "reporting": '{"summary": "detailed report text"}',
    "scanning": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path"}]}',
    "task_generation": '{"tasks": [{"title": "string", "description": "string", "task_type": "feature|bugfix|research", "priority": "P0|P1|P2|P3", "depends_on": [0, 1]}]}',
    "merge_resolution": '{"status": "ok|error", "summary": "string"}',
    "dependency_analysis": '{"edges": [{"from": "task_id_first", "to": "task_id_depends", "reason": "why"}]}',
    "general": '{"status": "ok|error", "summary": "string"}',
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


def build_step_prompt(
    *,
    task: Task,
    step: str,
    attempt: int,
    is_codex: bool,
    project_languages: list[str] | None = None,
    project_commands: dict[str, dict[str, str]] | None = None,
) -> str:
    """Build a prompt from Task fields with step-specific instructions."""
    category = _step_category(step)
    instruction = _CATEGORY_INSTRUCTIONS[category]

    # Special prompt for dependency analysis
    if category == "dependency_analysis" and isinstance(task.metadata, dict):
        parts = [_PREAMBLE, "", instruction, ""]

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
                    parts.append(f"  Labels: {', '.join(str(l) for l in labels)}")
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
        parts.append("- Use the exact task IDs from above.")
        parts.append("- If all tasks are independent, return an empty edges array.")
        parts.append("- Do not create circular dependencies.")

        parts.append("")
        parts.append(_GUARDRAILS)

        if not is_codex:
            schema = _CATEGORY_JSON_SCHEMAS["dependency_analysis"]
            parts.append("")
            parts.append(f"Respond with valid JSON matching this schema: {schema}")

        return "\n".join(parts)

    parts = [_PREAMBLE, "", instruction, ""]

    parts.append(f"Task: {task.title}")
    if task.description:
        parts.append(f"Description: {task.description}")
    parts.append(f"Type: {task.task_type}")
    parts.append(f"Priority: {task.priority}")
    parts.append(f"Step: {step}")
    if attempt > 1:
        parts.append(f"Attempt: {attempt}")

    # Inject outputs from prior pipeline steps.
    step_outputs = task.metadata.get("step_outputs") if isinstance(task.metadata, dict) else None
    if isinstance(step_outputs, dict) and step_outputs and category in _STEP_OUTPUT_INJECTION:
        inject_keys = _STEP_OUTPUT_INJECTION[category]
        if inject_keys is None:
            inject_keys = tuple(step_outputs.keys())
        for key in inject_keys:
            val = step_outputs.get(key)
            if isinstance(val, str) and val.strip():
                parts.append("")
                parts.append(f"## Output from prior '{key}' step")
                parts.append(val.strip())

    # Include review findings for fix steps
    review_findings = task.metadata.get("review_findings") if isinstance(task.metadata, dict) else None
    if review_findings and isinstance(review_findings, list):
        parts.append("")
        parts.append("Review findings to address:")
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
    if review_history and isinstance(review_history, list):
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
    if plan_for_generation and category == "task_generation":
        parts.append("")
        parts.append("## Plan to decompose into subtasks")
        parts.append(str(plan_for_generation))
        parts.append("")
        parts.append(
            "Decompose this plan into ordered subtasks. Use the depends_on field "
            "(array of task indices) to specify execution order where needed."
        )

    # Include context for iterative plan refinement.
    if category == "planning" and step == "plan_refine" and isinstance(task.metadata, dict):
        base_plan = str(task.metadata.get("plan_refine_base") or "").strip()
        feedback = str(task.metadata.get("plan_refine_feedback") or "").strip()
        instructions = str(task.metadata.get("plan_refine_instructions") or "").strip()
        if base_plan:
            parts.append("")
            parts.append("## Base plan")
            parts.append(base_plan)
        if feedback:
            parts.append("")
            parts.append("## Feedback to address")
            parts.append(feedback)
        if instructions:
            parts.append("")
            parts.append("## Additional instructions")
            parts.append(instructions)
        if not is_codex:
            parts.append("")
            parts.append("Return a full rewritten plan in the `plan` field.")
    if category == "planning":
        parts.append("")
        parts.append("## Planning output requirements")
        parts.append("- Return only the final plan body.")
        parts.append("- Do not include prefaces, tool logs, or follow-up questions.")
        parts.append("- For refinements, return the full rewritten plan, not just a change summary.")

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

        parts.append("")
        parts.append("Edit the conflicted files to resolve all conflicts. "
                      "Ensure BOTH this task's and the other task(s)' objectives are preserved.")

    # Include verify failure context for fix steps.
    if isinstance(task.metadata, dict) and category in ("implementation", "fix"):
        verify_failure = task.metadata.get("verify_failure")
        if verify_failure:
            parts.append("")
            parts.append("## Verification failure to fix")
            parts.append(f"  {verify_failure}")
        verify_output = task.metadata.get("verify_output")
        if isinstance(verify_output, str) and verify_output.strip():
            parts.append("")
            parts.append("## Verification output (test/lint/typecheck logs)")
            parts.append(verify_output.strip())

    # Inject human feedback from previous review or retry.
    if isinstance(task.metadata, dict):
        retry_guidance = task.metadata.get("retry_guidance")
        if isinstance(retry_guidance, dict):
            prev_error = str(retry_guidance.get("previous_error") or "").strip()
            if prev_error:
                parts.append("")
                parts.append("## Previous attempt error")
                parts.append(prev_error)
            text = str(retry_guidance.get("guidance") or "").strip()
            if text:
                parts.append("")
                parts.append("## Feedback from previous attempt")
                parts.append(text)
        requested_changes = task.metadata.get("requested_changes")
        if isinstance(requested_changes, dict):
            text = str(requested_changes.get("guidance") or "").strip()
            if text:
                parts.append("")
                parts.append("## Requested changes from reviewer")
                parts.append(text)

    # Inject language standards for implementation and review steps
    if project_languages and category in ("implementation", "fix", "review"):
        for lang in project_languages:
            lang_block = _LANGUAGE_STANDARDS.get(lang)
            if lang_block:
                parts.append("")
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
    parts.append(_GUARDRAILS)

    if not is_codex:
        # Add JSON schema instruction for ollama
        schema = _CATEGORY_JSON_SCHEMAS.get(category, _CATEGORY_JSON_SCHEMAS["general"])
        parts.append("")
        parts.append(f"Respond with valid JSON matching this schema: {schema}")

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
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


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
        prompt = build_step_prompt(
            task=task, step=step, attempt=attempt,
            is_codex=(spec.type in {"codex", "claude"}), project_languages=langs or None,
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
        self._container.tasks.upsert(task)

        def _check_task_cancelled() -> bool:
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
            self._container.tasks.upsert(task)

        # 4. Map result
        step_result = self._map_result(result, spec, step)

        # 5. For codex/claude verification steps that fell through to default "ok"
        #    (no structured summary), run a lightweight LLM formatter to parse the
        #    freeform output into pass/fail.
        category = _step_category(step)
        if (
            category == "verification"
            and spec.type in {"codex", "claude"}
            and result.response_text
            and step_result.status == "ok"
            and step_result.summary is None
        ):
            step_result = self._parse_verify_output(
                spec=spec,
                response_text=result.response_text,
                project_dir=project_dir,
                task=task,
            )

        # 6. For codex/claude review steps that fell through with no findings,
        #    run a lightweight LLM formatter to extract structured findings.
        if (
            category == "review"
            and spec.type in {"codex", "claude"}
            and result.response_text
            and step_result.status == "ok"
            and step_result.findings is None
        ):
            step_result = self._parse_review_output(
                spec=spec,
                response_text=result.response_text,
                project_dir=project_dir,
            )

        # 7. For codex/claude task generation steps that fell through with no
        #    generated_tasks, run a lightweight LLM formatter to extract them.
        if (
            category == "task_generation"
            and spec.type in {"codex", "claude"}
            and result.response_text
            and step_result.status == "ok"
            and step_result.generated_tasks is None
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
        formatter_prompt = (
            "You are a CI results classifier.\n\n"
            "Given the verification output below, respond with ONLY a JSON object:\n"
            '{"status": "pass|fail|skip|environment", "summary": "one-line summary of results"}\n\n'
            "Rules:\n"
            '- Use "fail" if a test, lint, or type-check command ran and '
            "produced failures (non-zero exit code from an actual check run).\n"
            '- Use "skip" if a command was not found, a tool is not installed, '
            "no test files exist, or a config file is missing (e.g. missing ESLint "
            "config, no pytest found). These are pre-existing environment gaps, "
            "not test failures.\n"
            '- Use "environment" if tests ran but failures are caused by OS, sandbox, '
            "permission, or infrastructure constraints — not by code logic. Examples: "
            "PermissionError on semaphores/sockets, resource limits, Docker/container "
            "restrictions, missing system libraries. These cannot be fixed by changing "
            "application code.\n"
            '- Use "pass" if every check that ran succeeded.\n'
            '- If some checks passed and others were skipped (tool not found), '
            'use "pass" — skipped checks are not failures.\n\n'
            "Verification output:\n"
            "---\n"
            f"{response_text[:8000]}\n"
            "---"
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
            logger.debug("Verify formatter call failed; returning default ok")
            return StepResult(status="ok", summary=response_text[:500])

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Verify formatter returned unparseable output")
            return StepResult(status="ok", summary=response_text[:500])

        status_val = str(parsed.get("status", "")).lower().strip()
        summary = parsed.get("summary")
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
        formatter_prompt = (
            "You are a code-review findings extractor.\n\n"
            "Given the review output below, respond with ONLY a JSON object:\n"
            '{"findings": [{"severity": "critical|high|medium|low|info", '
            '"category": "bug|security|performance|style|maintainability|other", '
            '"summary": "one-line description", '
            '"file": "path/to/file or empty string", '
            '"line": 0, '
            '"suggested_fix": "brief suggestion or empty string"}]}\n\n'
            "Rules:\n"
            "- Only include findings that represent actual issues requiring code changes.\n"
            "- EXCLUDE positive observations, praise, informational notes, and\n"
            "  intentional/documented design decisions that need no action.\n"
            "- If a finding is labeled 'positive', 'by design', or explicitly states\n"
            "  no change is needed, drop it.\n"
            "- Return an empty findings array if the review found no actionable issues.\n"
            "- Each finding must have at least severity, category, and summary.\n"
            "- Use the exact severity/category values listed above.\n\n"
            "Review output:\n"
            "---\n"
            f"{response_text[:8000]}\n"
            "---"
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
            logger.debug("Review formatter call failed; returning default ok")
            return StepResult(status="ok", summary=response_text[:500])

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Review formatter returned unparseable output")
            return StepResult(status="ok", summary=response_text[:500])

        findings = parsed.get("findings")
        if isinstance(findings, list):
            return StepResult(status="ok", findings=findings)

        return StepResult(status="ok", summary=response_text[:500])

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
        formatter_prompt = (
            "You are a task-list extractor.\n\n"
            "Given the task generation output below, respond with ONLY a JSON object:\n"
            '{"tasks": [{"title": "string", "description": "string", '
            '"task_type": "feature|bugfix|research|chore", '
            '"priority": "P0|P1|P2|P3", '
            '"depends_on": []}]}\n\n'
            "Rules:\n"
            "- Extract every distinct task/subtask mentioned in the output.\n"
            "- Each task must have at least title and description.\n"
            "- Use depends_on as zero-based indices into the tasks array to express ordering.\n"
            "- Return an empty tasks array only if the output truly contains no tasks.\n\n"
            "Task generation output:\n"
            "---\n"
            f"{response_text[:12000]}\n"
            "---"
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
            logger.debug("Task generation formatter call failed; returning default ok")
            return StepResult(status="ok", summary=response_text[:500])

        parsed = _extract_json(fmt_result.response_text or "")
        if not isinstance(parsed, dict):
            logger.debug("Task generation formatter returned unparseable output")
            return StepResult(status="ok", summary=response_text[:500])

        tasks = (
            parsed.get("tasks")
            or parsed.get("subtasks")
            or parsed.get("items")
        )
        if isinstance(tasks, list):
            return StepResult(status="ok", generated_tasks=tasks)

        return StepResult(status="ok", summary=response_text[:500])

    def _map_result(self, result: WorkerRunResult, spec: Any, step: str) -> StepResult:
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
                    pass
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
                    pass
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
                    pass
            return StepResult(status="error", summary=summary)

        # Dependency analysis: always parse response text (both codex and ollama)
        category = _step_category(step)
        if category == "dependency_analysis" and result.response_text:
            return self._parse_dep_analysis_output(result.response_text)

        if category == "planning" and result.response_text:
            parsed = _extract_json(result.response_text)
            if isinstance(parsed, dict):
                summary = parsed.get("plan") or parsed.get("summary")
                if summary:
                    return StepResult(status="ok", summary=_normalize_planning_text(str(summary))[:20000])
            return StepResult(status="ok", summary=_normalize_planning_text(result.response_text)[:20000])

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

        # Parse structured output for ollama
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

        # For planning, implementation, reporting — extract summary
        summary = parsed.get("summary") or parsed.get("plan")
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
            pass

        parts = [
            "You are a delivery summary writer.",
            "",
            f"Task: {task.title}",
        ]
        if task.description:
            parts.append(f"Description: {task.description[:500]}")
        parts.append(f"Type: {task.task_type}")
        parts.append("")
        parts.append("## Execution log")
        parts.extend(step_lines or ["(no steps recorded)"])

        if diff_stat:
            parts.append("")
            parts.append("## Git diff stat")
            parts.append(diff_stat)

        if task.error:
            parts.append("")
            parts.append(f"## Error: {task.error}")

        parts.append("")
        parts.append(
            "Given the execution log and diff stat above, produce a concise summary "
            "of what was implemented, what tests/checks passed or failed, and what "
            "requires human attention. You have access to the project files — read "
            "specific changed files if you need more detail about the implementation. "
            'Respond with ONLY a JSON object: {"summary": "markdown text"}'
        )

        prompt = "\n".join(parts)

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
