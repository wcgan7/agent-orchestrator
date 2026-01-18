"""Prompt builders for non-agentic workers (e.g., local models via Ollama).

These workers do not have filesystem or shell access. Prompts must request structured
outputs that the coordinator can apply (write artifacts or apply patches).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _agents_rules_block(agents_text: str) -> str:
    if not agents_text.strip():
        return ""
    return f"""
Repository rules (AGENTS.md):
{agents_text}
"""


def build_local_plan_prompt(
    *,
    prd_path: Path,
    prd_text: str,
    phase_plan_path: Path,
    task_queue_path: Path,
    repo_file_list: str,
    agents_text: str,
    user_prompt: Optional[str],
) -> str:
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    return f"""You are planning phases for a new feature.

You DO NOT have filesystem access. Do NOT claim to have edited files. Instead, return JSON only.

Inputs:
- PRD path: {prd_path}
- PRD content:
{prd_text}
{user_block}
{_agents_rules_block(agents_text)}
Repository file list (partial):
{repo_file_list}

Output (JSON only; no markdown fences, no extra text):
{{
  "phase_plan": {{
    "phases": [
      {{
        "id": "phase-1",
        "name": "Short phase name",
        "status": "todo",
        "description": "What this phase delivers",
        "acceptance_criteria": ["acceptance checks"],
        "branch": "feature/short-feature-name",
        "test_command": "optional command",
        "deps": ["optional phase ids this phase depends on"]
      }}
    ]
  }},
  "task_queue": {{
    "tasks": [
      {{
        "id": "phase-1",
        "type": "implement",
        "phase_id": "phase-1",
        "status": "todo",
        "lifecycle": "ready",
        "step": "plan_impl",
        "priority": 1,
        "deps": [],
        "description": "Short task description",
        "acceptance_criteria": ["..."],
        "test_command": "optional command",
        "branch": "feature/short-feature-name"
      }}
    ]
  }},
  "human_blocking_issues": [],
  "human_next_steps": []
}}

Notes:
- All phases should share the same branch name if you include 'branch'.
- The coordinator will write outputs to:
  - phase plan: {phase_plan_path}
  - task queue: {task_queue_path}
"""


def build_local_impl_plan_prompt(
    *,
    phase: dict[str, Any],
    prd_path: Path,
    prd_text: str,
    prd_markers: list[str],
    impl_plan_path: Path,
    user_prompt: Optional[str],
    agents_text: str,
    repo_file_list: str,
    test_command: Optional[str],
    plan_expansion_request: Optional[list[str]] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"
    test_block = test_command or "(none specified)"
    expansion_block = ""
    req = [str(x).strip() for x in (plan_expansion_request or []) if str(x).strip()]
    if req:
        requested = "\n".join(f"- {item}" for item in req)
        expansion_block = f"""
ALLOWLIST EXPANSION REQUIRED
The plan MUST cover ALL of these paths via files_to_change and/or new_files:
{requested}
"""

    phase_id = str(phase.get("id") or "phase-unknown")

    return f"""Produce an implementation plan for the phase below.

You DO NOT have filesystem access. Return JSON only; do not write files.

PRD path: {prd_path}
PRD content:
{prd_text}

PRD sections/IDs (cite in spec_summary):
{markers_block}

Phase: {phase.get("name") or phase_id}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}
Global/phase test command: {test_block}
{expansion_block}
{user_block}
{_agents_rules_block(agents_text)}
Repository file list (partial):
{repo_file_list}

Output (JSON only; no markdown fences, no extra text):
{{
  "impl_plan": {{
    "phase_id": "{phase_id}",
    "spec_summary": ["PRD requirements relevant to this phase (cite section headers/IDs)"],
    "files_to_change": ["path/a.py"],
    "new_files": ["path/new_file.py"],
    "technical_approach": ["1. ...", "2. ..."],
    "design_notes": {{
      "architecture": ["..."],
      "data_flow": ["..."],
      "invariants": ["..."],
      "edge_cases": ["..."]
    }},
    "test_plan": {{
      "commands": ["pytest -q"],
      "new_tests": ["tests/test_x.py::test_y"],
      "manual_checks": ["..."]
    }},
    "migration_or_rollout": ["(none)"],
    "open_questions": [],
    "assumptions": [],
    "plan_deviations": []
  }},
  "human_blocking_issues": [],
  "human_next_steps": []
}}

The coordinator will write the impl plan JSON to: {impl_plan_path}
"""


def build_local_review_prompt(
    *,
    phase: dict[str, Any],
    review_path: Path,
    prd_path: Path,
    prd_text: str,
    prd_markers: list[str],
    user_prompt: Optional[str],
    agents_text: str,
    changed_files: list[str],
    diff_text: str,
    diff_stat: str,
    status_text: str,
    impl_plan_text: str,
    tests_snapshot: Optional[dict[str, Any]],
    simple_review: bool,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    tests_block = json.dumps(tests_snapshot or {}, indent=2, sort_keys=True)

    schema_hint = (
        """{
  "mergeable": true,
  "phase_id": "phase-1",
  "issues": [
    {
      "severity": "low|medium|high|critical",
      "summary": "short summary",
      "rationale": "why this matters",
      "files": ["path"],
      "suggested_fix": "specific fix guidance"
    }
  ],
  "files_reviewed": ["..."],
  "evidence": ["..."]
}"""
        if not simple_review
        else """{
  "mergeable": true,
  "issues": [{"severity":"low|medium|high","text":"..."}]
}"""
    )

    return f"""Review the phase implementation and determine whether it's mergeable.

You DO NOT have filesystem access. Return JSON only; do not write files.

PRD path: {prd_path}
PRD content:
{prd_text}

PRD sections/IDs:
{markers_block}

Phase: {phase.get("name") or phase.get("id")}
Acceptance criteria:
{acceptance_block}
{user_block}
{_agents_rules_block(agents_text)}

Implementation plan (JSON excerpt):
{impl_plan_text}

Changed files:
{json.dumps(changed_files, indent=2)}

Diff stat:
{diff_stat}

Git status:
{status_text}

Diff:
{diff_text}

Last verification snapshot:
{tests_block}

Output (JSON only; no markdown fences, no extra text):
{{
  "review": {schema_hint},
  "human_blocking_issues": [],
  "human_next_steps": []
}}

The coordinator will write the review JSON to: {review_path}
"""


def build_local_implement_prompt(
    *,
    prd_path: Path,
    phase: dict[str, Any],
    task: dict[str, Any],
    impl_plan_path: Path,
    impl_plan_text: str,
    allowed_files: list[str],
    agents_text: str,
    repo_context_files: dict[str, str],
    user_prompt: Optional[str],
) -> str:
    phase_name = phase.get("name") or phase.get("id")
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none provided)"
    context_items = task.get("context", []) or []
    context_block = "\n".join(f"- {item}" for item in context_items) if context_items else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    allowed_block = "\n".join(f"- {path}" for path in allowed_files) if allowed_files else "- (none)"

    files_block = ""
    for path, content in repo_context_files.items():
        files_block += f"\n--- FILE: {path} ---\n{content}\n"

    return f"""Implement the COMPLETE phase described below by producing a unified diff patch.

You DO NOT have filesystem access. Do NOT describe changes without providing a patch.

PRD: {prd_path}
Phase: {phase_name}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}

Additional context from previous runs:
{context_block}
{user_block}
{_agents_rules_block(agents_text)}

Implementation plan file path: {impl_plan_path}
Implementation plan content:
{impl_plan_text}

Allowed files to change (patch must touch only these paths):
{allowed_block}

Repository file contents (for files you may need to edit):
{files_block}

Output (JSON only; no markdown fences, no extra text):
{{
  "patch": "diff --git a/path b/path\\n... (unified diff; include new files as needed)",
  "human_blocking_issues": [],
  "human_next_steps": [],
  "notes": "optional short note"
}}

Rules:
- Patch must apply cleanly with `git apply`.
- Do not include changes outside the allowed files list.
"""

