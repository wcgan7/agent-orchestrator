"""Build the text prompts passed to the Codex worker for each runner step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .constants import REVIEW_MIN_EVIDENCE_ITEMS
from .utils import _normalize_text


def _build_minimal_review_fix_prompt(
    *,
    review_blockers: list[str],
    review_blocker_files: list[str],
    allowed_files: Optional[list[str]] = None,
    events_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    heartbeat_seconds: Optional[int] = None,
) -> str:
    """Build a minimal prompt for addressing review blockers without full PRD context.

    This prompt is focused and efficient - it only includes information needed
    to address the specific review issues. It does NOT include PRD, acceptance
    criteria, or full implementation context.

    Args:
        review_blockers: List of blocking issues from the review.
        review_blocker_files: List of files implicated in the blockers.
        allowed_files: List of files the agent is allowed to edit.
        events_path: Path to events file for progress reporting.
        progress_path: Path to progress file for progress reporting.
        run_id: Current run identifier.
        heartbeat_seconds: Heartbeat interval for progress reporting.

    Returns:
        A minimal, focused prompt string for addressing review blockers.
    """
    # Build blockers list
    blockers_block = ""
    if review_blockers:
        blockers_list = "\n".join(f"- {blocker}" for blocker in review_blockers)
        blockers_block = f"""
Blocking issues to address:
{blockers_list}
"""

    # Build implicated files list
    files_block = ""
    if review_blocker_files:
        files_list = "\n".join(f"- {f}" for f in review_blocker_files)
        files_block = f"""
Files implicated in the blockers:
{files_list}
"""

    # Build allowed files section
    allowed_files = list(allowed_files) if allowed_files else []
    allowed_block = ""
    if allowed_files:
        allowed_list = "\n".join(f"- {path}" for path in allowed_files[:30])
        if len(allowed_files) > 30:
            allowed_list += f"\n- ... and {len(allowed_files) - 30} more files"
        allowed_block = f"""
Allowed files to edit:
{allowed_list}
"""

    # Progress contract
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = f"""
Progress contract (REQUIRED):
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, human_blocking_issues, human_next_steps, heartbeat.
{heartbeat_block}"""

    events_block = ""
    if events_path:
        events_block = f"- Append events to: {events_path}\n"

    return f"""ADDRESS REVIEW BLOCKERS

The previous implementation review found issues that must be addressed before merging.
{blockers_block}{files_block}
Instructions:
1. Fix ONLY the blocking issues listed above
2. Make minimal changes to address the review feedback
3. Do NOT add new features or refactor unrelated code
4. Focus on the specific files mentioned in the blockers
{allowed_block}
{events_block}{progress_block}"""


def _build_minimal_fix_prompt(
    *,
    prompt_mode: str,
    last_verification: dict[str, Any],
    allowed_files: Optional[list[str]] = None,
    events_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    heartbeat_seconds: Optional[int] = None,
) -> str:
    """Build a minimal prompt for fixing simple verification failures.

    This prompt is focused and efficient - it only includes information needed
    to fix the specific verification issue (format, lint, or typecheck).
    It does NOT include PRD, acceptance criteria, or implementation context.

    Args:
        prompt_mode: The prompt mode (fix_format, fix_lint, fix_typecheck).
        last_verification: Verification snapshot with command, error, failing paths.
        allowed_files: List of files the agent is allowed to edit.
        events_path: Path to events file for progress reporting.
        progress_path: Path to progress file for progress reporting.
        run_id: Current run identifier.
        heartbeat_seconds: Heartbeat interval for progress reporting.

    Returns:
        A minimal, focused prompt string for fixing the verification issue.
    """
    snapshot = last_verification or {}
    cmd = str(snapshot.get("command") or "").strip()
    log_path = str(snapshot.get("log_path") or "").strip()
    log_tail = str(snapshot.get("log_tail") or "").strip()
    exit_code = snapshot.get("exit_code")
    error_type = str(snapshot.get("error_type") or "").strip()
    stage = str(snapshot.get("stage") or "").strip()
    all_failing_paths = list(snapshot.get("all_failing_paths") or [])

    # Determine header and instructions based on prompt mode
    if prompt_mode == "fix_format":
        header = "FORMAT CHECK FAILING"
        instructions = """Instructions:
1. Fix ONLY the formatting issues listed above
2. Do NOT modify code logic or add features
3. Make minimal changes to satisfy the formatter
4. If the formatter has an auto-fix option, you may run it"""
    elif prompt_mode == "fix_lint":
        header = "LINT CHECK FAILING"
        instructions = """Instructions:
1. Fix ONLY the lint errors listed above
2. Do NOT modify code logic or add features
3. Make minimal changes to satisfy the linter
4. Common fixes: remove unused imports, fix line length, add missing type hints"""
    elif prompt_mode == "fix_typecheck":
        header = "TYPE CHECK FAILING"
        instructions = """Instructions:
1. Fix ONLY the type errors listed above
2. Do NOT modify code logic or add features
3. Make minimal changes to fix type annotations
4. Common fixes: add type hints, fix incompatible types, handle Optional properly"""
    else:
        # Fallback for unknown modes
        header = "VERIFICATION FAILING"
        instructions = """Instructions:
1. Fix the verification errors listed above
2. Make minimal changes to pass verification"""

    # Build failing files section
    failing_files_block = ""
    if all_failing_paths:
        failing_files_list = "\n".join(f"- {path}" for path in all_failing_paths)
        failing_files_block = f"""
Files with errors:
{failing_files_list}
"""

    # Build allowed files section (only include failing files that are allowed)
    allowed_files = list(allowed_files) if allowed_files else []
    if all_failing_paths and allowed_files:
        # Filter to only show the intersection
        relevant_allowed = [f for f in all_failing_paths if f in allowed_files or any(f.startswith(a.rstrip('/') + '/') for a in allowed_files if a.endswith('/'))]
        if not relevant_allowed:
            relevant_allowed = all_failing_paths  # Fall back to all failing paths
    else:
        relevant_allowed = all_failing_paths if all_failing_paths else allowed_files

    allowed_block = ""
    if relevant_allowed:
        allowed_list = "\n".join(f"- {path}" for path in relevant_allowed)
        allowed_block = f"""
Allowed files to edit:
{allowed_list}
"""

    # Progress contract (optional but recommended)
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = f"""
Progress contract (REQUIRED):
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, human_blocking_issues, human_next_steps, heartbeat.
{heartbeat_block}"""

    events_block = ""
    if events_path:
        events_block = f"- Append events to: {events_path}\n"

    return f"""{header}

Command: {cmd or "(unknown)"}
Exit code: {exit_code if exit_code is not None else "(unknown)"}
Log: {log_path or "(unknown)"}
{failing_files_block}
Error details:
{log_tail or "(no output captured)"}

{instructions}
{allowed_block}
{events_block}{progress_block}"""


def _build_resume_prompt(
    user_prompt: str,
    progress_path: Path,
    run_id: str,
    heartbeat_seconds: Optional[int] = None,
) -> str:
    """Build a standalone resume prompt that the agent must complete successfully."""
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
    return f"""You have been given the following instructions to complete:

{user_prompt}

Follow these instructions carefully. When you are done:
1. If you successfully completed the instructions, write a progress file indicating success.
2. If you cannot complete the instructions (e.g., need clarification, blocked by an issue),
   write a progress file with human_blocking_issues explaining what is blocking you.

Progress contract (REQUIRED):
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id="resume_prompt", phase="resume_prompt",
  actions (list of actions you took), claims (what you accomplished),
  next_steps, human_blocking_issues (if blocked), human_next_steps, heartbeat.
{heartbeat_block}
IMPORTANT: If you successfully complete the instructions, human_blocking_issues MUST be empty.
If you are blocked and cannot complete the instructions, human_blocking_issues MUST contain the blocking reason(s).
"""


def _build_plan_prompt(
    prd_path: Path,
    phase_plan_path: Path,
    task_queue_path: Path,
    events_path: Path,
    progress_path: Path,
    run_id: str,
    user_prompt: Optional[str],
    heartbeat_seconds: Optional[int] = None,
) -> str:
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
    return f"""Your task is to plan phases for a new feature.

Follow all repository rules in AGENTS.md.

Inputs:
- PRD: {prd_path}
- Repository: current working directory
{user_block}

Output files (write/update):
- Phase plan: {phase_plan_path}
  Schema:
  {{
    "updated_at": "ISO-8601",
    "phases": [
      {{
        "id": "phase-1",
        "name": "Short phase name",
        "status": "todo",
        "description": "What this phase delivers",
        "acceptance_criteria": ["list of acceptance checks"],
        "branch": "feature/short-feature-name",
        "test_command": "optional command for this phase"
      }}
    ]
  }}
  Branch naming:
  - Use a single shared branch for the whole PRD run (same value for every phase if provided).
  - Do NOT include the phase id (e.g. avoid `feature/phase-1-*`).
- Task queue: {task_queue_path}
  Include one task per phase with:
  id, type="implement", phase_id, status, priority, deps, description,
  acceptance_criteria, test_command, branch.

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, human_blocking_issues, human_next_steps, heartbeat.
{heartbeat_block}"""


def _build_phase_prompt(
    prd_path: Path,
    phase: dict[str, Any],
    task: dict[str, Any],
    events_path: Path,
    progress_path: Path,
    run_id: str,
    user_prompt: Optional[str],
    impl_plan_path: Optional[Path] = None,
    allowed_files: Optional[list[str]] = None,
    no_progress_attempts: int = 0,
    technical_approach_text: str = "",
    heartbeat_seconds: Optional[int] = None,
    prompt_mode: Optional[str] = None,
    last_verification: Optional[dict[str, Any]] = None,
    review_blockers: Optional[list[str]] = None,
    review_blocker_files: Optional[list[str]] = None,
) -> str:
    phase_name = phase.get("name") or phase.get("id")
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = (
        "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none provided)"
    )
    context_items = task.get("context", []) or []
    context_block = "\n".join(f"- {item}" for item in context_items) if context_items else "- (none)"

    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    plan_path_display = impl_plan_path or "(missing)"

    allowed_files = list(allowed_files) if allowed_files else []
    if "README.md" not in allowed_files:
        allowed_files.append("README.md")
    allowed_block = "\n".join(f"- {path}" for path in allowed_files) if allowed_files else "- (none)"

    no_progress_block = ""
    if no_progress_attempts > 0:
        no_progress_block = (
            "\nNOTE: Previous run made no code changes. You MUST edit files "
            "to implement the requirements.\n"
        )

    approach_block = ""
    if technical_approach_text:
        approach_block = f"\nTechnical Approach (from Plan):\n{technical_approach_text}\n"
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"

    banner_block = ""
    if prompt_mode in {"fix_tests", "fix_verify", "fix_format", "fix_lint", "fix_typecheck"}:
        snapshot = last_verification or {}
        cmd = str(snapshot.get("command") or "").strip()
        log_path = str(snapshot.get("log_path") or "").strip()
        log_tail = str(snapshot.get("log_tail") or "").strip()
        exit_code = snapshot.get("exit_code")
        all_failing_paths = list(snapshot.get("all_failing_paths") or [])

        # Stage-specific headers and instructions
        if prompt_mode == "fix_format":
            header = "FORMAT CHECK FAILING -- FIX THIS FIRST"
            priority_instruction = """Priority for this run:
1) Fix ONLY the formatting issues (minimal change).
2) Do NOT modify code logic - only fix formatting.
3) Only then continue implementing remaining acceptance criteria if needed."""
        elif prompt_mode == "fix_lint":
            header = "LINT CHECK FAILING -- FIX THIS FIRST"
            priority_instruction = """Priority for this run:
1) Fix ONLY the lint errors in the listed files (minimal change).
2) Do NOT modify code logic - only fix lint issues.
3) Only then continue implementing remaining acceptance criteria if needed."""
        elif prompt_mode == "fix_typecheck":
            header = "TYPE CHECK FAILING -- FIX THIS FIRST"
            priority_instruction = """Priority for this run:
1) Fix ONLY the type errors in the listed files (minimal change).
2) Do NOT modify code logic - only fix type annotations.
3) Only then continue implementing remaining acceptance criteria if needed."""
        elif prompt_mode == "fix_tests":
            header = "TESTS ARE FAILING -- FIX THIS FIRST"
            priority_instruction = """Priority for this run:
1) Fix the failing tests (this may require code logic changes).
2) Only then continue implementing remaining acceptance criteria."""
        else:
            header = "VERIFY IS FAILING -- FIX THIS FIRST"
            priority_instruction = """Priority for this run:
1) Fix the failing verification (minimal change).
2) Only then continue implementing remaining acceptance criteria."""

        # Include failing files if available
        failing_files_block = ""
        if all_failing_paths:
            failing_files_list = "\n".join(f"  - {path}" for path in all_failing_paths[:20])  # Limit to 20 files
            if len(all_failing_paths) > 20:
                failing_files_list += f"\n  ... and {len(all_failing_paths) - 20} more files"
            failing_files_block = f"""
Files with errors:
{failing_files_list}
"""

        banner_block = f"""
{header}
Command: {cmd or "(unknown)"}
Exit code: {exit_code if exit_code is not None else "(unknown)"}
Log: {log_path or "(unknown)"}
{failing_files_block}
Recent output:
{log_tail or "(no log output captured)"}

{priority_instruction}
"""
    elif prompt_mode == "address_review":
        blockers = review_blockers or []
        files = review_blocker_files or []
        blockers_block = "\n".join(f"- {item}" for item in blockers) if blockers else "- (none listed)"
        files_block = "\n".join(f"- {item}" for item in files) if files else "- (none listed)"
        banner_block = f"""
REVIEW BLOCKERS -- ADDRESS THESE FIRST
Blocking issues:
{blockers_block}

Files implicated:
{files_block}

Priority for this run:
1) Address the review blockers.
2) Re-check acceptance criteria after fixes.
"""
    else:
        test_failure = task.get("test_failure")
        if isinstance(test_failure, dict):
            cmd = str(test_failure.get("command") or "").strip()
            log_path = str(test_failure.get("log_path") or "").strip()
            log_tail = str(test_failure.get("log_tail") or "").strip()
            attempt = test_failure.get("attempt")
            max_attempts = test_failure.get("max_attempts")

            attempt_str = ""
            if isinstance(attempt, int) and isinstance(max_attempts, int) and max_attempts > 0:
                attempt_str = f"(attempt {attempt}/{max_attempts})"

            banner_block = f"""
TESTS ARE FAILING {attempt_str} -- FIX THIS FIRST
Command: {cmd or "(unknown)"}
Log: {log_path or "(unknown)"}
Recent output:
{log_tail or "(no log output captured)"}

Priority for this run:
1) Fix the failing tests (minimal change).
2) Only then continue implementing remaining acceptance criteria.
"""

    return f"""Implement the COMPLETE phase described below.

Follow all repository rules in AGENTS.md.

{banner_block}

PRD: {prd_path}
Phase: {phase_name}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}

Additional context from previous runs:
{context_block}
{user_block}

Implementation plan file: {plan_path_display}
{approach_block}

Rules:
- Work on the ENTIRE phase scope. Implement all necessary changes.
- Allowed files to read/edit:
{allowed_block}

Runner artifact files (ALWAYS allowed to edit for progress reporting; not part of the phase allowlist):
- {events_path}
- {progress_path}
{no_progress_block}

Do not run the full test suite. The coordinator runs tests. If you run any tests, run only a fast, targeted subset (≤60s).
If tests are failing (see banner/context), prioritize fixing them before any additional feature work.

Progress contract (REQUIRED):
- Append events to: {events_path}
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims, next_steps, human_blocking_issues, human_next_steps, heartbeat.
{heartbeat_block}"""


def _extract_prd_markers(prd_text: str, max_items: int = 20) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for line in prd_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip()
            normalized = _normalize_text(header)
            if header and normalized not in seen:
                markers.append(header)
                seen.add(normalized)
                if len(markers) >= max_items:
                    return markers
    return markers


def _build_minimal_allowlist_expansion_prompt(
    *,
    phase: dict[str, Any],
    impl_plan_path: Path,
    current_allowlist: list[str],
    expansion_paths: list[str],
    error_type: Optional[str] = None,
    log_tail: Optional[str] = None,
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    heartbeat_seconds: Optional[int] = None,
) -> str:
    """Build a minimal prompt for allowlist expansion without full PRD context.

    This prompt is focused and efficient - it only asks the agent to update the
    implementation plan to include files that failed verification. It does NOT
    include the full PRD content, which significantly reduces token usage.

    Args:
        phase: Phase metadata (id, name, description).
        impl_plan_path: Path where the updated plan should be written.
        current_allowlist: Files currently in the plan's allowlist.
        expansion_paths: Files that need to be added to the allowlist.
        error_type: Type of verification error (e.g., "typecheck_failed").
        log_tail: Recent verification output for context.
        progress_path: Path to progress file for progress reporting.
        run_id: Current run identifier.
        heartbeat_seconds: Heartbeat interval for progress reporting.

    Returns:
        A minimal, focused prompt string for allowlist expansion.
    """
    # Determine the failure type for context
    failure_context = ""
    if error_type:
        if "format" in error_type.lower():
            failure_context = "These files have **formatting errors** that need to be fixed."
        elif "lint" in error_type.lower():
            failure_context = "These files have **lint errors** that need to be fixed."
        elif "typecheck" in error_type.lower():
            failure_context = "These files have **type check errors** that need to be fixed."
        elif "test" in error_type.lower():
            failure_context = "These files are involved in **test failures** that need to be fixed."
        else:
            failure_context = f"These files failed verification ({error_type})."
    else:
        failure_context = "These files failed verification and need to be included in the plan."

    # Build the paths lists
    current_list = "\n".join(f"- {p}" for p in sorted(current_allowlist)[:50])
    if len(current_allowlist) > 50:
        current_list += f"\n- ... and {len(current_allowlist) - 50} more files"

    expansion_list = "\n".join(f"- {p}" for p in expansion_paths)

    # Log tail excerpt for context
    log_block = ""
    if log_tail:
        # Truncate to last 2000 chars for efficiency
        tail = log_tail[-2000:] if len(log_tail) > 2000 else log_tail
        log_block = f"""
Recent verification output (for context):
```
{tail}
```
"""

    # Progress contract
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = f"""
Progress contract (REQUIRED):
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id, phase, actions, claims,
  next_steps, human_blocking_issues, human_next_steps, heartbeat.
{heartbeat_block}"""

    return f"""UPDATE IMPLEMENTATION PLAN: Allowlist Expansion Required

Phase: {phase.get("name") or phase.get("id")}

Verification found failing files outside the current plan allowlist.
{failure_context}

🚨 FILES THAT MUST BE ADDED TO THE PLAN:
{expansion_list}

Current allowlist (files already in plan):
{current_list}
{log_block}
Your task:
1. Read the existing implementation plan at: {impl_plan_path}
2. Add ALL the files listed above to `files_to_change` (or `new_files` if they don't exist yet)
3. Write the updated plan back to the same file

Rules:
- You MUST include ALL the files listed above in your updated plan
- Do NOT remove any files that are already in the plan
- Do NOT add unrelated files beyond what is requested
- Keep the rest of the plan (spec_summary, technical_approach, etc.) unchanged
- The plan schema must remain valid JSON
{progress_block}
Output file (write JSON): {impl_plan_path}
"""


def _build_impl_plan_prompt(
    phase: dict[str, Any],
    prd_path: Path,
    prd_text: str,
    prd_truncated: bool,
    prd_markers: Optional[list[str]],
    impl_plan_path: Path,
    user_prompt: Optional[str],
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    test_command: Optional[str] = None,
    heartbeat_seconds: Optional[int] = None,
    plan_expansion_request: Optional[list[str]] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    prd_notice = ""
    if prd_truncated:
        prd_notice = (
            "\nNOTE: PRD content truncated. Open the PRD file to read the full spec.\n"
        )
    prd_markers = prd_markers or []
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, human_blocking_issues, human_next_steps, heartbeat.\n"
            f"{heartbeat_block}"
        )
    test_block = test_command or "(none specified)"
    expansion_block = ""
    req = [str(x).strip() for x in (plan_expansion_request or []) if str(x).strip()]
    if req:
        requested = "\n".join(f"- {item}" for item in req)
        expansion_block = f"""
🚨 ALLOWLIST EXPANSION REQUIRED

Verification found failing repo files outside the current plan allowlist.
Your plan MUST cover ALL of the following paths by including them in:
- files_to_change and/or new_files (preferred: exact paths), OR
- a narrow covering directory/glob (only if supported by this repo’s allowlist policy).

Do NOT add unrelated files or broad globs like "src/**" unless absolutely necessary.

Requested paths (MUST be covered):
{requested}
"""

    return f"""Produce an implementation plan for the phase below.

Follow all repository rules in AGENTS.md.

PRD: {prd_path}
PRD content (read first):
{prd_text}
{prd_notice}
PRD sections/IDs (cite in spec_summary):
{markers_block}

Phase: {phase.get("name") or phase.get("id")}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}
Global/phase test command: {test_block}
{expansion_block}
{user_block}
{progress_block}

Output file (write JSON): {impl_plan_path}

Plan schema (strict):
{{
  "phase_id": "{phase.get('id')}",
  "spec_summary": ["PRD requirements relevant to this phase (with section/IDs)"],
  "files_to_change": ["path/a.py", "path/b.ts"],
  "new_files": ["path/new_file.py"],
  "technical_approach": [
      "1. High-level step description",
      "2. Another step description",
      "3. Integration notes"
  ],
  "design_notes": {{
    "architecture": ["3-8 bullets"],
    "data_flow": ["..."],
    "invariants": ["..."],
    "edge_cases": ["..."]
  }},
  "test_plan": {{
    "commands": ["npm test", "pytest -q"],
    "new_tests": ["tests/test_x.py::test_y"],
    "manual_checks": ["..."]
  }},
  "migration_or_rollout": ["if applicable, otherwise '(none)'"],
  "open_questions": ["if any"],
  "assumptions": ["..."],
  "plan_deviations": []
}}

Rules:
- Cite PRD section headers/IDs in spec_summary where available.
- Focus on a coherent technical approach, not just a list of rigid steps.
- files_to_change must be non-empty unless the phase is docs-only.
- Set plan_deviations to an empty list in the initial plan.
- If an allowlist expansion request is present, files_to_change/new_files MUST cover every requested path.
"""


def _build_review_prompt(
    phase: dict[str, Any],
    review_path: Path,
    prd_path: Path,
    prd_text: str,
    prd_truncated: bool,
    prd_markers: Optional[list[str]],
    user_prompt: Optional[str],
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    changed_files: Optional[list[str]] = None,
    diff_text: str = "",
    diff_truncated: bool = False,
    diff_stat: str = "",
    diff_stat_truncated: bool = False,
    status_text: str = "",
    status_truncated: bool = False,
    impl_plan_text: str = "",
    impl_plan_truncated: bool = False,
    heartbeat_seconds: Optional[int] = None,
    tests_snapshot: Optional[dict[str, Any]] = None,
) -> str:
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, human_blocking_issues, human_next_steps, heartbeat.\n"
            f"{heartbeat_block}"
        )

    prd_notice = ""
    if prd_truncated:
        prd_notice = (
            "\nNOTE: PRD content truncated. Open the PRD file to read the full spec.\n"
        )
    prd_markers = prd_markers or []
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"

    changed_files = changed_files or []
    changed_block = "\n".join(f"- {path}" for path in changed_files) if changed_files else "- (none)"

    diff_notice = ""
    if diff_truncated:
        diff_notice = "\nNOTE: Diff truncated. Open git diff for full context.\n"

    diff_stat_notice = ""
    if diff_stat_truncated:
        diff_stat_notice = "\nNOTE: Diffstat truncated.\n"
    diff_stat_block = diff_stat if diff_stat else "(no diffstat output)"

    status_notice = ""
    if status_truncated:
        status_notice = "\nNOTE: Status truncated.\n"
    status_block = status_text if status_text else "(clean)"

    plan_notice = ""
    if impl_plan_truncated:
        plan_notice = "\nNOTE: Implementation plan truncated. Open the plan file for full details.\n"

    tests_block = "(no coordinator test run recorded)"
    if isinstance(tests_snapshot, dict):
        cmd = str(tests_snapshot.get("command") or "").strip() or "(unknown)"
        exit_code = tests_snapshot.get("exit_code")
        log_path = str(tests_snapshot.get("log_path") or "").strip() or "(unknown)"
        tail = str(tests_snapshot.get("log_tail") or "").strip()
        captured_at = str(tests_snapshot.get("captured_at") or "").strip()
        timestamp_block = f"Captured at: {captured_at}\n" if captured_at else ""
        tests_block = (
            f"Command: {cmd}\n"
            f"Exit code: {exit_code}\n"
            f"Log: {log_path}\n"
            f"{timestamp_block}"
            f"Recent output:\n{tail if tail else '(empty)'}"
        )

    return f"""Perform a thorough code review for the phase below and write JSON to {review_path}.

Follow all repository rules in AGENTS.md.

Phase: {phase.get("name") or phase.get("id")}
PRD: {prd_path}
PRD content (read first):
{prd_text}
{prd_notice}
PRD sections/IDs (for reference in spec_summary):
{markers_block}

Acceptance criteria:
{acceptance_block}
{user_block}
{progress_block}

Git status (from coordinator):
{status_block}
{status_notice}

Changed files (from coordinator):
{changed_block}

Diffstat (from coordinator):
{diff_stat_block}
{diff_stat_notice}

Diff (from coordinator):
{diff_text}
{diff_notice}

Coordinator test results:
{tests_block}

Implementation plan (from coordinator):
{impl_plan_text or "(missing)"}
{plan_notice}

Review output schema:
{{
  "phase_id": "{phase.get('id')}",
  "spec_summary": ["bullets restating PRD requirements relevant to this phase"],
  "design_assessment": {{
    "architecture_summary": ["3-8 bullets describing solution structure"],
    "key_components": ["components/modules touched and their roles"]
  }},
  "architecture_checklist": [
    {{
      "item": "architecture check item",
      "met": "yes|no|partial",
      "notes": "short evidence-based note",
      "files": ["path/to/file.ext"]
    }}
  ],
  "spec_traceability": [
    {{
      "requirement": "PRD requirement or acceptance criterion",
      "coverage": "yes|partial|no",
      "evidence": "specific code or diff reference",
      "files": ["path/to/file.ext"]
    }}
  ],
  "logic_risks": [
    {{
      "risk": "possible failure mode or edge case",
      "impact": "why it matters",
      "mitigation": "suggested fix or guard",
      "files": ["path/to/file.ext"]
    }}
  ],
  "acceptance_criteria_checklist": [
    {{
      "criterion": "exact acceptance criterion text",
      "met": "yes|no|partial",
      "evidence": "specific evidence from code or diff",
      "files": ["path/to/file.ext"]
    }}
  ],
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "summary": "One sentence",
      "rationale": "Why it matters",
      "files": ["path/to/file.ext"],
      "suggested_fix": "Actionable change"
    }}
  ],
  "summary": "Short summary",
  "changed_files": ["exact list from coordinator"],
  "files_reviewed": ["list of paths"],
  "evidence": ["at least two concrete observations with file/diff references"],
  "recommendations": ["actionable fixes"]
}}

Review instructions:
- Verify implementation aligns with the plan.
- If acceptance criteria are empty, include one checklist item with criterion "(none provided)".
- Provide at least {REVIEW_MIN_EVIDENCE_ITEMS} concrete evidence items tied to files/diff.
- Use the "Coordinator test results" section as the source of truth for baseline tests:
  - Exit code 0 => tests passed (do not raise issues just for lack of proof).
  - Non-zero exit => include at least one high/critical issue describing the failure and files involved.
- If tests passed (coordinator verified), assume logic works but check for code quality and specs.
"""


def _build_simple_review_prompt(
    phase: dict[str, Any],
    review_path: Path,
    prd_path: Path,
    prd_text: str,
    prd_truncated: bool,
    prd_markers: Optional[list[str]],
    user_prompt: Optional[str],
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    changed_files: Optional[list[str]] = None,
    diff_text: str = "",
    diff_truncated: bool = False,
    diff_stat: str = "",
    diff_stat_truncated: bool = False,
    status_text: str = "",
    status_truncated: bool = False,
    impl_plan_text: str = "",
    impl_plan_truncated: bool = False,
    heartbeat_seconds: Optional[int] = None,
    tests_snapshot: Optional[dict[str, Any]] = None,
) -> str:
    """Build a simplified review prompt with minimal output schema."""
    acceptance = phase.get("acceptance_criteria") or []
    acceptance_block = "\n".join(f"- {item}" for item in acceptance) if acceptance else "- (none)"
    user_block = f"\nSpecial instructions:\n{user_prompt}\n" if user_prompt else ""
    progress_block = ""
    if progress_path and run_id:
        heartbeat_block = ""
        if heartbeat_seconds:
            heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"
        progress_block = (
            "\nProgress contract (REQUIRED):\n"
            f"- Write snapshot to: {progress_path}\n"
            f"  Required fields: run_id={run_id}, task_id, phase, actions, claims, "
            "next_steps, human_blocking_issues, human_next_steps, heartbeat.\n"
            f"{heartbeat_block}"
        )

    prd_notice = ""
    if prd_truncated:
        prd_notice = (
            "\nNOTE: PRD content truncated. Open the PRD file to read the full spec.\n"
        )
    prd_markers = prd_markers or []
    markers_block = "\n".join(f"- {item}" for item in prd_markers) if prd_markers else "- (none found)"

    changed_files = changed_files or []
    changed_block = "\n".join(f"- {path}" for path in changed_files) if changed_files else "- (none)"

    diff_notice = ""
    if diff_truncated:
        diff_notice = "\nNOTE: Diff truncated. Open git diff for full context.\n"

    diff_stat_notice = ""
    if diff_stat_truncated:
        diff_stat_notice = "\nNOTE: Diffstat truncated.\n"
    diff_stat_block = diff_stat if diff_stat else "(no diffstat output)"

    status_notice = ""
    if status_truncated:
        status_notice = "\nNOTE: Status truncated.\n"
    status_block = status_text if status_text else "(clean)"

    plan_notice = ""
    if impl_plan_truncated:
        plan_notice = "\nNOTE: Implementation plan truncated. Open the plan file for full details.\n"

    tests_block = "(no coordinator test run recorded)"
    if isinstance(tests_snapshot, dict):
        cmd = str(tests_snapshot.get("command") or "").strip() or "(unknown)"
        exit_code = tests_snapshot.get("exit_code")
        log_path = str(tests_snapshot.get("log_path") or "").strip() or "(unknown)"
        tail = str(tests_snapshot.get("log_tail") or "").strip()
        captured_at = str(tests_snapshot.get("captured_at") or "").strip()
        timestamp_block = f"Captured at: {captured_at}\n" if captured_at else ""
        tests_block = (
            f"Command: {cmd}\n"
            f"Exit code: {exit_code}\n"
            f"Log: {log_path}\n"
            f"{timestamp_block}"
            f"Recent output:\n{tail if tail else '(empty)'}"
        )

    return f"""Perform a thorough code review for the phase below and write JSON to {review_path}.

Follow all repository rules in AGENTS.md.

Phase: {phase.get("name") or phase.get("id")}
PRD: {prd_path}
PRD content (read first):
{prd_text}
{prd_notice}
PRD sections/IDs (for reference):
{markers_block}

Acceptance criteria:
{acceptance_block}
{user_block}
{progress_block}

Git status (from coordinator):
{status_block}
{status_notice}

Changed files (from coordinator):
{changed_block}

Diffstat (from coordinator):
{diff_stat_block}
{diff_stat_notice}

Diff (from coordinator):
{diff_text}
{diff_notice}

Coordinator test results:
{tests_block}

Implementation plan (from coordinator):
{impl_plan_text or "(missing)"}
{plan_notice}

Review output schema (STRICT - only these fields):
{{
  "mergeable": true,
  "issues": [
    {{"severity": "high|medium|low", "text": "Description of the issue"}}
  ]
}}

Review instructions:
- Set "mergeable" to true if the implementation is ready to merge (tests pass, no critical issues).
- Set "mergeable" to false if there are blocking issues that must be fixed before merging.
- List issues with severity "high" for blockers, "medium" for should-fix, "low" for nice-to-have.
- If tests failed (non-zero exit code in coordinator test results), set mergeable=false and include a high severity issue.
- Focus on correctness, spec compliance, and critical code quality issues.
- Keep issue text concise but actionable.
"""
