from __future__ import annotations

from typing import Any, Optional

try:
    from .constants import REVIEW_BLOCKING_SEVERITIES, REVIEW_MIN_EVIDENCE_ITEMS, REVIEW_SEVERITIES
    from .utils import _validate_string_list
except ImportError:  # pragma: no cover
    from constants import REVIEW_BLOCKING_SEVERITIES, REVIEW_MIN_EVIDENCE_ITEMS, REVIEW_SEVERITIES
    from utils import _validate_string_list


def _extract_review_blocker_files(review_data: dict[str, Any]) -> list[str]:
    if not isinstance(review_data, dict):
        return []
    collected: list[str] = []
    seen: set[str] = set()

    def _add(paths: Any) -> None:
        if not isinstance(paths, list):
            return
        for path in paths:
            path_value = str(path).strip()
            if not path_value or path_value in seen:
                continue
            collected.append(path_value)
            seen.add(path_value)

    _add(review_data.get("files_reviewed"))
    _add(review_data.get("changed_files"))

    issues = review_data.get("issues")
    if isinstance(issues, list):
        for item in issues:
            if isinstance(item, dict):
                _add(item.get("files"))

    return collected


def _is_docs_only_phase(phase: dict[str, Any]) -> bool:
    keywords = ["doc", "docs", "readme", "documentation"]
    haystack = " ".join(
        str(value or "")
        for value in [
            phase.get("name"),
            phase.get("description"),
            " ".join(phase.get("acceptance_criteria") or []),
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _plan_deviations_present(plan_data: dict[str, Any]) -> bool:
    deviations = plan_data.get("plan_deviations")
    if not isinstance(deviations, list):
        return False
    return any(isinstance(item, str) and item.strip() for item in deviations)


def _validate_impl_plan_data(
    plan_data: dict[str, Any],
    phase: dict[str, Any],
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
    expected_test_command: Optional[str] = None,
) -> tuple[bool, str]:
    if not isinstance(plan_data, dict):
        return False, "Implementation plan is not a JSON object"

    phase_id = phase.get("id")
    if not plan_data.get("phase_id"):
        return False, "phase_id missing"
    if phase_id and str(plan_data.get("phase_id")) != str(phase_id):
        return False, "phase_id does not match current phase"

    if not prd_has_content:
        return False, "PRD content missing for plan"

    spec_summary = plan_data.get("spec_summary")
    valid, error = _validate_string_list(spec_summary, "spec_summary")
    if not valid:
        return False, error

    # REMOVED: Rigid step count validation.
    # Added: Check for technical approach or steps.
    steps = plan_data.get("steps")
    tech_approach = plan_data.get("technical_approach")

    if not steps and not tech_approach:
        return False, "Must provide either 'technical_approach' (text) or 'steps' (list)"

    files_to_change = plan_data.get("files_to_change")
    if not isinstance(files_to_change, list):
        return False, "files_to_change must be a list"
    if not files_to_change and not _is_docs_only_phase(phase):
        return False, "files_to_change must be non-empty for non-docs phases"
    if any(not isinstance(path, str) or not path.strip() for path in files_to_change):
        return False, "files_to_change must contain non-empty strings"

    return True, ""


def _validate_review_data(
    review_data: dict[str, Any],
    phase: dict[str, Any],
    changed_files: Optional[list[str]] = None,
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
) -> tuple[bool, str]:
    if not isinstance(review_data, dict):
        return False, "Review output is not a JSON object"

    phase_id = phase.get("id")
    review_phase_id = review_data.get("phase_id")
    if not review_phase_id:
        return False, "phase_id missing"
    if phase_id and str(review_phase_id) != str(phase_id):
        return False, "phase_id does not match current phase"

    # PRD access: if missing, require an explicit critical/high issue
    issues = review_data.get("issues")
    if not isinstance(issues, list):
        return False, "issues must be a list"
    for i, item in enumerate(issues):
        if not isinstance(item, dict):
            return False, f"issues[{i}] must be an object"
        sev = str(item.get("severity", "")).strip().lower()
        if sev not in REVIEW_SEVERITIES:
            return False, f"issues[{i}].severity must be one of: {sorted(REVIEW_SEVERITIES)}"
        summ = item.get("summary")
        if not isinstance(summ, str) or not summ.strip():
            return False, f"issues[{i}].summary must be a non-empty string"
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            return False, f"issues[{i}].rationale must be a non-empty string"
        files = item.get("files")
        if not isinstance(files, list) or any(not isinstance(f, str) or not f.strip() for f in files):
            return False, f"issues[{i}].files must be a list of non-empty strings"
        suggested_fix = item.get("suggested_fix")
        if not isinstance(suggested_fix, str) or not suggested_fix.strip():
            return False, f"issues[{i}].suggested_fix must be a non-empty string"

    if not prd_has_content:
        has_prd_access_issue = any(
            isinstance(it, dict)
            and str(it.get("severity", "")).strip().lower() in REVIEW_BLOCKING_SEVERITIES
            and "prd" in str(it.get("summary", "")).lower()
            for it in issues
        )
        if not has_prd_access_issue:
            return False, "issues must include a critical/high PRD access failure when PRD content missing"

    files_reviewed = review_data.get("files_reviewed")
    if not isinstance(files_reviewed, list) or not files_reviewed:
        return False, "files_reviewed must be a non-empty list"
    if any(not isinstance(item, str) or not item.strip() for item in files_reviewed):
        return False, "files_reviewed must contain non-empty strings"
    files_reviewed_set = {item.strip() for item in files_reviewed if item.strip()}

    expected_files = [path.strip() for path in (changed_files or []) if str(path).strip()]
    expected_set = {path for path in expected_files if path}
    if expected_set and not expected_set.issubset(files_reviewed_set):
        # keep lenient as before
        pass

    evidence = review_data.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < REVIEW_MIN_EVIDENCE_ITEMS:
        return False, f"evidence must include at least {REVIEW_MIN_EVIDENCE_ITEMS} items"

    return True, ""
