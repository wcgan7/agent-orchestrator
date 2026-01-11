from __future__ import annotations

from typing import Any, Optional

from ..validation import _validate_impl_plan_data


def validate_plan(
    plan_data: dict[str, Any],
    phase: dict[str, Any],
    *,
    prd_markers: Optional[list[str]] = None,
    prd_truncated: bool = False,
    prd_has_content: bool = True,
    expected_test_command: Optional[str] = None,
    plan_expansion_request: Optional[list[str]] = None,
) -> tuple[bool, str]:
    return _validate_impl_plan_data(
        plan_data,
        phase,
        prd_markers=prd_markers,
        prd_truncated=prd_truncated,
        prd_has_content=prd_has_content,
        expected_test_command=expected_test_command,
        plan_expansion_request=plan_expansion_request,
    )
