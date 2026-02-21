"""Workdoc templates and synchronization helpers for orchestrator tasks."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable

from ..domain.models import ReviewCycle, Task
from ..events.bus import EventBus
from ..storage.container import Container


class WorkdocManager:
    """Manage workdoc rendering and synchronization for task execution."""

    # Maps step names to (heading, placeholder_step) pairs.
    # placeholder_step is the step name used in the template placeholder text.
    _WORKDOC_SECTION_MAP: dict[str, tuple[str, str | None]] = {
        "plan": ("## Plan", "plan"),
        "initiative_plan": ("## Plan", "plan"),
        "analyze": ("## Analysis", "analyze"),
        "diagnose": ("## Analysis", "analyze"),
        "scan_deps": ("## Dependency Scan Findings", "scan_deps"),
        "scan_code": ("## Code Scan Findings", "scan_code"),
        "generate_tasks": ("## Generated Tasks", "generate_tasks"),
        "profile": ("## Profiling Baseline", "profile"),
        "implement": ("## Implementation Log", "implement"),
        "prototype": ("## Implementation Log", "implement"),
        "implement_fix": ("## Fix Log", None),  # placeholder uses different wording
        "verify": ("## Verification Results", "verify"),
        "benchmark": ("## Verification Results", "verify"),
        "reproduce": ("## Verification Results", "verify"),
        "report": ("## Final Report", "report"),
    }

    _FEATURE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Plan

_Pending: will be populated by the plan step._

## Implementation Log

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _GENERIC_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Plan

_Pending: will be populated by the plan step._

## Analysis

_Pending: will be populated by the analyze step._

## Profiling Baseline

_Pending: will be populated by the profile step._

## Implementation Log

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Final Report

_Pending: will be populated by the report step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _VERIFY_ONLY_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Verification Results

_Pending: will be populated by the verify step._

## Final Report

_Pending: will be populated by the report step._
"""

    _SECURITY_AUDIT_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Dependency Scan Findings

_Pending: will be populated by the scan_deps step._

## Code Scan Findings

_Pending: will be populated by the scan_code step._

## Security Report

_Pending: will be populated by the report step._

## Generated Remediation Tasks

_Pending: will be populated by the generate_tasks step._
"""

    _REPO_REVIEW_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Repository Analysis

_Pending: will be populated by the analyze step._

## Initiative Plan

_Pending: will be populated by the plan step._

## Generated Tasks

_Pending: will be populated by the generate_tasks step._
"""

    _RESEARCH_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Research Analysis

_Pending: will be populated by the analyze step._

## Final Report

_Pending: will be populated by the report step._
"""

    _REVIEW_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Review Analysis

_Pending: will be populated by the analyze step._

## Review Findings

_Pending: will be populated by the review step._

## Final Report

_Pending: will be populated by the report step._
"""

    _BUG_FIX_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Reproduction Evidence

_Pending: will be populated by the reproduce step._

## Diagnosis

_Pending: will be populated by the diagnose step._

## Fix Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _REFACTOR_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Refactor Analysis

_Pending: will be populated by the analyze step._

## Refactor Plan

_Pending: will be populated by the plan step._

## Refactor Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _PERFORMANCE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Profiling Baseline

_Pending: will be populated by the profile step._

## Optimization Plan

_Pending: will be populated by the plan step._

## Optimization Implementation

_Pending: will be populated by the implement step._

## Benchmark Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _TEST_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Coverage Analysis

_Pending: will be populated by the analyze step._

## Test Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _DOCS_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Documentation Analysis

_Pending: will be populated by the analyze step._

## Documentation Updates

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _HOTFIX_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Hotfix Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._

## Review Findings

_Pending: will be populated by the review step._

## Fix Log

_Pending: will be populated as needed._
"""

    _CHORE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Chore Implementation

_Pending: will be populated by the implement step._

## Verification Results

_Pending: will be populated by the verify step._
"""

    _SPIKE_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Spike Analysis

_Pending: will be populated by the analyze step._

## Prototype Notes

_Pending: will be populated by the implement step._

## Final Report

_Pending: will be populated by the report step._
"""

    _PLAN_ONLY_WORKDOC_TEMPLATE = """\
# Working Document: {title}

**Task ID:** {task_id}
**Type:** {task_type} | **Priority:** {priority}
**Created:** {created_at}

---

## Task Description

{description}

---

## Analysis

_Pending: will be populated by the analyze step._

## Plan

_Pending: will be populated by the plan step._

## Generated Tasks

_Pending: will be populated by the generate_tasks step._

## Final Report

_Pending: will be populated by the report step._
"""

    def __init__(
        self,
        container: Container,
        bus: EventBus,
        *,
        pipeline_id_resolver: Callable[[Task], str],
    ) -> None:
        """Initialize the manager with orchestrator dependencies."""
        self.container = container
        self.bus = bus
        self._pipeline_id_resolver = pipeline_id_resolver

    def workdoc_template_for_task(self, task: Task) -> str:
        """Select the default workdoc template for the task's pipeline."""
        pipeline_id = self._pipeline_id_resolver(task)
        template_by_pipeline: dict[str, str] = {
            "feature": self._FEATURE_WORKDOC_TEMPLATE,
            "bug_fix": self._BUG_FIX_WORKDOC_TEMPLATE,
            "refactor": self._REFACTOR_WORKDOC_TEMPLATE,
            "research": self._RESEARCH_WORKDOC_TEMPLATE,
            "docs": self._DOCS_WORKDOC_TEMPLATE,
            "test": self._TEST_WORKDOC_TEMPLATE,
            "repo_review": self._REPO_REVIEW_WORKDOC_TEMPLATE,
            "security_audit": self._SECURITY_AUDIT_WORKDOC_TEMPLATE,
            "review": self._REVIEW_WORKDOC_TEMPLATE,
            "performance": self._PERFORMANCE_WORKDOC_TEMPLATE,
            "hotfix": self._HOTFIX_WORKDOC_TEMPLATE,
            "spike": self._SPIKE_WORKDOC_TEMPLATE,
            "chore": self._CHORE_WORKDOC_TEMPLATE,
            "plan_only": self._PLAN_ONLY_WORKDOC_TEMPLATE,
            "verify_only": self._VERIFY_ONLY_WORKDOC_TEMPLATE,
        }
        return template_by_pipeline.get(pipeline_id, self._GENERIC_WORKDOC_TEMPLATE)

    def workdoc_section_for_step(self, task: Task, step: str) -> tuple[str, str | None] | None:
        """Resolve section heading and placeholder mapping for a step."""
        section = self._WORKDOC_SECTION_MAP.get(step)
        if not section:
            return None
        heading, placeholder_step = section
        pipeline_id = self._pipeline_id_resolver(task)
        section_overrides: dict[str, dict[str, tuple[str, str | None]]] = {
            "security_audit": {
                "report": ("## Security Report", "report"),
                "generate_tasks": ("## Generated Remediation Tasks", "generate_tasks"),
            },
            "repo_review": {
                "analyze": ("## Repository Analysis", "analyze"),
                "initiative_plan": ("## Initiative Plan", "plan"),
            },
            "research": {
                "analyze": ("## Research Analysis", "analyze"),
            },
            "review": {
                "analyze": ("## Review Analysis", "analyze"),
            },
            "bug_fix": {
                "reproduce": ("## Reproduction Evidence", "reproduce"),
                "diagnose": ("## Diagnosis", "diagnose"),
                "implement": ("## Fix Implementation", "implement"),
            },
            "refactor": {
                "analyze": ("## Refactor Analysis", "analyze"),
                "plan": ("## Refactor Plan", "plan"),
                "implement": ("## Refactor Implementation", "implement"),
            },
            "performance": {
                "plan": ("## Optimization Plan", "plan"),
                "implement": ("## Optimization Implementation", "implement"),
                "benchmark": ("## Benchmark Results", "verify"),
            },
            "test": {
                "analyze": ("## Coverage Analysis", "analyze"),
                "implement": ("## Test Implementation", "implement"),
            },
            "docs": {
                "analyze": ("## Documentation Analysis", "analyze"),
                "implement": ("## Documentation Updates", "implement"),
            },
            "hotfix": {
                "implement": ("## Hotfix Implementation", "implement"),
            },
            "chore": {
                "implement": ("## Chore Implementation", "implement"),
            },
            "spike": {
                "analyze": ("## Spike Analysis", "analyze"),
                "prototype": ("## Prototype Notes", "implement"),
            },
        }
        override = section_overrides.get(pipeline_id, {}).get(step)
        if override:
            heading, placeholder_step = override
        return heading, placeholder_step

    def workdoc_canonical_path(self, task_id: str) -> Path:
        """Return canonical workdoc path for task state."""
        return self.container.state_root / "workdocs" / f"{task_id}.md"

    @staticmethod
    def workdoc_worktree_path(project_dir: Path) -> Path:
        """Return worktree-local workdoc path."""
        return project_dir / ".workdoc.md"

    def init_workdoc(self, task: Task, project_dir: Path) -> Path:
        """Render the workdoc template and write canonical + worktree copies."""
        workdocs_dir = self.container.state_root / "workdocs"
        workdocs_dir.mkdir(parents=True, exist_ok=True)

        canonical = self.workdoc_canonical_path(task.id)
        template = self.workdoc_template_for_task(task)
        content = (
            template
            .replace("{title}", task.title)
            .replace("{task_id}", task.id)
            .replace("{task_type}", task.task_type)
            .replace("{priority}", task.priority)
            .replace("{created_at}", task.created_at)
            .replace("{description}", task.description or "(no description)")
        )
        canonical.write_text(content, encoding="utf-8")

        worktree_copy = self.workdoc_worktree_path(project_dir)
        shutil.copy2(str(canonical), str(worktree_copy))

        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["workdoc_path"] = str(canonical)
        return canonical

    @staticmethod
    def cleanup_workdoc_for_commit(project_dir: Path) -> None:
        """Remove the worktree .workdoc.md before commit."""
        workdoc = project_dir / ".workdoc.md"
        if workdoc.exists():
            workdoc.unlink()

    def refresh_workdoc(self, task: Task, project_dir: Path) -> None:
        """Copy canonical workdoc to worktree so worker sees the latest version."""
        canonical = self.workdoc_canonical_path(task.id)
        if not canonical.exists():
            return
        worktree_copy = self.workdoc_worktree_path(project_dir)
        shutil.copy2(str(canonical), str(worktree_copy))

    def sync_workdoc(
        self, task: Task, step: str, project_dir: Path, summary: str | None, attempt: int | None = None
    ) -> None:
        """Post-step sync: accept worker changes or fallback-append summary."""
        canonical = self.workdoc_canonical_path(task.id)
        if not canonical.exists():
            return
        worktree_copy = self.workdoc_worktree_path(project_dir)
        if not worktree_copy.exists():
            return

        canonical_text = canonical.read_text(encoding="utf-8")
        worktree_text = worktree_copy.read_text(encoding="utf-8")

        changed = False
        orchestrator_managed_steps = {"verify", "benchmark", "reproduce", "implement_fix", "report", "profile"}
        allow_worker_workdoc_write = step not in orchestrator_managed_steps
        if worktree_text != canonical_text and allow_worker_workdoc_write:
            canonical.write_text(worktree_text, encoding="utf-8")
            changed = True
        elif summary and summary.strip():
            section = self.workdoc_section_for_step(task, step)
            if not section:
                return
            heading, placeholder_step = section
            if placeholder_step:
                placeholder = f"_Pending: will be populated by the {placeholder_step} step._"
            else:
                placeholder = "_Pending: will be populated as needed._"
            trimmed = summary.strip()
            if step == "implement_fix":
                cycle_num = int(attempt or 1)
                trimmed = f"### Fix Cycle {cycle_num}\n{trimmed}"
            if placeholder in canonical_text:
                updated = canonical_text.replace(placeholder, trimmed, 1)
            else:
                idx = canonical_text.find(heading)
                if idx == -1:
                    return
                newline_after = canonical_text.find("\n", idx)
                if newline_after == -1:
                    updated = canonical_text + "\n\n" + trimmed
                else:
                    rest = canonical_text[newline_after + 1 :]
                    next_heading = re.search(r"^## ", rest, re.MULTILINE)
                    if next_heading:
                        insert_pos = newline_after + 1 + next_heading.start()
                        updated = canonical_text[:insert_pos] + trimmed + "\n\n" + canonical_text[insert_pos:]
                    else:
                        updated = canonical_text.rstrip() + "\n\n" + trimmed + "\n"
            canonical.write_text(updated, encoding="utf-8")
            worktree_copy.write_text(updated, encoding="utf-8")
            changed = True

        if changed:
            self.bus.emit(
                channel="tasks",
                event_type="workdoc.updated",
                entity_id=task.id,
                payload={"step": step},
            )

    def sync_workdoc_review(self, task: Task, cycle: ReviewCycle, project_dir: Path) -> None:
        """Append review cycle findings to canonical/worktree workdoc."""
        canonical = self.workdoc_canonical_path(task.id)
        if not canonical.exists():
            return

        text = canonical.read_text(encoding="utf-8")
        lines: list[str] = [
            f"### Review Cycle {cycle.attempt} — {cycle.decision}",
        ]
        counts = cycle.open_counts or {}
        lines.append(
            "Open findings: "
            f"critical={int(counts.get('critical', 0))}, "
            f"high={int(counts.get('high', 0))}, "
            f"medium={int(counts.get('medium', 0))}, "
            f"low={int(counts.get('low', 0))}"
        )
        for finding in cycle.findings:
            if finding.file and finding.line:
                loc = f" ({finding.file}:{finding.line})"
            elif finding.file:
                loc = f" ({finding.file})"
            else:
                loc = ""
            category = f"[{finding.category}] " if finding.category else ""
            lines.append(f"- **[{finding.severity}]** {category}{finding.summary}{loc}")
            if finding.suggested_fix:
                lines.append(f"  - Suggested fix: {finding.suggested_fix}")
            if finding.status and finding.status != "open":
                lines.append(f"  - Status: {finding.status}")
        block = "\n".join(lines)

        placeholder = "_Pending: will be populated by the review step._"
        if placeholder in text:
            updated = text.replace(placeholder, block, 1)
        else:
            heading = "## Review Findings"
            idx = text.find(heading)
            if idx == -1:
                updated = text.rstrip() + "\n\n" + heading + "\n\n" + block + "\n"
            else:
                next_heading = re.search(r"^## ", text[idx + len(heading) :], re.MULTILINE)
                if next_heading:
                    insert_pos = idx + len(heading) + next_heading.start()
                    updated = text[:insert_pos] + block + "\n\n" + text[insert_pos:]
                else:
                    updated = text.rstrip() + "\n\n" + block + "\n"

        canonical.write_text(updated, encoding="utf-8")
        worktree_copy = self.workdoc_worktree_path(project_dir)
        if worktree_copy.exists():
            worktree_copy.write_text(updated, encoding="utf-8")

        self.bus.emit(
            channel="tasks",
            event_type="workdoc.updated",
            entity_id=task.id,
            payload={"step": "review", "cycle": cycle.attempt},
        )
