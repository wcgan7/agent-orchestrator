"""PRD import generator — parses a markdown PRD into tasks on the board."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskType


class PrdImportGenerator:
    """Parse a PRD markdown file and create one task per phase."""

    name = "prd_import"
    description = "Import a PRD file and create tasks for each phase"

    def generate(
        self,
        project_dir: Path,
        *,
        prd_path: Optional[Path] = None,
        prd_content: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        _progress = on_progress or (lambda msg, frac: None)

        if prd_content is None and prd_path is not None:
            prd_content = prd_path.read_text(encoding="utf-8")
        if prd_content is None:
            return []

        _progress("Parsing PRD...", 0.3)
        phases = self._extract_phases(prd_content)
        acceptance = self._extract_acceptance_criteria(prd_content)

        _progress("Creating tasks...", 0.7)
        tasks: list[Task] = []
        parent_task = Task(
            title=self._extract_title(prd_content),
            description="Imported from PRD",
            task_type=TaskType.FEATURE,
            priority=TaskPriority.P1,
            source=TaskSource.PRD_IMPORT,
            labels=["prd"],
            acceptance_criteria=acceptance,
        )
        tasks.append(parent_task)

        prev_id: Optional[str] = None
        for i, (name, desc) in enumerate(phases):
            phase_task = Task(
                title=name,
                description=desc,
                task_type=TaskType.FEATURE,
                priority=TaskPriority.P1,
                source=TaskSource.PRD_IMPORT,
                parent_id=parent_task.id,
                labels=["prd", f"phase-{i + 1}"],
            )
            if prev_id:
                phase_task.blocked_by.append(prev_id)
            tasks.append(phase_task)
            parent_task.children_ids.append(phase_task.id)
            prev_id = phase_task.id

        _progress("Complete", 1.0)
        return tasks

    def _extract_title(self, content: str) -> str:
        """Pull the first H1 heading as the PRD title."""
        m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        if m:
            # Strip "Feature:" prefix if present
            title = m.group(1).strip()
            title = re.sub(r"^Feature:\s*", "", title, flags=re.IGNORECASE)
            return title
        return "Imported PRD"

    def _extract_phases(self, content: str) -> list[tuple[str, str]]:
        """Extract phases from the PRD (H3 or numbered list under "Phase" sections)."""
        phases: list[tuple[str, str]] = []

        # Try "### Phase N: Title" pattern
        phase_pattern = re.compile(
            r"###?\s*Phase\s+\d+[:\s]*(.+?)(?=\n###?\s*Phase|\n##\s|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        for m in phase_pattern.finditer(content):
            first_line = m.group(1).split("\n")[0].strip()
            body = m.group(1).strip()
            phases.append((first_line, body))

        if phases:
            return phases

        # Fallback: numbered list items under "Implementation" section
        impl_pattern = re.compile(
            r"##\s*.*(?:Implementation|Plan).*?\n(.*?)(?=\n##\s|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        m = impl_pattern.search(content)
        if m:
            section = m.group(1)
            for item in re.finditer(r"(?:^|\n)\d+\.\s*\*?\*?(.+?)(?=\n\d+\.|\Z)", section, re.DOTALL):
                line = item.group(1).strip().split("\n")[0]
                phases.append((line, item.group(1).strip()))

        return phases

    def _extract_acceptance_criteria(self, content: str) -> list[str]:
        """Extract acceptance criteria from the PRD."""
        criteria: list[str] = []
        ac_pattern = re.compile(
            r"##\s*.*(?:Acceptance|Criteria).*?\n(.*?)(?=\n##\s|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        m = ac_pattern.search(content)
        if m:
            section = m.group(1)
            for line in section.strip().split("\n"):
                line = line.strip().lstrip("-*• ")
                if line and not line.startswith("#"):
                    criteria.append(line)
        return criteria
