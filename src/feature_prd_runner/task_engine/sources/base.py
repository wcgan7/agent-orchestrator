"""Base class for task generators that create tasks from various sources."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, Optional

from ..model import Task


class TaskGenerator(abc.ABC):
    """Abstract base for one-click task generators.

    Subclasses implement :meth:`generate` to analyze a project and produce
    a list of :class:`Task` objects for the dynamic task board.
    """

    #: Human-readable name for this generator (e.g. "Repo Review").
    name: str = "base"

    #: Short description shown in the UI.
    description: str = ""

    @abc.abstractmethod
    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        """Analyze *project_dir* and return tasks.

        Parameters
        ----------
        project_dir:
            Root of the project to analyze.
        on_progress:
            Optional callback ``(message, fraction)`` for progress reporting.
            *fraction* is in ``[0.0, 1.0]``.

        Returns
        -------
        list[Task]
            Newly generated tasks (not yet persisted).
        """
        ...
