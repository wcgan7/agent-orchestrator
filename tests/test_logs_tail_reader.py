from __future__ import annotations

from pathlib import Path

from agent_orchestrator.runtime.api.router import _read_tail


def test_read_tail_drops_partial_first_line_when_truncated(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Tail starts in the middle of "line2", so the partial prefix is dropped.
    tail = _read_tail(path, max_chars=10)

    assert tail == "line3\n"


def test_read_tail_keeps_full_lines_when_cut_at_newline_boundary(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Tail starts exactly at "line2", so no line is partial and both lines remain.
    tail = _read_tail(path, max_chars=12)

    assert tail == "line2\nline3\n"
