from __future__ import annotations

from pathlib import Path

from agent_orchestrator.runtime.api.router import _read_from_offset, _read_tail


def test_read_tail_drops_partial_first_line_when_truncated(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Tail starts in the middle of "line2", so the partial prefix is dropped.
    tail, tail_start = _read_tail(path, max_chars=10)

    assert tail == "line3\n"
    assert tail_start == len("line1\nline2\n".encode("utf-8"))


def test_read_tail_keeps_full_lines_when_cut_at_newline_boundary(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Tail starts exactly at "line2", so no line is partial and both lines remain.
    tail, tail_start = _read_tail(path, max_chars=12)

    assert tail == "line2\nline3\n"
    assert tail_start == len("line1\n".encode("utf-8"))


def test_read_tail_returns_full_file_when_within_limit(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\n", encoding="utf-8")

    tail, tail_start = _read_tail(path, max_chars=200)

    assert tail == "line1\nline2\n"
    assert tail_start == 0


def test_read_from_offset_can_align_to_previous_newline(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Offset lands in the middle of line2; align mode should back up to line start.
    text, new_offset, chunk_start = _read_from_offset(
        path,
        offset=8,
        max_bytes=4,
        read_to=12,
        align_start_to_line=True,
    )

    assert text == "line2\n"
    assert chunk_start == 6
    assert new_offset == 12


def test_read_from_offset_reports_monotonic_chunk_start_for_backfill_windows(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    # First historical window: [line2, line3]
    chunk_b, new_offset_b, chunk_start_b = _read_from_offset(
        path,
        offset=8,
        max_bytes=6,
        read_to=18,
        align_start_to_line=True,
    )

    # Next historical window: [line1], bounded by prior chunk start.
    chunk_a, new_offset_a, chunk_start_a = _read_from_offset(
        path,
        offset=0,
        max_bytes=6,
        read_to=chunk_start_b,
        align_start_to_line=True,
    )

    assert chunk_b == "line2\nline3\n"
    assert chunk_start_b == 6
    assert new_offset_b == 18
    assert chunk_a == "line1\n"
    assert chunk_start_a == 0
    assert new_offset_a == 6
    assert chunk_start_a < chunk_start_b
