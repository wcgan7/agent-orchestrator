import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner import runner
from feature_prd_runner.io_utils import _load_data
from feature_prd_runner.state import _ensure_state_files


def _git_porcelain(project_dir: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def test_dry_run_is_read_only(tmp_path, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    _ensure_state_files(project_dir, prd_path)

    before = _git_porcelain(project_dir)

    try:
        runner.main(["dry-run", "--project-dir", str(project_dir), "--prd-file", str(prd_path)])
    except SystemExit as exc:
        assert exc.code == 0

    after = _git_porcelain(project_dir)
    assert after == before
    out = capsys.readouterr().out
    assert "Dry-run guarantees" in out


def test_doctor_json(tmp_path, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    _ensure_state_files(project_dir, prd_path)

    try:
        runner.main(["doctor", "--project-dir", str(project_dir), "--prd-file", str(prd_path), "--json"])
    except SystemExit as exc:
        assert exc.code in {0, 1, 2}

    out = capsys.readouterr().out
    assert '"checks"' in out
    payload = _load_data(project_dir / ".prd_runner" / "run_state.yaml", {})
    assert payload.get("prd_path") is not None
