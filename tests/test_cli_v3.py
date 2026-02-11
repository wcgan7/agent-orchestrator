from __future__ import annotations

from pathlib import Path

from feature_prd_runner.cli_v3 import main


def test_task_create_list_and_run(tmp_path: Path) -> None:
    rc = main(['--project-dir', str(tmp_path), 'task', 'create', 'CLI Task', '--priority', 'P1'])
    assert rc == 0

    rc = main(['--project-dir', str(tmp_path), 'task', 'list'])
    assert rc == 0


def test_project_pin_list_unpin(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()

    assert main(['--project-dir', str(tmp_path), 'project', 'pin', str(repo), '--project-id', 'repo-1']) == 0
    assert main(['--project-dir', str(tmp_path), 'project', 'list']) == 0
    assert main(['--project-dir', str(tmp_path), 'project', 'unpin', 'repo-1']) == 0


def test_quick_action_and_orchestrator_status(tmp_path: Path) -> None:
    assert main(['--project-dir', str(tmp_path), 'quick-action', 'Do quick thing']) == 0
    assert main(['--project-dir', str(tmp_path), 'orchestrator', 'status']) == 0
