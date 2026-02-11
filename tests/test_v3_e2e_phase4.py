from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from feature_prd_runner.server.api import create_app


def test_pin_create_run_review_approve_done(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()

    with TestClient(app) as client:
        pin = client.post('/api/v3/projects/pinned', json={'path': str(repo)})
        assert pin.status_code == 200

        created = client.post('/api/v3/tasks', json={'title': 'Feature task', 'metadata': {'scripted_findings': [[]]}}).json()['task']
        run = client.post(f"/api/v3/tasks/{created['id']}/run")
        assert run.status_code == 200
        assert run.json()['task']['status'] == 'in_review'

        approved = client.post(f"/api/v3/review/{created['id']}/approve", json={})
        assert approved.status_code == 200
        assert approved.json()['task']['status'] == 'done'


def test_import_dependency_execution_order(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        preview = client.post('/api/v3/import/prd/preview', json={'content': '- Step A\n- Step B'}).json()
        commit = client.post('/api/v3/import/prd/commit', json={'job_id': preview['job_id']}).json()
        first_id, second_id = commit['created_task_ids']

        first_run = client.post(f'/api/v3/tasks/{first_id}/run')
        assert first_run.status_code == 200

        blocked_second = client.post(f'/api/v3/tasks/{second_id}/run')
        assert blocked_second.status_code == 400

        client.post(f'/api/v3/review/{first_id}/approve', json={})
        second_run = client.post(f'/api/v3/tasks/{second_id}/run')
        assert second_run.status_code == 200


def test_quick_action_stays_off_board_until_promoted(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        quick = client.post('/api/v3/quick-actions', json={'prompt': 'One-off command'}).json()['quick_action']
        tasks_before = client.get('/api/v3/tasks').json()['tasks']
        assert tasks_before == []

        promote = client.post(f"/api/v3/quick-actions/{quick['id']}/promote", json={})
        assert promote.status_code == 200
        task_id = promote.json()['task']['id']

        tasks_after = client.get('/api/v3/tasks').json()['tasks']
        ids = [task['id'] for task in tasks_after]
        assert task_id in ids


def test_findings_loop_until_zero_open_then_done(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        task = client.post(
            '/api/v3/tasks',
            json={
                'title': 'Loop task',
                'approval_mode': 'auto_approve',
                'metadata': {
                    'scripted_findings': [
                        [{'severity': 'high', 'summary': 'Need fix'}],
                        [],
                    ]
                },
            },
        ).json()['task']
        run = client.post(f"/api/v3/tasks/{task['id']}/run")
        assert run.status_code == 200
        assert run.json()['task']['status'] == 'done'
        assert run.json()['task']['retry_count'] >= 1


def test_request_changes_reopens_task_with_feedback(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        task = client.post('/api/v3/tasks', json={'title': 'Needs feedback', 'metadata': {'scripted_findings': [[]]}}).json()['task']
        client.post(f"/api/v3/tasks/{task['id']}/run")

        changed = client.post(
            f"/api/v3/review/{task['id']}/request-changes",
            json={'guidance': 'Please add integration tests'},
        )
        assert changed.status_code == 200
        body = changed.json()['task']
        assert body['status'] == 'ready'
        assert body['metadata']['requested_changes']['guidance'] == 'Please add integration tests'


def test_single_run_branch_commits_in_task_order(tmp_path: Path) -> None:
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.email', 'ci@example.com'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'CI'], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / 'seed.txt').write_text('seed\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'seed.txt'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'commit', '-m', 'seed'], cwd=tmp_path, check=True, capture_output=True)

    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        first = client.post('/api/v3/tasks', json={'title': 'First', 'approval_mode': 'auto_approve'}).json()['task']
        second = client.post('/api/v3/tasks', json={'title': 'Second', 'approval_mode': 'auto_approve'}).json()['task']

        client.post(f"/api/v3/tasks/{first['id']}/run")
        client.post(f"/api/v3/tasks/{second['id']}/run")

    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    assert branch.startswith('orchestrator-run-')

    messages = subprocess.run(['git', 'log', '--pretty=%s', '-n', '2'], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.splitlines()
    assert messages[0].startswith(f"task({second['id']})")
    assert messages[1].startswith(f"task({first['id']})")
