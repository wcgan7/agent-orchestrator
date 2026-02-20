from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from agent_orchestrator.server.api import create_app
from agent_orchestrator.runtime.orchestrator import DefaultWorkerAdapter
from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult


class _FileWritingAdapter(DefaultWorkerAdapter):
    """DefaultWorkerAdapter that also creates a file during implement steps."""

    def run_step(self, *, task, step: str, attempt: int) -> StepResult:
        result = super().run_step(task=task, step=step, attempt=attempt)
        if step in ("implement", "implement_fix") and result.status == "ok":
            wt = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
            if wt:
                (Path(wt) / f"change-{task.id}-{attempt}.txt").write_text("impl\n")
        return result


class _PrdImportWorkerAdapter(DefaultWorkerAdapter):
    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        if step == "generate_tasks":
            parsed = task.metadata.get("parsed_prd") if isinstance(task.metadata, dict) else None
            candidates = parsed.get("task_candidates") if isinstance(parsed, dict) else None
            if not isinstance(candidates, list) or not candidates:
                lines = [line.strip() for line in str(task.description or "").splitlines()]
                fallback = [line[2:].strip() for line in lines if line.startswith("- ") and line[2:].strip()]
                candidates = [{"title": title, "priority": "P2"} for title in fallback]
            if isinstance(candidates, list):
                generated: list[dict[str, object]] = []
                prev: str | None = None
                for idx, item in enumerate(candidates, start=1):
                    if not isinstance(item, dict):
                        continue
                    ref_id = f"prd_{idx}"
                    generated.append(
                        {
                            "id": ref_id,
                            "title": str(item.get("title") or f"Imported PRD task {idx}"),
                            "task_type": "feature",
                            "priority": str(item.get("priority") or "P2"),
                            "depends_on": [prev] if prev else [],
                            "metadata": {
                                "generated_ref_id": ref_id,
                                "generated_depends_on": [prev] if prev else [],
                                "source_chunk_id": item.get("source_chunk_id"),
                                "source_section_path": item.get("source_section_path"),
                            },
                        }
                    )
                    prev = ref_id
                return StepResult(status="ok", generated_tasks=generated)
        return super().run_step(task=task, step=step, attempt=attempt)


def test_pin_create_run_review_approve_done(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()

    with TestClient(app) as client:
        pin = client.post('/api/projects/pinned', json={'path': str(repo)})
        assert pin.status_code == 200

        created = client.post('/api/tasks', json={'title': 'Feature task', 'metadata': {'scripted_findings': [[]]}}).json()['task']
        run = client.post(f"/api/tasks/{created['id']}/run")
        assert run.status_code == 200
        assert run.json()['task']['status'] == 'in_review'

        approved = client.post(f"/api/review/{created['id']}/approve", json={})
        assert approved.status_code == 200
        assert approved.json()['task']['status'] == 'done'


def test_import_dependency_execution_order(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=_PrdImportWorkerAdapter())
    with TestClient(app) as client:
        preview = client.post('/api/import/prd/preview', json={'content': '- Step A\n- Step B'}).json()
        commit = client.post('/api/import/prd/commit', json={'job_id': preview['job_id']}).json()
        first_id, second_id = commit['created_task_ids']

        first_run = client.post(f'/api/tasks/{first_id}/run')
        assert first_run.status_code == 200

        blocked_second = client.post(f'/api/tasks/{second_id}/run')
        assert blocked_second.status_code == 400

        client.post(f'/api/review/{first_id}/approve', json={})
        second_run = client.post(f'/api/tasks/{second_id}/run')
        assert second_run.status_code == 200


def test_terminal_session_stays_off_board(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        session = client.post('/api/terminal/session', json={}).json()['session']
        assert session['id']
        tasks_before = client.get('/api/tasks').json()['tasks']
        assert tasks_before == []


def test_findings_loop_until_zero_open_then_done(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            '/api/tasks',
            json={
                'title': 'Loop task',
                'status': 'backlog',
                'approval_mode': 'auto_approve',
                'metadata': {
                    'scripted_findings': [
                        [{'severity': 'high', 'summary': 'Need fix'}],
                        [],
                    ]
                },
            },
        ).json()['task']
        run = client.post(f"/api/tasks/{task['id']}/run")
        assert run.status_code == 200
        assert run.json()['task']['status'] == 'done'
        assert run.json()['task']['retry_count'] >= 1


def test_request_changes_reopens_task_with_feedback(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post('/api/tasks', json={'title': 'Needs feedback', 'status': 'backlog', 'metadata': {'scripted_findings': [[]]}}).json()['task']
        client.post(f"/api/tasks/{task['id']}/run")

        changed = client.post(
            f"/api/review/{task['id']}/request-changes",
            json={'guidance': 'Please add integration tests'},
        )
        assert changed.status_code == 200
        body = changed.json()['task']
        assert body['status'] == 'queued'
        assert body['metadata']['requested_changes']['guidance'] == 'Please add integration tests'
        assert len(body['human_review_actions']) == 1
        assert body['human_review_actions'][0]['action'] == 'request_changes'
        assert body['human_review_actions'][0]['guidance'] == 'Please add integration tests'


def test_single_run_branch_commits_in_task_order(tmp_path: Path) -> None:
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.email', 'ci@example.com'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'CI'], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / 'seed.txt').write_text('seed\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'seed.txt'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'commit', '-m', 'seed'], cwd=tmp_path, check=True, capture_output=True)

    app = create_app(project_dir=tmp_path, worker_adapter=_FileWritingAdapter())
    with TestClient(app) as client:
        first = client.post('/api/tasks', json={'title': 'First', 'approval_mode': 'auto_approve'}).json()['task']
        second = client.post('/api/tasks', json={'title': 'Second', 'approval_mode': 'auto_approve'}).json()['task']

        client.post(f"/api/tasks/{first['id']}/run")
        client.post(f"/api/tasks/{second['id']}/run")

    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    assert branch.startswith('orchestrator-run-')

    messages = subprocess.run(['git', 'log', '--pretty=%s', '-n', '2'], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.splitlines()
    assert messages[0].startswith(f"task({second['id']})")
    assert messages[1].startswith(f"task({first['id']})")
