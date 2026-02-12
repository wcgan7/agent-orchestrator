from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .server import create_app
from .v3.events import EventBus
from .v3.orchestrator import OrchestratorService
from .v3.storage import V3Container


def _resolve_project_dir(project_dir: Optional[str]) -> Path:
    return Path(project_dir).expanduser().resolve() if project_dir else Path.cwd().resolve()


def _ctx(project_dir: Optional[str]) -> tuple[V3Container, OrchestratorService]:
    container = V3Container(_resolve_project_dir(project_dir))
    bus = EventBus(container.events, container.project_id)
    orchestrator = OrchestratorService(container, bus)
    return container, orchestrator


def _project_pin(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    path = Path(args.path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        sys.stderr.write(f"Invalid path: {path}\n")
        return 1
    if not args.allow_non_git and not (path / '.git').exists():
        sys.stderr.write("Path must contain .git unless --allow-non-git is set\n")
        return 1
    cfg = container.config.load()
    pinned = [entry for entry in list(cfg.get('pinned_projects') or []) if str(entry.get('path')) != str(path)]
    project_id = args.project_id or f"pinned-{path.name}"
    pinned.append({'id': project_id, 'path': str(path)})
    cfg['pinned_projects'] = pinned
    container.config.save(cfg)
    sys.stdout.write(json.dumps({'id': project_id, 'path': str(path)}) + '\n')
    return 0


def _project_list(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    cfg = container.config.load()
    pinned = list(cfg.get('pinned_projects') or [])
    sys.stdout.write(json.dumps({'projects': pinned}, indent=2) + '\n')
    return 0


def _project_unpin(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    cfg = container.config.load()
    pinned = list(cfg.get('pinned_projects') or [])
    remaining = [entry for entry in pinned if entry.get('id') != args.project_id]
    cfg['pinned_projects'] = remaining
    container.config.save(cfg)
    sys.stdout.write(json.dumps({'removed': len(remaining) != len(pinned), 'project_id': args.project_id}) + '\n')
    return 0


def _task_create(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    from .v3.domain.models import Task

    task = Task(
        title=args.title,
        description=args.description or '',
        priority=args.priority,
        task_type=args.task_type,
        source='manual',
    )
    container.tasks.upsert(task)
    sys.stdout.write(json.dumps({'task': task.to_dict()}, indent=2) + '\n')
    return 0


def _task_list(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    tasks = container.tasks.list()
    if args.status:
        tasks = [task for task in tasks if task.status == args.status]
    sys.stdout.write(json.dumps({'tasks': [task.to_dict() for task in tasks]}, indent=2) + '\n')
    return 0


def _task_run(args: argparse.Namespace) -> int:
    _, orchestrator = _ctx(args.project_dir)
    try:
        task = orchestrator.run_task(args.task_id)
    except ValueError as exc:
        sys.stderr.write(str(exc) + '\n')
        return 1
    sys.stdout.write(json.dumps({'task': task.to_dict()}, indent=2) + '\n')
    return 0


def _quick_action(args: argparse.Namespace) -> int:
    container, _ = _ctx(args.project_dir)
    from .v3.domain.models import QuickActionRun, now_iso

    run = QuickActionRun(prompt=args.prompt, status='completed', started_at=now_iso(), finished_at=now_iso(), result_summary='Quick action executed')
    container.quick_actions.upsert(run)
    sys.stdout.write(json.dumps({'quick_action': run.to_dict()}, indent=2) + '\n')
    return 0


def _orchestrator_status(args: argparse.Namespace) -> int:
    _, orchestrator = _ctx(args.project_dir)
    sys.stdout.write(json.dumps(orchestrator.status(), indent=2) + '\n')
    return 0


def _orchestrator_control(args: argparse.Namespace) -> int:
    _, orchestrator = _ctx(args.project_dir)
    try:
        payload = orchestrator.control(args.action)
    except ValueError as exc:
        sys.stderr.write(str(exc) + '\n')
        return 1
    sys.stdout.write(json.dumps(payload, indent=2) + '\n')
    return 0


def _server(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("Install server extras: pip install 'feature-prd-runner[server]'\n")
        return 1

    app = create_app(project_dir=_resolve_project_dir(args.project_dir))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Feature PRD Runner v3 CLI (UI-first minimal)')
    parser.add_argument('--project-dir', default=None, help='Target project directory (default: current working directory)')
    subparsers = parser.add_subparsers(dest='command', required=True)

    server = subparsers.add_parser('server', help='Start the v3 web server')
    server.add_argument('--host', default='127.0.0.1')
    server.add_argument('--port', default=8000, type=int)
    server.add_argument('--reload', action='store_true')
    server.set_defaults(func=_server)

    project = subparsers.add_parser('project', help='Manage pinned projects')
    project_sub = project.add_subparsers(dest='project_cmd', required=True)
    ppin = project_sub.add_parser('pin', help='Pin a project path')
    ppin.add_argument('path')
    ppin.add_argument('--project-id', default=None)
    ppin.add_argument('--allow-non-git', action='store_true')
    ppin.set_defaults(func=_project_pin)
    plist = project_sub.add_parser('list', help='List pinned projects')
    plist.set_defaults(func=_project_list)
    punpin = project_sub.add_parser('unpin', help='Unpin a project by ID')
    punpin.add_argument('project_id')
    punpin.set_defaults(func=_project_unpin)

    task = subparsers.add_parser('task', help='Manage tasks')
    task_sub = task.add_subparsers(dest='task_cmd', required=True)
    tcreate = task_sub.add_parser('create', help='Create a task')
    tcreate.add_argument('title')
    tcreate.add_argument('--description', default='')
    tcreate.add_argument('--priority', default='P2', choices=['P0', 'P1', 'P2', 'P3'])
    tcreate.add_argument('--task-type', default='feature')
    tcreate.set_defaults(func=_task_create)
    tlist = task_sub.add_parser('list', help='List tasks')
    tlist.add_argument('--status', default=None)
    tlist.set_defaults(func=_task_list)
    trun = task_sub.add_parser('run', help='Run a task')
    trun.add_argument('task_id')
    trun.set_defaults(func=_task_run)

    quick = subparsers.add_parser('quick-action', help='Run an ephemeral quick action')
    quick.add_argument('prompt')
    quick.set_defaults(func=_quick_action)

    orchestrator = subparsers.add_parser('orchestrator', help='Inspect/control orchestrator')
    orch_sub = orchestrator.add_subparsers(dest='orchestrator_cmd', required=True)
    ostatus = orch_sub.add_parser('status', help='Show orchestrator status')
    ostatus.set_defaults(func=_orchestrator_status)
    ocontrol = orch_sub.add_parser('control', help='Control orchestrator')
    ocontrol.add_argument('action', choices=['pause', 'resume', 'drain', 'stop'])
    ocontrol.set_defaults(func=_orchestrator_control)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, 'func', None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args) or 0)
