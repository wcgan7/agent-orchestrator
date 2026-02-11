# Feature PRD Runner (Orchestrator-First v3)

Feature PRD Runner is an orchestration-first AI engineering control center.
`Task` is the primary unit. PRD import is one intake mode, not the core model.

## Product Model

Primary intake:
- `Create Task` (persistent, board-backed)
- `Import PRD` (preview graph, commit to board)
- `Quick Action` (ephemeral by default, promote explicitly)

Primary surfaces:
- `Board` (default)
- `Execution`
- `Review Queue`
- `Agents`
- `Settings`

## Runtime and State

- Backend: FastAPI (`/api/v3/*` only)
- Frontend: React/Vite, route-driven app
- State root: `.prd_runner/v3/`
- Storage: file-backed repositories with lock + atomic writes

On first v3 launch, legacy `.prd_runner` state is archived to:

- `.prd_runner_legacy_<timestamp>/`

No automatic migration of legacy tasks/runs is performed.

## API

Canonical API base: `/api/v3`

Domains:
- projects
- tasks
- import/prd
- quick-actions
- review queue/actions
- orchestrator status/control
- agents

WebSocket endpoint: `/ws` with v3 channels only:
- `tasks`, `queue`, `agents`, `review`, `quick_actions`, `notifications`, `system`

## CLI (UI-First Minimal)

```bash
feature-prd-runner server
feature-prd-runner project pin /abs/path/to/repo
feature-prd-runner project list
feature-prd-runner project unpin <project_id>
feature-prd-runner task create "Implement X"
feature-prd-runner task list
feature-prd-runner task run <task_id>
feature-prd-runner quick-action "summarize recent errors"
feature-prd-runner orchestrator status
feature-prd-runner orchestrator control pause
```

## Local Development

Backend:

```bash
python -m pip install -e ".[server,test]"
feature-prd-runner server --project-dir /path/to/workspace
```

Frontend:

```bash
npm --prefix web install
npm --prefix web run dev
```

## Quality Defaults

- Default approval mode: `human_review`
- Default quality gate: zero open findings across all severities (`critical/high/medium/low`)
- Auto-approve is opt-in per task

## License

MIT
