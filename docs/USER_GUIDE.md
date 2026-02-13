# User Guide

## What This Tool Does

Feature PRD Runner is an orchestration-first AI engineering control center.
The primary unit is a `Task`. You can create tasks directly, import them from a PRD,
or run one-off quick actions.

Primary UI surfaces:
- Board
- Execution
- Review Queue
- Agents
- Settings

## Quick Start

Install backend dependencies and run server:

```bash
python -m pip install -e ".[server]"
feature-prd-runner server --project-dir /absolute/path/to/your/repo
```

Run the web dashboard:

```bash
npm --prefix web install
npm --prefix web run dev
```

Open:
- Backend: `http://localhost:8080`
- Frontend: `http://localhost:3000`

## Core Concepts

- Task: persistent, board-backed work item.
- Import Job: PRD parse/preview/commit workflow that creates tasks + dependencies.
- Quick Action: ephemeral one-off run; can be promoted into a task.
- Review Queue: tasks waiting for human approval (`in_review`).
- Agent: worker process entry in orchestrator pool.

## Task Lifecycle

Statuses:
- `backlog`
- `ready`
- `in_progress`
- `in_review`
- `done`
- `blocked`
- `cancelled`

Default policy:
- New tasks start in `backlog`.
- Approval mode defaults to `human_review`.
- Quality gate defaults to zero open findings for `critical/high/medium/low`.

Dependency rules:
- Tasks with unresolved blockers cannot run.
- A blocker is considered resolved when its status is `done` or `cancelled`.

## Common Workflows

### 1. Create and Run a Task

1. Open `Create Work` -> `Create Task`.
2. Fill title/description/type/priority.
3. Move task to `ready` (or use run actions from task detail).
4. Monitor progress in `Execution` and task detail.
5. Approve in `Review Queue` if task reaches `in_review`.

### 2. Import a PRD

1. Open `Create Work` -> `Import PRD`.
2. Paste PRD text and click preview.
3. Validate generated nodes/edges.
4. Commit import job to create board tasks.
5. Track created tasks from board and import job details.

### 3. Run a Quick Action

1. Open `Create Work` -> `Quick Action`.
2. Submit prompt.
3. Review quick action result in detail panel.
4. Optionally click promote to create a board task.

## Project Management

In `Settings`:
- Pin/unpin repositories.
- Browse filesystem directories.
- Allow non-git directories explicitly when needed.

All API/UI operations target the currently selected project.

## Settings Reference

Config sections exposed in UI/API:

- `orchestrator`
- `agent_routing`
- `defaults.quality_gate`
- `workers`

Examples:
- Control concurrency and max review attempts.
- Route task types to agent roles.
- Override worker/provider by role or route.
- Tune quality gate thresholds.

## Realtime Behavior

WebSocket endpoint: `/ws`

Used channels:
- `tasks`
- `queue`
- `agents`
- `review`
- `quick_actions`
- `notifications`
- `system`

The UI subscribes and auto-refreshes mounted surfaces when relevant events arrive.

## Data Storage and Backups

State root:
- `.prd_runner/v3/`

Key files:
- `tasks.yaml`
- `runs.yaml`
- `review_cycles.yaml`
- `agents.yaml`
- `quick_actions.yaml`
- `events.jsonl`
- `config.yaml`

If legacy state exists, it is archived automatically to:
- `.prd_runner_legacy_<timestamp>/`

## Troubleshooting

- Check server health:
  - `GET /healthz`
  - `GET /readyz`
- Verify selected project:
  - `GET /` (returns project + project_id)
- If UI appears stale, confirm `/ws` connectivity and project selection.
- If a task won’t run, inspect blockers and status (`ready` required).
