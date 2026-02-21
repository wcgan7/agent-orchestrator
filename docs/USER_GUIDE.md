# User Guide

## Overview

Agent Orchestrator is a local AI delivery control center with task lifecycle
management, execution orchestration, review gates, and worker routing.

Primary surfaces in the web app:
- `Board`
- `Planning`
- `Execution`
- `Workers`
- `Settings`
- `Create Work` drawer (`Create Task`, `Import PRD`, `Terminal`)

Board ordering defaults:
- Active columns prioritize urgency (`P0` first) with status-specific recency signals.
- `Done` shows newest completions first (based on most recent `updated_at`).

## Quick Start

Start backend:

```bash
python -m pip install -e ".[server]"
agent-orchestrator server --project-dir /absolute/path/to/your/repo
```

Start frontend:

```bash
npm --prefix web install
npm --prefix web run dev
```

Open:
- Backend: `http://localhost:8080`
- Frontend: `http://localhost:3000`

## Core Concepts

- Task: unit of work with lifecycle state, priority, type, and pipeline.
- Pipeline: ordered steps chosen by task type (for example docs, feature, bug).
- Run: one orchestrator execution attempt for a task.
- Review Queue: tasks waiting for human decision.
- HITL mode: collaboration style for worker behavior.
- Project commands: language-specific test/lint/typecheck/format commands
  injected into worker prompts.

## Task Lifecycle

Statuses:
- `backlog`
- `queued`
- `in_progress`
- `in_review`
- `blocked`
- `done`
- `cancelled`

Transition rules:
- `backlog -> queued|cancelled`
- `queued -> backlog|cancelled`
- `in_progress -> cancelled`
- `in_review -> done|blocked|cancelled`
- `blocked -> queued|in_review|cancelled`
- `cancelled -> backlog`

Constraint:
- A task cannot move to `queued` while any blocker is unresolved.

Blockers are resolved when dependent tasks are `done` or `cancelled`.

## Typical Workflows

### 1. Create and run a task

1. Open `Create Work` -> `Create Task`.
2. Enter title, description, priority, type, and optional metadata.
3. Move to `queued`.
4. Start from task actions or let orchestrator pick it from queue.
5. Track step progression in task detail and `Execution`.
6. Handle review decision in `Review Queue` when status becomes `in_review`.

### 2. Import PRD into executable tasks

1. Open `Create Work` -> `Import PRD`.
2. Paste PRD content and run preview.
3. Review preview graph (nodes/edges and warnings).
4. Commit import.
5. Monitor generated parent task (`initiative_plan`) and created children.

### 3. Use embedded terminal

1. Open `Create Work` -> `Terminal`.
2. Start or attach to a session.
3. Send input, resize as needed, stream logs.
4. Stop session with `TERM` (default) or `KILL`.

## Planning and Task Generation

Each task has a plan document with revisions:
- `latest_revision_id`
- `committed_revision_id`
- `revisions[]`

Supported actions:
- Queue refine job from feedback.
- Add manual revision.
- Commit a revision.
- Generate child tasks from `latest`, `committed`, explicit `revision`, or
  inline `override` text.

Generation validation rules:
- `source=revision` requires `revision_id`.
- `source=override` requires `plan_override`.
- `revision_id` and `plan_override` are rejected for incompatible sources.

## Review Flow

Review queue includes all tasks in `in_review`.

Actions:
- `Approve`: marks task `done`; if preserved branch metadata exists, merge is
  attempted first.
- `Request changes`: returns task to `queued` and sets retry target to
  `implement`.

Both actions can include optional human guidance and are recorded in task
history/timeline metadata.

## HITL and Collaboration

Collaboration APIs and UI support:
- Mode catalog (`/api/collaboration/modes`)
- Timeline (system events + human review actions + feedback/comments)
- Feedback records (add/dismiss)
- Threaded comments (add/resolve)

Timeline entries also surface normalized human-blocking issues when available.

## Settings and Worker Routing

Settings sections:
- `orchestrator`: concurrency, auto dependency analysis, max review attempts
- `agent_routing`: default role and task-type role mapping
- `defaults`: quality gate thresholds and default dependency policy
- `workers`: default provider/model, heartbeats, providers, step routing
- `project.commands`: language-specific command overrides

Worker provider types:
- `codex` (`command`, optional `model`, optional `reasoning_effort`)
- `claude` (`command`, optional `model`, optional `reasoning_effort`)
- `ollama` (optional `endpoint`, `model`, `temperature`, `num_ctx`)

Routing behavior:
- Step-level routing uses explicit `workers.routing` first.
- Falls back to `workers.default`.

## Project Commands

Define explicit commands workers should run during implement/verify steps.

```yaml
# .agent_orchestrator/config.yaml
project:
  commands:
    python:
      test: ".venv/bin/pytest -n auto --tb=short"
      lint: ".venv/bin/ruff check ."
      typecheck: ".venv/bin/mypy ."
      format: ".venv/bin/ruff format ."
    typescript:
      test: "npm test"
      lint: "npx eslint ."
      typecheck: "npx tsc --noEmit"
```

Rules:
- Language keys are normalized to lowercase.
- Each field is optional.
- Empty string in PATCH removes a specific command.
- Extra languages may be stored but are ignored unless detected in project.

Example API patch:

```bash
curl -X PATCH http://localhost:8080/api/settings \
  -H 'Content-Type: application/json' \
  -d '{
    "project": {
      "commands": {
        "python": {
          "test": ".venv/bin/pytest -n auto",
          "lint": ".venv/bin/ruff check ."
        }
      }
    }
  }'
```

## Runtime Storage

State root in each selected project:
- `.agent_orchestrator/tasks.yaml`
- `.agent_orchestrator/runs.yaml`
- `.agent_orchestrator/review_cycles.yaml`
- `.agent_orchestrator/agents.yaml`
- `.agent_orchestrator/terminal_sessions.yaml`
- `.agent_orchestrator/plan_revisions.yaml`
- `.agent_orchestrator/plan_refine_jobs.yaml`
- `.agent_orchestrator/events.jsonl`
- `.agent_orchestrator/config.yaml`

Legacy migration behavior:
- incompatible old state is archived to `.agent_orchestrator_legacy_<timestamp>/`

## Diagnostics and Troubleshooting

Health checks:
- `GET /healthz`
- `GET /readyz`

Context checks:
- `GET /` confirms active project and schema version.
- `GET /api/settings` confirms effective runtime configuration.
- `GET /api/workers/health` validates provider availability.

Task/run diagnostics:
- `GET /api/tasks/{task_id}/logs` for stdout/stderr and step history.
- `GET /api/tasks/{task_id}/diff` for latest commit file changes.
- `GET /api/collaboration/timeline/{task_id}` for review and blocker context.

## Additional References

- `README.md`
- `docs/API_REFERENCE.md`
- `docs/CLI_REFERENCE.md`
- `web/README.md`
- `example/README.md`
