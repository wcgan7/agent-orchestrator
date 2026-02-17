# Agent Orchestrator

Agent Orchestrator is a local orchestration control center for AI-assisted software delivery.
It gives you a task board, execution controls, review gates, and agent operations in one place.

Orchestrator autonomously plans and executes work across repositories — parallelizing independent changes, sequencing dependencies, and resolving conflicts as delivery progress.
Execution runs under enforced coding standards and a continuous review-and-fix cycle, producing resilient, merge-ready results instead of fragile one-pass output.

![Agent Orchestrator Dashboard](web/public/homepage-screenshot.png)

## What You Can Do

- Manage a full task lifecycle on a board (`backlog` -> `ready` -> `in_progress` -> `in_review` -> `done`).
- Import PRDs into executable task graphs with dependency edges.
- Draft, refine, and commit task plans with full revision lineage before task generation.
- Run Quick Actions for one-off work and optionally promote results into tasks.
- Control orchestrator execution (`pause`, `resume`, `drain`, `stop`).
- Manage agent pool capacity and role/provider routing.
- Approve or request changes from a dedicated review queue.
- Observe realtime updates across board, execution, and task detail via WebSocket.

## Quick Start (2 Minutes)

### 1. Start the backend

```bash
python -m pip install -e ".[server]"
agent-orchestrator server --project-dir /absolute/path/to/your/repo
```

Backend URL:
- `http://localhost:8080`

### 2. Start the web UI

```bash
npm --prefix web install
npm --prefix web run dev
```

Frontend URL:
- `http://localhost:3000`

## Core Workflows

### Create and run a task

1. Open `Create Work` -> `Create Task`.
2. Fill task fields (title, type, priority, description).
3. Transition to `ready` or run from task detail.
4. Track progress in `Execution`.
5. Approve from `Review Queue` when it reaches `in_review`.

### Import a PRD into tasks

1. Open `Create Work` -> `Import PRD`.
2. Paste PRD content and preview generated tasks/dependencies.
3. Commit the import job.
4. Review and execute created tasks from the board.

### Run a quick action

1. Open `Create Work` -> `Quick Action`.
2. Submit prompt/command intent.
3. Inspect result details.
4. Promote to task if you want board-tracked follow-up work.

## API and CLI

- REST/WebSocket reference: `docs/API_REFERENCE.md`
- CLI reference: `docs/CLI_REFERENCE.md`
- End-to-end usage guide: `docs/USER_GUIDE.md`

API base path:
- `/api`

WebSocket endpoint:
- `/ws`

## Configuration and Runtime Data

Runtime state is stored in the selected project directory:
- `.agent_orchestrator/tasks.yaml`
- `.agent_orchestrator/runs.yaml`
- `.agent_orchestrator/review_cycles.yaml`
- `.agent_orchestrator/agents.yaml`
- `.agent_orchestrator/quick_actions.yaml`
- `.agent_orchestrator/plan_revisions.yaml`
- `.agent_orchestrator/plan_refine_jobs.yaml`
- `.agent_orchestrator/events.jsonl`
- `.agent_orchestrator/config.yaml`

Primary configurable areas:
- `orchestrator` (concurrency, auto deps, review attempts)
- `agent_routing` (default role, task-type role routing, provider overrides)
- `defaults.quality_gate`
- `workers` (default provider, routing, providers)
- `project.commands` (per-language test, lint, typecheck, format commands)

Claude provider example:
```json
{
  "workers": {
    "default": "claude",
    "providers": {
      "claude": {
        "type": "claude",
        "command": "claude -p",
        "model": "sonnet",
        "reasoning_effort": "medium"
      }
    }
  }
}
```

Notes:
- Claude CLI must be installed and authenticated locally.
- Reasoning effort flags are only passed when supported by your installed CLI version.

## Verify Locally

```bash
# Backend tests
pytest

# Optional integration tests (skipped by default and in CI)
AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 pytest tests/test_integration_worker_model_fallback.py
AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 pytest tests/test_integration_claude_provider.py

# Frontend checks
npm --prefix web run check

# Frontend smoke e2e
npm --prefix web run e2e:smoke
```

## Documentation

- `docs/README.md`: documentation index
- `docs/USER_GUIDE.md`: complete user guide
- `docs/API_REFERENCE.md`: endpoint and websocket reference
- `docs/CLI_REFERENCE.md`: CLI commands and options
- `web/README.md`: frontend-specific setup and test workflow
- `example/README.md`: sample project walkthrough

## License

MIT
