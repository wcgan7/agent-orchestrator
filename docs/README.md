# Docs Index (v3)

## Core

- `ORCHESTRATOR_FIRST_REVAMP_PLAN.md`: execution plan and implementation log.

## Operating Model

- Orchestrator-first control plane.
- Task-centric execution lifecycle.
- Human-review default with strict quality gate.

## API Reference (high-level)

Base path: `/api/v3`

- `/projects`, `/projects/pinned`
- `/tasks`, `/tasks/board`, `/tasks/{id}` and task actions
- `/import/prd/preview`, `/import/prd/commit`, `/import/{job_id}`
- `/quick-actions`, `/quick-actions/{id}/promote`
- `/review-queue`, `/review/{task_id}/approve`, `/review/{task_id}/request-changes`
- `/orchestrator/status`, `/orchestrator/control`
- `/agents` and agent control endpoints

WebSocket: `/ws` (v3 channels only)

## State Layout

- `.prd_runner/v3/tasks.yaml`
- `.prd_runner/v3/runs.yaml`
- `.prd_runner/v3/review_cycles.yaml`
- `.prd_runner/v3/agents.yaml`
- `.prd_runner/v3/quick_actions.yaml`
- `.prd_runner/v3/events.jsonl`
- `.prd_runner/v3/config.yaml`

Legacy state is archived on first v3 boot; no auto-migration.
