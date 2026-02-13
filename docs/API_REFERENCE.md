# API Reference (v3)

Base path: `/api/v3`

Project scoping:
- Most endpoints accept `project_dir` query parameter.
- If omitted, server default/current working directory is used.

## Health and Root

- `GET /` -> runtime metadata (`project`, `project_id`, `schema_version`)
- `GET /healthz`
- `GET /readyz`

## Projects

- `GET /projects`
- `GET /projects/pinned`
- `POST /projects/pinned`
- `DELETE /projects/pinned/{project_id}`
- `GET /projects/browse`

## Tasks

- `POST /tasks`
- `GET /tasks`
- `GET /tasks/board`
- `GET /tasks/execution-order`
- `GET /tasks/{task_id}`
- `PATCH /tasks/{task_id}`
- `POST /tasks/{task_id}/transition`
- `POST /tasks/{task_id}/run`
- `POST /tasks/{task_id}/retry`
- `POST /tasks/{task_id}/cancel`
- `POST /tasks/{task_id}/approve-gate`
- `POST /tasks/{task_id}/dependencies`
- `DELETE /tasks/{task_id}/dependencies/{dep_id}`
- `POST /tasks/analyze-dependencies`
- `POST /tasks/{task_id}/reset-dep-analysis`

## PRD Import

- `POST /import/prd/preview`
- `POST /import/prd/commit`
- `GET /import/{job_id}`

## Quick Actions

- `POST /quick-actions`
- `GET /quick-actions`
- `GET /quick-actions/{quick_action_id}`
- `POST /quick-actions/{quick_action_id}/promote`

## Review Queue

- `GET /review-queue`
- `POST /review/{task_id}/approve`
- `POST /review/{task_id}/request-changes`

## Orchestrator

- `GET /orchestrator/status`
- `POST /orchestrator/control` (`pause|resume|drain|stop`)

## Settings

- `GET /settings`
- `PATCH /settings`

Top-level settings payload sections:
- `orchestrator`
- `agent_routing`
- `defaults`
- `workers`

## Agents

- `GET /agents`
- `POST /agents/spawn`
- `POST /agents/{agent_id}/pause`
- `POST /agents/{agent_id}/resume`
- `POST /agents/{agent_id}/terminate`
- `GET /agents/types`

## Collaboration and Visibility

- `GET /metrics`
- `GET /phases`
- `GET /collaboration/modes`
- `GET /collaboration/presence`
- `GET /collaboration/timeline/{task_id}`
- `GET /collaboration/feedback/{task_id}`
- `POST /collaboration/feedback`
- `POST /collaboration/feedback/{feedback_id}/dismiss`
- `GET /collaboration/comments/{task_id}`
- `POST /collaboration/comments`
- `POST /collaboration/comments/{comment_id}/resolve`

## WebSocket

Endpoint: `/ws`

Supported channels:
- `tasks`
- `queue`
- `agents`
- `review`
- `quick_actions`
- `notifications`
- `system`

Event envelope:
- `id`
- `ts`
- `channel`
- `type`
- `entity_id`
- `payload`
- `project_id`
