# API Reference

Base path: `/api`

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

Task payload fields include optional `worker_model`:
- On `POST /tasks`, set `worker_model` to pin a model for that task.
- On `PATCH /tasks/{task_id}`, `worker_model` can be updated.

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
- `project`

`workers.providers.<name>` fields:
- codex: `type`, `command` (default `codex exec`), optional `model`, optional `reasoning_effort` (`low|medium|high`)
- claude: `type`, `command` (default `claude -p`), optional `model`, optional `reasoning_effort` (`low|medium|high`, mapped to Claude CLI `--effort`)
- ollama: `type`, `endpoint`, `model`, optional `temperature`, optional `num_ctx`

`workers` also supports:
- `default`: default worker provider name
- `default_model`: optional default model for codex workers (used when task has no `worker_model`)
- `routing`: per-step provider routing map

### Project Commands

The `project` section lets you declare per-language commands that workers use during
implementation and verification steps.

PATCH example — set Python commands:
```json
{
  "project": {
    "commands": {
      "python": {
        "test": ".venv/bin/pytest -n auto --tb=short",
        "lint": ".venv/bin/ruff check ."
      }
    }
  }
}
```

Fields per language: `test`, `lint`, `typecheck`, `format`. All optional.

Merge semantics:
- `null` / omitted field -> no change
- `""` (empty string) -> removes that command
- non-empty string -> sets the command

Language keys are normalized to lowercase on write. Only languages detected in the
project (via marker files like `pyproject.toml`, `tsconfig.json`, `go.mod`) are
injected into worker prompts — extra entries are stored but ignored at runtime.

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
