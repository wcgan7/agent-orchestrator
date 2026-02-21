# API Reference

Base path: `/api`

Project scoping:
- Most endpoints accept a `project_dir` query parameter.
- If omitted, the server default/current working directory is used.

## Root and Health (outside `/api`)

- `GET /` -> runtime metadata (`name`, `version`, `project`, `project_id`, `schema_version`)
- `GET /healthz`
- `GET /readyz`

## Projects

- `GET /api/projects`
  - Query: `include_non_git` (bool)
- `GET /api/projects/pinned`
- `POST /api/projects/pinned`
- `DELETE /api/projects/pinned/{project_id}`
- `GET /api/projects/browse`
  - Query: `path`, `include_hidden` (bool), `limit` (1-1000)

## Tasks

- `POST /api/tasks`
- `POST /api/tasks/classify-pipeline`
- `GET /api/tasks`
  - Query: `status`, `task_type`, `priority`
- `GET /api/tasks/board`
- `GET /api/tasks/execution-order`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/diff`
- `GET /api/tasks/{task_id}/logs`
- `PATCH /api/tasks/{task_id}`
- `POST /api/tasks/{task_id}/transition`
- `POST /api/tasks/{task_id}/run`
- `POST /api/tasks/{task_id}/retry`
- `POST /api/tasks/{task_id}/cancel`
- `POST /api/tasks/{task_id}/approve-gate`
- `POST /api/tasks/{task_id}/dependencies`
- `DELETE /api/tasks/{task_id}/dependencies/{dep_id}`
- `POST /api/tasks/analyze-dependencies`
- `POST /api/tasks/{task_id}/reset-dep-analysis`
- `GET /api/tasks/{task_id}/workdoc`
- `GET /api/tasks/{task_id}/plan`
- `POST /api/tasks/{task_id}/plan/refine`
- `GET /api/tasks/{task_id}/plan/jobs`
- `GET /api/tasks/{task_id}/plan/jobs/{job_id}`
- `POST /api/tasks/{task_id}/plan/commit`
- `POST /api/tasks/{task_id}/plan/revisions`
- `POST /api/tasks/{task_id}/generate-tasks`

Task payload fields include optional `worker_model`:
- On `POST /api/tasks`, set `worker_model` to pin a model for that task.
- On `PATCH /api/tasks/{task_id}`, `worker_model` can be updated.

### Iterative Planning Endpoints

`GET /api/tasks/{task_id}/plan` returns:
- `task_id`
- `latest_revision_id`
- `committed_revision_id`
- `revisions[]` (`id`, `source`, `parent_revision_id`, `step`, `feedback_note`, `provider`, `model`, `content`, `content_hash`, `status`, timestamps)
- `active_refine_job` when a refine job is in progress

Legacy compatibility fields are still included:
- `plans`
- `latest`

`POST /api/tasks/{task_id}/plan/refine` request:
```json
{
  "base_revision_id": "pr-abc123",
  "feedback": "Tighten rollout and risk controls",
  "instructions": "Keep auth and migration details",
  "priority": "normal"
}
```

`POST /api/tasks/{task_id}/plan/revisions` request:
```json
{
  "content": "Full revised plan text...",
  "parent_revision_id": "pr-abc123",
  "feedback_note": "manual edits before generate"
}
```

`POST /api/tasks/{task_id}/plan/commit` request:
```json
{
  "revision_id": "pr-abc123"
}
```

`POST /api/tasks/{task_id}/generate-tasks` request:
```json
{
  "source": "committed",
  "revision_id": "pr-abc123",
  "plan_override": "Manual plan text",
  "infer_deps": true
}
```

`source` values:
- `latest`
- `committed`
- `revision` (requires `revision_id`)
- `override` (requires `plan_override`)

## PRD Import

- `POST /api/import/prd/preview`
- `POST /api/import/prd/commit`
- `GET /api/import/{job_id}`

## Terminal

- `POST /api/terminal/session`
- `GET /api/terminal/session`
- `POST /api/terminal/session/{session_id}/input`
- `POST /api/terminal/session/{session_id}/resize`
- `POST /api/terminal/session/{session_id}/stop`
- `GET /api/terminal/session/{session_id}/logs`

## Review Queue

- `GET /api/review-queue`
- `POST /api/review/{task_id}/approve`
- `POST /api/review/{task_id}/request-changes`

## Orchestrator

- `GET /api/orchestrator/status`
- `POST /api/orchestrator/control` (`pause|resume|drain|stop`)

## Settings and Workers

- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/workers/health`
- `GET /api/workers/routing`

Top-level settings payload sections:
- `orchestrator`
- `agent_routing`
- `defaults`
- `workers`
- `project`

`workers.providers.<name>` fields:
- codex: `type`, `command` (default `codex exec`), optional `model`, optional `reasoning_effort` (`low|medium|high`)
- claude: `type`, `command` (default `claude -p`), optional `model`, optional `reasoning_effort` (`low|medium|high`, mapped to Claude CLI `--effort`)
- ollama: `type`, optional `endpoint`, optional `model`, optional `temperature`, optional `num_ctx`

`workers` also supports:
- `default`: default worker provider name
- `default_model`: optional default model for codex workers (used when task has no `worker_model`)
- `heartbeat_seconds`: provider heartbeat interval
- `heartbeat_grace_seconds`: timeout before a worker is marked stale
- `routing`: per-step provider routing map

### Project Commands

The `project` section lets you declare per-language commands workers use during implementation and verification.

PATCH example:
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

Fields per language: `test`, `lint`, `typecheck`, `format` (all optional).

Merge semantics:
- `null` / omitted field -> no change
- `""` (empty string) -> removes that command
- non-empty string -> sets the command

Language keys are normalized to lowercase. Only languages detected in the project are injected into worker prompts; extra entries are stored but ignored at runtime.

## Agents

- `GET /api/agents`
- `POST /api/agents/spawn`
- `POST /api/agents/{agent_id}/pause`
- `POST /api/agents/{agent_id}/resume`
- `POST /api/agents/{agent_id}/terminate`
- `DELETE /api/agents/{agent_id}`
- `POST /api/agents/{agent_id}/remove` (legacy-compatible alternative to DELETE)
- `GET /api/agents/types`

## Collaboration and Visibility

- `GET /api/metrics`
- `GET /api/phases`
- `GET /api/collaboration/modes`
- `GET /api/collaboration/presence`
- `GET /api/collaboration/timeline/{task_id}`
- `GET /api/collaboration/feedback/{task_id}`
- `POST /api/collaboration/feedback`
- `POST /api/collaboration/feedback/{feedback_id}/dismiss`
- `GET /api/collaboration/comments/{task_id}`
- `POST /api/collaboration/comments`
- `POST /api/collaboration/comments/{comment_id}/resolve`

## WebSocket

Endpoint: `/ws`

Supported channels:
- `tasks`
- `queue`
- `agents`
- `review`
- `terminal`
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
