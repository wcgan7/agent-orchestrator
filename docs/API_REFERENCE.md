# API Reference

Base path: `/api`

This document describes the HTTP and WebSocket contract exposed by the local
Agent Orchestrator server (`agent-orchestrator server`).

## Conventions

- Most endpoints accept `project_dir` as an optional query parameter.
- If `project_dir` is omitted, the server uses its startup project (or current
  working directory).
- JSON responses use object envelopes (for example `{ "task": ... }` or
  `{ "tasks": [...] }`).
- Error responses are standard FastAPI errors with HTTP status and `detail`.

## Root and Health Endpoints

These routes are outside `/api`.

### `GET /`
Returns runtime metadata for the active project context.

Response fields:
- `name`
- `version`
- `project`
- `project_id`
- `schema_version`

### `GET /healthz`
Liveness endpoint.

### `GET /readyz`
Readiness endpoint with project and orchestrator cache details.

## Projects

### `GET /api/projects`
List discovered and pinned projects.

Query:
- `include_non_git` (boolean, default `false`)

Response:
- `projects[]` with `id`, `path`, `source` (`discovered|pinned`), `is_git`

### `GET /api/projects/pinned`
Returns pinned projects from config.

Response:
- `items[]`

### `POST /api/projects/pinned`
Pin a project path.

Request body:
- `path` (required)
- `project_id` (optional)
- `allow_non_git` (optional, default `false`)

Notes:
- Returns `400` if path is invalid or non-git without `allow_non_git=true`.

### `DELETE /api/projects/pinned/{project_id}`
Remove a pinned project.

Response:
- `removed` (`true` if an entry was deleted)

### `GET /api/projects/browse`
Browse directories for project selection.

Query:
- `path` (optional; defaults to current user's home)
- `include_hidden` (boolean, default `false`)
- `limit` (integer, default `200`, range `1..1000`)

Response:
- `path`, `parent`, `current_is_git`
- `directories[]` with `name`, `path`, `is_git`
- `truncated` (`true` if results hit `limit`)

## Tasks

### `POST /api/tasks`
Create a task.

Request fields include:
- `title` (required)
- `description`, `task_type`, `priority`, `status`
- `labels[]`, `blocked_by[]`, `parent_id`
- `pipeline_template[]` (optional override)
- `approval_mode` (`human_review|auto_approve`)
- `hitl_mode`
- `dependency_policy` (`permissive|prudent|strict`)
- `source`, `worker_model`
- `metadata` (object)
- `project_commands` (per-task command overrides)
- optional classifier fields used by `task_type="auto"`

Behavior:
- `status` is only honored for `backlog` and `queued` at create time.
- `task_type="auto"` requires a high-confidence classifier result.
- If dependency policy is invalid/missing, server falls back to settings default,
  then `prudent`.

### `POST /api/tasks/classify-pipeline`
Classify free-form task text into a pipeline.

Request:
- `title`
- `description` (optional)
- `metadata` (optional)

Response:
- `pipeline_id`
- `task_type`
- `confidence` (`high|low`)
- `reason`
- `allowed_pipelines[]`

### `GET /api/tasks`
List tasks.

Query filters:
- `status`
- `task_type`
- `priority`

Response:
- `tasks[]`
- `total`

### `GET /api/tasks/board`
Returns board columns keyed by status.

Columns:
- `backlog`, `queued`, `in_progress`, `in_review`, `blocked`, `done`, `cancelled`

### `GET /api/tasks/execution-order`
Returns dependency-aware batches for non-terminal tasks.

Response:
- `batches` (list of task-id lists)
- `completed` (recently completed/cancelled task IDs)

### `GET /api/tasks/{task_id}`
Fetch one task. Returns `404` if not found.

Additive timing fields in task payload:
- `timing_summary.total_completed_seconds`: Sum of completed run durations in seconds.
- `timing_summary.active_run_started_at`: ISO timestamp of current in-progress run start, when present.
- `timing_summary.is_running`: `true` when an active run is in progress.
- `timing_summary.first_started_at`: Earliest valid run start timestamp in run history.
- `timing_summary.last_finished_at`: Latest valid run finish timestamp in run history.

### `PATCH /api/tasks/{task_id}`
Patch mutable fields.

Important constraints:
- Status changes are rejected (`400`).
- Use transition/retry/cancel/review actions for status updates.
- Empty `project_commands` clears per-task command overrides.

### `POST /api/tasks/{task_id}/transition`
Manual status transition with guardrails.

Valid transitions:
- `backlog -> queued|cancelled`
- `queued -> backlog|cancelled`
- `in_progress -> cancelled`
- `in_review -> done|blocked|cancelled`
- `blocked -> queued|in_review|cancelled`
- `cancelled -> backlog`

Notes:
- Transitioning to `queued` requires all blockers resolved (`done|cancelled`).

### `POST /api/tasks/{task_id}/run`
Queue/execute a task through the orchestrator.

### `POST /api/tasks/{task_id}/retry`
Reset task for another run.

Request (optional):
- `guidance`
- `start_from_step`

Behavior:
- Clears `error` and `pending_gate`.
- Increments `retry_count`.
- Stores retry guidance metadata/history when provided.

### `POST /api/tasks/{task_id}/cancel`
Set task status to `cancelled`.

### `POST /api/tasks/{task_id}/approve-gate`
Clear a pending human gate.

Request:
- `gate` (optional; if provided must match `task.pending_gate`)

### `POST /api/tasks/{task_id}/dependencies`
Add dependency edge.

Request:
- `depends_on` (task id)

### `DELETE /api/tasks/{task_id}/dependencies/{dep_id}`
Remove dependency edge.

### `POST /api/tasks/analyze-dependencies`
Run dependency inference across tasks and return inferred edges.

### `POST /api/tasks/{task_id}/reset-dep-analysis`
Remove inferred dependency metadata and inferred blocker links for a task.

### `GET /api/tasks/{task_id}/workdoc`
Return task work document payload.

### `GET /api/tasks/{task_id}/plan`
Return iterative plan document, including revisions and active refine job.

### `POST /api/tasks/{task_id}/plan/refine`
Queue async refinement of a plan revision.

Request:
- `feedback` (required)
- `base_revision_id` (optional)
- `instructions` (optional)
- `priority` (`normal|high`)

### `GET /api/tasks/{task_id}/plan/jobs`
List refine jobs for a task.

### `GET /api/tasks/{task_id}/plan/jobs/{job_id}`
Get one refine job.

### `POST /api/tasks/{task_id}/plan/commit`
Commit a chosen revision.

Request:
- `revision_id`

### `POST /api/tasks/{task_id}/plan/revisions`
Create a manual revision.

Request:
- `content` (required)
- `parent_revision_id` (optional)
- `feedback_note` (optional)

### `POST /api/tasks/{task_id}/generate-tasks`
Generate child tasks from plan content.

Request:
- `source`: `latest|committed|revision|override`
- `revision_id` (required when `source=revision`)
- `plan_override` (required when `source=override`)
- `infer_deps` (default `true`)

Validation:
- `revision_id` is only valid with `source=revision`.
- `plan_override` is only valid with `source=override`.

### `GET /api/tasks/{task_id}/diff`
Return latest commit diff/stat for the task.

Response:
- `commit` (or `null`)
- `files[]` (`path`, `changes`)
- `diff` (full unified diff)
- `stat` (git stat text)

### `GET /api/tasks/{task_id}/logs`
Read current or historical step logs.

Query:
- `max_chars` (`200..2000000`, default `12000`)
- `stdout_offset`, `stderr_offset` (for incremental reads)
- `backfill` (align chunk to line starts)
- `stdout_read_to`, `stderr_read_to` (optional byte limits)
- `step` (optional, fetch logs for a specific step attempt)

Response includes:
- `mode` (`active|last|history|none`)
- `stdout`, `stderr`
- offsets and chunk metadata
- `available_steps[]`, `step_execution_counts`
- `progress` JSON payload when present

## PRD Import

### `POST /api/import/prd/preview`
Parse PRD text and return a preview graph.

Request:
- `content` (required)
- `title` (optional)
- `default_priority` (default `P2`)

Response:
- `job_id`
- `preview.nodes[]`, `preview.edges[]`
- parsing metadata (`chunk_count`, `strategy`, `ambiguity_warnings`)

### `POST /api/import/prd/commit`
Commit previewed import and create tasks via initiative pipeline.

Request:
- `job_id`

Response:
- `job_id`
- `created_task_ids[]`
- `parent_task_id`

### `GET /api/import/{job_id}`
Fetch stored import job state.

## Review and Collaboration

### `GET /api/review-queue`
List tasks in `in_review`.

### `POST /api/review/{task_id}/approve`
Approve review; marks task `done` (and merges preserved branch when applicable).

Body:
- `guidance` (optional)

### `POST /api/review/{task_id}/request-changes`
Send task back to `queued` with review guidance.

Body:
- `guidance` (optional)

### `GET /api/collaboration/modes`
Returns available HITL mode definitions.

### `GET /api/collaboration/presence`
Presence feed (currently empty list).

### `GET /api/collaboration/timeline/{task_id}`
Task timeline combining status, events, feedback, comments, and human blockers.

### `GET /api/collaboration/feedback/{task_id}`
List feedback records for a task.

### `POST /api/collaboration/feedback`
Add a feedback item.

Required fields:
- `task_id`
- `summary`

Optional fields:
- `feedback_type`, `priority`, `details`, `target_file`

### `POST /api/collaboration/feedback/{feedback_id}/dismiss`
Mark feedback as addressed.

### `GET /api/collaboration/comments/{task_id}`
List comments for a task.

Query:
- `file_path` (optional exact match filter)

### `POST /api/collaboration/comments`
Create a threaded file comment.

Required fields:
- `task_id`, `file_path`, `body`

Optional fields:
- `line_number`, `line_type`, `parent_id`

### `POST /api/collaboration/comments/{comment_id}/resolve`
Mark a comment resolved.

## Terminal

### `POST /api/terminal/session`
Start a PTY session for the active project.

Request:
- `cols` (default `120`, range `2..500`)
- `rows` (default `36`, range `2..300`)
- `shell` (optional)

### `GET /api/terminal/session`
Get active session for project, if any.

### `POST /api/terminal/session/{session_id}/input`
Write UTF-8 input to session.

Request:
- `data`

### `POST /api/terminal/session/{session_id}/resize`
Resize PTY.

Request:
- `cols` (`2..500`)
- `rows` (`2..300`)

### `POST /api/terminal/session/{session_id}/stop`
Stop session with signal.

Request:
- `signal`: `TERM|KILL` (default `TERM`)

### `GET /api/terminal/session/{session_id}/logs`
Read terminal output.

Query:
- `offset` (default `0`)
- `max_bytes` (default `65536`)

Response:
- `output`
- `offset` (next cursor)
- `status`
- `finished_at`

## Orchestrator and Runtime Metrics

### `GET /api/orchestrator/status`
Return orchestrator state (queue, in-progress, control mode, etc.).

### `POST /api/orchestrator/control`
Control orchestrator state.

Request:
- `action`: `pause|resume|drain|stop`

### `GET /api/metrics`
Aggregated runtime metrics (API calls, wall time, queue depth, progress counters).

### `GET /api/phases`
Task progress snapshot used for execution phase visualization.

## Settings and Workers

### `GET /api/settings`
Read normalized runtime settings payload.

Top-level sections:
- `orchestrator`
- `agent_routing`
- `defaults`
- `workers`
- `project`

### `PATCH /api/settings`
Patch one or more settings sections.

Merge semantics:
- Omitted section: unchanged.
- Included section: shallow-merged by section rules.
- In `project.commands`, empty string removes a command.

Workers settings notes:
- Provider types: `codex`, `claude`, `ollama`.
- `workers.default` falls back to `codex` if invalid.
- `heartbeat_grace_seconds` is clamped to be at least
  `heartbeat_seconds`.

### `GET /api/workers/health`
Health check results for configured and common known providers.

### `GET /api/workers/routing`
Resolved step-to-provider routing table.

## Agents

### `GET /api/agents`
List agent records.

### `GET /api/agents/types`
List built-in agent role types and their capabilities.

Query parameters:
- `project_dir` (optional): project path used to resolve `task_type_affinity`
  from active `agent_routing.task_type_roles` settings.

Response:
- `types`: array of role descriptors.
  - `role`: machine role name.
  - `display_name`: human-friendly role label.
  - `description`: short role description.
  - `task_type_affinity`: sorted task types mapped to this role in settings.
  - `allowed_steps`: workflow steps the role can execute (`plan`,
    `implement`, `verify`, `review`).
  - `limits`: current runtime limits object.
    - `max_tokens`
    - `max_time_seconds`
    - `max_cost_usd`

### `POST /api/agents/spawn`
Create an agent record.

Request:
- `role` (default `general`)
- `capacity` (default `1`)
- `override_provider` (optional)

### `POST /api/agents/{agent_id}/pause`
Pause an agent.

### `POST /api/agents/{agent_id}/resume`
Resume an agent.

### `POST /api/agents/{agent_id}/terminate`
Mark an agent terminated.

### `DELETE /api/agents/{agent_id}`
Remove agent record.

### `POST /api/agents/{agent_id}/remove`
Legacy-compatible remove endpoint (same behavior as `DELETE`).

## WebSocket

Endpoint: `/ws`

Channels used by UI:
- `tasks`
- `queue`
- `agents`
- `review`
- `terminal`
- `notifications`
- `system`

Event envelope fields:
- `id`
- `ts`
- `channel`
- `type`
- `entity_id`
- `payload`
- `project_id`
