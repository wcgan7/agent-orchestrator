# Unified Task Orchestration Spec

> **Status (2026-02-11):** Implemented v2 baseline spec for the current codebase.
> **Forward plan:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

Status: Implemented
Owner: Core platform (backend + web)
Last updated: 2026-02-11

## 1) Goal

Make `Task` the primary unit across planning, execution, and tracking.

- `Create Task` and `Import PRD` both produce board-backed tasks.
- Orchestrator consumes `READY` tasks directly from the task engine.
- `Quick Action` is the only default entry point that does not create board tasks.

## 2) Product Entry Points

### 2.1 Create Task (primary)

Purpose: Add persistent work to the board.

Output:
- One `Task` in `backlog` or `ready`.

Core fields:
- `title` (required)
- `task_type` (required)
- `priority` (required)
- `description`, `acceptance_criteria`, `context_files`, `blocked_by`, `labels`, `assignee` (optional)

### 2.2 Import PRD (primary)

Purpose: Convert a PRD into an executable task graph.

Output:
- `ImportJob`
- N tasks
- Dependency edges

Controls:
- `granularity`: `coarse | balanced | fine`
- `auto_ready`: bool
- `max_parallelism_hint`: integer

UX requirement:
- Preview before commit, including tasks, dependencies, blocked roots, and estimated waves.

### 2.3 Quick Action (ephemeral)

Purpose: Run a one-off command/prompt without polluting the board.

Output:
- `QuickRun` record only.

Optional:
- `Promote to task` toggle/action.

Rule:
- Default behavior does not create task-board entries.

## 3) Domain Model

### 3.1 Task

Key fields:
- Identity: `id`, `title`, `description`
- Classification: `task_type`, `priority`, `status`, `labels`
- Graph: `blocked_by[]`, `blocks[]`, `parent_id`, `children_ids[]`
- Definition: `acceptance_criteria[]`, `context_files[]`, `pipeline_template`
- Runtime: `current_step`, `current_agent_id`, `run_ids[]`, `retry_count`, `error`
- Provenance: `source` (`manual | prd_import | generated | promoted_quick_action`)

### 3.2 QuickRun

Key fields:
- `id`, `prompt`, `status`, `started_at`, `finished_at`, `logs_ref`, `result_summary`
- `promoted_task_id` (nullable)

### 3.3 ImportJob

Key fields:
- `id`, `source`, `status`, `task_count`, `edge_count`, `warnings[]`, `created_task_ids[]`

## 4) State Machine

States:
- `backlog`, `ready`, `in_progress`, `in_review`, `done`, `blocked`, `cancelled`

Allowed transitions:
- `backlog -> ready | cancelled`
- `ready -> in_progress | blocked | backlog | cancelled`
- `in_progress -> in_review | blocked | ready | cancelled`
- `in_review -> done | in_progress | blocked`
- `blocked -> ready | backlog | cancelled`
- `done -> ready` (re-open)
- `cancelled -> backlog` (restore)

Config override:
- If `FEATURE_PRD_AUTO_APPROVE_REVIEW=true`, allow `in_progress -> done`.

Dependency guard:
- A task cannot enter `ready` or `in_progress` while blockers are unresolved.

Claiming guard:
- `in_progress` must be set by orchestrator claim lock, unless an explicit force-run path is used.

## 5) Orchestrator Contract

Scheduler:
- Pull candidates from `READY` with dependencies satisfied.
- Sort by priority then creation time.
- Respect worker and repo concurrency limits.
- Atomically claim: `ready -> in_progress`.

Execution:
- Resolve pipeline by `task_type`.
- Feature default pipeline:
  - `plan -> plan_impl -> implement -> verify -> review -> commit`
- Persist step-level events and task runtime metadata.

Completion:
- Success: advance to `in_review` or `done` depending on HITL mode.
- Failure: retry within policy; otherwise `blocked` with structured reason.
- Done: unblock dependents and re-evaluate ready queue.

## 6) API Shape (v2 target)

Tasks:
- `POST /api/v2/tasks`
- `GET /api/v2/tasks`
- `GET /api/v2/tasks/board`
- `POST /api/v2/tasks/{id}/transition`
- `POST /api/v2/tasks/{id}/dependencies`
- `DELETE /api/v2/tasks/{id}/dependencies/{dep_id}`
- `POST /api/v2/tasks/{id}/run` (new explicit run action)
- `POST /api/v2/tasks/{id}/retry` (new)
- `POST /api/v2/tasks/{id}/cancel` (new)

PRD import:
- `POST /api/v2/import/prd/preview`
- `POST /api/v2/import/prd/commit`
- `GET /api/v2/import/{job_id}`

Quick Action:
- `POST /api/v2/quick-runs`
- `GET /api/v2/quick-runs/{id}`
- `POST /api/v2/quick-runs/{id}/promote`

Events:
- `task.created`, `task.updated`, `task.claimed`, `task.step_started`, `task.step_completed`, `task.blocked`, `task.completed`
- `quick_run.started`, `quick_run.completed`, `quick_run.failed`, `quick_run.promoted`

Implemented event APIs:
- `GET /api/v2/tasks/events/recent`
- `GET /api/v2/tasks/{id}/events`
- `GET /api/v2/quick-runs/events/recent`
- `GET /api/v2/quick-runs/{id}/events`

## 7) UX Rules

- Tasks is default landing page.
- Execution shows queue/workers/health, not an alternate task system.
- Quick Action is clearly labeled as one-off.
- Blocked tasks display blocker reasons and unblock guidance.
- Invalid manual status transitions are prevented with explicit error reason.

## 8) Non-Functional Requirements

- Atomic claim and transitions.
- Idempotent start/run operations.
- Full audit trail for task/run events.
- Keyboard and screen-reader parity for core workflows.
- Backward-compatibility shim during migration window.

## 9) Milestones

- M1: API/state-machine freeze + `Quick Task -> Quick Action` rename.
- M2: Orchestrator consumes v2 ready queue + task claim lock.
- M3: PRD preview/commit importer + dependency-first execution.
- M4: Quick-run promote flow + telemetry + legacy-path deprecation.

## 10) Implementation Checklist

### Backend
- [x] Enforce dependency guard for `ready` and `in_progress` transitions.
- [x] Expose state-machine metadata endpoint for UI/test contract checks.
- [x] Add orchestrator adapter to consume `TaskEngine.get_ready_tasks()`.
  - M4 update: v2 adapter is now default-on. Legacy scheduler fallback is temporary via `FEATURE_PRD_DISABLE_V2_TASKS=true`.
- [x] Add atomic claim API/path for `ready -> in_progress`.
- [x] Add `POST /api/v2/tasks/{id}/run`.
- [x] Add retry/cancel endpoints and policies.
- [x] Persist runtime step/task events from pipeline execution.
- [x] Add PRD import preview/commit endpoints.
- [x] Add quick-run promote endpoint.
- [x] Add quick-run telemetry events + query endpoints.
- [x] Deprecate legacy scheduler path (v2 default-on with opt-out flag).

### Frontend
- [x] Rename `Quick Task` to `Quick Action`.
- [x] Add clear one-off helper text and promotion toggle.
- [x] Add `Run Now` in task detail view.
- [x] Make Tasks the default landing route/view.
- [x] Make Execution a monitor/control surface for active queue/workers.
- [x] Add blocked-task explainers and unblock shortcuts.

### Testing
- [x] E2E: create task -> ready -> orchestrator claim -> done.
- [x] E2E: PRD import graph executes in dependency order.
- [x] E2E: quick action does not create board task unless promoted.
- [x] Concurrency test: no dual-claim for same task.
- [x] A11y pass across Tasks/Execution/Agents.

## 11) Open Decisions

- [Resolved] Terminal dependency policy: `cancelled` satisfies blocker resolution by default.
- [Resolved] New task default: `backlog` (manual move to `ready` required).
- [Resolved] HITL default: tasks require human review by default (`in_progress -> in_review -> done`).
  - Config override: set `FEATURE_PRD_AUTO_APPROVE_REVIEW=true` to allow direct `in_progress -> done`.
- [Resolved] Promotion behavior: quick action promotion creates a new task by default.
