# Orchestrator-First Revamp (Big-Bang) Plan

## Summary

- Reposition the product as an orchestration-first AI engineering control center where `Task` is the primary unit and PRD import is one intake mode.
- Execute a big-bang rewrite on the existing stack (Python/FastAPI + React/Vite) with a hard reset of runtime state.
- Keep file persistence, but behind strict repository interfaces so DB migration is easy later.
- Keep Quick Action as ephemeral by default, with explicit promotion to task.
- Default policy: implementation tasks must resolve all severities (critical/high/medium/low), then go to human review queue unless task-level auto-approve is enabled.
- Git strategy: one orchestrator run branch, task-level commits on that branch.
- CLI becomes UI-first/minimal; web app is the primary control plane.

## Grounded Current-State Problems To Eliminate

- Product model split: legacy PRD-runner and v2 task/orchestration coexist in core paths (`src/feature_prd_runner/orchestrator.py`, `src/feature_prd_runner/server/api.py`).
- UI model split: one monolithic app (`web/src/App.tsx`) combines multiple competing workflows and mixes legacy + v2 endpoints.
- API inconsistency: legacy `/api/*` and v2 `/api/v2/*` both drive production UI.
- Execution inconsistency: v2 tasks are adapter/fallback paths, not the dominant scheduler/executor path.
- Project selection gap: no robust persisted manual repo registry in the web flow.

## Product Contract (Post-Revamp)

Primary intake methods:
- `Create Task` (persistent, board-backed).
- `Import PRD` (preview graph, then commit to board tasks).
- `Quick Action` (ephemeral quick run, optional promote).

Primary surfaces:
- `Board` (default landing).
- `Execution` (queue/workers/health, no alternate task model).
- `Review Queue` (approve/request-changes for `in_review` tasks).
- `Agents` (pool, roles, capacity, overrides).
- `Settings` (projects, policies, providers, diagnostics).

## Architecture Blueprint

### Backend Modules (new canonical path: `v3`)

- Create `v3` backend package under `src/feature_prd_runner/v3/`.
- `v3/domain/`: task, quick action, run, review cycle, agent session types.
- `v3/storage/`: repository interfaces + file-backed implementation.
- `v3/orchestrator/`: scheduler, claim lock, pipeline executor, review/refine loop, handoff manager.
- `v3/api/`: FastAPI routers only for v3 contracts.
- `v3/events/`: event bus + websocket publisher.
- Existing legacy modules remain only for archival/reference in this phase and are removed from runtime wiring.

### Storage Contract (file-backed, abstracted)

- New state root: `.prd_runner/v3/`.
- Files:
- `.prd_runner/v3/tasks.yaml`
- `.prd_runner/v3/runs.yaml`
- `.prd_runner/v3/review_cycles.yaml`
- `.prd_runner/v3/agents.yaml`
- `.prd_runner/v3/quick_actions.yaml`
- `.prd_runner/v3/events.jsonl`
- `.prd_runner/v3/config.yaml`
- Strong transactional writes with lock + atomic temp-file rename.
- Repository interfaces:
- `TaskRepository`, `RunRepository`, `AgentRepository`, `QuickActionRepository`, `EventRepository`.
- All orchestration logic depends on interfaces, never direct YAML I/O.

### Orchestrator Runtime

Scheduler loop:
- Pull runnable tasks (`ready` + dependencies resolved).
- Sort by priority, then aging, then creation time.
- Enforce concurrency caps and repo-level conflict guards.
- Claim lock transition `ready -> in_progress` must be atomic.

Implementation pipeline default:
- `plan -> implement -> verify -> review`
- If review has open findings above threshold: `implement_fix -> verify -> review` loop.

Exit conditions:
- No open findings => `commit -> in_review` (or `done` if task auto-approve true).
- Loop attempt cap exceeded => `blocked` with structured blocker reason.

Agent model:
- Role-specialized prompts/capabilities with shared default worker.
- Optional role-level provider overrides in config.

Human review:
- `in_review` queue is first-class.
- `approve` => `done`.
- `request_changes` => back to `ready` with injected guidance artifacts.

## Public API Changes (Canonical v3)

### Projects

- `GET /api/v3/projects` returns discovered + pinned projects.
- `POST /api/v3/projects/pinned` adds manual project path.
- `DELETE /api/v3/projects/pinned/{project_id}` removes pinned path.
- `GET /api/v3/projects/pinned` lists pinned paths.
- Path validation:
- Must exist, must be directory, must be readable, must contain `.git` (or explicit override flag).

### Tasks

- `POST /api/v3/tasks`
- `GET /api/v3/tasks`
- `GET /api/v3/tasks/board`
- `GET /api/v3/tasks/{id}`
- `PATCH /api/v3/tasks/{id}`
- `POST /api/v3/tasks/{id}/transition`
- `POST /api/v3/tasks/{id}/run`
- `POST /api/v3/tasks/{id}/retry`
- `POST /api/v3/tasks/{id}/cancel`
- `POST /api/v3/tasks/{id}/dependencies`
- `DELETE /api/v3/tasks/{id}/dependencies/{dep_id}`

### PRD Import

- `POST /api/v3/import/prd/preview`
- `POST /api/v3/import/prd/commit`
- `GET /api/v3/import/{job_id}`

### Quick Action

- `POST /api/v3/quick-actions`
- `GET /api/v3/quick-actions`
- `GET /api/v3/quick-actions/{id}`
- `POST /api/v3/quick-actions/{id}/promote`

### Review Queue

- `GET /api/v3/review-queue`
- `POST /api/v3/review/{task_id}/approve`
- `POST /api/v3/review/{task_id}/request-changes`

### Orchestration + Agents

- `GET /api/v3/orchestrator/status`
- `POST /api/v3/orchestrator/control` (`pause|resume|drain|stop`)
- `GET /api/v3/agents`
- `POST /api/v3/agents/spawn`
- `POST /api/v3/agents/{id}/pause`
- `POST /api/v3/agents/{id}/resume`
- `POST /api/v3/agents/{id}/terminate`

### Events/WebSocket

- One multiplexed websocket endpoint remains.
- v3 channels only: `tasks`, `queue`, `agents`, `review`, `quick_actions`, `notifications`, `system`.
- Event schema standardized:
- `id`, `ts`, `channel`, `type`, `entity_id`, `payload`, `project_id`.

## Public Type Changes

### Task (v3)

Core fields:
- `id`, `title`, `description`, `task_type`, `priority`, `status`, `labels`.

Dependency fields:
- `blocked_by`, `blocks`, `parent_id`, `children_ids`.

Execution fields:
- `pipeline_template`, `current_step`, `current_agent_id`, `run_ids`, `retry_count`, `error`.

Policy fields:
- `quality_gate` (default zero open findings across all severities).
- `approval_mode` (`human_review` default, `auto_approve` optional).

Source fields:
- `source` (`manual`, `prd_import`, `promoted_quick_action`, etc.).

### ReviewFinding (new)

- `id`, `task_id`, `severity`, `category`, `summary`, `file`, `line`, `suggested_fix`, `status`.

### ReviewCycle (new)

- `id`, `task_id`, `attempt`, `findings`, `open_counts`, `decision`, `created_at`.

### QuickActionRun (v3)

- `id`, `prompt`, `status`, `started_at`, `finished_at`, `result_summary`, `promoted_task_id`.

## UI Revamp Plan

### Information Architecture

- Replace multi-role scattered cockpit with one clear model:
- `Board` (default)
- `Execution`
- `Review Queue`
- `Agents`
- `Settings`
- Remove legacy PRD-runner-first entry widgets from primary navigation.

### Main UX Changes

- Replace `web/src/App.tsx` monolith with route-driven composition.
- Replace current mixed launcher with single `Create Work` modal:
- Tab 1: Create Task
- Tab 2: Import PRD
- Tab 3: Quick Action
- Board-first workbench:
- Left: Kanban columns.
- Center: Task detail and activity.
- Right: Queue/agent context.
- Add manual repo pinning UI in project switcher:
- Search discovered projects.
- Add absolute path manually.
- Persist pinned list.
- Quick Action remains visible but explicitly labeled ephemeral.

### Component Rewrite Targets

Replace or heavily refactor:
- `web/src/components/TaskLauncher.tsx`
- `web/src/components/TasksPanel.tsx`
- `web/src/components/RunsPanel.tsx`
- `web/src/components/ProjectSelector.tsx`
- `web/src/App.tsx`

Keep/adapt:
- Kanban components under `web/src/components/KanbanBoard/`
- Agent components under `web/src/components/AgentCard/`
- Unified websocket context.

## CLI Contract (UI-First Minimal)

- Keep only minimal commands:
- `server`
- `project pin/list/unpin`
- `task create/list/run`
- `quick-action`
- `orchestrator status/control`
- Mark legacy PRD-runner linear commands as deprecated and non-default.
- README and docs rewritten to present web UI as primary.

## Cutover and Rollout (Big-Bang + Hard Reset)

### Cutover Mechanics

- On first launch of revamp:
- If `.prd_runner` exists and is not v3, archive to `.prd_runner_legacy_<timestamp>`.
- Create fresh `.prd_runner/v3/` state.
- No automatic migration of old tasks/runs.
- Version stamp:
- `.prd_runner/v3/config.yaml` contains `schema_version: 3`.

### Rollout Steps

1. Build v3 backend modules and routes behind new app wiring.
2. Rewire frontend to only v3 APIs.
3. Remove runtime references to legacy `/api/*` task/run orchestration endpoints.
4. Ship cutover command and startup archive logic.
5. Update README/docs as orchestration-first.
6. Run full test matrix + smoke E2E.
7. Release as a single breaking version.

## Testing and Acceptance Criteria

### Backend Tests

State machine:
- Valid and invalid transitions for all statuses.

Dependency guard:
- Task cannot run or become ready with unresolved blockers.

Claim lock:
- Concurrent claims cannot dual-claim same task.

Scheduler:
- Priority order, dependency order, conflict avoidance, concurrency cap.

Review loop:
- Findings reopen implement loop until threshold satisfied.

Quick Action:
- No task created by default; promotion creates exactly one task.

Project registry:
- Manual pin/unpin/list with path validation and persistence.

### Frontend Tests

- Default landing is Board.
- Create Work modal covers Task, PRD import, Quick Action.
- Manual project add persists and appears in selector.
- Task detail review actions transition correctly.
- Review queue approve/request-changes works.
- Websocket event updates all active surfaces without polling regressions.

### E2E Scenarios

- Manual repo pin -> create feature task -> orchestrator executes -> review queue -> approve -> done.
- Import PRD preview -> commit graph -> dependency-ordered execution.
- Quick Action execution remains off-board -> promote -> appears in backlog.
- Task with findings loops implement/review until zero open findings.
- Request changes in review queue reopens task with feedback.
- Single-run branch receives per-task commits in expected order.

### Environment/CI Gate

- Enforce Python `>=3.10` in CI runtime and local dev checks.
- Fail build if any UI uses legacy `/api/` orchestration endpoints instead of `/api/v3/`.

## Delivery Plan (Decision-Complete)

### Phase 1 (Week 1-2): Core Backend v3

- Implement domain models, storage interfaces, file repositories.
- Implement tasks, quick actions, imports, review queue, projects APIs.
- Implement event model + websocket channel contract.
- Add hard-reset archival boot logic.

### Phase 2 (Week 3-4): Orchestrator Engine

- Implement scheduler + claim lock + execution worker adapter.
- Implement implementation/review refinement loop and quality gate enforcement.
- Implement single-run branch + per-task commit integration.
- Add agent role routing defaults + optional overrides.

### Phase 3 (Week 5-6): UI Rebuild

- Replace app shell/navigation around orchestrator-first surfaces.
- Ship Create Work modal and board-first workbench.
- Ship pinned project management (manual add + discovery merge).
- Remove legacy UI panels/routes not mapped to v3 model.

### Phase 4 (Week 7): Stabilization and Release

- Full backend/frontend/E2E pass.
- Docs rewrite (`README`, web docs, API docs).
- Breaking release with explicit reset behavior notice.
- Post-release cleanup of dead legacy runtime paths.

## Assumptions and Defaults Locked

- Deployment: local-first.
- Migration style: big-bang rewrite.
- Stack: keep FastAPI + React/Vite.
- Persistence: file store with repository abstraction.
- Existing state handling: hard reset with archival backup.
- Approval default: human review queue.
- Quality gate default: zero open findings at all severities.
- Agent model routing: shared default worker with role-level overrides optional.
- Git model: single orchestrator run branch with task-level commits.
- Quick Action: ephemeral by default, explicit promotion to task.
- CLI direction: UI-first minimal CLI.

## Implementation Progress Log

### Execution Rules Applied

- Working mode: `implement -> document -> test -> review -> fix -> repeat` per phase.
- Commit policy: one commit per completed phase + final consolidation commit.
- Runtime strategy: move production wiring to v3 modules; keep legacy code only as non-wired reference during migration.

### Phase 1 (Core Backend v3) - Complete

Status: `COMPLETE`

Completed:
- Added new canonical backend package: `src/feature_prd_runner/v3/`.
- Implemented v3 domain types for tasks, runs, review cycles/findings, agents, quick actions.
- Implemented repository interfaces and file-backed implementations with lock + atomic writes.
- Implemented v3 state bootstrap/cutover:
  - archives legacy `.prd_runner` to `.prd_runner_legacy_<timestamp>` on first v3 launch when legacy state is detected.
  - initializes `.prd_runner/v3/` and `schema_version: 3` config.
- Implemented v3 event bus + websocket publisher with v3 channels:
  - `tasks`, `queue`, `agents`, `review`, `quick_actions`, `notifications`, `system`.
- Implemented v3 API router (`/api/v3/*`) for:
  - projects (list/pin/unpin),
  - tasks CRUD/board/dependencies/run/retry/cancel/transition,
  - PRD import preview/commit/job,
  - quick actions + promote,
  - review queue + approve/request-changes,
  - orchestrator status/control,
  - agents list/spawn/pause/resume/terminate.
- Rewired FastAPI runtime to v3-only API and v3 websocket hub endpoint.

Testing completed:
- Added and passed `tests/test_v3_backend_phase1.py` (8 tests):
  - cutover archival + schema init,
  - dependency guard,
  - quick action promotion singleton behavior,
  - project pin validation + persistence,
  - PRD preview/commit dependency chain creation,
  - claim-lock single-claim guarantee under concurrency,
  - transition valid/invalid edge enforcement,
  - review queue request-changes/approve flow.

Review findings and fixes:
- Gap found: `POST /api/v3/tasks/{id}/run` could execute a different ready task.
  - Fix: added explicit orchestrator `run_task(task_id)` path.
- Gap found: transition endpoint accepted invalid state edges.
  - Fix: added strict transition matrix validation.

Outstanding for future phases:
- Scheduler sophistication (aging/concurrency strategy tuning) and deeper worker integration are covered in Phase 2.

### Phase 2 (Orchestrator Engine) - Complete

Status: `COMPLETE`

Completed:
- Implemented orchestrator service with:
  - scheduler tick loop,
  - atomic claim flow through repository lock,
  - repo conflict guard,
  - concurrency cap enforcement,
  - task-specific run API path.
- Added execution worker adapter abstraction (`WorkerAdapter`) with default deterministic adapter.
- Implemented implementation/review refinement loop:
  - `plan -> implement -> verify -> review`,
  - when findings exceed quality gate, loop `implement_fix -> verify -> review`,
  - cap attempts, then mark task `blocked`.
- Implemented single run branch + per-task commits:
  - one branch per orchestrator run lifecycle,
  - task-level commits with task-stamped commit messages.
- Implemented agent role routing:
  - task-type to role mapping via config,
  - role-level provider override injection into task metadata.

Testing completed:
- Added and passed `tests/test_v3_orchestrator_phase2.py` (6 tests):
  - review-loop retry until findings clear,
  - review-attempt cap to blocked,
  - role routing + provider override,
  - single run branch + per-task commit ordering,
  - scheduler priority/dependency/repo-conflict behavior,
  - scheduler concurrency cap behavior.

Review findings and fixes:
- Gap found: dependency checks treated missing blockers as resolved.
  - Fix: missing blocker now considered unresolved in claim, transition, and explicit run paths.

Outstanding for future phases:
- Frontend route-shell rewrite, Create Work modal, and v3-only UI flows are covered in Phase 3.

### Phase 3 (UI Rebuild) - Complete

Status: `COMPLETE`

Completed:
- Replaced monolithic app shell with route-driven orchestrator-first UI:
  - `Board` (default landing),
  - `Execution`,
  - `Review Queue`,
  - `Agents`,
  - `Settings`.
- Implemented single `Create Work` modal with required tabs:
  - `Create Task`,
  - `Import PRD` (preview -> commit),
  - `Quick Action` (explicitly labeled ephemeral).
- Implemented board-first workbench and v3 API data binding.
- Implemented project management in settings:
  - discovered + pinned project listing,
  - manual absolute path pin with non-git override option,
  - selected project persistence in local storage.
- Wired realtime updates through websocket subscription to v3 channels.
- Added new dedicated UI stylesheet (`orchestrator.css`) for revamp layout and responsive behavior.

Testing completed:
- Updated and passed frontend app tests:
  - `web/src/App.defaultView.test.tsx`,
  - `web/src/App.accessibility.test.tsx`.
- Verified frontend production compile/build:
  - `npm --prefix web run build` passed.

Review findings and fixes:
- Gap found: test ambiguity on duplicate `Create Work` and `Create Task` button labels.
  - Fix: updated tests to use explicit `getAllByRole` assertions where duplicate labels are intentional.

Outstanding for future phases:
- Stabilization pass to align docs/CLI contracts and run combined backend/frontend verification is covered in Phase 4.

### Phase 4 (Stabilization and Release) - Complete

Status: `COMPLETE`

Completed:
- Introduced UI-first minimal CLI implementation:
  - new entrypoint module: `src/feature_prd_runner/cli_v3.py`,
  - supported commands:
    - `server`
    - `project pin/list/unpin`
    - `task create/list/run`
    - `quick-action`
    - `orchestrator status/control`.
- Switched package script entrypoint to minimal CLI:
  - `feature-prd-runner = feature_prd_runner.cli_v3:main`.
- Rewrote root docs for orchestrator-first positioning:
  - `README.md`
  - `docs/README.md`.
- Added CI gate tests:
  - Python runtime floor (`>=3.10`),
  - UI runtime sources must use `/api/v3/*` endpoints.
- Added end-to-end backend scenario tests for the revamp contract.

Testing completed:
- Backend suites:
  - `tests/test_v3_backend_phase1.py`
  - `tests/test_v3_orchestrator_phase2.py`
  - `tests/test_cli_v3.py`
  - `tests/test_revamp_ci_gates.py`
  - `tests/test_v3_e2e_phase4.py`
  - Result: `25 passed`.
- Frontend suites:
  - `web/src/App.defaultView.test.tsx`
  - `web/src/App.accessibility.test.tsx`
  - Result: `5 passed`.
- Frontend build:
  - `npm --prefix web run build` passed.

Full revamp-plan gap review cycles:
- Review cycle 1 identified gaps:
  - missing explicit CI gate for non-v3 UI endpoints,
  - missing explicit E2E scenarios for pinned repo -> run -> review -> approve, PRD dependency order, quick-action promotion, request-changes reopen.
  - Fixes applied: CI gate tests + dedicated E2E coverage.
- Review cycle 2 identified gaps:
  - frontend regression test did not assert websocket-driven refresh behavior,
  - project pinning UI test did not verify selector persistence visibility.
  - Fixes applied: websocket refresh test + project selector persistence assertion.
- Review cycle 3:
  - no remaining contract gaps found against this revamp plan.
- Post-completion review cycle 4 (follow-up):
  - gap found: CI gate scope only covered two UI files.
    - fix: gate now scans all non-test TS/TSX files under `web/src`.
  - gap found: frontend compile scope was narrowed and could hide regressions.
    - fix: restored `web/tsconfig.json` include to full `src`.
  - gap found: runtime UI still had legacy `/api/*` endpoint usage in non-test components.
    - fix: migrated remaining non-test component endpoint strings to `/api/v3/*`.
  - gap found: helper inconsistency for task correction method and execution-order route.
    - fix: changed helper to `PATCH /api/v3/tasks/{id}` and added `GET /api/v3/tasks/execution-order`.
  - verification after fixes:
    - backend suites passed (`25 passed`),
    - frontend tests passed (`5 passed`),
    - frontend build passed.

Release readiness:
- Breaking reset behavior implemented and documented.
- Runtime is v3-first and legacy runtime wiring is removed.

## Post-Completion Gap Findings (2026-02-12)

This section records the follow-up full repo review findings and resolution decisions.

### Decisions Applied

- Remove legacy tests/components that are not part of the v3 orchestrator-first runtime contract.
- Enforce guarded state transitions: task `status` is no longer mutable via `PATCH /api/v3/tasks/{id}`.

### Findings Identified

1. Full matrix claim mismatch:
   - Focused revamp suites passed, but full backend/frontend test matrices still included legacy failing suites.
2. Transition guard gap:
   - `PATCH /api/v3/tasks/{id}` previously allowed direct status mutation, bypassing transition/dependency checks.
3. Retry guard gap:
   - `POST /api/v3/tasks/{id}/retry` previously set `ready` without blocker validation.
4. Review action guard gap:
   - Review actions (`approve` / `request-changes`) previously allowed on tasks not in `in_review`.
5. Cutover strictness gap:
   - Hard-reset/archive semantics were weaker than contract wording for non-v3 legacy state edge cases.
6. UI contract drift:
   - Board-first center/right workbench depth and PRD preview graph remained simplified.
7. Project UX gap:
   - Settings included manual pinning/persistence, but no explicit discovered-project search UX.
8. Legacy UI surface still present in repo:
   - Obsolete components/tests remained on disk and introduced non-canonical endpoint assumptions.
9. Legacy package-level reference:
   - Top-level package import still referenced legacy orchestrator module.

### Remediation Work Directed

- Legacy frontend components/tests and legacy backend API test suites were removed from active repo coverage where they were outside the v3 contract.
- v3 API transition hardening implemented:
  - reject `status` updates in task `PATCH`,
  - enforce blocker checks on `retry`,
  - require `in_review` state for review actions.

### Follow-up Remediation Complete (2026-02-12, later pass)

- Bridged remaining backend cutover gaps:
  - archive now triggers for any non-v3 `.prd_runner` root, not only a legacy artifact subset.
  - existing v3 config is normalized to `schema_version: 3`.
- Bridged remaining UI contract gaps:
  - board-first workbench now renders left Kanban + center task detail + right queue/agent context.
  - settings now includes project search over discovered+pinned items.
  - PRD import preview now includes a visible node/edge graph summary before commit.
- Bridged remaining legacy wiring gap:
  - removed package-level import of legacy orchestrator from top-level package init.
- Full current matrix after cleanup/bridging:
  - backend: `890 passed`,
  - frontend: `50 passed`,
  - frontend build: passed.
