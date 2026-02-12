# Web UI vs Backend Capability Gaps (v3)

Status: Updated after integration pass  
Date: 2026-02-12  
Scope: Mounted v3 web app (`web/src/App.tsx`) against v3 backend router (`src/feature_prd_runner/v3/api/router.py`).

## Update (2026-02-12)

Implemented in mounted UI:
- Task lifecycle actions: `run`, `retry`, `cancel`, and manual `transition`.
- Dependency management: add/remove blocker IDs from task detail.
- Quick Action improvements: recent quick-action list and `promote` action.
- Project management: pinned project list + unpin action.
- Execution visibility: execution-order batches shown in Execution view.
- Extended payload support:
  - Create Task now supports `task_type`, `labels`, `blocked_by`, `approval_mode`, `parent_id`, `pipeline_template`, and `metadata`.
  - Review actions now send optional `guidance`.
  - Agent spawn now supports configurable `role`, `capacity`, and `override_provider`.

## Method

1. Enumerated all v3 backend routes from `/api/v3/*`.
2. Enumerated all mounted UI calls in `web/src/App.tsx`.
3. Compared capability parity and noted partial payload support.

## Endpoint Coverage Summary

Now implemented in mounted UI (called from `web/src/App.tsx`):
- `GET /projects`
- `GET /projects/pinned`
- `POST /projects/pinned`
- `DELETE /projects/pinned/{project_id}`
- `GET /projects/browse`
- `POST /tasks`
- `GET /tasks/board`
- `GET /tasks/execution-order`
- `POST /tasks/{task_id}/transition`
- `POST /tasks/{task_id}/run`
- `POST /tasks/{task_id}/retry`
- `POST /tasks/{task_id}/cancel`
- `POST /tasks/{task_id}/dependencies`
- `DELETE /tasks/{task_id}/dependencies/{dep_id}`
- `POST /import/prd/preview`
- `POST /import/prd/commit`
- `POST /quick-actions`
- `GET /quick-actions`
- `POST /quick-actions/{quick_action_id}/promote`
- `GET /review-queue`
- `POST /review/{task_id}/approve`
- `POST /review/{task_id}/request-changes`
- `GET /orchestrator/status`
- `POST /orchestrator/control`
- `GET /agents`
- `POST /agents/spawn`
- `POST /agents/{agent_id}/pause`
- `POST /agents/{agent_id}/resume`
- `POST /agents/{agent_id}/terminate`

Still backend-supported but not yet exposed in mounted UI:
- None in current mounted v3 surface.

## Residual Payload/UX Gaps

1. Error states are still lightweight.
- Inline retry controls now exist for Task Explorer, Import Job detail, and Quick Action detail.
- Full structured failure analytics UX remains future work.
- References: `web/src/components/AppPanels/TaskExplorerPanel.tsx`, `web/src/components/AppPanels/ImportJobPanel.tsx`, `web/src/components/AppPanels/QuickActionDetailPanel.tsx`

## Additional Drift Risk (Codebase Hygiene)

Compatibility endpoints were added in v3 router for legacy/unmounted component contracts:
- `/api/v3/metrics`
- `/api/v3/phases`
- `/api/v3/agents/types`
- `/api/v3/collaboration/*` (modes, presence, timeline, feedback, comments)

This removes endpoint-contract drift while mounted UI remains focused on `web/src/App.tsx`.

## Recommended Implementation Order

1. Task lifecycle controls (`run`, `retry`, `cancel`, `transition`) + explicit task detail fetch/update.
2. Dependency graph editor (add/remove blockers).
3. Quick Action history + promote flow.
4. Create Task advanced fields (`approval_mode`, `blocked_by`, labels/metadata).
5. Project unpin/list-pinned UX.
6. Execution-order visualization.
7. Remove or migrate unmounted legacy components to avoid further backend/UI drift.
