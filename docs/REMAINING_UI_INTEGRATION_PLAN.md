# Remaining UI Integration Plan (v3)

Status: Implemented (mounted UI)  
Date: 2026-02-12  
Scope: Close the final mounted-UI gaps after task detail/read-write parity.

## Remaining Gaps

1. No blocking backend API parity gaps remain in mounted v3 UI.
2. Remaining work is non-blocking UX polish (for example richer failure analytics views).

## Design Goals

- Keep `Board` as visual execution surface, not overloaded as a search/report view.
- Keep `Create Work` flow lightweight, with drill-down pages for history/details.
- Reuse existing styling and route model in `web/src/App.tsx` to avoid design drift.
- Add capability without introducing a second, conflicting task model.

## Direction 1: Task Explorer (GET /tasks)

### UX

- Add a new panel in `Board` page right rail or top strip: `Task Explorer`.
- Controls:
  - Search text.
  - Status filter.
  - Type filter.
  - Priority filter.
  - Optional toggle: `Only blocked` (maps to status `blocked`).
- Results:
  - Paginated/simple list sorted by priority + recency.
  - Clicking a row selects task and opens/refreshes Task Detail.

### API

- Call `GET /api/v3/tasks` with query params:
  - `status`, `task_type`, `priority`.
- For free-text search (backend currently lacks explicit search in v3 router):
  - Use client-side filter over fetched results as interim.
  - Future backend extension can replace this with server-side search.

### Implementation Notes

- Add local state:
  - `taskQuery`, `taskFilterStatus`, `taskFilterType`, `taskFilterPriority`.
  - `taskListResults`, `taskListLoading`, `taskListError`.
- Add `loadTaskList()` function and call on filter changes with debounce.
- Keep `board` fetch for kanban unchanged; Task Explorer is supplemental.

## Direction 2: Import Job Detail (GET /import/{job_id})

### UX

- In Create Work > Import PRD:
  - After preview, show `Job ID` with `View details` link/button.
  - After commit, keep a compact `Recent Import Jobs` list.
- Add detail card:
  - Job status.
  - Task count.
  - Created task IDs.
  - Original title/source metadata.

### API

- Use `GET /api/v3/import/{job_id}` on:
  - Explicit click, and
  - Optional short polling while status is non-terminal (`preview_ready`/`committing`).

### Implementation Notes

- Add state:
  - `selectedImportJobId`, `selectedImportJob`, `importJobLoading`, `importJobError`.
  - `recentImportJobIds` (persist in local state/session for UX continuity).
- Polling policy:
  - 2s interval, max 60s, stop on terminal state or modal close.

## Direction 3: Quick Action Detail (GET /quick-actions/{id})

### UX

- In Create Work > Quick Action:
  - Keep existing recent list.
  - Add `View details` action per row.
  - Show detail drawer/card with:
    - Prompt
    - Status
    - Started/finished timestamps
    - Result summary
    - Promotion status and linked task id (if promoted)

### API

- Continue `GET /api/v3/quick-actions` for list.
- Add `GET /api/v3/quick-actions/{quick_action_id}` for focused detail.

### Implementation Notes

- Add state:
  - `selectedQuickActionId`, `selectedQuickAction`, `quickActionDetailLoading`, `quickActionDetailError`.
- Refresh behavior:
  - Reload detail after promote to reflect `promoted_task_id`.

## UI Structure Refactor (Minimal, Safe)

To keep integration neat without large rewrite:

1. Extract three presentational subcomponents from `App.tsx`:
   - `TaskExplorerPanel`
   - `ImportJobPanel`
   - `QuickActionDetailPanel`
2. Keep data-fetching/orchestration in `App.tsx` first.
3. Later, move fetch logic into small hooks if complexity grows:
   - `useTaskList`, `useImportJob`, `useQuickActionDetail`.

This avoids a risky full-architecture refactor while reducing `App.tsx` growth.

## Sequence (Implemented)

1. Implemented Task Explorer (`GET /tasks`) with filters, search, blocked-only toggle, and pagination.
2. Implemented Quick Action detail (`GET /quick-actions/{id}`) with refresh after promote.
3. Implemented Import Job detail (`GET /import/{job_id}`) with recent jobs and created-task-id display after commit.
4. Extracted presentational panels in mounted UI:
   - `TaskExplorerPanel`
   - `ImportJobPanel`
   - `QuickActionDetailPanel`
5. Added compatibility endpoints in backend v3 router for legacy/unmounted component contracts (`metrics`, `phases`, `agents/types`, `collaboration/*`) to eliminate API drift.

## Validation Plan

- Unit/UI tests:
  - Filter changes trigger correct `/tasks` query composition.
  - Selecting task in explorer updates Task Detail.
  - Quick action detail fetch and promote state refresh.
  - Import job detail fetch and status display.
- Regression:
  - Existing `App.defaultView.test.tsx`.
  - Build (`npm run build`) and targeted tests (`vitest`).

## Definition of Done

- All three remaining backend capabilities are visible and usable from mounted UI.
- No duplicated task model or alternate workflow introduced.
- Existing Board/Create Work/Execution flows remain intact.
- Docs updated in:
  - `docs/WEB_UI_FEATURE_GAPS.md` (gap list now fully closed or reduced to future enhancements).

Current status: complete for the API-integration scope above.
