# Web UI Feature Gaps

> **Status (2026-02-11):** Historical point-in-time gap tracker (v2 transition period).
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

CLI features that are not yet available through the web dashboard.
Tracked here so we can implement them incrementally.

---

## Fully Missing

### 1. `dry-run` — Preview Next Action
- **CLI**: `feature-prd-runner dry-run [--json]`
- **What it does**: Shows what the runner would do next without making any changes. Reports next task, next step, and any issues (PRD mismatch, stale state, etc.).
- **What's needed**:
  - [ ] Backend: `GET /api/dry-run` endpoint (read-only, returns next action + warnings)
  - [ ] Frontend: "Dry Run" button/panel on Dashboard that displays the preview

### 2. `doctor` — System Diagnostics
- **CLI**: `feature-prd-runner doctor [--check-codex] [--json]`
- **What it does**: Checks system health — git status, config validity, codex availability, Python/Node versions, state file integrity.
- **What's needed**:
  - [ ] Backend: `GET /api/doctor` endpoint (runs all checks, returns structured results)
  - [ ] Frontend: "Doctor" panel or modal showing pass/fail for each check with remediation hints

### 3. `workers list` — List Worker Providers
- **CLI**: `feature-prd-runner workers list`
- **What it does**: Lists all configured worker providers (codex, ollama, etc.) with their config.
- **What's needed**:
  - [ ] Backend: `GET /api/workers` endpoint
  - [ ] Frontend: Workers section (could be a tab or panel in Agents view)

### 4. `workers test` — Test Worker Provider
- **CLI**: `feature-prd-runner workers test <worker>`
- **What it does**: Sends a test prompt to a worker provider and reports success/failure.
- **What's needed**:
  - [ ] Backend: `POST /api/workers/{worker}/test` endpoint
  - [ ] Frontend: "Test" button next to each worker in the workers list

### 5. `explain` — Explain Blocked Task
- **CLI**: `feature-prd-runner explain <task_id>`
- **What it does**: Prints a human-readable explanation of why a task is blocked, including error type, resolution steps, and attempt counts.
- **What's needed**:
  - [ ] Backend: `GET /api/tasks/{task_id}/explain` endpoint
  - [ ] Frontend: "Why blocked?" button on blocked task cards (TaskDetail slide-over or Kanban card)

### 6. `inspect` — Deep Task State
- **CLI**: `feature-prd-runner inspect <task_id> [--json]`
- **What it does**: Dumps detailed internal task state — current step, lifecycle, error history, attempt counts, worker assignments, verification results, review results.
- **What's needed**:
  - [ ] Backend: `GET /api/tasks/{task_id}/inspect` endpoint (returns full state)
  - [ ] Frontend: "Inspect" tab or expandable section in TaskDetail slide-over

---

## Partially Missing

### 7. Advanced `run` Options in TaskLauncher
- **Current web UI `StartRunRequest`**: mode, content, test_command, build_command, verification_profile, auto_approve_plans, auto_approve_changes, auto_approve_commits
- **Missing CLI options not exposed**:
  - [ ] `--language` (auto/python/typescript/javascript/nextjs/go/rust)
  - [ ] `--reset-state` (archive and recreate .prd_runner before starting)
  - [ ] `--require-clean` / `--no-require-clean` (git cleanliness check)
  - [ ] `--commit` / `--no-commit` (enable/disable git commit step)
  - [ ] `--push` / `--no-push` (enable/disable git push)
  - [ ] `--interactive` (step-by-step approval gates)
  - [ ] `--parallel` / `--max-workers` (parallel execution)
  - [ ] `--ensure-ruff` (ruff helper for Python)
  - [ ] `--ensure-deps` / `--ensure-deps-command` (dependency install)
  - [ ] `--shift-minutes` (timebox per worker run)
  - [ ] `--heartbeat-seconds` / `--heartbeat-grace-seconds`
  - [ ] `--max-task-attempts` / `--max-review-attempts` / `--max-auto-resumes`
  - [ ] `--worker` (worker provider override)
  - [ ] `--codex-command` (custom codex command)
- **What's needed**:
  - [ ] Backend: Extend `StartRunRequest` model with these fields
  - [ ] Frontend: "Advanced Options" collapsible section in TaskLauncher

### 8. Structured `correct` Command
- **CLI**: `feature-prd-runner correct <task_id> --file <path> --issue <desc> [--fix <suggestion>]`
- **Current web UI**: Freeform chat message only
- **What's needed**:
  - [ ] Backend: `POST /api/tasks/{task_id}/correct` endpoint (or extend messages API with structured type)
  - [ ] Frontend: "Send Correction" form with file path, issue description, and suggested fix fields

### 9. Structured `require` Command
- **CLI**: `feature-prd-runner require <requirement> [--task-id <id>] [--priority high|medium|low]`
- **Current web UI**: Freeform chat message only
- **What's needed**:
  - [ ] Backend: `POST /api/requirements` endpoint (or extend messages API)
  - [ ] Frontend: "Add Requirement" form with priority selector and optional task scope

### 10. `logs` by Task ID and Step
- **CLI**: `feature-prd-runner logs <task_id> [--step <step>] [--lines N]`
- **Current web API**: `GET /api/logs/{run_id}` — only by run_id
- **What's needed**:
  - [ ] Backend: `GET /api/tasks/{task_id}/logs` endpoint (resolves task → run_id internally, supports `--step` filter)
  - [ ] Frontend: "Logs" tab in TaskDetail slide-over with step filter dropdown

### 11. Metrics Export
- **CLI**: `feature-prd-runner metrics --export csv|html [--output <path>]`
- **Current web UI**: In-browser display only
- **What's needed**:
  - [ ] Backend: `GET /api/metrics/export?format=csv|html` endpoint (returns downloadable file)
  - [ ] Frontend: "Export" dropdown button in MetricsPanel (CSV / HTML options)

### 12. Parallel Execution Plan View
- **CLI**: `feature-prd-runner plan-parallel [--tree]`
- **Current web UI**: DependencyGraph shows relationships but not execution batches
- **What's needed**:
  - [ ] Backend: Already has `GET /api/v2/tasks/execution-order` (returns batches) — may need a phase-level equivalent
  - [ ] Frontend: "Execution Plan" view showing batches/waves with tasks grouped by parallel execution stage

---

## Out of Scope (CLI-only is fine)

- **`example`** — Project scaffolding. Only useful from terminal.
- **`server`** — Starts the web server itself. Inherently CLI-only.

---

## Implementation Priority Suggestion

**High value, low effort:**
1. `explain` (#5) — blocked tasks are confusing without this
2. `inspect` (#6) — debugging requires seeing full state
3. Advanced `run` options (#7) — language/reset-state/parallel are frequently needed

**Medium value:**
4. `doctor` (#2) — helpful onboarding/troubleshooting
5. `dry-run` (#1) — safety before committing to a run
6. Structured `correct` (#8) — better than freeform chat
7. `logs` by task (#10) — natural way users think about logs

**Lower priority:**
8. `workers` management (#3, #4) — most users have one worker
9. Structured `require` (#9) — chat works as workaround
10. Metrics export (#11) — nice-to-have
11. Parallel plan view (#12) — DependencyGraph partially covers this
