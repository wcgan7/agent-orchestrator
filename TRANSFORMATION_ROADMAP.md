# Feature PRD Runner → AI Engineering Orchestrator: Transformation Roadmap

## Current State Assessment

A **single-agent, single-PRD, linear pipeline runner** with a monitoring dashboard. The core loop is: parse PRD → plan phases → for each phase: implement → verify → review → commit. One agent (Codex/Ollama) does everything sequentially. The UI is a read-heavy dashboard — lots of panels showing state, but limited ability to actually *drive* work.

**The fundamental problem:** it's a pipeline executor pretending to be an orchestrator. There's no multi-agent coordination, no dynamic task creation, no feedback loops that actually change agent behavior mid-run, and no way to use it for anything other than "implement this PRD."

**Codebase stats:** ~15k LOC backend (Python), ~4.5k LOC frontend (React/TypeScript), 30+ API endpoints, FSM-based pipeline, file-based state management.

---

## The 5 Transformations

### 1. Multi-Agent Orchestration (the biggest unlock)

**Today:** One worker subprocess runs at a time per phase. "Parallel" mode just runs independent phases in threads, but each phase is still one agent.

**The vision:** A proper agent pool where multiple specialized agents work concurrently, each with different roles, models, and capabilities.

**What to build:**

- **Agent Registry** — define agent types (implementer, reviewer, researcher, tester, architect) with different models, system prompts, and tool access
- **Agent Pool Manager** — spin up N agents, assign tasks, monitor health, redistribute on failure
- **Shared Context Bus** — agents can read each other's progress, artifacts, and decisions without going through files
- **Agent-to-Agent handoff** — reviewer agent passes feedback directly to implementer agent, not through a file-based retry loop
- **Per-agent resource limits** — token budgets, time budgets, cost caps per agent

```
Architect Agent ──→ plans phases
                      ↓
Implementer Agent 1 ──→ phase-1  ──→ Reviewer Agent ──→ feedback loop
Implementer Agent 2 ──→ phase-2  ──→ Reviewer Agent ──→ feedback loop
Researcher Agent    ──→ answers questions from implementers
Tester Agent        ──→ runs verification independently
```

**Why it moves the needle:** This is the difference between "a script that calls an LLM" and "an AI team." Parallel specialized agents with handoff is 3-5x faster and produces better results because each agent has focused context.

**Implementation details:**

- `src/feature_prd_runner/agents/registry.py` — Agent type definitions with model, prompt template, tool access, resource limits
- `src/feature_prd_runner/agents/pool.py` — Agent lifecycle management (spawn, monitor, kill, reassign)
- `src/feature_prd_runner/agents/context_bus.py` — Shared memory / message passing between agents
- `src/feature_prd_runner/agents/handoff.py` — Structured handoff protocol (reviewer → implementer, researcher → implementer)
- New API endpoints: `GET /api/agents`, `POST /api/agents/{id}/pause`, `POST /api/agents/{id}/redirect`
- New WebSocket channel: `/ws/agents` — real-time agent activity stream

---

### 2. Dynamic Task Engine (Jira-like, not pipeline-like)

**Today:** Tasks are derived from PRD phases at plan time. Fixed. Linear. Can't add, split, or reprioritize mid-run.

**The vision:** A dynamic task board where tasks can be created from *any source* — PRDs, repo analysis, user input, agent discoveries — and the system figures out execution order.

**What to build:**

- **Task Model overhaul:**
  - Tasks as first-class entities with types: `feature`, `bug`, `refactor`, `research`, `review`, `test`, `docs`
  - Priority (P0-P3), effort estimate, assignee (agent or human)
  - Labels, parent/child relationships, blockers
  - Rich description with acceptance criteria, context files, related tasks

- **Task Sources (one-click generators):**
  - **Repo Review** → analyzes codebase, generates improvement tasks automatically
  - **PRD Import** → today's flow, but tasks land on the board instead of a rigid pipeline
  - **Bug Scan** → runs tests, linting, type checking, creates fix tasks for failures
  - **Security Audit** → scans for OWASP issues, dependency vulnerabilities
  - **Performance Audit** → identifies slow paths, large bundles, N+1 queries
  - **Enhancement Brainstorm** → AI generates feature ideas based on codebase analysis
  - **Manual Creation** → user creates tasks directly in UI

- **Smart Scheduling:**
  - Dependency graph computed dynamically as tasks are added
  - Priority queue with preemption (P0 bug interrupts P2 feature)
  - Agent affinity (assign review tasks to reviewer agent, not implementer)
  - Parallel dispatch of independent tasks to available agents

**Why it moves the needle:** This turns it from "run one feature PRD" into "manage an entire engineering backlog with AI agents." Users come back daily, not just when they have a PRD to run.

**Implementation details:**

- `src/feature_prd_runner/tasks/model.py` — Enhanced task dataclass with type, priority, labels, parent/child, effort
- `src/feature_prd_runner/tasks/engine.py` — Task CRUD, dependency resolution, scheduling
- `src/feature_prd_runner/tasks/sources/` — Task generators:
  - `repo_review.py` — Scan codebase, generate improvement tasks
  - `prd_import.py` — Parse PRD into tasks (replaces current plan step)
  - `bug_scan.py` — Run tests/lint, create fix tasks
  - `security_audit.py` — OWASP + dependency scanning
  - `performance_audit.py` — Profiling and bottleneck detection
  - `enhancement_brainstorm.py` — AI-generated feature ideas
- `src/feature_prd_runner/tasks/scheduler.py` — Priority queue with preemption, agent affinity, parallel dispatch
- New API endpoints: full CRUD for tasks, `POST /api/tasks/generate/{source}`, `POST /api/tasks/{id}/assign`

---

### 3. Real-Time Control Center (not a monitoring dashboard)

**Today:** 15 separate panels, all polling on different intervals (3-10s), no real-time updates except logs. The UI shows state but doesn't let you *drive*.

**The vision:** A command center where you see all agents working in real-time and can intervene instantly — like a mission control, not a status page.

**What to build:**

- **Unified WebSocket for all state** — kill all polling, push everything through a single multiplexed WebSocket connection
- **Agent Activity Feed** — real-time stream showing what each agent is doing right now (reading file X, writing function Y, running tests)
- **Kanban Board view** — tasks flow through columns: Backlog → In Progress → Review → Done
- **Agent Cards** — each running agent gets a live card showing:
  - Current task, current file, token usage, cost so far
  - Live streaming output (not just logs — actual agent reasoning)
  - Inline intervention: pause, redirect, give feedback, reassign
- **Split-pane workspace:**
  - Left: task board / agent overview
  - Center: active work area (diff viewer, log stream, agent output)
  - Right: context panel (file tree, related tasks, chat)
- **Command Palette** (Cmd+K) — quick actions: create task, assign agent, run review, search tasks, jump to file
- **Notifications & Alerts** — agent blocked, review needed, task completed, budget exceeded — with sound/desktop notifications
- **Dark mode** (the CSS variables system is already partially there)

**Why it moves the needle:** Users currently have to scroll through 15 panels to understand what's happening. A proper control center means you can manage 5 agents across 20 tasks from one screen.

**Implementation details:**

- `web/src/contexts/WebSocketContext.tsx` — Single multiplexed WebSocket connection replacing all polling
- `web/src/components/KanbanBoard/` — Drag-and-drop task board (Backlog, Ready, In Progress, Review, Done)
- `web/src/components/AgentCard/` — Live agent status card with streaming output and inline controls
- `web/src/components/CommandPalette/` — Cmd+K quick action overlay
- `web/src/components/SplitPane/` — Resizable split-pane layout (left: board, center: work area, right: context)
- `web/src/components/ActivityFeed/` — Unified real-time event stream for all agents and tasks
- `web/src/components/NotificationCenter/` — Desktop notifications, sound alerts, notification drawer
- `web/src/styles/themes/` — Dark mode theme with toggle
- `server/websocket.py` — Multiplexed WebSocket with channels: `tasks`, `agents`, `logs`, `metrics`, `notifications`

---

### 4. Flexible Pipeline Engine (task-driven, not hardcoded)

**Today:** Every task goes through the same 5-step pipeline: plan_impl → implement → verify → review → commit. A documentation task still gets "implementation" and "verification."

**The vision:** Pipeline steps are determined by the task type and can be customized, skipped, or extended.

**What to build:**

- **Pipeline Templates:**
  ```yaml
  feature:
    steps: [plan, implement, verify, review, commit]
  bug_fix:
    steps: [reproduce, diagnose, fix, verify, review, commit]
  refactor:
    steps: [analyze, plan, implement, verify, review, commit]
  research:
    steps: [gather, analyze, summarize, report]
  docs:
    steps: [analyze_code, write, review, commit]
  repo_review:
    steps: [scan, analyze, generate_tasks]
  security_audit:
    steps: [scan_deps, scan_code, generate_report, generate_tasks]
  ```
- **Step Registry** — pluggable steps that can be mixed and matched
- **Conditional Steps** — skip verify if task is docs-only, skip review if change is < 10 lines
- **Custom Steps** — users define their own steps with custom prompts and verification
- **Pipeline Composition** — chain pipelines (research → feature → test)
- **Mid-run pipeline modification** — add a step, skip a step, reorder based on what the agent discovers

**Why it moves the needle:** Not everything is a "feature implementation." Research tasks, bug fixes, refactors, docs — they all need different workflows. Flexible pipelines make the tool useful for *all* engineering work, not just greenfield features.

**Implementation details:**

- `src/feature_prd_runner/pipelines/registry.py` — Pipeline template definitions and registration
- `src/feature_prd_runner/pipelines/steps/` — Individual step implementations:
  - `reproduce.py`, `diagnose.py` — Bug-specific steps
  - `gather.py`, `summarize.py`, `report.py` — Research-specific steps
  - `scan_deps.py`, `scan_code.py` — Security audit steps
  - `analyze_code.py` — Docs generation step
- `src/feature_prd_runner/pipelines/engine.py` — Pipeline executor with conditional logic and mid-run modification
- `src/feature_prd_runner/pipelines/templates/` — YAML pipeline template files
- Config: `.prd_runner/pipelines/` — User-defined custom pipeline templates

---

### 5. Human-AI Collaboration Layer (not just approval gates)

**Today:** Chat panel is hidden behind a button. Approval gates are binary approve/reject. Feedback goes into a text field and hopes the agent reads it.

**The vision:** Rich, contextual collaboration where humans and agents work together fluidly.

**What to build:**

- **Inline Code Review** — comment on specific lines in diffs (like GitHub PR review), agents respond to and address comments
- **Agent Reasoning Viewer** — see *why* the agent made a decision, not just *what* it did. Show the agent's plan, considered alternatives, and trade-offs
- **Structured Feedback** — not free text, but actionable: "change approach to X", "use library Y instead", "this file shouldn't be modified"
- **Feedback that sticks** — feedback persists and is injected into future agent prompts for the same task (today feedback is fire-and-forget)
- **Human-in-the-loop modes:**
  - **Autopilot** — agents run freely, humans review after
  - **Supervised** — agents propose, humans approve each step
  - **Collaborative** — humans and agents work on the same task, taking turns
  - **Review-only** — agents implement, humans do all code review
- **Multi-user support** — multiple humans reviewing/guiding different agents simultaneously
- **Activity Timeline** — unified chronological view of all human and agent actions on a task

**Why it moves the needle:** The current approval gate model is "rubber stamp or reject." Real collaboration means the AI gets better with human input, and humans trust it more because they can see and shape reasoning.

**Implementation details:**

- `src/feature_prd_runner/collaboration/feedback.py` — Structured feedback model (type, target, action, persistence)
- `src/feature_prd_runner/collaboration/modes.py` — Human-in-the-loop mode definitions and enforcement
- `src/feature_prd_runner/collaboration/review.py` — Inline code review with line-level comments
- `src/feature_prd_runner/collaboration/reasoning.py` — Agent reasoning capture and display
- `web/src/components/InlineReview/` — GitHub-style line-level commenting on diffs
- `web/src/components/ReasoningViewer/` — Collapsible agent reasoning tree
- `web/src/components/FeedbackPanel/` — Structured feedback form (approach change, library swap, file restriction)
- `web/src/components/ActivityTimeline/` — Unified human + agent action timeline
- New API endpoints: `POST /api/tasks/{id}/feedback`, `GET /api/tasks/{id}/reasoning`, `POST /api/tasks/{id}/review-comment`

---

## Priority & Sequencing

| Priority | Transformation | Effort | Impact | Why This Order |
|----------|---------------|--------|--------|----------------|
| **P0** | Dynamic Task Engine | Medium | Very High | Everything else builds on having a proper task model |
| **P0** | Real-Time Control Center | Medium | Very High | Users need to see and control before they trust multi-agent |
| **P1** | Multi-Agent Orchestration | High | Very High | The core differentiator, but needs task engine first |
| **P1** | Flexible Pipeline Engine | Medium | High | Unlocks non-feature use cases (review, research, etc.) |
| **P2** | Human-AI Collaboration | Medium | High | Polishes the experience once the core engine works |

### Implementation Phases

**Phase A: Foundation (Task Engine + Control Center)**
1. Enhanced task model with types, priorities, labels, parent/child
2. Task CRUD API endpoints
3. Unified WebSocket replacing all polling
4. Kanban board UI
5. One-click task generators (repo review, bug scan, manual creation)
6. Command palette (Cmd+K)
7. Split-pane layout

**Phase B: Intelligence (Multi-Agent + Pipelines)**
1. Agent registry and type definitions
2. Agent pool manager with health monitoring
3. Pipeline template system
4. Step registry with pluggable steps
5. Smart scheduler with priority queue and agent affinity
6. Agent cards with live streaming output
7. Agent-to-agent handoff protocol

**Phase C: Collaboration (Human-AI Layer)**
1. Inline code review on diffs
2. Structured feedback model
3. Agent reasoning viewer
4. Human-in-the-loop mode selector
5. Persistent feedback injection into prompts
6. Activity timeline
7. Multi-user support

---

## Quick Wins (high impact, low effort)

These can be done immediately and deliver value fast:

1. **One-click Repo Review** — add a `/api/runs/repo-review` endpoint that scans the repo and generates a list of improvement tasks. Uses existing worker infrastructure with a new prompt. Wire a "Review Repo" button in the UI. ~1 day.

2. **WebSocket for all state** — replace the 6 different polling intervals with one multiplexed WebSocket. The WebSocket infrastructure already exists for logs. ~1 day.

3. **Task creation from UI** — let users create tasks manually (not just from PRDs). Simple form: title, description, type, priority, files. ~0.5 day.

4. **Kanban view** — render existing task_queue.yaml as a kanban board (columns: ready/running/waiting/done). The data is already there. ~1 day.

5. **Cost tracking** — the events.jsonl already has timing data. Add actual token counting from worker output and show real costs in the metrics panel. ~0.5 day.

---

## Architecture Changes

```
Current:                          Target:

CLI/API → Orchestrator            CLI/API → Task Engine → Scheduler
    ↓                                 ↓           ↓
Single Worker                     Agent Pool Manager
    ↓                                 ↓
FSM (5 steps)                     Pipeline Engine (pluggable steps)
    ↓                                 ↓
File-based state                  Event Store + State Manager
    ↓                                 ↓
Polling UI                        WebSocket-driven Control Center
```

The existing FSM, state management, and worker infrastructure are solid foundations. The key architectural shifts are:

1. **Decouple task creation from pipeline execution** — tasks come from many sources, not just PRDs
2. **Add an agent pool between the scheduler and workers** — multiple agents, specialized roles
3. **Move the UI from polling to push** — single WebSocket, real-time everything
4. **Make pipelines pluggable** — step registry, templates, conditional execution

---

## Success Metrics

After implementation, the system should support:

- [ ] 5+ agents working concurrently on different tasks
- [ ] Tasks created from 5+ sources (PRD, repo review, bug scan, manual, agent-discovered)
- [ ] Real-time updates across all UI panels (< 100ms latency)
- [ ] 5+ pipeline templates for different task types
- [ ] Inline code review with agent response
- [ ] Per-agent cost tracking and budgeting
- [ ] One-click repo review generating actionable tasks
- [ ] Kanban board with drag-and-drop task management
- [ ] Command palette for power users
- [ ] Human-in-the-loop mode selection per task

---

## Detailed Implementation Checklist

### Phase A: Foundation (Task Engine + Control Center)

#### A1. Enhanced Task Model
- [ ] Define new `TaskModel` dataclass in `src/feature_prd_runner/tasks/model.py`
  - [ ] `id`, `title`, `description` (basics)
  - [ ] `type` enum: feature, bug, refactor, research, review, test, docs
  - [ ] `priority` enum: P0 (critical), P1 (high), P2 (medium), P3 (low)
  - [ ] `status` enum: backlog, ready, in_progress, in_review, done, cancelled
  - [ ] `assignee` field (agent ID or human username, nullable)
  - [ ] `labels` list (free-form tags)
  - [ ] `parent_id` / `children_ids` for task hierarchy
  - [ ] `blocked_by` / `blocks` for dependency tracking
  - [ ] `acceptance_criteria` list
  - [ ] `context_files` list (relevant file paths)
  - [ ] `related_tasks` list
  - [ ] `effort_estimate` enum: XS, S, M, L, XL
  - [ ] `pipeline_template` (which pipeline to use, nullable = auto-detect from type)
  - [ ] `created_at`, `updated_at`, `completed_at` timestamps
  - [ ] `created_by` (source: user, prd_import, repo_review, bug_scan, agent, etc.)
  - [ ] `metadata` dict for extensible key-value data
- [ ] Write migration utility to convert existing `task_queue.yaml` tasks into new model
- [ ] Add JSON schema validation for new task model
- [ ] Unit tests for task model serialization/deserialization

#### A2. Task Engine (CRUD + Persistence)
- [ ] Create `src/feature_prd_runner/tasks/engine.py`
  - [ ] `create_task()` — validate, assign ID, persist
  - [ ] `get_task(id)` — fetch by ID
  - [ ] `list_tasks(filters)` — filter by status, type, priority, assignee, labels
  - [ ] `update_task(id, changes)` — partial update with validation
  - [ ] `delete_task(id)` — soft delete (set status=cancelled)
  - [ ] `bulk_create_tasks(tasks)` — for generators that produce multiple tasks
  - [ ] `reorder_tasks(task_ids)` — manual priority reordering
- [ ] Create `src/feature_prd_runner/tasks/store.py`
  - [ ] File-based storage (YAML) for backward compat
  - [ ] Thread-safe read/write with file locking
  - [ ] Atomic writes (write temp → rename)
  - [ ] Index by ID, status, type for fast lookups
- [ ] Dependency graph management
  - [ ] `add_dependency(task_id, depends_on_id)` with cycle detection
  - [ ] `remove_dependency(task_id, depends_on_id)`
  - [ ] `get_ready_tasks()` — tasks with all deps satisfied
  - [ ] `get_blocked_tasks()` — tasks waiting on dependencies
  - [ ] Topological sort for execution ordering
- [ ] Unit tests for all CRUD operations
- [ ] Unit tests for dependency graph cycle detection
- [ ] Integration test: create → update → list → delete flow

#### A3. Task API Endpoints
- [ ] `GET /api/v2/tasks` — list with filters (status, type, priority, assignee, label, search)
- [ ] `POST /api/v2/tasks` — create single task
- [ ] `GET /api/v2/tasks/{id}` — get task detail
- [ ] `PATCH /api/v2/tasks/{id}` — partial update
- [ ] `DELETE /api/v2/tasks/{id}` — soft delete
- [ ] `POST /api/v2/tasks/bulk` — bulk create
- [ ] `POST /api/v2/tasks/{id}/assign` — assign to agent or human
- [ ] `POST /api/v2/tasks/{id}/unassign` — remove assignment
- [ ] `POST /api/v2/tasks/{id}/transition` — status transition with validation
- [ ] `GET /api/v2/tasks/{id}/dependencies` — get dependency graph for a task
- [ ] `POST /api/v2/tasks/{id}/dependencies` — add dependency
- [ ] `DELETE /api/v2/tasks/{id}/dependencies/{dep_id}` — remove dependency
- [ ] `GET /api/v2/tasks/board` — board view (tasks grouped by status columns)
- [ ] `POST /api/v2/tasks/reorder` — reorder within a column
- [ ] Request/response validation with Pydantic models
- [ ] API tests for all endpoints
- [ ] Backward compat: keep existing `/api/tasks` working during migration

#### A4. Task Generators (One-Click Sources)
- [ ] Create `src/feature_prd_runner/tasks/sources/base.py`
  - [ ] `TaskGenerator` base class with `generate(project_dir) -> list[Task]` interface
  - [ ] Progress callback for long-running generators
  - [ ] Error handling and partial results
- [ ] `src/feature_prd_runner/tasks/sources/repo_review.py`
  - [ ] Scan repo structure (file organization, naming conventions)
  - [ ] Analyze code quality signals (complexity, duplication, test coverage)
  - [ ] Check for missing tests, docs, type annotations
  - [ ] Identify dead code and unused dependencies
  - [ ] Generate prioritized improvement tasks
  - [ ] Use AI worker to analyze and categorize findings
- [ ] `src/feature_prd_runner/tasks/sources/prd_import.py`
  - [ ] Parse markdown PRD into phases (reuse existing plan logic)
  - [ ] Convert each phase into a task with acceptance criteria
  - [ ] Set up parent/child relationships (PRD → phases)
  - [ ] Preserve dependency ordering from PRD
- [ ] `src/feature_prd_runner/tasks/sources/bug_scan.py`
  - [ ] Run test suite, collect failures
  - [ ] Run linter, collect violations
  - [ ] Run type checker, collect errors
  - [ ] Create one task per failure/violation with context
  - [ ] Deduplicate (same root cause → one task)
- [ ] `src/feature_prd_runner/tasks/sources/security_audit.py`
  - [ ] Scan dependencies for known vulnerabilities (pip-audit / npm audit / govulncheck)
  - [ ] Check for hardcoded secrets patterns
  - [ ] Identify common security anti-patterns (SQL injection, XSS, etc.)
  - [ ] Generate fix tasks with severity ratings
- [ ] `src/feature_prd_runner/tasks/sources/enhancement_brainstorm.py`
  - [ ] Analyze codebase capabilities and architecture
  - [ ] Use AI to brainstorm feature ideas and improvements
  - [ ] Generate tasks with descriptions and effort estimates
  - [ ] Categorize by type (feature, refactor, performance, UX)
- [ ] `POST /api/v2/tasks/generate/{source}` endpoint for each generator
- [ ] Progress streaming via WebSocket during generation
- [ ] Unit tests for each generator with mock repos

#### A5. Unified WebSocket (Replace All Polling)
- [ ] Design multiplexed WebSocket protocol
  - [ ] Channel-based: `{ channel: "tasks" | "agents" | "logs" | "metrics" | "notifications", event: string, data: any }`
  - [ ] Client subscribes to channels: `{ action: "subscribe", channels: ["tasks", "agents"] }`
  - [ ] Server pushes on any state change (not on timer)
- [ ] Refactor `server/websocket.py`
  - [ ] Single WebSocket endpoint: `WS /ws`
  - [ ] Channel multiplexing with subscription management
  - [ ] Connection registry (track connected clients)
  - [ ] Heartbeat/keepalive (30s ping/pong)
  - [ ] Automatic reconnection protocol (client sends last_event_id)
  - [ ] Backpressure handling for slow clients
- [ ] Server-side event emission
  - [ ] Hook into task engine: emit on create/update/delete/transition
  - [ ] Hook into agent pool: emit on spawn/progress/complete/error
  - [ ] Hook into pipeline: emit on step start/complete/error
  - [ ] Hook into metrics: emit on token usage, cost update
  - [ ] Notification events: approval needed, agent blocked, task complete, budget warning
- [ ] Create `web/src/contexts/WebSocketContext.tsx`
  - [ ] Single WebSocket connection manager
  - [ ] Auto-reconnect with exponential backoff
  - [ ] Channel subscription hooks: `useChannel("tasks", callback)`
  - [ ] Connection status indicator (connected/reconnecting/disconnected)
  - [ ] Event deduplication (by event ID)
  - [ ] Offline queue (buffer actions while disconnected)
- [ ] Remove all polling from existing components
  - [ ] `TasksPanel` — replace `setInterval` with WebSocket subscription
  - [ ] `RunsPanel` — replace `setInterval` with WebSocket subscription
  - [ ] `MetricsPanel` — replace `setInterval` with WebSocket subscription
  - [ ] `PhaseTimeline` — replace `setInterval` with WebSocket subscription
  - [ ] `ApprovalGate` — replace `setInterval` with WebSocket subscription
  - [ ] `BreakpointsPanel` — replace `setInterval` with WebSocket subscription
  - [ ] `DependencyGraph` — replace `setInterval` with WebSocket subscription
  - [ ] `Chat` — replace `setInterval` with WebSocket subscription
- [ ] Integration tests: WebSocket connection, subscription, event delivery
- [ ] Load test: 10 concurrent clients, 100 events/sec

#### A6. Kanban Board UI
- [ ] Create `web/src/components/KanbanBoard/KanbanBoard.tsx`
  - [ ] Column layout: Backlog | Ready | In Progress | In Review | Done
  - [ ] Task cards within columns showing: title, type badge, priority badge, assignee avatar, labels
  - [ ] Card expand → task detail panel (description, acceptance criteria, context files, activity)
  - [ ] Drag-and-drop between columns (status transitions)
  - [ ] Drag-and-drop within columns (priority reordering)
  - [ ] Column task counts and progress indicators
- [ ] Create `web/src/components/KanbanBoard/TaskCard.tsx`
  - [ ] Compact card: title, type icon, priority color bar, assignee
  - [ ] Hover: show description preview, labels, effort estimate
  - [ ] Click: open task detail sidebar
  - [ ] Context menu: assign, change priority, add label, delete
  - [ ] Visual indicators: blocked (lock icon), has children (tree icon), running agent (pulse)
- [ ] Create `web/src/components/KanbanBoard/TaskDetail.tsx`
  - [ ] Full task view in slide-over panel
  - [ ] Editable fields: title, description, type, priority, labels, acceptance criteria
  - [ ] Dependency viewer: shows blocked_by and blocks tasks
  - [ ] Activity log: status changes, comments, agent actions
  - [ ] File context: list of relevant files with quick-open
  - [ ] Assign/unassign controls
  - [ ] Pipeline step progress (if task is in progress)
- [ ] Create `web/src/components/KanbanBoard/CreateTaskModal.tsx`
  - [ ] Form: title, description, type (dropdown), priority (dropdown)
  - [ ] Optional: labels, acceptance criteria, context files, parent task
  - [ ] Optional: effort estimate, pipeline template override
  - [ ] Submit → POST /api/v2/tasks
- [ ] Create `web/src/components/KanbanBoard/BoardFilters.tsx`
  - [ ] Filter by: type, priority, assignee, label, search text
  - [ ] Sort by: priority, created date, updated date, effort
  - [ ] Group by: type, assignee, priority (alternative to kanban columns)
  - [ ] Save filter presets
- [ ] Wire to WebSocket for real-time updates (cards appear/move/update live)
- [ ] Keyboard shortcuts: N (new task), / (search), arrow keys (navigate), Enter (open)
- [ ] Unit tests for TaskCard, TaskDetail, CreateTaskModal
- [ ] Integration test: create task → appears on board → drag to in_progress → WebSocket update

#### A7. Command Palette
- [ ] Create `web/src/components/CommandPalette/CommandPalette.tsx`
  - [ ] Trigger: Cmd+K (Mac) / Ctrl+K (Windows/Linux)
  - [ ] Fuzzy search across commands and tasks
  - [ ] Keyboard navigation: arrow keys, Enter to select, Escape to close
  - [ ] Recent commands history
- [ ] Command categories
  - [ ] **Tasks**: Create task, Search tasks, Open task by ID
  - [ ] **Generators**: Run repo review, Run bug scan, Run security audit, Run enhancement brainstorm
  - [ ] **Agents**: View agents, Pause all agents, Resume all agents
  - [ ] **Navigation**: Go to Kanban board, Go to metrics, Go to logs, Go to settings
  - [ ] **Actions**: Start new run, Stop run, Retry task, Skip step
  - [ ] **View**: Toggle dark mode, Toggle split pane, Toggle full screen
- [ ] Command registration system (components register their commands)
- [ ] Styled: floating overlay, dimmed background, search input at top, results below
- [ ] Tests for fuzzy search, keyboard navigation

#### A8. Split-Pane Layout
- [ ] Create `web/src/components/Layout/SplitPane.tsx`
  - [ ] Resizable panes with drag handle
  - [ ] Left pane: task board (Kanban) or agent overview
  - [ ] Center pane: active work area (contextual based on selection)
  - [ ] Right pane: context panel (collapsible)
  - [ ] Persist pane sizes in localStorage
  - [ ] Collapse/expand panes with double-click on divider
- [ ] Create `web/src/components/Layout/WorkArea.tsx`
  - [ ] Shows content based on current selection:
    - Task selected → task detail + agent output + diff viewer
    - Agent selected → agent live output + current task + file changes
    - No selection → dashboard overview / activity feed
  - [ ] Tab bar for multiple open items
  - [ ] Split horizontal for side-by-side views
- [ ] Create `web/src/components/Layout/ContextPanel.tsx`
  - [ ] File tree for current project
  - [ ] Related tasks for selected item
  - [ ] Chat/feedback panel for selected task
  - [ ] Quick actions relevant to current context
- [ ] Refactor `App.tsx` from single-column scroll to split-pane layout
- [ ] Responsive: collapse to single pane on narrow screens
- [ ] Tests for pane resizing, collapse/expand, content switching

#### A9. Dark Mode
- [ ] Extend `web/src/styles/variables.css` with dark theme variables
  - [ ] Background colors (surface, card, input, hover)
  - [ ] Text colors (primary, secondary, muted, inverse)
  - [ ] Border colors
  - [ ] Status colors (success, error, warning, info) — adjusted for dark backgrounds
  - [ ] Syntax highlighting colors for diff viewer
- [ ] Create `web/src/contexts/ThemeContext.tsx`
  - [ ] Theme state: light | dark | system
  - [ ] Toggle function
  - [ ] Persist preference in localStorage
  - [ ] Respect `prefers-color-scheme` media query for system mode
  - [ ] Apply `data-theme` attribute to document root
- [ ] Update all component CSS to use theme variables (not hardcoded colors)
- [ ] Theme toggle in header and command palette
- [ ] Test: toggle theme, refresh persists, system preference detected

---

### Phase B: Intelligence (Multi-Agent + Pipelines)

#### B1. Agent Registry
- [ ] Create `src/feature_prd_runner/agents/registry.py`
  - [ ] `AgentType` dataclass: name, role, model, system_prompt, tools, resource_limits
  - [ ] Built-in agent types:
    - [ ] `architect` — plans phases, breaks down work, designs architecture
    - [ ] `implementer` — writes code, applies changes
    - [ ] `reviewer` — reviews code against criteria, finds issues
    - [ ] `tester` — writes and runs tests, verifies behavior
    - [ ] `researcher` — gathers information, reads docs, analyzes patterns
    - [ ] `debugger` — diagnoses failures, reads logs, identifies root causes
  - [ ] `register_agent_type(agent_type)` — add custom agent types
  - [ ] `get_agent_type(name)` — lookup by name
  - [ ] `list_agent_types()` — all registered types
- [ ] Agent type configuration in `.prd_runner/agents.yaml`
  - [ ] Override model, prompts, limits per project
  - [ ] Define custom agent types
  - [ ] Set default agent assignments per task type
- [ ] Resource limit model
  - [ ] `max_tokens_per_task` — token budget per task
  - [ ] `max_cost_per_task` — dollar cost cap
  - [ ] `max_time_per_task` — wall clock timeout
  - [ ] `max_concurrent_files` — file edit concurrency limit
  - [ ] `max_retries` — retry limit before escalation
- [ ] Unit tests for registration, lookup, config loading

#### B2. Agent Pool Manager
- [ ] Create `src/feature_prd_runner/agents/pool.py`
  - [ ] `AgentInstance` dataclass: id, type, status, current_task, started_at, metrics
  - [ ] `spawn_agent(agent_type, task)` — create new agent subprocess/worker
  - [ ] `kill_agent(agent_id)` — terminate agent
  - [ ] `pause_agent(agent_id)` — pause (save state, stop processing)
  - [ ] `resume_agent(agent_id)` — resume from saved state
  - [ ] `reassign_agent(agent_id, new_task)` — redirect to different task
  - [ ] `list_agents()` — all active agents with status
  - [ ] `get_agent(id)` — single agent detail
- [ ] Health monitoring
  - [ ] Heartbeat check (configurable interval)
  - [ ] Memory/CPU usage tracking per agent
  - [ ] Token usage tracking per agent (real-time)
  - [ ] Cost accumulation per agent
  - [ ] Auto-restart on crash (up to max_retries)
  - [ ] Dead agent detection and cleanup
- [ ] Pool sizing
  - [ ] `max_agents` — total agent limit
  - [ ] `max_agents_per_type` — per-type limit
  - [ ] Dynamic scaling: spawn more agents when task queue grows
  - [ ] Graceful shutdown: finish current task before killing
- [ ] Agent output streaming
  - [ ] Capture stdout/stderr in real-time
  - [ ] Parse progress.json updates
  - [ ] Forward to WebSocket channel
  - [ ] Store in run artifacts
- [ ] API endpoints
  - [ ] `GET /api/v2/agents` — list all agents
  - [ ] `GET /api/v2/agents/{id}` — agent detail + live metrics
  - [ ] `POST /api/v2/agents/{id}/pause` — pause agent
  - [ ] `POST /api/v2/agents/{id}/resume` — resume agent
  - [ ] `POST /api/v2/agents/{id}/kill` — terminate agent
  - [ ] `POST /api/v2/agents/{id}/reassign` — reassign to new task
  - [ ] `POST /api/v2/agents/{id}/message` — send guidance to agent
- [ ] WebSocket events: agent_spawned, agent_progress, agent_completed, agent_error, agent_killed
- [ ] Unit tests for spawn, kill, health monitoring
- [ ] Integration test: spawn agent → assign task → monitor → complete → cleanup

#### B3. Agent Context Bus
- [ ] Create `src/feature_prd_runner/agents/context_bus.py`
  - [ ] Shared artifact store (agents read/write named artifacts)
  - [ ] `publish(agent_id, topic, data)` — agent publishes context
  - [ ] `subscribe(agent_id, topic, callback)` — agent subscribes to updates
  - [ ] `query(topic, filters)` — pull-based context lookup
  - [ ] Topics: `decisions`, `discoveries`, `file_changes`, `test_results`, `questions`
- [ ] Cross-agent visibility
  - [ ] Implementer can see reviewer feedback without file round-trip
  - [ ] Researcher results available to all implementers
  - [ ] Tester results trigger reviewer re-evaluation
  - [ ] Architect decisions cascade to all agents
- [ ] Context persistence (survives agent restart)
- [ ] Unit tests for publish/subscribe, query, persistence

#### B4. Agent-to-Agent Handoff
- [ ] Create `src/feature_prd_runner/agents/handoff.py`
  - [ ] `HandoffProtocol`: structured message from one agent to another
  - [ ] Handoff types:
    - [ ] `review_feedback` — reviewer → implementer (issues to fix)
    - [ ] `research_result` — researcher → implementer (context found)
    - [ ] `test_failure` — tester → debugger (failure to diagnose)
    - [ ] `architecture_decision` — architect → implementer (design to follow)
    - [ ] `escalation` — any agent → human (can't proceed)
  - [ ] Handoff includes: source agent, target agent/type, payload, priority, context
  - [ ] Auto-routing: system determines best target agent for handoff
- [ ] Direct feedback loop (no file-based retry)
  - [ ] Reviewer creates issues → directly injected into implementer prompt
  - [ ] Implementer addresses issues → reviewer re-evaluates only changed items
  - [ ] Feedback history preserved across iterations
- [ ] Unit tests for handoff protocol, routing, feedback loops

#### B5. Agent Cards UI
- [ ] Create `web/src/components/AgentCard/AgentCard.tsx`
  - [ ] Agent avatar/icon by type (architect, implementer, reviewer, etc.)
  - [ ] Status indicator: running (pulse), paused (gray), idle (dim), error (red)
  - [ ] Current task: title, type, phase
  - [ ] Live metrics: tokens used, cost so far, elapsed time
  - [ ] Current file being edited (if applicable)
  - [ ] Mini progress bar for current step
- [ ] Create `web/src/components/AgentCard/AgentStream.tsx`
  - [ ] Live streaming agent output (reasoning, actions, file edits)
  - [ ] Collapsible sections for different output types
  - [ ] Syntax-highlighted code blocks
  - [ ] Scroll lock / auto-scroll toggle
- [ ] Create `web/src/components/AgentCard/AgentControls.tsx`
  - [ ] Pause / Resume button
  - [ ] Kill button (with confirmation)
  - [ ] Reassign button (opens task picker)
  - [ ] Send message button (inline text input)
  - [ ] View full logs button
- [ ] Create `web/src/components/AgentOverview/AgentOverview.tsx`
  - [ ] Grid of all agent cards
  - [ ] Summary: total agents, running, idle, errored
  - [ ] Total cost / tokens across all agents
  - [ ] "Spawn Agent" button (select type + task)
- [ ] Wire to WebSocket agent channel for real-time updates
- [ ] Tests for AgentCard rendering, controls, streaming

#### B6. Pipeline Template System
- [ ] Create `src/feature_prd_runner/pipelines/registry.py`
  - [ ] `PipelineTemplate` dataclass: name, description, steps, conditions
  - [ ] Built-in templates:
    - [ ] `feature` — plan → implement → verify → review → commit
    - [ ] `bug_fix` — reproduce → diagnose → fix → verify → review → commit
    - [ ] `refactor` — analyze → plan → implement → verify → review → commit
    - [ ] `research` — gather → analyze → summarize → report
    - [ ] `docs` — analyze_code → write → review → commit
    - [ ] `repo_review` — scan → analyze → generate_tasks
    - [ ] `security_audit` — scan_deps → scan_code → generate_report → generate_tasks
    - [ ] `test_writing` — analyze_coverage → plan_tests → implement_tests → verify → commit
  - [ ] `register_template(template)` — add custom templates
  - [ ] `get_template(name)` — lookup
  - [ ] `list_templates()` — all available
  - [ ] `resolve_template(task)` — auto-select based on task type
- [ ] Custom templates from YAML files in `.prd_runner/pipelines/`
- [ ] Unit tests for template resolution

#### B7. Step Registry
- [ ] Create `src/feature_prd_runner/pipelines/steps/base.py`
  - [ ] `PipelineStep` base class: `name`, `execute(context) -> StepResult`
  - [ ] `StepResult`: status, artifacts, next_step_override, skip_remaining
  - [ ] Step context: task, agent, project_dir, previous_results, config
- [ ] Implement all step types
  - [ ] `plan` — generate implementation plan (existing plan_impl logic)
  - [ ] `implement` — run worker to make changes (existing implement logic)
  - [ ] `verify` — run tests/lint/format/typecheck (existing verify logic)
  - [ ] `review` — structured code review (existing review logic)
  - [ ] `commit` — git commit and push (existing commit logic)
  - [ ] `reproduce` — attempt to reproduce a bug from description
  - [ ] `diagnose` — analyze logs/errors to find root cause
  - [ ] `gather` — collect information from codebase/docs/web
  - [ ] `analyze` — analyze collected information
  - [ ] `summarize` — create concise summary
  - [ ] `report` — generate structured report
  - [ ] `scan_deps` — scan dependencies for vulnerabilities
  - [ ] `scan_code` — static analysis for security issues
  - [ ] `generate_tasks` — create tasks from analysis results
  - [ ] `analyze_coverage` — check test coverage gaps
  - [ ] `plan_tests` — plan which tests to write
- [ ] Conditional step logic
  - [ ] Skip conditions: `skip_if: "change_lines < 10"`, `skip_if: "task.type == 'docs'"`
  - [ ] Retry conditions: `retry_if: "exit_code != 0"`, `max_retries: 3`
  - [ ] Branching: `on_failure: "diagnose"`, `on_success: "commit"`
- [ ] Unit tests for each step type
- [ ] Integration test: custom pipeline with mixed steps

#### B8. Smart Scheduler
- [ ] Create `src/feature_prd_runner/tasks/scheduler.py`
  - [ ] Priority queue implementation
    - [ ] Sort by: priority (P0 first), then created_at (oldest first)
    - [ ] Preemption: P0 task can interrupt P2 agent (with state save)
    - [ ] Fair scheduling: prevent starvation of low-priority tasks
  - [ ] Agent affinity matching
    - [ ] Task type → preferred agent type mapping
    - [ ] History-based affinity (agent already has context for this area)
    - [ ] Load balancing across available agents
  - [ ] Parallel dispatch
    - [ ] Compute independent tasks (no dependency conflicts)
    - [ ] Respect max_agents limit
    - [ ] File conflict detection (don't assign two tasks that edit same files)
  - [ ] Scheduling loop
    - [ ] `schedule_next()` — pick highest priority ready task, assign to best available agent
    - [ ] `on_task_complete(task_id)` — unblock dependents, schedule next
    - [ ] `on_agent_free(agent_id)` — find next task for this agent
    - [ ] `rebalance()` — reassign tasks after priority changes
- [ ] API endpoint: `GET /api/v2/scheduler/queue` — view scheduling queue and decisions
- [ ] Unit tests for priority ordering, affinity matching, conflict detection
- [ ] Integration test: 10 tasks, 3 agents, dependency graph → correct execution order

---

### Phase C: Collaboration (Human-AI Layer)

#### C1. Inline Code Review
- [ ] Create `web/src/components/InlineReview/DiffViewer.tsx`
  - [ ] Side-by-side diff view (old vs new)
  - [ ] Unified diff view (toggle)
  - [ ] Syntax highlighting per language
  - [ ] Line numbers for both sides
  - [ ] Collapsible unchanged regions
  - [ ] File navigation (prev/next file)
- [ ] Create `web/src/components/InlineReview/LineComment.tsx`
  - [ ] Click on line number → inline comment form appears
  - [ ] Comment input with markdown support
  - [ ] Comment actions: Submit, Cancel, Edit, Delete
  - [ ] Thread replies (nested comments on same line)
  - [ ] Resolve/unresolve thread
- [ ] Create `web/src/components/InlineReview/ReviewSummary.tsx`
  - [ ] Overall review status: approved, changes_requested, pending
  - [ ] Comment count by file
  - [ ] Unresolved thread count
  - [ ] Submit review button (approve / request changes)
- [ ] Backend: `src/feature_prd_runner/collaboration/review.py`
  - [ ] `ReviewComment` model: file, line, content, author, created_at, resolved
  - [ ] `ReviewThread` model: comments grouped by file+line
  - [ ] Persist review comments per task
  - [ ] Inject unresolved comments into agent prompt on retry
- [ ] API endpoints
  - [ ] `GET /api/v2/tasks/{id}/review/comments` — all review comments
  - [ ] `POST /api/v2/tasks/{id}/review/comments` — add comment (file, line, content)
  - [ ] `PATCH /api/v2/tasks/{id}/review/comments/{cid}` — edit/resolve comment
  - [ ] `DELETE /api/v2/tasks/{id}/review/comments/{cid}` — delete comment
  - [ ] `POST /api/v2/tasks/{id}/review/submit` — submit review (approve/request_changes)
- [ ] Tests for diff rendering, comment CRUD, thread resolution

#### C2. Structured Feedback Model
- [ ] Create `src/feature_prd_runner/collaboration/feedback.py`
  - [ ] `Feedback` dataclass: type, target, content, action, persistent, created_by, created_at
  - [ ] Feedback types:
    - [ ] `approach_change` — "use X approach instead of Y"
    - [ ] `library_preference` — "use library X instead of Y"
    - [ ] `file_restriction` — "do not modify file X"
    - [ ] `style_preference` — "follow pattern X for this"
    - [ ] `requirement_clarification` — "the requirement means X"
    - [ ] `general_guidance` — free-form guidance
  - [ ] Persistence levels:
    - [ ] `task` — applies only to current task
    - [ ] `project` — applies to all tasks in project
    - [ ] `session` — applies to current session only
  - [ ] `get_active_feedback(task_id)` — all feedback applicable to a task
  - [ ] Feedback injection: append to agent prompt as "Human Guidance" section
- [ ] Create `web/src/components/FeedbackPanel/FeedbackPanel.tsx`
  - [ ] Feedback type selector (dropdown or tabs)
  - [ ] Type-specific form fields
  - [ ] Active feedback list (with edit/delete)
  - [ ] Persistence scope selector
  - [ ] Visual indicator when feedback is being applied to an agent
- [ ] API endpoints
  - [ ] `GET /api/v2/tasks/{id}/feedback` — list feedback for task
  - [ ] `POST /api/v2/tasks/{id}/feedback` — add feedback
  - [ ] `PATCH /api/v2/feedback/{fid}` — edit feedback
  - [ ] `DELETE /api/v2/feedback/{fid}` — delete feedback
  - [ ] `GET /api/v2/projects/{dir}/feedback` — project-level feedback
- [ ] Unit tests for feedback model, injection, persistence

#### C3. Agent Reasoning Viewer
- [ ] Create `src/feature_prd_runner/collaboration/reasoning.py`
  - [ ] `ReasoningEntry` model: step, thought, decision, alternatives, confidence, timestamp
  - [ ] Capture reasoning from agent progress.json `claims` and `next_steps`
  - [ ] Parse structured reasoning from agent output (when available)
  - [ ] Store reasoning chain per task per agent
- [ ] Create `web/src/components/ReasoningViewer/ReasoningViewer.tsx`
  - [ ] Collapsible tree view of reasoning chain
  - [ ] Each node: step name, decision made, confidence level
  - [ ] Expand: see alternatives considered, trade-offs, rationale
  - [ ] Timeline view: reasoning over time
  - [ ] Highlight decision points where human input could help
- [ ] API endpoint: `GET /api/v2/tasks/{id}/reasoning` — reasoning chain for task
- [ ] Tests for reasoning capture, tree rendering

#### C4. Human-in-the-Loop Mode Selector
- [ ] Create `src/feature_prd_runner/collaboration/modes.py`
  - [ ] Mode definitions:
    - [ ] `autopilot` — agents run freely, human reviews at end
    - [ ] `supervised` — agents propose each step, human approves before execution
    - [ ] `collaborative` — human and agents alternate on task steps
    - [ ] `review_only` — agents implement, human reviews all code changes
  - [ ] Mode enforcement in pipeline engine
    - [ ] Insert approval gates based on mode
    - [ ] `supervised`: gate before every step
    - [ ] `review_only`: gate only before commit
    - [ ] `collaborative`: gate at designated handoff points
  - [ ] Mode can be set per-task or project-wide
  - [ ] Mode can be changed mid-task
- [ ] Create `web/src/components/ModeSelector/ModeSelector.tsx`
  - [ ] Dropdown or segmented control: Autopilot | Supervised | Collaborative | Review Only
  - [ ] Mode description tooltip
  - [ ] Current mode indicator in header
  - [ ] Per-task mode override in task detail
- [ ] API endpoints
  - [ ] `GET /api/v2/settings/mode` — current mode
  - [ ] `PUT /api/v2/settings/mode` — change mode
  - [ ] `PUT /api/v2/tasks/{id}/mode` — per-task mode override
- [ ] Tests for mode enforcement, gate insertion, mode switching

#### C5. Persistent Feedback Injection
- [ ] Modify prompt builders in `src/feature_prd_runner/prompts.py`
  - [ ] Add `## Human Guidance` section to all prompts
  - [ ] Include task-level feedback
  - [ ] Include project-level feedback
  - [ ] Include inline review comments (unresolved only)
  - [ ] Include structured feedback (approach changes, preferences)
  - [ ] Priority ordering: most recent and most specific first
  - [ ] Token-aware: truncate oldest feedback if context window is tight
- [ ] Feedback effectiveness tracking
  - [ ] Track if agent addressed feedback in next iteration
  - [ ] Surface unaddressed feedback to human
  - [ ] Auto-escalate repeated unaddressed feedback
- [ ] Tests for prompt injection, truncation, effectiveness tracking

#### C6. Activity Timeline
- [ ] Create `web/src/components/ActivityTimeline/ActivityTimeline.tsx`
  - [ ] Chronological list of all events on a task
  - [ ] Event types: status_change, agent_action, human_feedback, code_change, review_comment, approval, error
  - [ ] Each event: timestamp, actor (agent or human), type icon, description
  - [ ] Expandable details for complex events (diffs, error logs, reasoning)
  - [ ] Filter by: event type, actor
  - [ ] Real-time updates via WebSocket
- [ ] Backend: `src/feature_prd_runner/collaboration/timeline.py`
  - [ ] Aggregate events from: task state changes, agent progress, feedback, reviews, commits
  - [ ] Unified event format for API
- [ ] API endpoint: `GET /api/v2/tasks/{id}/timeline` — paginated timeline
- [ ] Tests for event aggregation, rendering, real-time updates

#### C7. Multi-User Support
- [ ] Enhance `server/auth.py`
  - [ ] User model: id, username, display_name, role (admin, developer, reviewer)
  - [ ] Role-based permissions: who can create tasks, assign agents, approve, kill agents
  - [ ] Session management (multiple concurrent sessions)
- [ ] User presence
  - [ ] Track which users are online (via WebSocket connections)
  - [ ] Show online users in header
  - [ ] Show who is viewing/editing a task (collaborative awareness)
- [ ] Per-user views
  - [ ] "My tasks" filter (tasks assigned to me)
  - [ ] "My reviews" filter (tasks awaiting my review)
  - [ ] Notification preferences per user
- [ ] API changes
  - [ ] `GET /api/v2/users` — list users
  - [ ] `GET /api/v2/users/me` — current user detail
  - [ ] `GET /api/v2/users/online` — online users
  - [ ] User ID attached to all actions (feedback, comments, approvals)
- [ ] Tests for auth, permissions, presence tracking

---

### Ongoing / Cross-Cutting

#### X1. Testing & Quality
- [ ] Unit test coverage target: 80%+ for all new modules
- [ ] Integration test suite for full workflows (task create → agent assign → execute → review → complete)
- [ ] E2E test with Playwright for critical UI flows (kanban, command palette, agent cards)
- [ ] Performance benchmarks: WebSocket throughput, task scheduling latency, UI render time
- [ ] CI pipeline: lint + type check + unit tests + integration tests on every PR

#### X2. Documentation
- [ ] Update README with new architecture overview
- [ ] API documentation (OpenAPI/Swagger for all v2 endpoints)
- [ ] Agent type configuration guide
- [ ] Pipeline template authoring guide
- [ ] User guide: Kanban board, command palette, agent management
- [ ] Migration guide: v1 → v2 task model

#### X3. Cost & Token Tracking
- [ ] Real token counting from worker API responses (not estimates)
- [ ] Per-agent cost accumulation
- [ ] Per-task cost accumulation
- [ ] Project-wide budget limits with alerts
- [ ] Cost breakdown dashboard (by agent type, task type, pipeline step)
- [ ] Budget exceeded → pause agents → notify human

#### X4. Notifications System
- [ ] Backend notification model: type, severity, message, target_user, read status
- [ ] Notification triggers:
  - [ ] Agent blocked / needs human input
  - [ ] Task completed / failed
  - [ ] Review requested
  - [ ] Budget warning (80% / 100%)
  - [ ] Agent health issue (crash, timeout)
- [ ] Desktop notifications (via Notification API)
- [ ] Sound alerts (configurable, off by default)
- [ ] Notification drawer in UI (bell icon with unread count)
- [ ] API: `GET /api/v2/notifications`, `POST /api/v2/notifications/{id}/read`
