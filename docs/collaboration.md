# Human-AI Collaboration (Phase C)

This document describes the Phase C human-AI collaboration features that enable rich, bidirectional interaction between humans and agents during task execution. These features go beyond the basic HITL steering and approval gates (see [HUMAN_IN_THE_LOOP.md](HUMAN_IN_THE_LOOP.md)) to provide inline code review, structured feedback, agent reasoning transparency, and real-time collaboration tooling.

## Overview

The collaboration system is built around several interconnected components:

- **Inline Review** -- Line-level commenting on code diffs, similar to GitHub pull request reviews.
- **Structured Feedback** -- Typed, prioritized guidance that agents can programmatically incorporate into their prompts.
- **Reasoning Viewer** -- Step-by-step display of agent thinking for transparency.
- **HITL Modes** -- Configurable levels of human involvement (autopilot through fully supervised).
- **Activity Timeline** -- Unified chronological view aggregating all events on a task.
- **Approval Gate** -- Blocking mechanism for actions that require human sign-off.
- **Notification Center** -- Real-time notifications via WebSocket with desktop and sound alerts.
- **User Management & Presence** -- Role-based user profiles and online presence tracking.

All API endpoints live under `/api/v2/collaboration/` (with agent reasoning under `/api/v2/agents/reasoning/`). The web UI components consume these endpoints and communicate in real time via WebSocket channels.

---

## Inline Review

**Backend:** `ReviewComment` model in `src/feature_prd_runner/collaboration/feedback.py`
**Frontend:** `web/src/components/InlineReview/InlineReview.tsx`
**API prefix:** `/api/v2/collaboration/comments`

The inline review system enables line-level commenting on code diffs, similar to how GitHub pull request reviews work. Users can click on any diff line to leave a comment, start threaded discussions, and resolve conversations when issues are addressed.

### Features

- **Line-level commenting** -- Click any line in a unified or split diff view to add a comment. Each comment records the file path, line number, and line type (added, removed, or context).
- **Threaded replies** -- Comments support threading via `parent_id`. Replies are nested under their parent comment, enabling focused conversations on specific code changes.
- **Resolving comments** -- Individual comments can be marked as resolved. The UI shows a count of open (unresolved) comments per file.
- **Side-by-side diff view** -- Toggle between unified and split diff rendering. The split view pairs removed and added lines side by side for easier comparison.
- **Syntax highlighting** -- Basic keyword and string highlighting for common languages (JavaScript, Python, TypeScript).

### ReviewComment Data Model

```python
@dataclass
class ReviewComment:
    id: str               # Auto-generated, e.g. "rc-a1b2c3d4"
    task_id: str
    file_path: str
    line_number: int
    line_type: str        # "added", "removed", "context"
    body: str
    resolved: bool
    parent_id: str | None # Links to another ReviewComment for threading
    author: str           # Username or agent_id
    author_type: str      # "human" or "agent"
    created_at: str       # ISO 8601
    resolved_at: str | None
```

### API Usage

**Add a comment:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/comments \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "phase-1",
    "file_path": "src/auth/login.py",
    "line_number": 42,
    "body": "This should validate the token expiry before proceeding.",
    "line_type": "added",
    "author": "alice"
  }'
```

**Add a threaded reply:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/comments \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "phase-1",
    "file_path": "src/auth/login.py",
    "line_number": 42,
    "body": "Good point, I will add an expiry check.",
    "parent_id": "rc-a1b2c3d4"
  }'
```

**Get comments for a task (optionally filtered by file):**

```bash
# All comments
curl http://localhost:8000/api/v2/collaboration/comments/phase-1

# Only unresolved comments for a specific file
curl "http://localhost:8000/api/v2/collaboration/comments/phase-1?file_path=src/auth/login.py&unresolved_only=true"
```

**Get replies to a comment:**

```bash
curl http://localhost:8000/api/v2/collaboration/comments/rc-a1b2c3d4/replies
```

**Resolve a comment:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/comments/rc-a1b2c3d4/resolve
```

---

## Structured Feedback

**Backend:** `Feedback` model and `FeedbackStore` in `src/feature_prd_runner/collaboration/feedback.py`
**Frontend:** `web/src/components/FeedbackPanel/FeedbackPanel.tsx`
**API prefix:** `/api/v2/collaboration/feedback`

Structured feedback is the primary mechanism for humans to give actionable, typed guidance to agents. Unlike free-form text, each feedback item has a specific type, priority, and action that agents can programmatically incorporate into their prompts. Feedback persists across task retries so agents do not repeat the same mistakes.

### Feedback Types

| Type | Description | Example |
|------|-------------|---------|
| `approach_change` | Request a different implementation approach | "Use a decorator pattern instead of inheritance" |
| `library_swap` | Swap one library for another | "Use `httpx` instead of `requests`" |
| `file_restriction` | Prohibit modification of a file | "Do not modify `config/production.yaml`" |
| `style_preference` | Coding style guidance | "Prefer list comprehensions over map/filter" |
| `bug_report` | Report a specific bug in the code | "Off-by-one error in pagination logic" |
| `general` | Free-form guidance | Any other instruction |

### Priority Levels

| Priority | Meaning | Agent behavior |
|----------|---------|----------------|
| `must` | Mandatory requirement | Agent must follow this instruction |
| `should` | Strong preference | Agent should follow unless there is a compelling reason not to |
| `suggestion` | Nice to have | Agent may incorporate at its discretion |

### Feedback Lifecycle

Each feedback item has a status:

- **`active`** -- Feedback is live and will be included in agent prompts.
- **`addressed`** -- Agent has acted on the feedback. An optional `agent_response` field records how it was addressed.
- **`dismissed`** -- Human decided the feedback is no longer relevant.

### Token-Aware Prompt Generation

The `FeedbackStore.get_prompt_instructions()` method converts all active feedback for a task into a block of text that is injected into the agent's prompt. It is token-aware:

- Feedback items are sorted most-recent-first so the freshest guidance is prioritized.
- If the total character length exceeds the `max_chars` limit (default 4000, roughly 1000 tokens), older items are truncated with a `[Earlier feedback truncated for context window]` marker.

```python
store = FeedbackStore()
# ... add feedback items ...

# Generate prompt block (default max 4000 chars)
instructions = store.get_prompt_instructions("phase-1", max_chars=4000)
# Returns something like:
# Human feedback to incorporate:
#   - [MUST] Use 'httpx' instead of 'requests'
#   - [SHOULD] Style: Prefer list comprehensions over map/filter
#   - [SUGGESTION] Add more inline comments for complex logic
```

### Effectiveness Tracking

Track how well agents are responding to feedback:

```bash
curl http://localhost:8000/api/v2/collaboration/feedback/phase-1/effectiveness
```

Returns:

```json
{
  "total": 5,
  "addressed": 3,
  "dismissed": 1,
  "active": 1,
  "addressed_rate": 0.6,
  "unaddressed_items": [
    {
      "id": "fb-e1f2a3b4",
      "summary": "Add retry logic to API calls",
      "created_at": "2026-01-20T14:30:00Z"
    }
  ]
}
```

### API Usage

**Submit feedback:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "phase-1",
    "feedback_type": "library_swap",
    "priority": "must",
    "summary": "Use httpx instead of requests",
    "original_value": "requests",
    "replacement_value": "httpx",
    "created_by": "alice"
  }'
```

**Get feedback for a task:**

```bash
# All feedback
curl http://localhost:8000/api/v2/collaboration/feedback/phase-1

# Only active feedback of a specific type
curl "http://localhost:8000/api/v2/collaboration/feedback/phase-1?active_only=true&feedback_type=bug_report"
```

**Mark feedback as addressed (by the agent):**

```bash
curl -X POST "http://localhost:8000/api/v2/collaboration/feedback/fb-e1f2a3b4/address?agent_response=Switched%20to%20httpx"
```

**Dismiss feedback:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/feedback/fb-e1f2a3b4/dismiss
```

**Get prompt instructions for agent injection:**

```bash
curl http://localhost:8000/api/v2/collaboration/feedback/phase-1/prompt
```

Returns:

```json
{
  "instructions": "Human feedback to incorporate:\n  - [MUST] Use 'httpx' instead of 'requests'\n  - [SHOULD] Style: Prefer list comprehensions"
}
```

---

## Reasoning Viewer

**Backend:** `ReasoningStore` in `src/feature_prd_runner/collaboration/reasoning.py`
**Frontend:** `web/src/components/ReasoningViewer/ReasoningViewer.tsx`
**API prefix:** `/api/v2/agents/reasoning`

The reasoning viewer provides transparency into agent decision-making by displaying step-by-step reasoning chains. Each agent working on a task records why it made certain decisions, what it planned, and what it produced at each pipeline step.

### Data Model

**ReasoningStep** -- One step in an agent's reasoning chain:

```python
@dataclass
class ReasoningStep:
    step_name: str                   # e.g. "plan_impl", "implement", "verify"
    status: str                      # pending | running | completed | failed | skipped
    reasoning: str                   # Why the agent decided to take this action
    output: str                      # Key output or artifacts from the step
    started_at: float | None         # Unix timestamp
    completed_at: float | None       # Unix timestamp
    duration_ms: float | None        # Computed property (completed - started)
```

**AgentReasoning** -- Complete trace for one agent on one task:

```python
@dataclass
class AgentReasoning:
    agent_id: str
    agent_role: str                  # e.g. "implementer", "reviewer", "verifier"
    task_id: str
    pipeline_id: str
    steps: list[ReasoningStep]
    current_step: str | None         # Name of the currently running step
```

### Real-Time Updates

The `ReasoningViewer` frontend component subscribes to the `agents` WebSocket channel. When an agent starts or completes a step, the server broadcasts an event and the viewer automatically refreshes to show the latest state. There is no polling; updates are pushed in real time.

### UI Features

- **Expandable agent panels** -- When multiple agents are working on a task, each gets its own collapsible panel showing role and progress (e.g., "3/5 steps").
- **Step-by-step display** -- Each step shows its status icon, name, and duration. Click a step to expand its reasoning text and output.
- **Status indicators** -- Steps are visually distinguished by status: pending (circle), running (play), completed (check), failed (cross), skipped (arrow).

### API Usage

**Get reasoning for a task:**

```bash
curl http://localhost:8000/api/v2/agents/reasoning/phase-1
```

Returns:

```json
{
  "reasonings": [
    {
      "agent_id": "agent-abc12345",
      "agent_role": "implementer",
      "task_id": "phase-1",
      "pipeline_id": "pipeline-001",
      "steps": [
        {
          "step_name": "plan_impl",
          "status": "completed",
          "reasoning": "Analyzed PRD requirements and identified 3 files to modify.",
          "output": "Plan: modify auth/login.py, auth/session.py, add auth/middleware.py",
          "started_at": 1706000000.0,
          "completed_at": 1706000005.2,
          "duration_ms": 5200.0
        },
        {
          "step_name": "implement",
          "status": "running",
          "reasoning": "Implementing JWT-based authentication as specified in plan.",
          "output": "",
          "started_at": 1706000006.0,
          "completed_at": null,
          "duration_ms": null
        }
      ],
      "current_step": "implement"
    }
  ]
}
```

**Record a step start (used by agents):**

```bash
curl -X POST "http://localhost:8000/api/v2/agents/reasoning/phase-1/step/start?agent_id=agent-abc12345&agent_role=implementer&step_name=verify&reasoning=Running%20test%20suite"
```

**Record a step completion (used by agents):**

```bash
curl -X POST "http://localhost:8000/api/v2/agents/reasoning/phase-1/step/complete?agent_id=agent-abc12345&step_name=verify&status=completed&output=All%2012%20tests%20passed"
```

---

## HITL Modes

**Backend:** `HITLMode` and `ModeConfig` in `src/feature_prd_runner/collaboration/modes.py`
**Frontend:** `web/src/components/HITLModeSelector/HITLModeSelector.tsx`
**API prefix:** `/api/v2/collaboration/modes`

HITL (Human-in-the-Loop) modes control how much human involvement is required during task execution. Each mode defines which approval gates are active, whether agents can run unattended, and whether reasoning display is required.

### Available Modes

| Mode | Display Name | Description | Approval Gates | Unattended | Reasoning |
|------|-------------|-------------|----------------|------------|-----------|
| `autopilot` | Autopilot | Agents run freely. Review results when they finish. | None | Yes | No |
| `supervised` | Supervised | Agents propose at each step. You approve before they continue. | Plan, Implement, Commit | No | Yes |
| `collaborative` | Collaborative | You and agents work together. Review implementation before commit. | After Implement, Commit | No | Yes |
| `review_only` | Review Only | Agents implement. You review all changes before commit. | After Implement, Commit | Yes | No |

### Mode Configuration Details

Each mode is defined by a `ModeConfig` dataclass:

```python
@dataclass(frozen=True)
class ModeConfig:
    mode: HITLMode
    display_name: str
    description: str
    approve_before_plan: bool       # Gate before planning
    approve_before_implement: bool  # Gate before implementation starts
    approve_before_commit: bool     # Gate before git commit
    approve_after_implement: bool   # Gate after implementation completes
    allow_unattended: bool          # Can agent proceed without human presence
    require_reasoning: bool         # Must agent explain reasoning at each step
```

### Project-Level vs Task-Level Modes

Modes can be set at two levels:

1. **Project level** -- Default mode for all tasks. Set via `PUT /api/v2/collaboration/modes`.
2. **Task level** -- Override for a specific task. Set via `PUT /api/v2/collaboration/modes/task/{task_id}`. The task-level mode takes precedence when set.

The effective mode for a task is determined by: task-level override if set, otherwise project-level default.

### Checking Gate Requirements

The `should_gate()` function determines whether a specific approval gate should fire:

```python
from feature_prd_runner.collaboration.modes import should_gate

# Returns True if 'collaborative' mode requires a gate before commit
should_gate("collaborative", "before_commit")   # True
should_gate("autopilot", "before_commit")        # False
```

### API Usage

**List all available modes:**

```bash
curl http://localhost:8000/api/v2/collaboration/modes
```

**Get current project mode:**

```bash
curl http://localhost:8000/api/v2/collaboration/modes/current
```

**Set project mode:**

```bash
curl -X PUT http://localhost:8000/api/v2/collaboration/modes \
  -H "Content-Type: application/json" \
  -d '{"mode": "collaborative"}'
```

**Set task-level mode override:**

```bash
curl -X PUT http://localhost:8000/api/v2/collaboration/modes/task/phase-1 \
  -H "Content-Type: application/json" \
  -d '{"mode": "supervised"}'
```

**Get effective mode for a task:**

```bash
curl http://localhost:8000/api/v2/collaboration/modes/task/phase-1
```

Returns:

```json
{
  "task_id": "phase-1",
  "task_mode": "supervised",
  "project_mode": "collaborative",
  "effective_mode": "supervised",
  "config": {
    "mode": "supervised",
    "display_name": "Supervised",
    "approve_before_plan": true,
    "approve_before_implement": true,
    "approve_before_commit": true,
    "approve_after_implement": false,
    "allow_unattended": false,
    "require_reasoning": true
  }
}
```

**Clear a task-level override (reverts to project mode):**

```bash
curl -X DELETE http://localhost:8000/api/v2/collaboration/modes/task/phase-1
```

---

## Activity Timeline

**Backend:** `TimelineAggregator` and `StateChangeStore` in `src/feature_prd_runner/collaboration/timeline.py`
**Frontend:** `web/src/components/ActivityTimeline/ActivityTimeline.tsx`
**API prefix:** `/api/v2/collaboration/timeline`

The activity timeline provides a unified chronological view of everything that has happened on a task. It aggregates events from multiple sources into a single stream, sorted newest-first.

### Event Sources

The `TimelineAggregator` collects events from four stores:

1. **Feedback events** -- From `FeedbackStore`. Each feedback item becomes a timeline event showing its type, summary, priority, and status.
2. **Review comment events** -- From `FeedbackStore` (comments). Shows file path, line number, author, and resolution status.
3. **Reasoning events** -- From `ReasoningStore`. Each agent pipeline step (start/complete) appears as a timeline event with duration.
4. **State change events** -- From `StateChangeStore`. Includes task status transitions (e.g., "running -> blocked"), git commits, and file modifications.

### TimelineEvent Data Model

```python
@dataclass
class TimelineEvent:
    id: str
    type: str          # status_change | agent_output | feedback | comment |
                       # file_change | commit | reasoning
    timestamp: str     # ISO 8601
    actor: str         # Username or agent_id
    actor_type: str    # "human" | "agent" | "system"
    summary: str
    details: str
    metadata: dict | None
```

### State Change Recording

The `StateChangeStore` records three types of events:

- **Status transitions** -- `record_state_change(task_id, old_status, new_status)` for task lifecycle changes.
- **Git commits** -- `record_commit(task_id, commit_hash, message)` when agents commit code.
- **File changes** -- `record_file_change(task_id, file_path, change_type)` for individual file modifications.

### API Usage

**Get timeline for a task:**

```bash
# Default: up to 100 events, newest first
curl http://localhost:8000/api/v2/collaboration/timeline/phase-1

# Limit to 20 events
curl "http://localhost:8000/api/v2/collaboration/timeline/phase-1?limit=20"
```

Returns:

```json
{
  "events": [
    {
      "id": "sc-3",
      "type": "status_change",
      "timestamp": "2026-01-20T15:00:00Z",
      "actor": "system",
      "actor_type": "system",
      "summary": "Status changed: running -> completed",
      "details": "",
      "metadata": {"old_status": "running", "new_status": "completed"}
    },
    {
      "id": "cm-2",
      "type": "commit",
      "timestamp": "2026-01-20T14:58:00Z",
      "actor": "agent-abc12345",
      "actor_type": "agent",
      "summary": "Commit: Add JWT authentication module",
      "details": "Add JWT authentication module with login, logout, and token refresh",
      "metadata": {"commit_hash": "a1b2c3d"}
    },
    {
      "id": "fb-e1f2a3b4",
      "type": "feedback",
      "timestamp": "2026-01-20T14:30:00Z",
      "actor": "alice",
      "actor_type": "human",
      "summary": "library_swap: Use httpx instead of requests",
      "details": "",
      "metadata": {"priority": "must", "status": "addressed"}
    }
  ]
}
```

---

## Approval Gate

**Frontend:** `web/src/components/ApprovalGate.tsx`
**API endpoints:** `/api/approvals` and `/api/approvals/respond`

The approval gate component provides a UI for reviewing and responding to blocking approval requests. When the runner or an agent reaches a configured gate (such as "before commit"), execution pauses until a human approves, rejects, or provides feedback.

### How It Works

1. **Gate fires** -- The runner reaches an approval point and creates a pending approval request.
2. **UI displays the request** -- The `ApprovalGate` component polls `/api/approvals` and displays all pending requests as cards.
3. **Human reviews context** -- Each approval card shows the gate type, task/phase IDs, timestamp, and contextual information (diff, plan, test results, review output) depending on gate configuration.
4. **Human responds** -- The user can write optional feedback and click Approve or Reject.
5. **Runner continues or stops** -- The response is sent to `/api/approvals/respond` and the runner proceeds or halts accordingly.

### Real-Time Updates

The component subscribes to the `approvals` WebSocket channel. When a new approval request is created or an existing one is resolved, the component automatically refreshes.

### Context Display

Approval cards conditionally render context sections based on the gate's configuration flags:

| Flag | Shows |
|------|-------|
| `show_diff` | Git diff of changes |
| `show_plan` | Implementation plan |
| `show_tests` | Test execution results |
| `show_review` | Code review output |

Any additional context data beyond these four keys is rendered as formatted JSON under "Additional Context".

### API Usage

**List pending approvals:**

```bash
curl http://localhost:8000/api/approvals
```

**Respond to an approval:**

```bash
curl -X POST http://localhost:8000/api/approvals/respond \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "approval-uuid-here",
    "approved": true,
    "feedback": "Looks good, proceed with commit"
  }'
```

---

## Notification Center

**Frontend:** `web/src/components/NotificationCenter/NotificationCenter.tsx`
**WebSocket channel:** `notifications`

The notification center provides real-time alerts to keep users informed about task progress, approval requests, errors, and other events. It renders as a bell icon in the UI header with a dropdown panel showing recent notifications.

### Notification Types

| Type | Icon | Use Case |
|------|------|----------|
| `info` | i | General status updates |
| `success` | checkmark | Task completions, successful operations |
| `warning` | ! | Non-blocking issues, approaching timeouts |
| `error` | cross | Failures, blocked tasks, errors |

### Features

- **Real-time delivery via WebSocket** -- Subscribes to the `notifications` channel. New notifications appear instantly without polling.
- **Desktop browser notifications** -- On first interaction with the bell icon, the component requests browser notification permission. When granted, incoming notifications also trigger native desktop popups using the browser Notification API.
- **Sound alerts** -- Each notification plays a short tone via the Web Audio API. The tone frequency varies by severity (440 Hz for errors, 520 Hz for warnings, 660 Hz for success, 600 Hz for info). A mute toggle allows users to silence sounds.
- **Unread badge** -- The bell icon shows a count of unread notifications (capped at "99+").
- **Mark all read / Clear all** -- Bulk actions to manage notification state.
- **Persistent list** -- Maintains up to 50 recent notifications in memory, with relative timestamps ("3m ago", "2h ago").

### WebSocket Message Format

The notification center expects messages on the `notifications` channel with this shape:

```json
{
  "id": "notif-abc123",
  "type": "success",
  "title": "Task phase-1 completed",
  "message": "All 12 tests passed. Changes committed to branch feature/auth.",
  "timestamp": "2026-01-20T15:00:00Z"
}
```

All fields except `id` and `timestamp` are required. Missing `id` is auto-generated via `crypto.randomUUID()`. Missing `timestamp` defaults to the current time.

---

## User Management & Presence

**Backend:** `UserStore` and `PresenceTracker` in `src/feature_prd_runner/server/users.py`
**API prefix:** `/api/v2/collaboration/users` and `/api/v2/collaboration/presence`

The user management system provides role-based access control and online presence tracking for collaborative workflows.

### User Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| `admin` | view, launch_run, control_run, feedback, review, manage_agents, manage_users, configure | Full access to all features |
| `developer` | view, launch_run, control_run, feedback, review, manage_agents | Can run tasks, give feedback, manage agents |
| `reviewer` | view, feedback, review | Can review code and give feedback; cannot launch tasks |
| `viewer` | view | Read-only access |

A default `admin` user is automatically created when the `UserStore` is initialized.

### UserProfile Data Model

```python
@dataclass
class UserProfile:
    id: str               # Auto-generated, e.g. "user-a1b2c3d4"
    username: str
    display_name: str
    role: UserRole         # admin | developer | reviewer | viewer
    avatar_color: str      # CSS color for visual identification
    created_at: str        # ISO 8601
    last_seen: str | None  # Updated on each presence ping
    active: bool           # False if deactivated
```

### Presence Tracking

The `PresenceTracker` maintains real-time information about which users are online and what they are currently viewing:

```json
{
  "username": "alice",
  "online": true,
  "last_active": "2026-01-20T15:00:00Z",
  "viewing": "task-detail",
  "task_id": "phase-1"
}
```

Clients send periodic presence updates to keep their status current. When a user stops sending updates, they can be marked offline.

### API Usage

**List all active users:**

```bash
curl http://localhost:8000/api/v2/collaboration/users
```

**Create a user:**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "display_name": "Alice Chen",
    "role": "developer"
  }'
```

**Get a user profile:**

```bash
curl http://localhost:8000/api/v2/collaboration/users/alice
```

**Get online users:**

```bash
curl http://localhost:8000/api/v2/collaboration/presence
```

**Update presence (heartbeat):**

```bash
curl -X POST http://localhost:8000/api/v2/collaboration/presence \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "viewing": "task-detail",
    "task_id": "phase-1"
  }'
```

---

## API Reference

Quick reference of all collaboration and related endpoints.

### Feedback Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/collaboration/feedback` | Add structured feedback for a task |
| `GET` | `/api/v2/collaboration/feedback/{task_id}` | List feedback for a task (supports `active_only`, `feedback_type` query params) |
| `POST` | `/api/v2/collaboration/feedback/{feedback_id}/address` | Mark feedback as addressed by the agent |
| `POST` | `/api/v2/collaboration/feedback/{feedback_id}/dismiss` | Dismiss feedback |
| `GET` | `/api/v2/collaboration/feedback/{task_id}/prompt` | Get agent prompt instructions from active feedback |
| `GET` | `/api/v2/collaboration/feedback/{task_id}/effectiveness` | Get feedback effectiveness report |

### Review Comment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/collaboration/comments` | Add a review comment (supports `parent_id` for threading) |
| `GET` | `/api/v2/collaboration/comments/{task_id}` | List comments for a task (supports `file_path`, `unresolved_only` query params) |
| `GET` | `/api/v2/collaboration/comments/{comment_id}/replies` | Get threaded replies to a comment |
| `POST` | `/api/v2/collaboration/comments/{comment_id}/resolve` | Resolve a comment |

### HITL Mode Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/collaboration/modes` | List all available HITL modes with their configurations |
| `GET` | `/api/v2/collaboration/modes/current` | Get the current project-level mode |
| `GET` | `/api/v2/collaboration/modes/{mode}` | Get configuration details for a specific mode |
| `PUT` | `/api/v2/collaboration/modes` | Set the project-level HITL mode |
| `PUT` | `/api/v2/collaboration/modes/task/{task_id}` | Set a task-level mode override |
| `GET` | `/api/v2/collaboration/modes/task/{task_id}` | Get the effective mode for a task (shows task, project, and effective modes) |
| `DELETE` | `/api/v2/collaboration/modes/task/{task_id}` | Clear a task-level mode override |

### Timeline Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/collaboration/timeline/{task_id}` | Get unified activity timeline (supports `limit` query param, default 100) |

### Agent Reasoning Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/agents/reasoning/{task_id}` | Get all agent reasoning traces for a task |
| `POST` | `/api/v2/agents/reasoning/{task_id}/step/start` | Record that an agent started a pipeline step |
| `POST` | `/api/v2/agents/reasoning/{task_id}/step/complete` | Record that an agent completed a pipeline step |

### User Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/collaboration/users` | List all active users |
| `POST` | `/api/v2/collaboration/users` | Create a new user |
| `GET` | `/api/v2/collaboration/users/{username}` | Get a user profile |

### Presence Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/collaboration/presence` | Get all currently online users |
| `POST` | `/api/v2/collaboration/presence` | Update user presence (heartbeat) |

### Approval Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/approvals` | List all pending approval requests |
| `POST` | `/api/approvals/respond` | Approve or reject a pending approval |

---

## Related Documentation

- [HUMAN_IN_THE_LOOP.md](HUMAN_IN_THE_LOOP.md) -- Basic steering, approval gates, and message bus
- [DEBUGGING.md](DEBUGGING.md) -- Error analysis and task debugging
- [CUSTOM_EXECUTION.md](CUSTOM_EXECUTION.md) -- Ad-hoc execution and superadmin mode

## Source Files

**Backend (Python):**
- `src/feature_prd_runner/collaboration/feedback.py` -- Feedback and ReviewComment models, FeedbackStore
- `src/feature_prd_runner/collaboration/modes.py` -- HITL mode definitions and gate logic
- `src/feature_prd_runner/collaboration/reasoning.py` -- Agent reasoning capture and retrieval
- `src/feature_prd_runner/collaboration/timeline.py` -- Timeline aggregation and state change recording
- `src/feature_prd_runner/server/collaboration_api.py` -- FastAPI router for all collaboration endpoints
- `src/feature_prd_runner/server/users.py` -- User profiles, roles, and presence tracking

**Frontend (TypeScript/React):**
- `web/src/components/InlineReview/InlineReview.tsx` -- Diff viewer with line-level comments
- `web/src/components/FeedbackPanel/FeedbackPanel.tsx` -- Structured feedback submission and display
- `web/src/components/ReasoningViewer/ReasoningViewer.tsx` -- Agent reasoning step viewer
- `web/src/components/HITLModeSelector/HITLModeSelector.tsx` -- Mode selection dropdown
- `web/src/components/ActivityTimeline/ActivityTimeline.tsx` -- Unified event timeline
- `web/src/components/ApprovalGate.tsx` -- Approval request cards with approve/reject
- `web/src/components/NotificationCenter/NotificationCenter.tsx` -- Bell icon with notification dropdown
