# Feature PRD Runner

An AI engineering orchestrator that coordinates multiple specialized agents to execute software development tasks — from feature implementation to bug fixes, refactors, security audits, and more.

Built on a dynamic task engine, configurable pipeline templates, and real-time collaboration features, it goes beyond single-agent PRD execution to provide a full-stack platform for AI-assisted engineering.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Dashboard (React)                    │
│  KanbanBoard · AgentCards · CommandPalette · NotificationCenter │
├─────────────────────────────────────────────────────────────┤
│                  FastAPI Server + WebSocket Hub              │
│  REST API · Real-time channels · Auth · Presence tracking   │
├──────────┬──────────┬────────────┬──────────────────────────┤
│  Task    │  Agent   │  Pipeline  │  Collaboration           │
│  Engine  │  Pool    │  Engine    │  Layer                   │
│          │          │            │                          │
│  CRUD    │  6 roles │  8 templates│  Feedback · Comments    │
│  Priority│  Spawn   │  18 steps  │  HITL modes · Reasoning │
│  Deps    │  Schedule│  Conditions│  Timeline · Presence     │
│  Labels  │  Handoff │  Retries   │  Notifications           │
├──────────┴──────────┴────────────┴──────────────────────────┤
│              File-based YAML persistence + FileLock          │
└─────────────────────────────────────────────────────────────┘
```

**Key packages:**

| Package | Purpose |
|---------|---------|
| `task_engine/` | Task model, CRUD engine, dependency resolution, YAML store |
| `agents/` | Agent registry (6 types), pool manager, scheduler, handoff/context bus |
| `pipelines/` | Pipeline templates (8), step registry (18 steps), execution engine |
| `collaboration/` | Feedback, review comments, HITL modes, reasoning traces, timeline |
| `server/` | FastAPI app with REST + WebSocket APIs, auth, presence tracking |
| `workers/` | Worker provider config (Codex, Ollama), diagnostics, runtime management |

## Requirements

- Python 3.10+
- Codex CLI installed and authenticated
- Git installed; a configured `origin` remote if you want automatic pushes

## Install

From this repo:

```bash
python -m pip install -e .
```

If you prefer `uv` (install `uv` first via `brew install uv`, `pipx install uv`, or `python -m pip install uv`):

```bash
uv pip install -e .
```

## Tests

Python unit tests (840 tests):

```bash
python -m pip install -e ".[test]"
pytest
```

If you prefer `uv`:

```bash
uv pip install -e ".[test]"
pytest
```

Web dashboard tests (101 tests, Vitest):

```bash
cd web
npm install
npm test
```

## Quick Start (in your target project)

The `example/` folder in this repository contains a starter `AGENTS.md` and a sample PRD.

1) Add an `AGENTS.md` to your project root (recommended)

```bash
cp /path/to/feature-prd-runner/example/AGENTS.md /path/to/your/project/AGENTS.md
```

2) Write a PRD (example provided in this repo)

```bash
mkdir -p /path/to/your/project/docs
cp /path/to/feature-prd-runner/example/test_feature_prd.md /path/to/your/project/docs/feature_prd.md
```

3) Run the coordinator

```bash
feature-prd-runner --project-dir /path/to/your/project --prd-file /path/to/your/project/docs/feature_prd.md
```

If you prefer module invocation:

```bash
python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md
```

If you prefer calling the Python API:

```python
from pathlib import Path

from feature_prd_runner.orchestrator import run_feature_prd

run_feature_prd(project_dir=Path("."), prd_path=Path("./docs/feature_prd.md"))
```

## Multi-Agent Orchestration

The orchestrator manages a pool of specialized agents, each with distinct roles, models, and capabilities.

### Agent Types

| Role | Description | Allowed Steps |
|------|-------------|---------------|
| **Implementer** | Writes code from plans | plan, plan_impl, implement, commit |
| **Reviewer** | Reviews code quality | review |
| **Researcher** | Gathers context and analyzes | gather, analyze, summarize, report |
| **Tester** | Runs tests, writes test cases | verify, implement |
| **Architect** | Plans high-level architecture | plan, plan_impl, analyze |
| **Debugger** | Diagnoses failures, finds root causes | reproduce, diagnose, implement, verify |

Each agent has configurable resource limits (token budgets, time budgets, cost caps, max retries), tool access permissions, and task affinity settings.

### Agent Pool

The agent pool manager handles:
- **Spawning** agents on demand based on task requirements
- **Health monitoring** with automatic restart on failure
- **Task assignment** via the scheduler (priority-based with agent affinity)
- **Agent-to-agent handoff** — structured context passing between agents (e.g., reviewer feedback flows directly to the implementer)
- **Shared context bus** — agents can read each other's progress and artifacts without file-based intermediaries
- **Pause/resume/terminate** controls via API and dashboard

### Agent API

```
GET    /api/v2/agents              # List all agents with status
POST   /api/v2/agents              # Spawn a new agent
GET    /api/v2/agents/{id}         # Get agent details
POST   /api/v2/agents/{id}/pause   # Pause agent
POST   /api/v2/agents/{id}/resume  # Resume agent
POST   /api/v2/agents/{id}/terminate  # Terminate agent
GET    /api/v2/agents/types        # List agent type blueprints
```

## Pipeline Templates

Tasks are executed through configurable pipeline templates. Each template defines a sequence of steps with conditions, retries, and agent role preferences.

### Built-in Templates

| Template | Steps | Use Case |
|----------|-------|----------|
| **feature** | Plan → Plan Impl → Implement → Verify → Review → Commit | New feature development |
| **bug_fix** | Reproduce → Diagnose → Fix → Verify → Review → Commit | Bug fixes |
| **refactor** | Analyze → Plan → Implement → Verify → Review → Commit | Code refactoring |
| **research** | Gather → Analyze → Summarize → Report | Research tasks |
| **docs** | Analyze Code → Write Docs → Review → Commit | Documentation |
| **test** | Analyze Coverage → Write Tests → Run Tests → Review → Commit | Test writing |
| **repo_review** | Scan → Analyze → Generate Tasks | Repository analysis |
| **security_audit** | Scan Deps → Scan Code → Report → Generate Fix Tasks | Security scanning |

### Pipeline Engine

The engine resolves the correct template for a task, evaluates step conditions, and drives execution:
- **Condition-based step skipping** — skip steps based on task type, change size, or previous step results
- **Retry logic** — configurable retry limits per step
- **HITL approval gates** — pause before plan, implement, or commit for human review
- **Runtime modification** — insert or skip steps on a running pipeline
- **Reasoning traces** — records agent thinking at each step for transparency
- **Event notifications** — fires events for task completion, failure, approval needs, and budget warnings

## Dynamic Task Engine

Tasks are first-class entities with rich metadata, not just pipeline phases.

### Task Model

Each task has:
- **Type**: feature, bug, refactor, research, review, test, docs
- **Priority**: P0 (critical) through P3 (low)
- **Status**: pending, in_progress, blocked, completed, failed, cancelled
- **Labels, parent/child relationships, blockers**
- **Acceptance criteria, context files, related tasks**
- **Source tracking** (PRD import, repo review, manual, bug scan, etc.)

### Task API

```
GET    /api/v2/tasks               # List tasks (filterable)
POST   /api/v2/tasks               # Create task
GET    /api/v2/tasks/{id}          # Get task details
PUT    /api/v2/tasks/{id}          # Update task
DELETE /api/v2/tasks/{id}          # Delete task
POST   /api/v2/tasks/{id}/assign   # Assign to agent
POST   /api/v2/tasks/{id}/move     # Move between statuses
GET    /api/v2/tasks/board         # Kanban board view
```

### Smart Scheduling

The scheduler assigns tasks to agents based on:
- Priority ordering (P0 preempts P2)
- Dependency graph resolution
- Agent role affinity (review tasks go to reviewer agents)
- Resource availability

## Collaboration Features

### Human-in-the-Loop (HITL) Modes

Four collaboration modes control the level of human involvement:

| Mode | Description |
|------|-------------|
| **Autopilot** | Agents work independently, humans observe |
| **Supervised** | Agents work, gates pause for approval at key points |
| **Collaborative** | Agents and humans work together, frequent checkpoints |
| **Review Only** | Agents do all work, humans review before commit |

Modes can be set globally (project-level) or per-task. The pipeline engine enforces approval gates based on the active mode.

```
GET    /api/v2/collaboration/modes              # List available modes
PUT    /api/v2/collaboration/modes              # Set project mode
PUT    /api/v2/collaboration/modes/task/{id}    # Set task-specific mode
GET    /api/v2/collaboration/modes/task/{id}    # Get effective mode for task
```

### Structured Feedback

Humans can provide typed, prioritized feedback on agent work:

- **Feedback types**: general, requirement, style, performance, security, architecture
- **Priority levels**: must, should, could, nice_to_have
- **Feedback lifecycle**: active → addressed / dismissed
- **Prompt injection**: active feedback is compiled into agent prompts
- **Effectiveness tracking**: reports on feedback addressed vs. dismissed

### Inline Review Comments

Line-level commenting on code diffs, similar to GitHub PR reviews:
- File + line number targeting
- Threaded replies
- Resolution tracking
- Triggers review notifications

### Activity Timeline

Unified chronological view of all events for a task: feedback, comments, state changes, agent reasoning steps — aggregated from multiple stores.

### Reasoning Viewer

Agents record their step-by-step thinking during pipeline execution. The reasoning viewer exposes this for transparency and debugging.

### User Management & Presence

- User profiles with roles (admin, lead, developer, viewer)
- Real-time online presence tracking
- Last-seen timestamps

## Real-Time Notifications

The notification system pushes events from backend to frontend via WebSocket:

- **Task lifecycle**: completed, failed, blocked
- **Agent events**: spawned, error, terminated, auto-restarted
- **Collaboration**: approval needed, review requested, mode changed
- **Budget warnings**: agent approaching cost limit (80%+ threshold)

The `NotificationCenter` component in the dashboard shows a notification feed with dismiss and clear-all controls.

## Web Dashboard

A React/TypeScript dashboard for monitoring and controlling the orchestrator.

```bash
# Start the backend server
feature-prd-runner server --port 8080

# In another terminal, start the frontend (requires Node.js 18+)
cd web
npm install
npm run dev
```

Access the dashboard at http://localhost:3000

### Dashboard Components

| Component | Description |
|-----------|-------------|
| **RunDashboard** | Main view with run status, progress, and controls |
| **KanbanBoard** | Drag-drop task board with slide-over detail view |
| **AgentCard** | Agent panel with stream, controls, and overview subcomponents |
| **CommandPalette** | Cmd+K fuzzy search across tasks, agents, and actions |
| **NotificationCenter** | Real-time notification feed |
| **PhaseTimeline** | Visual phase progress with dependencies |
| **DependencyGraph** | Task dependency visualization |
| **MetricsPanel** | API usage, costs, timing, code change metrics |
| **CostBreakdown** | Per-agent and per-task cost analysis |
| **ApprovalGate** | Approve/reject interface for HITL gates |
| **FileReview** | File-by-file diff review |
| **InlineReview** | Line-level code commenting |
| **ReasoningViewer** | Agent step-by-step thinking viewer |
| **HITLModeSelector** | Mode picker (autopilot/supervised/collaborative/review_only) |
| **FeedbackPanel** | Structured feedback form and list |
| **ActivityTimeline** | Unified event chronology |
| **LiveLog** | Real-time log streaming via WebSocket |
| **BreakpointsPanel** | Breakpoint management |
| **TaskLauncher** | Quick task creation dialog |
| **ControlPanel** | Run control actions (retry, skip, resume, stop) |
| **Chat** | Live collaboration messaging |
| **ProjectSelector** | Multi-project switching |
| **Login** | Authentication |

### Frontend Architecture

- **React 18** with TypeScript and Vite
- **Plain CSS** (no framework) with theme support (light/dark/system)
- **WebSocket** for real-time updates (channels: runs, logs, agents, notifications)
- **Contexts**: ThemeContext, WebSocketContext, ToastContext
- **Testing**: Vitest + React Testing Library

See [web/README.md](web/README.md) for detailed frontend setup and development instructions.

## Usage

The runner core is a FSM with these steps:

1. `PLAN`: derive phases from PRD + repo context.
2. For each phase:
   - `PLAN_IMPL`: write a per-phase implementation plan JSON (including allowed files).
   - `IMPLEMENT`: run Codex and enforce the plan allowlist.
   - `VERIFY`: run your configured command (tests/lint) and capture logs.
   - `REVIEW`: run a structured review against acceptance criteria and evidence.
   - `COMMIT`: commit and push to `origin` when clean.

Note: the `COMMIT` step runs `git commit` and `git push -u origin <branch>` by default. Use
`--no-commit` and/or `--no-push` to disable those behaviors.

## Language Support

The runner supports multiple programming languages with optimized verification output parsing:

| Language   | Test Framework  | Linter           | Formatter        | Type Checker |
|------------|-----------------|------------------|------------------|--------------|
| Python     | pytest          | ruff             | ruff             | mypy         |
| TypeScript | jest, vitest    | eslint           | prettier         | tsc          |
| Next.js    | jest, vitest    | next lint        | prettier         | next build   |
| JavaScript | jest, vitest    | eslint           | prettier         | -            |
| Go         | go test         | golangci-lint    | gofmt            | -            |
| Rust       | cargo test      | clippy           | cargo fmt        | -            |

### Language Detection

The runner auto-detects your project language from manifest files:
- `package.json` with `next` dependency or `next.config.*` → Next.js
- `package.json` with TypeScript dependency → TypeScript
- `package.json` without TypeScript → JavaScript
- `pyproject.toml`, `setup.py`, `requirements.txt` → Python
- `go.mod` → Go
- `Cargo.toml` → Rust

Or specify explicitly:

```bash
feature-prd-runner run my-feature.md --language typescript
```

### TypeScript/JavaScript Quick Start

```bash
# Auto-detected from package.json
feature-prd-runner run my-feature.md

# Or explicitly
feature-prd-runner run my-feature.md --language typescript --verify-profile typescript
```

Configure in `.prd_runner/config.yaml`:

```yaml
language: typescript
verify_profile: typescript

test_command: npm test
lint_command: npx eslint .
typecheck_command: npx tsc --noEmit
format_command: npx prettier --check .

ensure_deps: install
```

### Next.js Quick Start

```bash
# Auto-detected from package.json with next dependency
feature-prd-runner run my-feature.md

# Or explicitly
feature-prd-runner run my-feature.md --language nextjs --verify-profile nextjs
```

Configure in `.prd_runner/config.yaml`:

```yaml
language: nextjs
verify_profile: nextjs

test_command: npm test
lint_command: npx next lint
typecheck_command: npx next build
format_command: npx prettier --check .

ensure_deps: install
```

### Creating Example Projects

Generate a starter project for any supported language:

```bash
feature-prd-runner example --output ./my-ts-project --language typescript
feature-prd-runner example --output ./my-py-project --language python
feature-prd-runner example --output ./my-go-project --language go
```

## Status

Inspect the runner's durable state without starting a run:

```bash
feature-prd-runner status --project-dir /path/to/your/project
```

For machine-readable output:

```bash
feature-prd-runner status --project-dir /path/to/your/project --json
```

## Dry Run

Preview what the runner would do next without making any changes (no git changes, no `.prd_runner` writes, no Codex, no tests):

```bash
feature-prd-runner dry-run --project-dir /path/to/your/project --prd-file /path/to/your/project/docs/feature_prd.md
```

## Doctor

Run read-only diagnostics (git + state + PRD consistency checks):

```bash
feature-prd-runner doctor --project-dir /path/to/your/project --prd-file /path/to/your/project/docs/feature_prd.md
```

For machine-readable output:

```bash
feature-prd-runner doctor --project-dir /path/to/your/project --prd-file /path/to/your/project/docs/feature_prd.md --json
```

## List and Resume

List phases and tasks:

```bash
feature-prd-runner list --project-dir /path/to/your/project
```

Resume a blocked task by id (optionally overriding the step):

```bash
feature-prd-runner resume phase-1 --project-dir /path/to/your/project --step implement
```

Other control-plane commands (these update `.prd_runner/task_queue.yaml`):

```bash
feature-prd-runner retry phase-1 --project-dir /path/to/your/project
feature-prd-runner rerun-step phase-1 --project-dir /path/to/your/project --step verify
feature-prd-runner skip-step phase-1 --project-dir /path/to/your/project --step verify
```

Each step writes progress to durable files for easy resume. Verification evidence is
recorded and fed into review prompts to avoid "not evidenced" failures.

If a review returns blocking issues, the runner routes back to `IMPLEMENT` with an
"address review issues" banner and the implicated files.

## CLI Options

Common options:

- `--test-command "..."`: command run during `VERIFY` after each phase.
- `--format-command "..."`: optional format-check command run during `VERIFY` (before lint/tests).
- `--lint-command "..."`: optional lint command run during `VERIFY` (before tests).
- `--typecheck-command "..."`: optional typecheck command run during `VERIFY` (before tests).
- `--language {auto,python,typescript,javascript,go,rust}`: project language for verification parsing (default: auto-detect).
- `--verify-profile {none,python,typescript,javascript,go,rust}`: enables preset defaults for the language (auto-detects tools).
- `--ensure-ruff {off,warn,install,add-config}`: helper behavior when using ruff-based commands (default: off).
- `--ensure-deps {off,install}`: optional helper to run an install step before verification (default: off).
- `--ensure-deps-command "..."`: install command used by `--ensure-deps install` (defaults to `python -m pip install -e ".[test]"` with fallback to `python -m pip install -e .`).
- `--new-branch` / `--no-new-branch`: create/switch to a new git branch once at the start of the run (default: True). If `--no-new-branch`, you must already be on a named branch (not detached HEAD).
- `--codex-command "..."`: Codex CLI command used to run the worker (default: `codex exec -`).
- `--worker NAME`: worker provider name override (e.g., `codex`, `ollama`). Overrides `.prd_runner/config.yaml` routing.
- `--shift-minutes N`: timebox per worker run.
- `--reset-state`: archive and recreate `.prd_runner/` before starting.
- `--require-clean` / `--no-require-clean`: refuse to run if there are git changes outside `.prd_runner/` (default: True).
- `--stop-on-blocking-issues` / `--no-stop-on-blocking-issues`: whether to stop when a phase is blocked.
- `--resume-blocked` / `--no-resume-blocked`: whether to auto-resume the most recent blocked task.
- `--custom-prompt "..."`: run a standalone "do this first" prompt once before continuing the normal cycle.
- `--simple-review` / `--no-simple-review`: toggle a simplified review schema.
- `--commit` / `--no-commit`: enable/disable `git commit` in the `COMMIT` step (default: True).
- `--push` / `--no-push`: enable/disable `git push` in the `COMMIT` step (default: True).

Resume blocked tasks: by default, the runner will auto-resume the most recent blocked
task on startup and replay the last step (plan, implement, review, etc.). Disable this
behavior with `--no-resume-blocked`. Each task is capped at 10 auto-resumes to avoid
infinite loops.

## State Files

All state lives in `.prd_runner/` inside the project directory:

- `run_state.yaml`: current status, active task/phase, last error
- `task_queue.yaml`: tasks derived from phases
- `phase_plan.yaml`: planned phases
- `config.yaml`: optional runner configuration (see below)
- `artifacts/`: events, tests, plans, reviews
  - `impl_plan_<phase_id>.json`: implementation plan per phase
  - `review_<phase_id>.json`: structured review output
  - `tests_<phase_id>.log`: test logs
- `runs/`: per-run logs, prompts, and progress snapshots (including review runs)

The coordinator will refuse to commit if `.prd_runner/` is tracked or not ignored. If the
repo is clean, it will try to add `.prd_runner/` to `.gitignore` automatically.

## Config

Optional config file: `.prd_runner/config.yaml` (kept out of git by default).

Example:

```yaml
verify:
  format_command: "ruff format --check ."
  lint_command: "ruff check ."
  typecheck_command: "mypy ."
  test_command: "pytest -q"

workers:
  default: codex
  providers:
    codex:
      type: codex
      command: "codex exec -"
    local:
      type: ollama   # alias: "local"
      endpoint: "http://localhost:11434"
      model: "qwen2.5-coder:7b"
      temperature: 0.2
      num_ctx: 8192
  routing:
    plan: local
    plan_impl: local
    review: local
    implement: codex
```

CLI flags override `config.yaml`.

## Codex Command Notes

`--codex-command` must accept the prompt either via stdin (`-`) or via a placeholder:

- stdin: `codex exec -` (default)
- file placeholder: `codex exec --prompt-file {prompt_file}`
- inline placeholder: `codex exec {prompt}`

Available placeholders: `{prompt_file}`, `{project_dir}`, `{run_dir}`, `{prompt}`.

## Workers (Codex + Ollama)

List configured workers for a project:

```bash
feature-prd-runner workers list --project-dir /path/to/your/project
```

Test a worker (checks Codex binary, or Ollama endpoint/model):

```bash
feature-prd-runner workers test local --project-dir /path/to/your/project
```

## Custom Prompts & Ad-Hoc Tasks

### The `exec` Command

Execute ad-hoc custom prompts outside the normal workflow:

```bash
# Basic usage
feature-prd-runner exec "Update all copyright headers to 2026"

# With superadmin mode (bypass AGENTS.md rules)
feature-prd-runner exec "Emergency fix: patch security vulnerability" --override-agents

# With context
feature-prd-runner exec "Add logging" --context-files "src/auth.py,src/api.py"
```

**Superadmin mode** (`--override-agents`) allows you to bypass AGENTS.md rules when you need full control:
- Skip file allowlists
- Bypass documentation/testing requirements
- Emergency hotfixes
- Administrative changes

See [docs/CUSTOM_EXECUTION.md](docs/CUSTOM_EXECUTION.md) for full documentation.

### Custom Prompt Before Run

Use `--custom-prompt` to run one standalone worker prompt before the main loop. This is
useful for "setup" steps like updating dependencies, regenerating locks, or resolving a
known failure before continuing phased implementation.

```bash
feature-prd-runner run --prd-file ./docs/feature_prd.md \
  --custom-prompt "Regenerate lockfiles and ensure tests pass"

# With superadmin mode
feature-prd-runner run --prd-file ./docs/feature_prd.md \
  --custom-prompt "Update dependencies" \
  --override-agents
```

If the worker reports `human_blocking_issues` in the run's `progress.json`, the runner
stops and surfaces the issues and suggested next steps.

## Human-in-the-Loop Control

The runner supports active human involvement through steering, approval gates, and bidirectional communication.

### Interactive Mode

Enable step-by-step approval gates at key checkpoints:

```bash
feature-prd-runner run --prd-file feature.md --interactive
```

This pauses execution before implement, after implement, and before commit for human approval.

### Steering Commands

**Send guidance to running workers:**

```bash
# Single message
feature-prd-runner steer "Focus on error handling"

# Interactive mode
feature-prd-runner steer
> Add more logging
> Check edge cases
```

**Approve/reject pending approval gates:**

```bash
feature-prd-runner approve
feature-prd-runner approve --feedback "Looks good"
feature-prd-runner reject --reason "Need more tests first"
```

### Configuration

Configure approval gates in `.prd_runner/config.yaml`:

```yaml
approval_gates:
  enabled: true
  gates:
    before_implement:
      enabled: true
      show_plan: true
      timeout: 300
    after_implement:
      enabled: true
      show_diff: true
      timeout: 300
    before_commit:
      enabled: true
      show_diff: true
      show_tests: true
      required: true  # Cannot skip
```

See [docs/HUMAN_IN_THE_LOOP.md](docs/HUMAN_IN_THE_LOOP.md) for comprehensive documentation on:
- Steering running workers
- Approval gate configuration
- Message bus architecture
- Interactive mode details
- Advanced usage patterns

## Enhanced Error Messages & Debugging

The runner provides comprehensive debugging tools for troubleshooting failures:

### Debug Commands

**Explain why a task is blocked:**
```bash
feature-prd-runner explain phase-1
```

**Inspect full task state:**
```bash
feature-prd-runner inspect phase-1
feature-prd-runner inspect phase-1 --json  # JSON output
```

**Trace event history:**
```bash
feature-prd-runner trace phase-1
feature-prd-runner trace phase-1 --limit 20  # Last 20 events
```

**View detailed logs:**
```bash
feature-prd-runner logs phase-1
feature-prd-runner logs phase-1 --step verify --lines 200
```

### Rich Error Reports

Errors include:
- Root cause analysis
- Files involved
- Actionable suggestions with commands
- Quick fixes

See [docs/DEBUGGING.md](docs/DEBUGGING.md) for full documentation.

## Parallel Phase Execution

Execute independent phases concurrently to reduce total execution time:

### Enable Parallel Execution

```bash
# Run with parallel execution (experimental)
feature-prd-runner run --prd-file feature.md --parallel

# Limit concurrent workers
feature-prd-runner run --prd-file feature.md --parallel --max-workers 2
```

### Visualize Execution Plan

```bash
# Show execution batches
feature-prd-runner plan-parallel

# Show dependency tree
feature-prd-runner plan-parallel --tree
```

### How It Works

Phases can specify dependencies in `phase_plan.yaml`:

```yaml
phases:
  - id: database-schema
    deps: []  # No dependencies

  - id: frontend-components
    deps: []  # Independent

  - id: api-endpoints
    deps: ["database-schema"]  # Depends on database

  - id: integration
    deps: ["api-endpoints", "frontend-components"]  # Depends on both
```

The parallel executor:
- Detects circular dependencies
- Uses topological sort to create execution batches
- Executes independent phases in parallel
- Tracks progress for all running phases

See [docs/PARALLEL_EXECUTION.md](docs/PARALLEL_EXECUTION.md) for full documentation.

## Troubleshooting

- `Codex command must include {prompt_file}, {prompt}, or '-'`: update `--codex-command` to accept input.
- `No heartbeat received within grace period`: increase `--heartbeat-grace-seconds`, or fix the worker so it updates `progress.json`.
- `.prd_runner is tracked` / `.prd_runner is not ignored`: remove it from git history and ensure `.gitignore` includes `.prd_runner/`.
- `git push failed`: verify `origin` remote and authentication; push the branch manually, then re-run.
- `VERIFY` logs are huge: the runner analyzes a bounded excerpt and saves it under `.prd_runner/runs/<run_id>/verify_output.txt` (or `pytest_failures.txt` for pytest).
- `state_corrupt`: a `.prd_runner/*.yaml` file is not parseable; restore it or re-run with `--reset-state` (archives the old state).
- `state_reset_failed`: `--reset-state` could not archive `.prd_runner/`; close other runners/fix permissions, or move/delete it manually.
- `prd_mismatch`: existing state was created for a different PRD (path or content); re-run with `--reset-state`.
- `state_invalid`: `.prd_runner/phase_plan.yaml` or `.prd_runner/task_queue.yaml` has an invalid schema (duplicate ids, missing deps, etc.); fix it or re-run with `--reset-state`.
