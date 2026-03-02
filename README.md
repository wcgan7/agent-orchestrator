[![Backend CI](https://github.com/wcgan7/agent-orchestrator/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/wcgan7/agent-orchestrator/actions/workflows/backend-ci.yml)
[![Web CI](https://github.com/wcgan7/agent-orchestrator/actions/workflows/web-ci.yml/badge.svg)](https://github.com/wcgan7/agent-orchestrator/actions/workflows/web-ci.yml)
[![Version](https://img.shields.io/github/v/release/wcgan7/agent-orchestrator)](https://github.com/wcgan7/agent-orchestrator/releases)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

# Agent Orchestrator

Deterministic orchestration for AI-assisted software delivery.

Agent Orchestrator sequences planning, implementation, validation, and review into a controlled execution pipeline. It runs multi-step AI workflows with dependency awareness, governance gates, and full traceability.

Designed for teams building controlled, repeatable AI-driven development systems — not single-pass code generation.

## Core Principles

- **Deterministic execution over ad-hoc prompting**
- **Quality gates before commits**
- **Dependency-aware parallelism**
- **Provider-agnostic routing**
- **Complete execution traceability**

## Execution Model

Agent Orchestrator runs tasks through **pipeline templates** chosen by task intent (or auto-routed).

```
Task
  ↓
Select pipeline template (feature / bug_fix / refactor / docs / test / research / …)
  ↓
Execute staged steps (template-defined)
  ↓
Apply governance gates (approvals, severity thresholds, retry limits)
  ↓
Outcome (commit, report, or task generation)
```

Pipeline templates define step flow; governance policies determine how and when work progresses.

### Reference Pipeline: `feature`

The `feature` pipeline represents the full delivery loop:

plan → implement → verify → review → commit

It includes structured planning, automated verification, severity-scored review findings, and iterative fix cycles until quality gates are satisfied.

---

<!-- Regenerate screenshot: npm --prefix web run screenshot:homepage -->
![Agent Orchestrator Dashboard](web/public/homepage-screenshot.png)

---

## When to Use Agent Orchestrator

Choose Agent Orchestrator when you need:

- Multi-step AI-driven feature delivery with review loops
- Governance controls over automated code changes
- Cross-repository task coordination
- Auditable, policy-controlled execution pipelines
- Structured decomposition of large PRDs into executable tasks

Not designed for lightweight or single-pass code generation workflows.

## What You Can Do

### Deliver high-quality code with intent-specific pipelines
- Write concise, intent-focused prompts.
- Select or let the orchestrator route work through the most suitable built-in pipeline (for example `feature`, `bug_fix`, `refactor`, `hotfix`, `docs`, `test`).
- Each pipeline step applies intent-specific guidance (`plan/analyze`, `implement`, `verify`, `review`, `commit`) to produce higher-quality outcomes than single-pass generation.
- Pipelines iterate review-and-fix loops until findings meet configured tolerance thresholds.
- Full PRDs can be imported and automatically decomposed into dependency-aware execution batches.

### Enforce quality and governance
- Choose a Human-in-the-Loop mode per task: **Autopilot**, **Supervised**, or **Review Only**.
  - **Autopilot**: runs end-to-end without approvals.
  - **Supervised**: requires plan approval and human review before commit.
  - **Review Only**: pauses before commit and adds a final completion gate.
- Configure severity thresholds (`critical`, `high`, `medium`, `low`) globally or per task to control pass/fail tolerance.
- Draft, refine, and commit plan revisions with full lineage before implementation.

### Scale execution across repositories
- Run independent tasks in parallel, with automatic isolated git worktree provisioning for same-repo execution.
- Automatically integrate branches and trigger conflict-resolution flows when collisions occur.
- Route pipeline steps across providers (Codex, Claude, Ollama) with configurable step-to-provider mapping.
- Use the embedded interactive terminal to intervene manually without leaving the orchestrator UI.

### Audit and trace every task
- Track full task history from prompt to completion: task state transitions, plan/workdoc revisions, review decisions, and gate approvals.
- Every task produces a persistent workdoc, plus per-step runtime evidence (`stdout.log`, `stderr.log`, `progress.json`) and event timeline.
- Inspect execution summaries, step outcomes, review findings, total runtime, and commit SHAs in task detail.

## Quick Start

### 1. Start the backend

```bash
python -m pip install -e ".[server]"
agent-orchestrator server
```

Backend runs at `http://localhost:8080` by default.

### 2. Start the web UI

```bash
npm --prefix web install
npm --prefix web run dev
```

Frontend runs at `http://localhost:3000` by default (proxies `/api` to the backend).

## Navigation

| Tab | Purpose |
|---|---|
| **Board** | Kanban columns with task cards, inline detail/edit, and task explorer |
| **Planning** | Plan creation, iterative refinement, and revision history per task |
| **Execution** | Orchestrator status, queue depth, execution batches, pause/resume/drain/stop controls |
| **Workers** | Provider health (Codex/Claude/Ollama), step-to-provider routing table, active task monitoring |
| **Settings** | Project selector, concurrency, auto-deps, quality gates, worker config, project commands |

## Core Workflows

### Create and run a task

1. Open **Create Work** → **Create Task**.
2. Fill task fields (title, type, priority, description). Use `auto` type when you want the orchestrator to select the best pipeline.
3. Transition to `queued` or run from task detail.
4. Track progress in **Execution**.
5. Review from the task detail modal when it reaches `in_review`.

### Import a PRD into tasks

1. Open **Create Work** → **Import PRD**.
2. Paste PRD content and preview generated tasks/dependencies.
3. Commit the import job.
4. Review and execute created tasks from the board.

### Use embedded terminal

1. Click **Toggle terminal** in the main UI.
2. Start or attach to the active project terminal session.
3. Run commands interactively with live output and ANSI support.
4. Stop the session from the UI when done.

## Task Lifecycle

```
backlog → queued → in_progress → in_review → done
                        ↓               ↓
                     blocked       (request changes → queued)
```

Tasks support dependency graphs (validated for cycles), automatic dependency inference, parallel execution with configurable concurrency, merge-aware completion for worktree runs, and review cycles with severity-based findings.

## Pipeline Templates

Tasks execute through pipeline templates matched to their type. These templates are designed for code quality and delivery reliability, with specialized step flows by task intent:

| Pipeline | Use case / intent | Steps / flow |
|---|---|---|
| `feature` | Standard feature delivery with planning, quality checks, and commit. | `plan → implement → verify → review → commit` |
| `bug_fix` | Diagnose a bug, fix, verify, and commit. | `diagnose → implement → verify → review → commit` |
| `refactor` | Structured refactor with analysis and explicit plan first. | `analyze → plan → implement → verify → review → commit` |
| `hotfix` | Fast-path production fix without dedicated diagnosis step. | `implement → verify → review → commit` |
| `docs` | Documentation updates with quality verification and review. | `analyze → implement → verify → review → commit` |
| `test` | Add/adjust tests with validation and review before commit. | `analyze → implement → verify → review → commit` |
| `research` | Investigate and produce a report (no commit step). | `analyze → report` |
| `repo_review` | Assess repository state, form an initiative plan, then generate execution tasks. | `analyze → initiative_plan → generate_tasks` |
| `security_audit` | Scan dependencies/code for security issues, report, and emit remediation tasks. | `scan_deps → scan_code → report → generate_tasks` |
| `review` | Analyze and review existing changes, then produce a report. | `analyze → review → report` |
| `performance` | Profile baseline, optimize, benchmark, then review and commit. | `profile → plan → implement → benchmark → review → commit` |
| `spike` | Timeboxed exploratory prototype and recommendation report. | `analyze → prototype → report` |
| `chore` | Mechanical maintenance work with verification and commit. | `implement → verify → commit` |
| `plan_only` | Initiative-level planning and decomposition into executable tasks. | `analyze → initiative_plan → generate_tasks` |
| `verify_only` | Run checks and report status without making code changes. | `verify → report` |

## API and CLI

- REST/WebSocket reference: `docs/API_REFERENCE.md`
- CLI reference: `docs/CLI_REFERENCE.md`
- End-to-end usage guide: `docs/USER_GUIDE.md`

API base path: `/api`
WebSocket endpoint: `/ws`

### CLI quick reference

```bash
# Start server
agent-orchestrator server --project-dir /path/to/repo

# Task management
agent-orchestrator task create "My task" --priority P1 --task-type feature
agent-orchestrator task list --status queued
agent-orchestrator task run <task_id>

# Orchestrator control
agent-orchestrator orchestrator status
agent-orchestrator orchestrator control pause

# Project management
agent-orchestrator project pin /path/to/repo
agent-orchestrator project list
agent-orchestrator project unpin <project_id>
```

## Configuration and Runtime Data

Runtime state is stored in the selected project directory:
- `.agent_orchestrator/runtime.db` (canonical runtime state store)
- `.agent_orchestrator/workdocs/<task_id>.md` (canonical task workdocs synced with per-worktree `.workdoc.md`)
- `.agent_orchestrator_archive/state_<timestamp>/` (archived runtime snapshots on clear)

Execution metadata also records per-step log artifact locations (for example `stdout.log`, `stderr.log`, and `progress.json`) in task run details.

Primary configurable areas:
- `orchestrator` (concurrency, auto deps, review attempts)
- `agent_routing` (default role, task-type role routing, provider overrides)
- `defaults.quality_gate`
- `workers` (default provider, routing, providers)
- `project.commands` (per-language test, lint, typecheck, format commands)
- `project.prompt_injections` (per-step additive prompt text appended to worker instructions)

Claude provider example:
```json
{
  "workers": {
    "default": "claude",
    "providers": {
      "claude": {
        "type": "claude",
        "command": "claude -p",
        "model": "sonnet",
        "reasoning_effort": "medium"
      }
    }
  }
}
```

Notes:
- Claude CLI must be installed and authenticated locally.
- Reasoning effort flags are only passed when supported by your installed CLI version.

## Verify Locally

```bash
# Use Python 3.10+ and a local virtualenv
python3 -m venv .venv
.venv/bin/pip install -e ".[server,test,dev]"

# Backend tests
.venv/bin/pytest -q

# Optional integration tests (skipped by default and in CI)
AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 .venv/bin/pytest tests/test_integration_worker_model_fallback.py
AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 .venv/bin/pytest tests/test_integration_claude_provider.py

# Frontend checks
npm --prefix web run check

# Frontend smoke e2e
npm --prefix web run e2e:smoke
```

Local pushes are gated by `.githooks/pre-push` and run:
- `.venv/bin/ruff check .`
- `.venv/bin/pytest -q`
- `npm --prefix web run check`

Enable hooks once per clone:
```bash
git config core.hooksPath .githooks
```

## Documentation

- `docs/README.md`: documentation index
- `docs/USER_GUIDE.md`: complete user guide
- `docs/API_REFERENCE.md`: endpoint and WebSocket reference
- `docs/CLI_REFERENCE.md`: CLI commands and options
- `web/README.md`: frontend-specific setup and test workflow
- `example/README.md`: sample project walkthrough

## Versioning

Agent Orchestrator follows Semantic Versioning.

During `v0.x`, the primary compatibility surface is the CLI and configuration schema. The REST/WebSocket API and UI are evolving and may change between minor releases.

## License

Released under the MIT License.
