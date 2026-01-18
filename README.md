# Feature PRD Runner

A standalone coordinator for long-running feature development driven by a PRD.
It uses Codex CLI as the “worker” and keeps durable state in local files so runs
can resume safely across restarts or interruptions.

This project is intentionally opinionated:
- Phases are planned from the PRD.
- Each phase is implemented with a file allowlist (derived from an implementation plan).
- Each phase is verified, reviewed against acceptance criteria, then committed and pushed.

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

Python unit tests:

```bash
python -m pip install -e ".[test]"
pytest
```

If you prefer `uv`:

```bash
uv pip install -e ".[test]"
pytest
```

Web dashboard tests (Vitest):

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

## Usage

The runner is a small FSM with these steps:

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

The runner is language-agnostic: it coordinates Codex + git and runs whatever `--test-command` you provide, so it can be used on repos in any language.

`VERIFY` is currently optimized for Python/pytest output parsing. For non-pytest commands it uses a more generic log excerpt + path extraction approach, and allowlist expansion signals are conservative.

## Status

Inspect the runner’s durable state without starting a run:

```bash
feature-prd-runner status --project-dir /path/to/your/project
```

For machine-readable output:

```bash
feature-prd-runner status --project-dir /path/to/your/project --json
```

## Web Dashboard

Monitor and control runs through a modern web interface:

```bash
# Start the backend server
feature-prd-runner server --port 8080

# In another terminal, start the frontend (requires Node.js)
cd web
npm install
npm run dev
```

Access the dashboard at http://localhost:3000

**Features**:
- Real-time run status and progress
- Tasks and runs tables (per project)
- Phase timeline with dependencies
- Live log streaming via WebSocket
- Metrics: API usage, costs, timing, code changes
- Approval gates (approve/reject) and file-by-file review
- Breakpoint management
- Live collaboration chat (guidance/requirements/corrections)
- Responsive design for mobile/desktop

**Requirements**:
- Backend: `pip install 'feature-prd-runner[server]'` (or `uv pip install 'feature-prd-runner[server]'`)
- Frontend: Node.js 18+ and npm

See [web/README.md](web/README.md) for detailed setup and development instructions.

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
recorded and fed into review prompts to avoid “not evidenced” failures.

If a review returns blocking issues, the runner routes back to `IMPLEMENT` with an
“address review issues” banner and the implicated files.

## CLI Options

Common options:

- `--test-command "..."`: command run during `VERIFY` after each phase.
- `--format-command "..."`: optional format-check command run during `VERIFY` (before lint/tests).
- `--lint-command "..."`: optional lint command run during `VERIFY` (before tests).
- `--typecheck-command "..."`: optional typecheck command run during `VERIFY` (before tests).
- `--verify-profile {none,python}`: enables preset defaults for Python projects (auto-detects `ruff`/`pytest`/`mypy`).
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
- `--custom-prompt "..."`: run a standalone “do this first” prompt once before continuing the normal cycle.
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

Example error output:
```
❌ Error: phase-1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error Type: test_failed
Verification failed: 2 tests failed

Root Cause:
  Tests failed during verification.

Files Involved:
  1. tests/test_auth.py

Suggested Actions:
  [1] Review test failures
      $ feature-prd-runner logs phase-1 --step verify

  [2] Retry after reviewing
      $ feature-prd-runner retry phase-1

Quick Fixes:
  • View full test output
    $ cat .prd_runner/runs/*/tests_phase-1.log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

See [docs/DEBUGGING.md](docs/DEBUGGING.md) for full documentation on:
- All debugging commands
- Error analysis features
- Root cause identification
- Programmatic access
- Best practices

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

**Example execution**:
```
Batch 1 (parallel): database-schema, frontend-components
Batch 2: api-endpoints (waits for database-schema)
Batch 3: integration (waits for both)
```

**Current Implementation Status**:
- ✅ Dependency analysis with topological sort
- ✅ Circular dependency detection
- ✅ Execution plan visualization
- ✅ Thread pool infrastructure
- ✅ Full parallel execution (fully implemented!)
- ✅ Thread-safe git operations
- ✅ Thread-safe state management
- ✅ Partial failure handling

When `--parallel` is enabled, the system analyzes dependencies, creates an optimal execution plan, and **executes independent phases in parallel**. Each phase runs through its complete task lifecycle (plan_impl → implement → verify → review → commit) while maintaining thread safety for git operations and state updates.

See [docs/PARALLEL_EXECUTION.md](docs/PARALLEL_EXECUTION.md) for:
- Dependency resolution details
- Configuration options
- Examples and best practices
- Progress tracking
- Error handling
- Troubleshooting

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

## Testing

### Python

```bash
python -m pip install -e ".[test]"
python -m pytest
```

With `uv`:

```bash
uv pip install -e ".[test]"
uv run pytest
```

### Web Dashboard (Frontend)

```bash
cd web
npm install
npm test
```
