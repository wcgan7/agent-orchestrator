# Feature PRD Runner

Standalone helper module for long-running feature development driven by a PRD.
It uses Codex CLI as a worker and keeps durable state in local files so runs can
resume safely across restarts or interruptions.

## Requirements

- Python 3.10+
- Codex CLI installed and authenticated

## Quick Start

The template lives in `example/AGENTS.md`; copy it into your project root as `AGENTS.md`:

```bash
cp ./example/AGENTS.md ./AGENTS.md
```

```bash
python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md
```

Optional flags:

```bash
python -m feature_prd_runner.runner \
  --project-dir . \
  --prd-file ./docs/feature_prd.md \
  --test-command "npm test" \
  --no-stop-on-blocking-issues \
  --no-resume-blocked \
  --resume-prompt "Focus on error handling first"
```

## What It Does

1. Plans phases from the PRD and repository context.
2. Creates one task per phase.
3. For each phase:
   - Generates a per-phase implementation plan (`PLAN_IMPL`).
   - Implements the phase with Codex (`IMPLEMENT`), enforcing the plan allowlist.
   - Runs coordinator tests/lints and stores a verification artifact (`VERIFY`).
   - Runs a review against PRD requirements, acceptance criteria, and the plan (`REVIEW`).
   - Commits and pushes when clean (`COMMIT`).

Each step writes progress to durable files for easy resume. Verification evidence is
recorded and fed into review prompts to avoid “not evidenced” failures.

Review-fix mode: if a review returns blocking issues, the runner routes back to IMPLEMENT
with a “address review issues” banner and tracked blocker files.

Blocking issues: if a run reports blocking issues that require human intervention, the
runner stops and prints the issues plus proposed resolve steps. Disable this behavior
with `--no-stop-on-blocking-issues`.

Resume blocked tasks: by default, the runner will auto-resume the most recent blocked
task on startup and replay the last step (plan, implement, review, etc.). Disable this
behavior with `--no-resume-blocked`. Each task is capped at 10 auto-resumes to avoid
infinite loops.

## State Files

All state lives in `.prd_runner/` inside the project directory:

- `run_state.yaml`: current status, active task/phase, last error
- `task_queue.yaml`: tasks derived from phases
- `phase_plan.yaml`: planned phases
- `artifacts/`: events, tests, plans, reviews
  - `impl_plan_<phase_id>.json`: implementation plan per phase
  - `review_<phase_id>.json`: structured review output
  - `tests_<phase_id>.log`: test logs
- `runs/`: per-run logs, prompts, and progress snapshots (including review runs)

The coordinator will refuse to commit if `.prd_runner/` is tracked or not ignored.

## Resume Prompts

Use `--resume-prompt` to inject special instructions into the next agent run:

```bash
python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md \
  --resume-prompt "Prioritize fixing CI failures"
```

The prompt is applied once to the next run and then cleared.

## Testing

```bash
python -m pip install -e ".[test]"
python -m pytest
```
