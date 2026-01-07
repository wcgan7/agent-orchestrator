# Feature PRD Runner

Standalone helper module for long-running feature development driven by a PRD.
It uses Codex CLI as a worker and keeps durable state in local files so runs can
resume safely across restarts or interruptions.

## Requirements

- Python 3.10+
- Codex CLI installed and authenticated

## Quick Start

```bash
python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md
```

Optional flags:

```bash
python -m feature_prd_runner.runner \
  --project-dir . \
  --prd-file ./docs/feature_prd.md \
  --test-command "npm test" \
  --resume-prompt "Focus on error handling first"
```

## What It Does

1. Plans phases from the PRD and repository context.
2. Creates one task per phase.
3. For each phase:
   - Checks out a branch for the phase.
   - Runs Codex CLI to implement the phase.
   - Runs tests and fixes failures.
   - Runs a review against PRD requirements and acceptance criteria.
   - Commits and pushes when clean.

Each step writes progress to durable files for easy resume.

## State Files

All state lives in `.prd_runner/` inside the project directory:

- `run_state.yaml`: current status, active task/phase, last error
- `task_queue.yaml`: tasks derived from phases
- `phase_plan.yaml`: planned phases
- `artifacts/`: events, progress snapshot, tests, reviews
- `runs/`: per-run logs and prompts

## Resume Prompts

Use `--resume-prompt` to inject special instructions into the next agent run:

```bash
python -m feature_prd_runner.runner --project-dir . --prd-file ./docs/feature_prd.md \
  --resume-prompt "Prioritize fixing CI failures"
```

The prompt is applied once to the next run and then cleared.
