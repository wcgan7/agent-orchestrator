# Debugging & Error Analysis

> **Status (2026-02-11):** Legacy runtime guide for the pre-v3 architecture.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document describes the enhanced debugging and error analysis features for troubleshooting blocked tasks and understanding failures.

## Overview

The Feature PRD Runner provides comprehensive debugging tools to help you:

- Understand why tasks are blocked or failing
- View detailed error analysis with root cause and suggestions
- Inspect task state at any point
- Trace event history for a task
- View detailed logs from runs
- Get actionable recommendations for fixing issues

## Commands

### `explain` - Explain Why Task is Blocked

Get a human-readable explanation of why a task is blocked.

```bash
feature-prd-runner explain TASK_ID [options]
```

**Arguments:**
- `TASK_ID`: Task identifier (e.g., `phase-1`)

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)

**Example:**

```bash
$ feature-prd-runner explain phase-1

Task 'phase-1' is blocked and requires human intervention.

Status: blocked
Step: implement
Attempts: 3

Last Error:
  Verification failed: 2 tests failed

Error Type: test_failed

Blocking Reason:
  Tests failed after 3 attempts

Blocking Issues:
  • Test test_auth.py::test_login failed with AssertionError
  • Test test_auth.py::test_logout failed with AttributeError

Suggested Next Steps:
  • Review test failures in logs
  • Fix failing tests and retry
  • Consider skipping tests if they're not critical

To Resume:
  feature-prd-runner resume phase-1
  (will resume at step: implement)
```

### `inspect` - Inspect Task State

Inspect detailed state of a task including all metadata and context.

```bash
feature-prd-runner inspect TASK_ID [options]
```

**Arguments:**
- `TASK_ID`: Task identifier

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--json`: Output as JSON for programmatic access

**Example:**

```bash
$ feature-prd-runner inspect phase-1

Task State: phase-1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lifecycle:      waiting_human
Step:           implement
Status:         blocked
Attempts:       3

Last Error:
Verification failed: 2 tests failed

Error Type: test_failed

Context:
  • Task started at 2026-01-16T10:30:00Z
  • Completed plan_impl successfully
  • Implemented auth module changes
  • Verification failed on attempt 1
  • Reimplemented with fixes
  • Verification failed again on attempt 2
  ... and 3 more

Metadata:
  type: implement
  phase_id: authentication
  prompt_mode: None
  auto_resume_count: 2
  manual_resume_attempts: 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**JSON Output:**

```bash
$ feature-prd-runner inspect phase-1 --json
{
  "task_id": "phase-1",
  "lifecycle": "waiting_human",
  "step": "implement",
  "status": "blocked",
  "attempts": 3,
  "last_error": "Verification failed: 2 tests failed",
  "last_error_type": "test_failed",
  "context": [
    "Task started at 2026-01-16T10:30:00Z",
    ...
  ],
  "metadata": {
    "type": "implement",
    "phase_id": "authentication",
    ...
  }
}
```

### `trace` - Trace Event History

View the full event history for a task to understand what happened.

```bash
feature-prd-runner trace TASK_ID [options]
```

**Arguments:**
- `TASK_ID`: Task identifier

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--json`: Output as JSON
- `--limit N`: Maximum number of events to show (default: 50)

**Example:**

```bash
$ feature-prd-runner trace phase-1 --limit 10

Event History for phase-1 (showing 10 events):
================================================================================

1. [2026-01-16T10:30:15Z] task_started
   Run ID: 20260116T103015Z-abc123-implement

2. [2026-01-16T10:30:20Z] worker_succeeded
   Run ID: 20260116T103015Z-abc123-plan_impl

3. [2026-01-16T10:32:45Z] worker_succeeded
   Run ID: 20260116T103245Z-def456-implement

4. [2026-01-16T10:35:10Z] verification_result
   Run ID: 20260116T103510Z-ghi789-verify
   Status: ✗ FAILED

5. [2026-01-16T10:35:15Z] worker_failed
   Run ID: 20260116T103245Z-def456-implement
   Error: test_failed
   Detail: Verification failed: 2 tests failed

6. [2026-01-16T10:40:30Z] task_resumed
   Run ID: 20260116T104030Z-jkl012-implement

7. [2026-01-16T10:43:15Z] worker_succeeded
   Run ID: 20260116T104030Z-jkl012-implement

8. [2026-01-16T10:45:00Z] verification_result
   Run ID: 20260116T104500Z-mno345-verify
   Status: ✗ FAILED

9. [2026-01-16T10:45:05Z] worker_failed
   Run ID: 20260116T104030Z-jkl012-implement
   Error: test_failed
   Detail: Verification still failing after fix

10. [2026-01-16T10:45:10Z] task_blocked
    Reason: Max verification attempts reached
```

### `logs` - View Detailed Logs

View detailed logs from task execution, including stdout, stderr, and test output.

```bash
feature-prd-runner logs TASK_ID [options]
```

**Arguments:**
- `TASK_ID`: Task identifier

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--step STEP`: Show logs for specific step (e.g., `verify`, `implement`)
- `--lines N`: Number of lines to show (default: 100)
- `--run-id ID`: Specific run ID (auto-detects most recent if not specified)

**Examples:**

```bash
# View all logs for task
$ feature-prd-runner logs phase-1

# View only verification logs
$ feature-prd-runner logs phase-1 --step verify

# View last 200 lines
$ feature-prd-runner logs phase-1 --lines 200

# View logs from specific run
$ feature-prd-runner logs phase-1 --run-id 20260116T103015Z-abc123-verify
```

**Output:**

```
Logs for task phase-1 (run: 20260116T103015Z-abc123-verify)
================================================================================

--- stdout ---
(showing last 100 lines)

Starting verification...
Running tests: pytest -v tests/
...
collected 12 items

tests/test_auth.py::test_login FAILED                                    [ 8%]
tests/test_auth.py::test_logout FAILED                                   [16%]
tests/test_auth.py::test_register PASSED                                 [25%]
...

--- stderr ---
(no errors)

--- Test Output ---
============================= test session starts ==============================
...
FAILED tests/test_auth.py::test_login - AssertionError: assert 401 == 200
FAILED tests/test_auth.py::test_logout - AttributeError: 'User' has no attribute 'session'
=========================== short test summary info ============================
...
```

## Error Analysis

The error analyzer provides comprehensive analysis of failures with:

### Root Cause Analysis

For each error type, the analyzer attempts to identify the root cause:

- **Test Failures**: Identifies which tests failed and why
- **Worker Failures**: Analyzes why the Codex worker failed
- **Allowlist Violations**: Shows which files were modified outside the plan
- **No Progress**: Detects when worker completes but makes no changes
- **Review Failures**: Summarizes code review issues
- **Git Failures**: Analyzes git push/commit problems

### Suggested Actions

Every error includes actionable suggestions:

```
Suggested Actions:
  [1] Review test failures
      $ feature-prd-runner logs phase-1 --step verify

  [2] Retry after reviewing
      $ feature-prd-runner retry phase-1

  [3] Skip verification (not recommended)
      $ feature-prd-runner skip-step phase-1 --step verify
```

### Quick Fixes

Common fixes are highlighted:

```
Quick Fixes:
  • View full test output
    $ cat .prd_runner/runs/*/tests_phase-1.log

  • Debug interactively
    $ feature-prd-runner debug phase-1
```

## Programmatic Access

All debugging commands support `--json` output for scripting:

```python
import json
import subprocess

# Get task state
result = subprocess.run(
    ["feature-prd-runner", "inspect", "phase-1", "--json"],
    capture_output=True,
    text=True
)
state = json.loads(result.stdout)

print(f"Task {state['task_id']} is {state['lifecycle']}")
print(f"Last error: {state['last_error']}")

# Get event history
result = subprocess.run(
    ["feature-prd-runner", "trace", "phase-1", "--json"],
    capture_output=True,
    text=True
)
events = json.loads(result.stdout)

for event in events:
    print(f"{event['timestamp']}: {event['event_type']}")
```

## Error Types

The system recognizes these error types:

| Error Type | Description | Common Causes |
|------------|-------------|---------------|
| `test_failed` | Tests failed during verification | Bugs in implementation, incorrect tests |
| `worker_failed` | Codex worker failed to complete | Timeout, API errors, invalid prompts |
| `allowlist_violation` | Modified files outside plan | Worker scope creep, incorrect plan |
| `no_progress` | No code changes introduced | Task already done, worker confused |
| `review_failed` | Code review found issues | Quality problems, missing requirements |
| `git_push_failed` | Cannot push to remote | Auth issues, network problems |
| `no_heartbeat` | Worker stopped updating progress | Crash, hang, resource exhaustion |
| `plan_missing` | Implementation plan not found | Plan step failed or skipped |
| `state_corrupt` | State files corrupted | File system issues, manual edits |

## Best Practices

### When a Task is Blocked

1. **Explain first**: Start with `explain` to understand why
2. **Inspect state**: Use `inspect` to see full task state
3. **View logs**: Check `logs` for detailed error messages
4. **Trace history**: Use `trace` to understand what led to the block
5. **Fix and retry**: Make changes, then use `retry` or `resume`

### Debugging Test Failures

```bash
# 1. View test output
feature-prd-runner logs phase-1 --step verify --lines 500

# 2. Understand the error
feature-prd-runner explain phase-1

# 3. Check what changed
cd /path/to/project
git diff

# 4. Fix the issues manually or let runner retry
feature-prd-runner retry phase-1
```

### Understanding Worker Failures

```bash
# 1. Check worker output
feature-prd-runner logs phase-1 --step implement

# 2. Look for errors in stderr
cat .prd_runner/runs/*/stderr.log

# 3. Check if it was a timeout or API issue
feature-prd-runner trace phase-1 | grep worker_failed

# 4. Retry with more time
# (edit .prd_runner/config.yaml to increase shift_minutes)
feature-prd-runner retry phase-1
```

### Tracking Down State Issues

```bash
# View full state
feature-prd-runner inspect phase-1 --json > phase-1-state.json

# Compare with previous run
feature-prd-runner inspect phase-1 --json > now.json
# (get previous state from backup)
diff previous.json now.json

# Check event history for transitions
feature-prd-runner trace phase-1 --limit 100
```

## Integration with Other Tools

### With `status` Command

```bash
# Get overview
feature-prd-runner status

# If blocked tasks found, investigate
feature-prd-runner explain phase-1
```

### With `resume` Command

```bash
# Explain why blocked
feature-prd-runner explain phase-1

# Fix issues, then resume
feature-prd-runner resume phase-1
```

### With Human-in-the-Loop

```bash
# Worker is running, check progress
feature-prd-runner logs phase-1 --lines 50

# Steer if needed
feature-prd-runner steer "Add more error handling"
```

## Advanced Usage

### Custom Error Analysis

You can use the ErrorAnalyzer programmatically:

```python
from pathlib import Path
from feature_prd_runner.debug import ErrorAnalyzer

analyzer = ErrorAnalyzer(Path("."))

# Analyze an error
report = analyzer.analyze_error(
    task_id="phase-1",
    error_type="test_failed",
    error_detail="2 tests failed",
    context={
        "failed_tests": ["test_auth.py::test_login", "test_auth.py::test_logout"],
        "exit_code": 1,
    }
)

# Format for display
formatted = analyzer.format_error_report(report, verbose=True)
print(formatted)

# Get suggestions
for action in report.suggested_actions:
    print(f"- {action['action']}: {action['command']}")
```

### Automated Monitoring

Create a monitoring script:

```python
#!/usr/bin/env python3
import json
import subprocess
import time

def check_tasks():
    result = subprocess.run(
        ["feature-prd-runner", "status", "--json"],
        capture_output=True,
        text=True
    )
    status = json.loads(result.stdout)

    # Check for blocked tasks
    blocked = status.get("blocking_tasks", [])
    if blocked:
        for task in blocked:
            task_id = task["id"]
            print(f"Task {task_id} blocked!")

            # Get explanation
            result = subprocess.run(
                ["feature-prd-runner", "explain", task_id],
                capture_output=True,
                text=True
            )
            print(result.stdout)

            # Send alert (email, Slack, etc.)
            # send_alert(task_id, result.stdout)

if __name__ == "__main__":
    while True:
        check_tasks()
        time.sleep(60)  # Check every minute
```

## Troubleshooting

### "Task not found"

```bash
# List all tasks
feature-prd-runner list --tasks

# Check if task ID is correct
# Task IDs are usually phase-1, phase-2, etc.
```

### "No events found"

Events are written to `.prd_runner/artifacts/events.jsonl`. If this file doesn't exist or is empty, no events have been recorded yet.

```bash
# Check if events file exists
ls -la .prd_runner/artifacts/events.jsonl

# Verify state directory structure
tree .prd_runner/
```

### "No runs found for task"

The task may not have started yet, or run directories may have been cleaned up.

```bash
# Check what runs exist
ls -la .prd_runner/runs/

# Check task queue
feature-prd-runner list --tasks
```

## Related Documentation

- [README.md](../README.md) - General usage and CLI reference
- [HUMAN_IN_THE_LOOP.md](HUMAN_IN_THE_LOOP.md) - Interactive steering and approval
- [CUSTOM_EXECUTION.md](CUSTOM_EXECUTION.md) - Ad-hoc execution
- [ROADMAP.md](../ROADMAP.md) - Future enhancements

## Feedback

Have suggestions for improving debugging features? Open an issue on GitHub!
