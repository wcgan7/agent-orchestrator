# Human-in-the-Loop Control

> **Status (2026-02-11):** Legacy runtime guide for the pre-v3 architecture.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document describes the human-in-the-loop features that enable active steering, approval gates, and bidirectional communication with running workers.

## Overview

The Feature PRD Runner supports several modes of human involvement:

1. **Steering**: Send guidance messages to running workers in real-time
2. **Approval Gates**: Pause execution at key points for human approval
3. **Interactive Mode**: Enable step-by-step approval at major checkpoints
4. **Message Bus**: Bidirectional communication channel between human and worker

## Quick Start

### Interactive Mode

Run with step-by-step approval gates enabled:

```bash
feature-prd-runner run --prd-file feature.md --interactive
```

This enables approval gates before implement, after implement, and before commit.

### Steering a Running Worker

Send guidance to a running worker:

```bash
# Single message
feature-prd-runner steer "Focus on error handling in the auth module"

# Interactive mode
feature-prd-runner steer
> Add more logging
> Check edge cases
> [Ctrl+C to exit]
```

### Approving/Rejecting Gates

When the runner pauses for approval, use these commands:

```bash
# Approve
feature-prd-runner approve
feature-prd-runner approve --feedback "Looks good, proceed"

# Reject
feature-prd-runner reject --reason "Need more tests before continuing"
```

## Commands

### `steer` - Send Steering Messages

Send guidance messages to a running worker.

```bash
feature-prd-runner steer [MESSAGE] [options]
```

**Arguments:**
- `MESSAGE`: Optional steering message to send. If omitted, enters interactive mode.

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--run-id ID`: Specific run ID to steer (auto-detects if not specified)

**Examples:**

```bash
# Send single message
feature-prd-runner steer "Add type hints to all functions"

# Interactive steering
feature-prd-runner steer
=== Interactive Steering Mode ===
Enter messages to send to the worker (Ctrl+C to exit)

> Focus on the payment module
✓ Message sent to worker
> Add comprehensive error handling
✓ Message sent to worker
^C
Exiting steering mode
```

**How it works:**
- Messages are written to `progress.json` for the worker to pick up
- Worker receives messages and can incorporate guidance into its work
- Use this to course-correct mid-execution without stopping the worker

### `approve` - Approve Pending Gate

Approve a pending approval request and allow the runner to continue.

```bash
feature-prd-runner approve [options]
```

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--run-id ID`: Specific run ID (auto-detects if not specified)
- `--feedback TEXT`: Optional feedback message to include with approval

**Examples:**

```bash
# Simple approval
feature-prd-runner approve

# Approval with feedback
feature-prd-runner approve --feedback "Good implementation, proceed to verification"
```

### `reject` - Reject Pending Gate

Reject a pending approval request and stop execution.

```bash
feature-prd-runner reject --reason REASON [options]
```

**Options:**
- `--project-dir PATH`: Project directory (default: current directory)
- `--run-id ID`: Specific run ID (auto-detects if not specified)
- `--reason TEXT`: **Required** - Reason for rejection

**Examples:**

```bash
feature-prd-runner reject --reason "Need to add unit tests before proceeding"
feature-prd-runner reject --reason "Implementation doesn't match specification"
```

## Configuration

### Config File Setup

You can configure approval gates in `.prd_runner/config.yaml`:

```yaml
approval_gates:
  enabled: true
  gates:
    before_implement:
      enabled: true
      message: "Review implementation plan before proceeding?"
      show_plan: true
      timeout: 300

    after_implement:
      enabled: true
      message: "Review code changes before verification?"
      show_diff: true
      allow_edit: true
      timeout: 300

    before_commit:
      enabled: true
      message: "Review and approve commit?"
      show_diff: true
      show_tests: true
      show_review: true
      required: true  # Cannot skip or timeout

    after_review_issues:
      enabled: false
      message: "Review found issues. Continue fixing?"
      show_review: true
      timeout: 300
```

### Gate Types

Available approval gate types:

- `before_plan_impl`: Before planning implementation (shows task context)
- `before_implement`: Before implementation starts (shows plan)
- `after_implement`: After implementation completes (shows diff)
- `before_verify`: Before running verification (shows changes)
- `after_verify`: After verification completes (shows test results)
- `before_review`: Before running review (shows verification results)
- `after_review_issues`: After review finds blocking issues (shows issues)
- `before_commit`: Before git commit/push (shows diff, tests, review)

### Gate Configuration Options

Each gate supports these options:

```yaml
gate_name:
  enabled: bool              # Whether this gate is active
  message: string            # Custom prompt message
  show_diff: bool            # Show git diff
  show_plan: bool            # Show implementation plan
  show_tests: bool           # Show test results
  show_review: bool          # Show review results
  timeout: int               # Auto-approve timeout in seconds (0 = no timeout)
  required: bool             # If true, cannot skip or timeout
  allow_edit: bool           # Allow human to request edits
```

## Use Cases

### Review Before Implementation

Enable gate to review the implementation plan before the worker starts coding:

```yaml
approval_gates:
  enabled: true
  gates:
    before_implement:
      enabled: true
      message: "Review implementation plan before coding?"
      show_plan: true
```

When the runner reaches this gate:

```
======================================================================
APPROVAL REQUIRED: before_implement
======================================================================

Review implementation plan before coding?

Task:    phase-1
Phase:   authentication
Step:    plan_impl

Implementation Plan

Files to change: 5
New files: 2

Sample files: ['src/auth/login.py', 'src/auth/session.py', 'src/auth/middleware.py']

Waiting for approval...
Use 'feature-prd-runner approve' or 'feature-prd-runner reject' to respond
Auto-approve in 300s if no response
```

Then in another terminal:

```bash
feature-prd-runner approve --feedback "Plan looks good"
```

### Review Changes Before Commit

Ensure human approval before committing changes to git:

```yaml
approval_gates:
  enabled: true
  gates:
    before_commit:
      enabled: true
      message: "Review and approve commit?"
      show_diff: true
      show_tests: true
      show_review: true
      required: true  # Must approve, no timeout
```

This shows a comprehensive summary before commit:

```
======================================================================
APPROVAL REQUIRED: before_commit
======================================================================

Review and approve commit?

Files changed: 8
  • src/auth/login.py
  • src/auth/session.py
  • src/auth/models.py
  ... and 5 more

Changes
[git diff output, truncated to 2000 chars]

Test Results
Status: PASSED
Exit code: 0

Review Results
Status: MERGEABLE
Issues found: 0

Waiting for approval...
Use 'feature-prd-runner approve' or 'feature-prd-runner reject' to respond
Required - cannot skip or timeout
```

### Steer During Implementation

Monitor the running worker and provide guidance:

**Terminal 1:** Running worker
```bash
feature-prd-runner run --prd-file feature.md
...
[worker is implementing changes]
```

**Terminal 2:** Steering
```bash
feature-prd-runner steer
=== Interactive Steering Mode ===
Enter messages to send to the worker (Ctrl+C to exit)

> Make sure to add comprehensive docstrings to all new functions
✓ Message sent to worker
> Focus on error handling for edge cases
✓ Message sent to worker
> Add type hints for better IDE support
✓ Message sent to worker
```

The worker receives these messages and incorporates them into its decision-making.

### Request Explanation

Ask the worker to explain its decisions or approach:

```bash
feature-prd-runner steer "Please explain your approach to implementing authentication"
```

The worker will see this message and can provide context in its progress updates.

## Interactive Mode Details

When you run with `--interactive`, the runner automatically enables these gates:

- `before_implement`: Review plan before coding starts
- `after_implement`: Review changes before verification
- `before_commit`: Review everything before committing

This provides checkpoints at major phase transitions.

**Example workflow:**

```bash
# Start in interactive mode
feature-prd-runner run --prd-file feature.md --interactive

# Runner pauses at "before_implement"
# (in another terminal)
feature-prd-runner approve

# Worker implements changes...
# Runner pauses at "after_implement"
feature-prd-runner steer "Add more error handling"
feature-prd-runner approve

# Verification and review run...
# Runner pauses at "before_commit"
feature-prd-runner approve --feedback "Looks good, commit it"

# Changes committed and pushed
```

## Message Bus Architecture

The bidirectional communication uses `progress.json` as the transport layer:

### Human → Worker Messages

Structure stored in `progress.json`:

```json
{
  "messages_from_human": [
    {
      "id": "guidance-1234567890",
      "type": "guidance",
      "content": "Focus on error handling",
      "timestamp": "2026-01-16T10:30:00Z",
      "metadata": {}
    }
  ]
}
```

Message types:
- `guidance`: Steering advice for the worker
- `clarification_request`: Question for the worker
- `approval_request`: Request for human approval

### Worker → Human Messages

Structure stored in `progress.json`:

```json
{
  "messages_to_human": [
    {
      "id": "status-1234567890",
      "type": "clarification_response",
      "content": "I'm implementing authentication using JWT tokens...",
      "timestamp": "2026-01-16T10:31:00Z",
      "metadata": {
        "in_reply_to": "guidance-1234567890"
      }
    }
  ]
}
```

### Approval Flow

When a gate requires approval:

1. Runner writes `approval_pending` to `progress.json`:
   ```json
   {
     "approval_pending": {
       "id": "approval-uuid",
       "gate_type": "before_commit",
       "message": "Review and approve commit?",
       "context": { ... },
       "timeout": 300,
       "created_at": "2026-01-16T10:30:00Z"
     }
   }
   ```

2. Runner displays prompt to user and polls for response

3. Human responds via CLI:
   ```bash
   feature-prd-runner approve --feedback "Looks good"
   ```

4. CLI writes `approval_response` to `progress.json`:
   ```json
   {
     "approval_response": {
       "request_id": "approval-uuid",
       "approved": true,
       "feedback": "Looks good",
       "responded_at": "2026-01-16T10:31:00Z"
     }
   }
   ```

5. Runner reads response and continues or stops

## Troubleshooting

### No Pending Approval Found

```
$ feature-prd-runner approve
No pending approval request
```

**Solutions:**
- Check if the runner is actually waiting for approval
- Verify you're in the correct project directory
- Check `.prd_runner/runs/*/progress.json` for pending requests

### Cannot Find Active Run

```
$ feature-prd-runner steer "message"
No active run found
```

**Solutions:**
- Verify a runner is actually running (`ps aux | grep feature-prd-runner`)
- Check `.prd_runner/run_state.yaml` for active run status
- Specify `--run-id` explicitly if auto-detection fails

### Messages Not Being Received

If the worker doesn't seem to receive steering messages:

1. Check the worker is reading messages (this depends on Codex/worker implementation)
2. Verify messages are written to `progress.json`:
   ```bash
   cat .prd_runner/runs/*/progress.json | jq '.messages_from_human'
   ```
3. Check worker logs for message processing

### Approval Timeout

If approval request times out:

- Gates auto-approve by default after timeout (configurable)
- Set `required: true` to prevent timeout
- Increase `timeout` value in gate configuration

## Advanced Usage

### Programmatic Control

You can interact with the message bus programmatically:

```python
from pathlib import Path
from feature_prd_runner.messaging import MessageBus, Message
from datetime import datetime, timezone

# Connect to running worker
progress_path = Path(".prd_runner/runs/20260116T103000Z-abcd1234/progress.json")
bus = MessageBus(progress_path)

# Send guidance
bus.send_guidance("Focus on edge case handling")

# Send custom message
msg = Message(
    id="custom-123",
    type="guidance",
    content="Custom instruction",
    timestamp=datetime.now(timezone.utc).isoformat(),
    metadata={"priority": "high"}
)
bus.send_to_worker(msg)

# Read messages from worker
messages = bus.receive_from_worker()
for m in messages:
    print(f"{m.type}: {m.content}")
```

### Custom Gate Configuration

Create custom gates by extending the configuration:

```yaml
approval_gates:
  enabled: true
  gates:
    # Custom gate before specific operations
    custom_checkpoint:
      enabled: true
      message: "Custom checkpoint - approve?"
      timeout: 600
      show_diff: true
      required: false
```

Then integrate into your workflow by modifying the orchestrator code to call:

```python
from feature_prd_runner.approval_gates import GateType

# At your custom checkpoint
if approval_manager:
    response = approval_manager.request_approval(
        GateType.CUSTOM_CHECKPOINT,
        progress_path,
        context={"custom_data": "..."}
    )
    if not response.approved:
        # Handle rejection
        pass
```

## Future Enhancements

Planned improvements to human-in-the-loop:

1. **Two-way Clarification**: Worker can ask questions and wait for human response
2. **Edit Requests**: Human can request specific edits without rejecting
3. **Conditional Gates**: Enable gates based on conditions (e.g., only if tests fail)
4. **Web UI**: Browser-based approval interface with richer visualizations
5. **Notification Integration**: Slack/email/webhook notifications for approvals
6. **Approval History**: Track all approval decisions and outcomes
7. **Multi-user Approval**: Require approvals from multiple people
8. **Approval Policies**: Configure which users can approve which gates

See [ROADMAP.md](../ROADMAP.md) for full details.

## Related Documentation

- [README.md](../README.md) - General usage and CLI reference
- [CUSTOM_EXECUTION.md](CUSTOM_EXECUTION.md) - Ad-hoc execution and superadmin mode
- [ROADMAP.md](../ROADMAP.md) - Future features and enhancements
- [example/AGENTS.md](../example/AGENTS.md) - Example agent rules file

## Feedback

Have suggestions for improving human-in-the-loop features? Open an issue on GitHub!
