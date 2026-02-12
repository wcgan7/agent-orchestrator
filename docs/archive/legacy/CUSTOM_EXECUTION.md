# Custom Execution & Flexible Prompts

> **Status (2026-02-11):** Legacy runtime guide for the pre-v3 architecture.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document describes how to use custom prompts and flexible execution modes with the Feature PRD Runner.

## Overview

The Feature PRD Runner supports flexible execution beyond the standard PLAN → IMPLEMENT → VERIFY → REVIEW → COMMIT cycle. You can:

- Execute ad-hoc custom prompts
- Bypass AGENTS.md rules when needed (superadmin mode)
- Run one-off tasks without full workflow
- Provide context for targeted execution

## The `exec` Command

### Basic Usage

Execute a custom prompt as a standalone task:

```bash
feature-prd-runner exec "Update all copyright headers to 2026"
```

### With Project Context

```bash
feature-prd-runner exec "Refactor the auth module for clarity" \
  --project-dir /path/to/project
```

### Superadmin Mode

Enable superadmin mode to bypass AGENTS.md rules:

```bash
feature-prd-runner exec "Quick fix: update version number" \
  --override-agents
```

**When superadmin mode is enabled:**
- File allowlists are not enforced
- Documentation/testing requirements can be skipped
- Normal AGENTS.md rules can be bypassed
- You have full control as the "superadmin"

**Use cases for superadmin mode:**
- Emergency hotfixes
- Quick administrative changes
- Experimental modifications
- When you know better than the rules

### With Context

Provide context to focus the execution:

```bash
# Focus on specific task files
feature-prd-runner exec "Add logging to error paths" \
  --context-task phase-2

# Focus on specific files
feature-prd-runner exec "Fix type errors" \
  --context-files "src/auth.py,src/models/user.py"
```

### Continue After Execution

Run a custom prompt and then continue with normal workflow:

```bash
feature-prd-runner exec "Regenerate lockfiles" \
  --then-continue
```

### Full Example

```bash
feature-prd-runner exec \
  "Add comprehensive error handling to the payment module" \
  --project-dir . \
  --override-agents \
  --context-files "src/payments/*.py" \
  --shift-minutes 30 \
  --heartbeat-seconds 60
```

## Integration with `run` Command

### Custom Prompt Before Run

Use `--custom-prompt` with the `run` command to execute a custom prompt before starting the normal workflow:

```bash
feature-prd-runner run \
  --prd-file feature.md \
  --custom-prompt "Update dependencies and ensure tests pass"
```

### With Superadmin Mode

```bash
feature-prd-runner run \
  --prd-file feature.md \
  --custom-prompt "Emergency: fix critical security vulnerability" \
  --override-agents
```

The `--override-agents` flag applies to the custom prompt, allowing you to bypass AGENTS.md rules for the initial task while maintaining normal rules for the PRD implementation.

## Command Reference

### `exec` Command Options

```
feature-prd-runner exec PROMPT [options]

Arguments:
  PROMPT                        Custom prompt/instructions to execute

Options:
  --project-dir PATH            Project directory (default: current directory)
  --codex-command CMD           Codex CLI command (default: codex exec -)
  --override-agents             Enable superadmin mode - bypass AGENTS.md rules
  --then-continue               Continue to normal cycle after completion
  --context-task TASK_ID        Task ID for context (limits scope to task files)
  --context-files FILES         Comma-separated list of files to focus on
  --shift-minutes N             Timebox for execution in minutes (default: 45)
  --heartbeat-seconds N         Heartbeat interval (default: 120)
  --heartbeat-grace-seconds N   Heartbeat grace period (default: 300)
  --log-level LEVEL             Logging verbosity (default: info)
```

### `run` Command Options (Custom Prompt)

```
feature-prd-runner run --prd-file FILE [options]

Custom Prompt Options:
  --custom-prompt PROMPT        Standalone prompt to execute before implementation
  --override-agents             Enable superadmin mode for custom-prompt
```

## Superadmin Mode Details

### What Gets Overridden

When `--override-agents` is enabled, the AI worker receives a special prompt prefix:

```
IMPORTANT - SUPERADMIN MODE:
You are operating in SUPERADMIN mode. You have special privileges:
- You may bypass normal AGENTS.md rules if necessary to complete the task
- You may modify any files needed (not restricted by allowlists)
- You may skip documentation/testing requirements if time-sensitive
- However, still follow best practices where reasonable
- Explain any rule bypasses in your progress report
```

### When to Use Superadmin Mode

**Appropriate use cases:**
- ✅ Emergency hotfixes requiring immediate action
- ✅ Administrative tasks (version bumps, config changes)
- ✅ Experimental features where rules don't apply
- ✅ Quick prototypes or proof-of-concepts
- ✅ Situations where you (the human) know better than the rules

**Inappropriate use cases:**
- ❌ Regular feature development (use normal workflow)
- ❌ To avoid writing tests (tests are important!)
- ❌ To skip code review (quality matters)
- ❌ Because you're lazy (rules exist for good reasons)

### Safety Considerations

- **Use sparingly**: Superadmin mode should be the exception, not the rule
- **Review carefully**: Always review changes made in superadmin mode
- **Document reasons**: Include rationale for bypassing rules
- **Consider consequences**: Skipped tests/docs may cause problems later

## Examples

### Example 1: Emergency Security Fix

```bash
# Critical security vulnerability discovered
feature-prd-runner exec \
  "Apply security patch from CVE-2025-1234 to auth module" \
  --override-agents \
  --context-files "src/auth/**/*.py"
```

**Why superadmin mode?** Time-sensitive security fix requires immediate action without full workflow overhead.

### Example 2: Dependency Update

```bash
# Update dependencies before starting new feature
feature-prd-runner run \
  --prd-file new-feature.md \
  --custom-prompt "Update all dependencies to latest versions and fix breaking changes" \
  --override-agents
```

**Why superadmin mode?** Dependency updates may touch many files outside normal allowlists.

### Example 3: Code Cleanup

```bash
# Quick cleanup task
feature-prd-runner exec \
  "Remove all TODO comments and unused imports across the codebase"
```

**Why no superadmin mode?** This is a routine cleanup that should follow normal rules.

### Example 4: Experimental Feature

```bash
# Try out a new approach
feature-prd-runner exec \
  "Experiment with Redis caching for API responses" \
  --override-agents \
  --context-files "src/cache.py,src/api/*.py"
```

**Why superadmin mode?** Experimental feature where standard rules may not apply yet.

### Example 5: Configuration Change

```bash
# Simple config update
feature-prd-runner exec \
  "Update database connection string in config.yaml to use new endpoint" \
  --override-agents
```

**Why superadmin mode?** Administrative change that doesn't need full workflow.

## Best Practices

1. **Prefer normal workflow**: Use `exec` and superadmin mode only when truly needed
2. **Be specific**: Provide clear, detailed prompts for better results
3. **Use context**: Limit scope with `--context-files` or `--context-task`
4. **Review output**: Always review changes made with superadmin mode
5. **Document rationale**: Include comments explaining why rules were bypassed
6. **Test afterward**: Even with superadmin mode, verify changes work correctly

## Troubleshooting

### Worker Blocked Despite Superadmin Mode

If the worker still reports blocking issues in superadmin mode, it may be encountering technical limitations (not rules). Review the `human_blocking_issues` in the progress file to understand what's blocking.

### Changes to Wrong Files

Use `--context-files` to limit the scope:

```bash
feature-prd-runner exec "Fix bug" --context-files "src/specific_module.py"
```

### Prompt Too Long

Break long prompts into multiple steps:

```bash
feature-prd-runner exec "Step 1: Refactor module A"
feature-prd-runner exec "Step 2: Update tests"
feature-prd-runner exec "Step 3: Update documentation"
```

### Want to See What Would Happen

Currently there's no dry-run mode for `exec`. Consider:
1. Running on a separate branch
2. Using a checkpoint/backup before execution
3. Reviewing the prompt carefully before execution

## Future Enhancements

Planned features for custom execution:

- Custom workflows (define your own step sequences)
- Step injection (insert custom steps into running workflows)
- Interactive mode (approve each action)
- Conditional execution (run based on conditions)
- Parallel custom tasks (run multiple prompts concurrently)

See `ROADMAP.md` for full details on upcoming features.

## Related Documentation

- [ROADMAP.md](../ROADMAP.md) - Full feature roadmap including flexible step progression
- [README.md](../README.md) - General usage documentation
- [AGENTS.md](../example/AGENTS.md) - Example AGENTS.md file with rules

## Feedback

Have suggestions for improving custom execution? Open an issue on GitHub!
