# Workdoc Retry Intent and Behavioral Specification

## Purpose
Define the intended retry behavior so task context is preserved end-to-end and the workdoc remains the primary source of execution history.

## Problem Statement
Today, retry re-queues execution but the run path reinitializes the workdoc from template. This can erase prior narrative context from the canonical workdoc and force workers to proceed with partial context.

## Intended Outcome
Retry should resume execution with full prior context. The workdoc should be append-only across attempts for the same task.

## Core Principles
- The canonical workdoc is the task's historical record.
- Retry is a continuation of a task, not a reset of task context.
- Historical context must remain readable and attributable by attempt.

## Behavioral Invariants
- Canonical workdoc must never be reset after first initialization.
- Retry must not rewrite workdoc from template.
- Retry must append new attempt context and step outputs.
- Prior attempt content must remain intact.
- Worker must always receive the latest canonical workdoc before each step.

## Retry Reset vs Preserve Rules

### Allowed to Reset (Execution Control)
- `task.status` to `queued`
- `task.pending_gate` to `None`
- current run transient error field if needed for restart (optionally archived into history)
- other strictly run-ephemeral execution flags

### Must Preserve (Task Context)
- canonical workdoc content
- `step_outputs`
- review/fix history
- retry guidance history
- preserved branch metadata and related retry metadata
- prior run records and review cycles

## Workdoc Append Semantics
- On retry start, append an attempt marker with timestamp and optional guidance.
- New step output is appended under existing section headings with attempt-scoped subheadings.
- No placeholder/template rehydration if canonical already exists.
- For orchestrator-managed steps, orchestrator appends summaries as today.
- For worker-managed steps, accepted worker edits should merge without removing prior history.

## Retry From Step Semantics
- `retry_from_step` controls execution start point only.
- Earlier steps may be skipped for execution, but historical workdoc content remains.
- New output appends from selected start step onward.

## Failure and Recovery Policy
- If canonical workdoc is missing during retry for a previously-run task, block with explicit error.
- Do not silently regenerate from template for retry.
- If malformed/unreadable workdoc is encountered, block with diagnostic context.

## Observability
- Emit retry lifecycle context (attempt/start step/guidance) in events.
- Keep existing `task.retry` and `workdoc.updated` compatibility behavior.

## Non-Goals
- Rewriting or compacting historical workdoc attempts.
- Deleting prior attempt history.
- Changing broad worker prompting strategy beyond full-context behavior.

## Acceptance Criteria
- Retry preserves prior workdoc content and appends only new attempt material.
- Worker sees preserved context on every retried step.
- Missing canonical workdoc on retry produces explicit block, not silent reset.
