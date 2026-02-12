# Parallel Phase Execution

> **Status (2026-02-11):** Legacy runtime guide for the pre-v3 architecture.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document describes the parallel execution feature that allows independent phases to run concurrently, significantly reducing overall execution time.

## Overview

The Feature PRD Runner can execute independent phases in parallel using dependency resolution and topological sorting. This is particularly beneficial for projects with multiple independent modules or features that can be developed simultaneously.

## Key Features

1. **Automatic Dependency Resolution**: Analyzes phase dependencies and creates optimal execution plan
2. **Topological Sorting**: Orders phases based on dependencies using Kahn's algorithm
3. **Circular Dependency Detection**: Prevents deadlocks by detecting cycles
4. **Worker Pool Management**: Configurable number of parallel workers
5. **Progress Tracking**: Real-time status updates for all running phases
6. **Failure Handling**: Continues execution of independent phases even if one fails

## Quick Start

### Enable Parallel Execution

```bash
feature-prd-runner run --prd-file feature.md --parallel
```

### Limit Parallel Workers

```bash
feature-prd-runner run --prd-file feature.md --parallel --max-workers 2
```

### Visualize Execution Plan

```bash
# Show execution plan with batches
feature-prd-runner plan-parallel

# Show dependency tree
feature-prd-runner plan-parallel --tree
```

## How It Works

### Dependency Resolution

Phases specify dependencies in `phase_plan.yaml`:

```yaml
phases:
  - id: database-schema
    description: "Set up database schema"
    deps: []  # No dependencies - can run immediately

  - id: frontend-components
    description: "Build UI components"
    deps: []  # Independent of database

  - id: api-endpoints
    description: "Create API endpoints"
    deps: ["database-schema"]  # Depends on database

  - id: integration
    description: "Integrate frontend and backend"
    deps: ["api-endpoints", "frontend-components"]  # Depends on both
```

### Execution Batches

The parallel executor creates batches of phases that can run concurrently:

```
Batch 1 (parallel):
  ├─ database-schema
  └─ frontend-components

Batch 2 (after Batch 1 completes):
  └─ api-endpoints (waits for database-schema)

Batch 3 (after Batch 2 completes):
  └─ integration (waits for api-endpoints and frontend-components)
```

### Topological Sort

The executor uses **Kahn's algorithm** for topological sorting:

1. Calculate in-degree (number of dependencies) for each phase
2. Start with phases that have zero dependencies
3. Execute them in parallel as a batch
4. Remove completed phases from dependency graph
5. Repeat until all phases are executed

### Circular Dependency Detection

Before execution, the system checks for circular dependencies:

```
Phase A depends on Phase B
Phase B depends on Phase C
Phase C depends on Phase A  ❌ CIRCULAR!
```

If detected, execution stops with a clear error message showing the cycle.

## Configuration

### In `.prd_runner/config.yaml`

```yaml
parallel:
  enabled: true
  max_workers: 3
  resource_limits:
    max_memory_mb: 8192
    max_cpu_percent: 80
  fail_fast: false  # Continue even if one phase fails
```

### Phase Dependencies

Define dependencies in phases:

```yaml
phases:
  - id: phase-1
    deps: []

  - id: phase-2
    deps: ["phase-1"]

  - id: phase-3
    deps: ["phase-1"]

  - id: phase-4
    deps: ["phase-2", "phase-3"]
```

### Parallel Groups (Optional)

Group related phases:

```yaml
phases:
  - id: backend-db
    parallel_group: "backend"
    deps: []

  - id: backend-api
    parallel_group: "backend"
    deps: ["backend-db"]

  - id: frontend-ui
    parallel_group: "frontend"
    deps: []
```

## Commands

### `run` with `--parallel`

Execute phases in parallel:

```bash
# Basic parallel execution
feature-prd-runner run --prd-file feature.md --parallel

# With custom worker count
feature-prd-runner run --prd-file feature.md --parallel --max-workers 5

# Combined with other flags
feature-prd-runner run \\
  --prd-file feature.md \\
  --parallel \\
  --max-workers 3 \\
  --interactive
```

### `plan-parallel`

Visualize execution plan before running:

```bash
# Show execution batches
$ feature-prd-runner plan-parallel

Parallel Execution Plan
Total phases: 4
Batches: 3
Max parallelism: 2

Batch 1: (2 phase(s) in parallel)
  • database-schema
    Set up database schema
  • frontend-components
    Build UI components

Batch 2: (1 phase(s) in parallel)
  • api-endpoints (depends on: database-schema)
    Create API endpoints

Batch 3: (1 phase(s) in parallel)
  • integration (depends on: api-endpoints, frontend-components)
    Integrate frontend and backend
```

### Dependency Tree Visualization

```bash
$ feature-prd-runner plan-parallel --tree

Phase Dependency Tree
├─ database-schema
│  └─ api-endpoints
│     └─ integration
└─ frontend-components
   └─ integration
```

## Examples

### Example 1: Microservices Development

```yaml
phases:
  # Independent services
  - id: user-service
    description: "Develop user service"
    deps: []

  - id: order-service
    description: "Develop order service"
    deps: []

  - id: payment-service
    description: "Develop payment service"
    deps: []

  # API gateway depends on all services
  - id: api-gateway
    description: "Set up API gateway"
    deps: ["user-service", "order-service", "payment-service"]

  # Integration tests depend on gateway
  - id: integration-tests
    description: "Run integration tests"
    deps: ["api-gateway"]
```

**Execution**:
- Batch 1: user-service, order-service, payment-service (parallel)
- Batch 2: api-gateway
- Batch 3: integration-tests

**Time Savings**: If each service takes 30 minutes, sequential = 150 minutes, parallel = 90 minutes (40% reduction)

### Example 2: Full-Stack Feature

```yaml
phases:
  # Database layer
  - id: database-migrations
    description: "Create database migrations"
    deps: []

  # Backend layers (depend on database)
  - id: backend-models
    description: "Create data models"
    deps: ["database-migrations"]

  - id: backend-api
    description: "Build API endpoints"
    deps: ["backend-models"]

  # Frontend (independent of backend initially)
  - id: frontend-components
    description: "Build UI components"
    deps: []

  - id: frontend-state
    description: "Set up state management"
    deps: ["frontend-components"]

  # Integration
  - id: integration
    description: "Connect frontend to backend"
    deps: ["backend-api", "frontend-state"]

  # Final testing
  - id: e2e-tests
    description: "End-to-end tests"
    deps: ["integration"]
```

**Execution**:
- Batch 1: database-migrations, frontend-components (parallel)
- Batch 2: backend-models, frontend-state (parallel)
- Batch 3: backend-api
- Batch 4: integration
- Batch 5: e2e-tests

### Example 3: Documentation and Testing

```yaml
phases:
  # Core implementation
  - id: implement-feature
    description: "Implement the feature"
    deps: []

  # Parallel documentation and testing
  - id: write-docs
    description: "Write documentation"
    deps: ["implement-feature"]

  - id: write-tests
    description: "Write unit tests"
    deps: ["implement-feature"]

  - id: write-integration-tests
    description: "Write integration tests"
    deps: ["implement-feature"]

  # Final review
  - id: final-review
    description: "Final review"
    deps: ["write-docs", "write-tests", "write-integration-tests"]
```

**Execution**:
- Batch 1: implement-feature
- Batch 2: write-docs, write-tests, write-integration-tests (parallel)
- Batch 3: final-review

## Progress Tracking

While phases run in parallel, you can monitor progress:

```bash
# Status updates are logged automatically
[INFO] Parallel execution plan: 3 batches, max parallelism: 2
[INFO] Executing batch 1/3 with 2 phase(s)
[INFO] Phase database-schema completed: success=True
[INFO] Phase frontend-components completed: success=True
[INFO] Executing batch 2/3 with 1 phase(s)
[INFO] Phase api-endpoints completed: success=True
[INFO] Executing batch 3/3 with 1 phase(s)
[INFO] Phase integration completed: success=True
```

## Error Handling

### Failure in Batch

If a phase fails:

1. Other phases in the same batch continue
2. Dependent phases in future batches are skipped
3. Independent phases continue execution

Example:

```
Batch 1:
  ✓ database-schema (success)
  ✗ frontend-components (failed)

Batch 2:
  ✓ api-endpoints (success - depends only on database-schema)
  ⊘ frontend-state (skipped - depends on failed frontend-components)

Batch 3:
  ⊘ integration (skipped - depends on frontend-state)
```

### Circular Dependencies

```bash
$ feature-prd-runner run --prd-file feature.md --parallel
ERROR: Circular dependency detected: phase-a -> phase-b -> phase-c -> phase-a

Phases involved in cycle:
  1. phase-a depends on phase-c
  2. phase-b depends on phase-a
  3. phase-c depends on phase-b

Fix: Remove one dependency to break the cycle.
```

## Best Practices

### 1. Design for Parallelism

Structure your PRD with independent phases:

**Good** (parallelizable):
```yaml
phases:
  - frontend-ui
  - backend-api
  - database-schema
```

**Bad** (sequential chain):
```yaml
phases:
  - step-1
  - step-2 (depends: step-1)
  - step-3 (depends: step-2)
  - step-4 (depends: step-3)
```

### 2. Minimize Dependencies

Only specify necessary dependencies:

**Good**:
```yaml
- id: api-tests
  deps: ["api-implementation"]  # Only depends on API
```

**Bad**:
```yaml
- id: api-tests
  deps: ["api-implementation", "frontend", "docs"]  # Unnecessary deps
```

### 3. Balance Batch Sizes

Aim for balanced batches:

**Good** (2-2-2 phases):
```yaml
Batch 1: phase-1, phase-2
Batch 2: phase-3, phase-4
Batch 3: phase-5, phase-6
```

**Bad** (4-1-1 phases):
```yaml
Batch 1: phase-1, phase-2, phase-3, phase-4
Batch 2: phase-5
Batch 3: phase-6
```

### 4. Consider Resource Limits

Don't overload:

```bash
# Good for 8-core machine
--max-workers 3

# Too aggressive for 4-core machine
--max-workers 8
```

### 5. Test Sequentially First

Before enabling parallel:

```bash
# First, ensure sequential execution works
feature-prd-runner run --prd-file feature.md

# Then try parallel
feature-prd-runner run --prd-file feature.md --parallel
```

## Implementation Status

### What's Working Now

✅ **Dependency Resolution**: Fully implemented with topological sort and circular dependency detection

✅ **Execution Plan Visualization**: Shows how phases would be parallelized with `plan-parallel` command

✅ **Thread Pool Infrastructure**: Complete `ParallelExecutor` class with worker pool management

✅ **Progress Tracking**: Real-time status updates for parallel phases

✅ **CLI Integration**: `--parallel` and `--max-workers` flags integrated into run command

✅ **Phase Grouping**: Automatic grouping of tasks by phase_id

✅ **Batch Analysis**: Creates execution batches based on dependencies

✅ **Full Task Execution**: PhaseExecutor runs complete task lifecycle (plan_impl → implement → verify → review → commit)

✅ **Thread-Safe Git Operations**: GitCoordinator serializes git operations across parallel phases

✅ **Thread-Safe State Management**: FileLock ensures safe concurrent state updates

✅ **Partial Failure Handling**: Continues independent phases even when others fail

### Current Status

✅ **Parallel Execution**: **Fully implemented and operational!**

When you run with `--parallel`:
- The system analyzes phase dependencies
- Creates an optimal execution plan
- **Executes independent phases in parallel**
- Reports results and handles partial failures

Each phase executor:
- Runs through complete FSM lifecycle for all tasks in the phase
- Uses thread-safe git operations via GitCoordinator
- Updates state safely with FileLock
- Reports success/failure to the orchestrator

### How Parallel Execution Works

The parallel execution system consists of three key components:

#### 1. PhaseExecutor

Runs all tasks for a single phase through the complete FSM lifecycle:
- **plan_impl**: Creates implementation plan with file allowlist
- **implement**: Runs Codex worker to make changes
- **verify**: Runs tests, linting, type checking
- **review**: Reviews changes against acceptance criteria
- **commit**: Commits and pushes changes

Each PhaseExecutor:
- Uses thread-safe git operations via GitCoordinator
- Updates state files safely with FileLock
- Handles all error cases and state transitions
- Reports success/failure to orchestrator

#### 2. GitCoordinator

Serializes git operations across all parallel phases:
- Singleton pattern ensures one global lock
- Uses threading.RLock for reentrant locking
- Prevents concurrent git operations that could corrupt repository
- Operations like checkout, commit, push are automatically serialized

#### 3. ParallelExecutor

Manages the thread pool and execution batches:
- Uses concurrent.futures.ThreadPoolExecutor
- Executes batches of independent phases in parallel
- Waits for batch completion before starting next batch
- Aggregates results and reports failures

### Limitations & Future Work

### Current Limitations

1. **Static Dependencies**: Dependencies must be specified upfront in phase plan

2. **No Dynamic Resource Management**: Does not automatically limit CPU/memory usage per phase

3. **No Phase Checkpointing**: If a phase fails, must restart phase from beginning

4. **Sequential Within Phase**: Tasks within a phase run sequentially (as required by FSM)

5. **Manual Dependency Management**: Phase dependencies must be declared explicitly

### Planned Enhancements

- **Dynamic Resource Allocation**: Adjust worker count based on system resources
- **Phase Checkpointing**: Resume from failed task within phase
- **Real-time Progress UI**: Web dashboard showing parallel execution status
- **Cost Optimization**: Balance parallelism with API cost considerations
- **Smarter Dependency Detection**: Auto-detect dependencies from file changes
- **Phase Priority**: Allow prioritizing critical phases for faster execution

## Troubleshooting

### "Circular dependency detected"

```bash
# Visualize dependency tree to find cycle
feature-prd-runner plan-parallel --tree

# Look for cycles in dependencies
# Fix by removing or reordering dependencies
```

### "No phases found in phase plan"

```bash
# Ensure phase plan exists
ls .prd_runner/phase_plan.yaml

# If missing, run planner first
feature-prd-runner run --prd-file feature.md
```

### Phases Not Running in Parallel

Currently, parallel mode analyzes dependencies but executes sequentially. Full parallel execution will be implemented in a future version. Use `plan-parallel` to verify dependency analysis is working correctly.

### High CPU/Memory Usage

```bash
# Reduce worker count
feature-prd-runner run --prd-file feature.md --parallel --max-workers 2

# Or disable parallel execution
feature-prd-runner run --prd-file feature.md
```

## Related Documentation

- [README.md](../README.md) - General usage
- [HUMAN_IN_THE_LOOP.md](HUMAN_IN_THE_LOOP.md) - Interactive control
- [DEBUGGING.md](DEBUGGING.md) - Error analysis
- [ROADMAP.md](../ROADMAP.md) - Future features

## Feedback

Have suggestions for improving parallel execution? Open an issue on GitHub!
