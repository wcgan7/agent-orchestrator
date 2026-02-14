# CLI Reference

Executable:

```bash
agent-orchestrator
```

Global option:
- `--project-dir <path>`: target project (defaults to current directory)

## Server

```bash
agent-orchestrator [--project-dir PATH] server [--host 127.0.0.1] [--port 8080] [--reload]
```

## Project Commands

Pin a project:

```bash
agent-orchestrator [--project-dir PATH] project pin /absolute/path/to/repo [--project-id ID] [--allow-non-git]
```

List pinned projects:

```bash
agent-orchestrator [--project-dir PATH] project list
```

Unpin:

```bash
agent-orchestrator [--project-dir PATH] project unpin <project_id>
```

## Task Commands

Create task:

```bash
agent-orchestrator [--project-dir PATH] task create "Title" [--description TEXT] [--priority P0|P1|P2|P3] [--task-type feature]
```

List tasks:

```bash
agent-orchestrator [--project-dir PATH] task list [--status backlog|ready|in_progress|in_review|done|blocked|cancelled]
```

Run task:

```bash
agent-orchestrator [--project-dir PATH] task run <task_id>
```

## Quick Action

```bash
agent-orchestrator [--project-dir PATH] quick-action "prompt text"
```

Quick actions are ephemeral by default and do not create board tasks unless promoted later.

## Orchestrator

Status:

```bash
agent-orchestrator [--project-dir PATH] orchestrator status
```

Control:

```bash
agent-orchestrator [--project-dir PATH] orchestrator control pause
agent-orchestrator [--project-dir PATH] orchestrator control resume
agent-orchestrator [--project-dir PATH] orchestrator control drain
agent-orchestrator [--project-dir PATH] orchestrator control stop
```
