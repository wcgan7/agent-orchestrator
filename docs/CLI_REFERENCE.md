# CLI Reference

Executable:

```bash
feature-prd-runner
```

Global option:
- `--project-dir <path>`: target project (defaults to current directory)

## Server

```bash
feature-prd-runner [--project-dir PATH] server [--host 127.0.0.1] [--port 8080] [--reload]
```

## Project Commands

Pin a project:

```bash
feature-prd-runner [--project-dir PATH] project pin /absolute/path/to/repo [--project-id ID] [--allow-non-git]
```

List pinned projects:

```bash
feature-prd-runner [--project-dir PATH] project list
```

Unpin:

```bash
feature-prd-runner [--project-dir PATH] project unpin <project_id>
```

## Task Commands

Create task:

```bash
feature-prd-runner [--project-dir PATH] task create "Title" [--description TEXT] [--priority P0|P1|P2|P3] [--task-type feature]
```

List tasks:

```bash
feature-prd-runner [--project-dir PATH] task list [--status backlog|ready|in_progress|in_review|done|blocked|cancelled]
```

Run task:

```bash
feature-prd-runner [--project-dir PATH] task run <task_id>
```

## Quick Action

```bash
feature-prd-runner [--project-dir PATH] quick-action "prompt text"
```

Quick actions are ephemeral by default and do not create board tasks unless promoted later.

## Orchestrator

Status:

```bash
feature-prd-runner [--project-dir PATH] orchestrator status
```

Control:

```bash
feature-prd-runner [--project-dir PATH] orchestrator control pause
feature-prd-runner [--project-dir PATH] orchestrator control resume
feature-prd-runner [--project-dir PATH] orchestrator control drain
feature-prd-runner [--project-dir PATH] orchestrator control stop
```
