Feature: Rich CLI Task Manager

1. Overview

We want to build a standalone command-line task manager tool. The goal is to manage a local todo list with persistent storage, using modern Python libraries (typer and rich) to provide a user-friendly and visually appealing terminal interface.

2. Technical Stack

Language: Python 3.9+

CLI Framework: typer

UI Library: rich (for tables and coloring)

Testing: pytest

3. Requirements

3.1 Core Data Logic (REQ-CORE)

Persistence: Tasks must be stored in a file named tasks.json in the current directory.

Data Structure: Each task is a dictionary containing:

id (integer, auto-incrementing)

title (string)

status (string: "pending" or "done")

Operations:

add_task(title): Adds a new task and saves to disk.

get_tasks(show_all=False): Returns a list of tasks.

complete_task(id): Marks a specific task as "done".

3.2 CLI Interface (REQ-CLI)

Command: Add

Usage: python main.py add "Buy groceries"

Output: "Task 'Buy groceries' added with ID 1."

Command: List

Usage: python main.py list

Behavior: Displays a rich Table with columns: ID, Title, Status.

Styling: "done" tasks should appear green; "pending" tasks should appear yellow.

Command: Done

Usage: python main.py done 1

Output: "Task 1 marked as done."

4. Phased Implementation Plan

The feature must be implemented in two distinct phases to ensure separation of concerns.

Phase 1: Core Logic & Storage

Goal: Implement the data handling without the CLI interface.

Create task_manager.py class/module.

Implement JSON load/save logic.

Implement add/list/complete methods.

Verification: Write unit tests in tests/test_manager.py verifying data persistence and logic (e.g., adding a task increases count, completing a task changes status).

Phase 2: CLI Integration

Goal: Connect the core logic to a terminal interface.

Create main.py using typer.

Import task_manager logic.

Implement the rich table rendering in the list command.

Verification: Write integration tests in tests/test_cli.py invoking the main.py commands (using typer.testing.CliRunner) to ensure output contains expected strings.
