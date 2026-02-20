Generate executable subtasks from the provided context.

Mission:
- Decompose the provided context into a complete set of independently implementable tasks.
- Preserve intended sequencing and dependency relationships.
- Avoid overlap, duplication, and vague task definitions.

Task generation rules:
- Use only the provided context as source of truth.
- Produce tasks that are concrete, scoped, and directly actionable.
- Prefer minimal, testable increments over broad mixed-scope tasks.
- Assign each task a unique `id` (stable within this output).
- Encode ordering constraints via `depends_on` task IDs.
- If no dependencies are required, use an empty array.
- When the context is security-focused, generate remediation tasks tied to
  specific findings and avoid unrelated backlog items.

Output requirements:
- Return JSON only; no conversational text or markdown fences.
- Respond with valid JSON matching this schema:
  `{"tasks": [{"id": "string", "title": "string", "description": "string", "task_type": "feature|bugfix|research|chore", "priority": "P0|P1|P2|P3", "depends_on": ["id-1"]}]}`
- Each task must include:
  - `id`
  - `title`
  - `description`
  - `task_type`
  - `priority`
  - `depends_on` (array of task IDs)
- Ensure coverage of the full plan scope with no redundant tasks.
