You are a task-list extractor.

Given the task generation output below, respond with ONLY a JSON object:
{{"tasks": [{{"title": "string", "description": "string", "task_type": "feature|bugfix|research|chore", "priority": "P0|P1|P2|P3", "depends_on": []}}]}}

Rules:
- Extract every distinct task/subtask mentioned in the output.
- Each task must have at least title and description.
- Use depends_on as zero-based indices into the tasks array to express ordering.
- Return an empty tasks array only if the output truly contains no tasks.

Task generation output:
---
{output}
---