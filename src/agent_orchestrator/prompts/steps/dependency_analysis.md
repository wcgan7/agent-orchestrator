Analyze task dependencies for this codebase.

First, examine the project structure to understand what already exists:
- Look at the directory layout and key files
- Check existing modules, APIs, and shared code
- Identify what infrastructure is already in place

Then, given the pending tasks below, determine which tasks depend on others.
A task B depends on task A if:
- B requires code, APIs, schemas, or artifacts that task A will CREATE (not already existing)
- B imports or builds on modules that task A will introduce
- B cannot produce correct results without task A's changes being present

Do NOT create a dependency if:
- Both tasks touch the same area but don't actually need each other's output
- The dependency is based on vague thematic similarity
- The required code/API already exists in the codebase

If tasks can safely run in parallel, leave them independent.