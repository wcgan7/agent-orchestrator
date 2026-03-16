Review the pull request changes against the PR description and branch context.

Your job is to analyze the diff for correctness, bugs, missing edge cases, test coverage gaps, security issues, and adherence to the PR description. Then write a concrete remediation plan so the subsequent implement step can fix every issue you find.

## Review checklist

1. **Requirement adherence** — Does the diff fully satisfy the PR description? Are any requirements missing or only partially implemented?
2. **Logic errors & bugs** — Are there off-by-one errors, incorrect conditions, wrong variable references, race conditions, or broken control flow?
3. **Edge cases** — Are boundary conditions, empty inputs, null/undefined values, and error paths handled?
4. **Test coverage** — Are there tests for the new/changed behavior? Do the tests cover edge cases and failure modes?
5. **Security** — Are there injection vectors, improper input validation, leaked secrets, or unsafe operations?
6. **Code quality** — Are there naming issues, dead code, duplicated logic, or overly complex constructions that hurt maintainability?

## Truncated diffs

If the diff was truncated (indicated by a `[DIFF TRUNCATED]` notice), consult the `--stat` summary above to identify files whose changes are not included in the diff. Use `git diff <base>...<head> -- <file>` to read each missing file's changes individually before completing the review. Prioritize source files over generated or vendored files.

## Output

Write your findings and remediation plan into the workdoc's **## Plan** section using this structure:

```
## Plan

### Findings

1. **[severity: high/medium/low]** Brief title
   - File: `path/to/file.ext` (lines N-M)
   - Issue: Description of the problem
   - Fix: Specific remediation action

2. ...

### Fix tasks

For each finding above, describe the concrete change needed:
1. In `path/to/file.ext`: [what to change and why]
2. In `path/to/test_file.ext`: [what test to add/fix]
...
```

If no issues are found, write "No issues found — pull request looks correct." in the Plan section.

Rules:
- Be specific: reference exact file paths, line ranges, variable names.
- Prioritize findings by severity (high first).
- Each finding must have a concrete fix action, not just a description of the problem.
- Do not modify any code yourself — only write the plan.
- Return only the review body (no preamble, tool logs, or follow-up questions).
