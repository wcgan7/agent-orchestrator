# AGENTS.md

### Purpose

This repository uses an autonomous **Feature PRD Runner** to plan, implement, test, review, and commit changes.

Agents operating in this repository **must follow the rules in this file**.
Violations may cause runs to be blocked or reverted.

---

## 1. Source of truth

* **PRD files** define *what* to build.
* **Implementation plans** define *how* to build within a phase.
* **The coordinator (runner)** is the final authority on task state, commits, and blocking.

Agents must not override or bypass coordinator decisions.

## Terminology

- **human_blocking_issues** (in `progress.json`):
  Used only when work cannot proceed without a human decision. Stops the runner.

- **issues[]** (in `review_*.json`):
  Code review findings with severity.
  `critical/high` loop back for fixes; `medium/low` may proceed.

---

## 2. Progress reporting (MANDATORY)

When instructed, agents must write `progress.json` snapshots with:

* `run_id`
* `task_id`
* `phase`
* `actions`
* `claims`
* `next_steps`
* `heartbeat`

### Human intervention blockers

Use **`human_blocking_issues`** in `progress.json` **ONLY** when:

* The agent cannot proceed without a human decision
* The spec is missing, contradictory, or ambiguous
* Required credentials, access, or approvals are unavailable

**Do NOT** use `human_blocking_issues` for:

* Code quality concerns
* Refactors
* Design improvements
* Review feedback

Human blockers **stop the runner**.

---

## 3. Planning rules

### Phase planning

* Phases must be scoped and independently testable
* Each phase should have clear acceptance criteria
* Planning output must not modify repository code

### Implementation plans

* Must list files to change (`files_to_change`, `new_files`)
* Must describe a coherent technical approach
* Must not assume unstated infrastructure or permissions

---

## 4. Implementation rules

* Implement the **entire phase**, not partial work
* Modify **only allowed files** specified in the implementation plan
* Update `README.md` when behavior changes
* Agents must not commit or push. Only the coordinator may commit/push after review passes.

### Repo modification policy by mode

- PLAN / PLAN_IMPL / REVIEW: do not modify repository files outside `.prd_runner/`
- IMPLEMENT: modify only files listed in the implementation plan allowlist (runner artifacts under `.prd_runner/`, like `progress.json` and `artifacts/events.ndjson`, are always allowed for progress reporting)

---

## 5. Testing rules

* Fix failing tests before proceeding
* If tests fail repeatedly, expect the task to be blocked
* Do not bypass tests without explicit instruction

---

## 6. Review rules (CRITICAL)

Review output must be written to `review_*.json`.

### Issues and severity

All code review concerns must be listed under:

```json
"issues": [
  {
    "severity": "critical|high|medium|low",
    "summary": "...",
    "rationale": "...",
    "files": ["..."],
    "suggested_fix": "..."
  }
]
```

Severity guidance:

* **critical**: correctness, security, data loss, unsafe state
* **high**: unacceptable design, missing failure handling, poor observability
* **medium**: refactors, test improvements, minor edge cases
* **low**: style, naming, documentation

**critical/high issues must be fixed before commit.**
**medium/low issues may proceed to commit.**

Do not down-rank issues to allow a commit.

### Acceptance criteria

* Each acceptance criterion must be evaluated explicitly
* If any criterion is not met, include at least one high or critical issue explaining why

### Evidence

* Provide concrete evidence tied to files or diffs
* Do not make speculative claims

---

## 7. What NOT to do

Agents must NOT:

* Modify `.prd_runner/` logic unless explicitly instructed
* Change task status or queue files manually
* Commit, push, or rebase branches
* Suppress issues to “let the runner pass”
* Use human blockers for fixable code issues

---

## 8. Failure handling philosophy

* Prefer looping back for fixable issues
* Escalate to human blockers only when truly necessary
* Be explicit and honest about risks and uncertainty
