# Verification Loop Efficiency Review

> **Status (2026-02-11):** Historical implementation review for a specific optimization pass.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document identifies inefficiencies in the verification-to-implementation retry loop and proposes improvements.

---

## Implementation Status

| Item | Status | Notes |
|------|--------|-------|
| Add `stage` and `all_failing_paths` to `VerificationResult` | ✅ DONE | `models.py` updated |
| Add granular `PromptMode` values (`FIX_FORMAT`, `FIX_LINT`, `FIX_TYPECHECK`) | ✅ DONE | `models.py` updated |
| Store `error_type`, `stage`, `all_failing_paths` in `last_verification` | ✅ DONE | `fsm.py` updated |
| Add `_select_fix_prompt_mode()` to FSM | ✅ DONE | `fsm.py` updated |
| Pass `stage` and `all_failing_paths` in `VerificationResult` | ✅ DONE | `run_verify.py` updated |
| Create stage-specific banners in `_build_phase_prompt()` | ✅ DONE | `prompts.py` updated |
| Create `_build_minimal_fix_prompt()` for lightweight fix prompts | ✅ DONE | `prompts.py` updated |
| Include failing file list in prompts | ✅ DONE | Banners now show files |
| Truncate long file lists (>20 files) | ✅ DONE | Shows "and N more files" |
| Comprehensive test coverage | ✅ DONE | 22 new tests in `test_verification_efficiency.py` |
| All existing tests pass | ✅ DONE | 419 tests pass |
| Create minimal allowlist expansion prompt | ✅ DONE | `prompts.py` updated |
| Use minimal prompt for expansion in run_worker.py | ✅ DONE | No full PRD for expansion |
| **CRITICAL FIX**: Use minimal fix prompt for IMPLEMENT step | ✅ DONE | `run_worker.py` updated |
| Create minimal review fix prompt | ✅ DONE | `prompts.py` updated |
| Use minimal prompt for ADDRESS_REVIEW mode | ✅ DONE | `run_worker.py` updated |

---

## Executive Summary

The current implementation has several inefficiencies when verification fails (format/lint/typecheck/tests). The main issues are:

1. **Generic prompts waste context** - The same full implementation prompt is used regardless of failure type
2. **Failing file paths not communicated** - Extracted paths are stored but not passed to the agent
3. **No differentiation by failure type** - Format, lint, typecheck, and test failures all get the same treatment
4. **Unnecessary "continue implementing" instructions** - Simple lint fixes don't need implementation context

---

## Detailed Findings

### Issue 1: Failing Paths Not Passed to Agent

**Location:** `actions/run_verify.py:444-481`, `models.py:367`

**Problem:** The system extracts `failing_repo_paths` using language-aware parsers for each verification stage:
- Format: `format_parser(excerpt_text, project_dir)` (line 447)
- Lint: `lint_parser(excerpt_text, project_dir)` (line 451)
- Typecheck: `typecheck_parser(excerpt_text, project_dir)` (line 455)
- Tests: Combined from test parser, traceback parser, and suspect files (lines 458-470)

However, these paths are **only used** to determine `needs_allowlist_expansion` (lines 477-481). The `VerificationResult.failing_paths` field is set to `expansion_paths`, which only includes paths **outside** the allowlist (line 500).

**Impact:** When ruff fails on files that are already in the allowlist, the agent receives:
- The log tail (which may be truncated)
- The command that failed
- But NOT a clean list of files to fix

**Recommendation:**
1. Add `all_failing_paths: list[str]` field to `VerificationResult`
2. Store `failing_repo_paths` in `last_verification` dict passed to prompt builder
3. Include the file list in the agent prompt for targeted fixes

---

### Issue 2: Generic Prompt for All Verification Failures

**Location:** `prompts.py:146-164`

**Problem:** The banner is identical for all verification failure types:

```python
if prompt_mode in {"fix_tests", "fix_verify"}:
    header = "TESTS ARE FAILING -- FIX THIS FIRST" if prompt_mode == "fix_tests" \
             else "VERIFY IS FAILING -- FIX THIS FIRST"
```

All verification failures (format, lint, typecheck, tests) get `prompt_mode = FIX_VERIFY` and the generic "VERIFY IS FAILING" header.

**Impact:**
- Format failures need: "Run the formatter" or "Fix formatting in these files"
- Lint failures need: "Fix lint errors in these specific files"
- Typecheck failures need: "Fix type errors in these files"
- Test failures need: Understanding of the test logic

But all get the same generic treatment.

**Recommendation:**
1. Add granular `PromptMode` values: `FIX_FORMAT`, `FIX_LINT`, `FIX_TYPECHECK`, `FIX_TESTS`
2. Store `error_type` (e.g., "lint_failed") in `last_verification` for prompt differentiation
3. Create stage-specific banners with targeted instructions

---

### Issue 3: Wasteful Context in Fix Prompts

**Location:** `prompts.py:207-243`

**Problem:** When `prompt_mode == "fix_verify"`, the full implementation prompt is still constructed:

```python
return f"""Implement the COMPLETE phase described below.

Follow all repository rules in AGENTS.md.

{banner_block}  # verification failure info

PRD: {prd_path}
Phase: {phase_name}
Description: {phase.get("description", "")}
Acceptance criteria:
{acceptance_block}
...
Implementation plan file: {plan_path_display}
{approach_block}
...
```

For a simple ruff fix, none of this context is needed.

**Impact:**
- Wasted tokens on irrelevant context
- Agent may be confused by implementation instructions when task is just "fix lint"
- Slower execution due to larger prompts

**Recommendation:**
1. Create a minimal `_build_fix_verification_prompt()` function for pure fix scenarios
2. Only include: failing command, error output, list of failing files, and the specific fix instructions
3. Skip PRD, acceptance criteria, technical approach, etc. for lint/format fixes

---

### Issue 4: "Continue Implementing" Instruction

**Location:** `prompts.py:161-163`

**Problem:** The verification failure banner includes:

```
Priority for this run:
1) Fix the failing verification (minimal change).
2) Only then continue implementing remaining acceptance criteria.
```

For pure lint/format/typecheck failures that don't affect functionality, instruction #2 is misleading. The agent might think it needs to do additional implementation work.

**Impact:**
- Agent may make unnecessary changes beyond fixing the verification issue
- Increased risk of introducing new bugs or scope creep
- Longer execution time

**Recommendation:**
1. For format/lint failures: "Fix only the indicated issues. Do not modify code logic."
2. For typecheck failures: "Fix only the type annotations. Do not modify code logic."
3. For test failures: Keep the current instruction (may need implementation changes)

---

### Issue 5: Missing Error Type in last_verification

**Location:** `fsm.py:207-213`

**Problem:** The `last_verification` dict stored in task state includes:

```python
task.last_verification = {
    "command": event.command,
    "exit_code": int(event.exit_code),
    "log_path": event.log_path,
    "log_tail": event.log_tail,
    "captured_at": event.captured_at,
}
```

But NOT `error_type` (e.g., "lint_failed", "format_failed", "typecheck_failed", "tests_failed").

**Impact:** The prompt builder can't differentiate between failure types and must use generic messaging.

**Recommendation:** Add `error_type` to `last_verification`:

```python
task.last_verification = {
    "command": event.command,
    "exit_code": int(event.exit_code),
    "log_path": event.log_path,
    "log_tail": event.log_tail,
    "captured_at": event.captured_at,
    "error_type": event.error_type,  # ADD THIS
    "failing_paths": failing_repo_paths,  # ADD THIS (all paths, not just expansion)
}
```

---

### Issue 6: No Stage Name in VerificationResult

**Location:** `models.py:357-374`

**Problem:** `VerificationResult` doesn't include which stage failed. The `error_type` field contains values like `"lint_failed"` but this is constructed as a side effect (line 432 in run_verify.py) and not prominently used.

**Recommendation:** Add explicit `stage: str` field to `VerificationResult`:

```python
@dataclass
class VerificationResult(Event):
    run_id: str
    passed: bool
    command: Optional[str]
    exit_code: int
    log_path: Optional[str]
    log_tail: str
    stage: Optional[str] = None  # ADD: "format", "lint", "typecheck", "tests"
    ...
```

---

### Issue 7: Inefficient Retry Strategy for Auto-Fixable Issues

**Problem:** Some verification failures are trivially auto-fixable:
- `ruff format --check .` failure → just run `ruff format .`
- Some lint issues have `--fix` options
- Import sorting issues can be auto-fixed

Currently, the system asks the agent to fix these manually.

**Recommendation:**
1. Detect auto-fixable failures (format check, certain lint rules)
2. Attempt auto-fix before involving the agent
3. Only fall back to agent if auto-fix fails or causes new issues

---

## Proposed Architecture Changes

### Short-term (Quick Wins)

1. **Add `error_type` and `failing_paths` to `last_verification`** (fsm.py)
2. **Create stage-specific banners** in `_build_phase_prompt()` (prompts.py)
3. **Remove "continue implementing" for lint/format failures** (prompts.py)

### Medium-term

4. **Create `_build_fix_verification_prompt()`** - minimal prompt for verification fixes
5. **Add granular PromptMode values** - `FIX_FORMAT`, `FIX_LINT`, `FIX_TYPECHECK`
6. **Pass failing file list to agent** - explicit "Fix these files:" section

### Long-term

7. **Implement auto-fix stage** - run formatters/fixers before involving agent
8. **Differentiate verification retries** - format/lint get fewer retries than tests
9. **Smart retry strategy** - if same files keep failing, escalate differently

---

## Example: Ideal Prompt for Ruff Lint Failure

Current (inefficient):
```
Implement the COMPLETE phase described below.

Follow all repository rules in AGENTS.md.

VERIFY IS FAILING -- FIX THIS FIRST
Command: ruff check .
Exit code: 1
Log: /path/to/lint.log
Recent output:
src/foo.py:10:1: F401 `os` imported but unused
src/bar.py:25:5: E501 Line too long (120 > 100)
...

Priority for this run:
1) Fix the failing verification (minimal change).
2) Only then continue implementing remaining acceptance criteria.

PRD: /path/to/prd.md
Phase: phase-1
Description: Add user authentication
Acceptance criteria:
- Users can log in
- Users can log out
...
[500+ more tokens of context]
```

Proposed (efficient):
```
LINT CHECK FAILING -- FIX THESE ISSUES

Command: ruff check .
Exit code: 1

Files with errors:
- src/foo.py
- src/bar.py

Error details:
src/foo.py:10:1: F401 `os` imported but unused
src/bar.py:25:5: E501 Line too long (120 > 100)

Instructions:
1. Fix ONLY the lint errors listed above
2. Do NOT modify code logic or add features
3. Make minimal changes to satisfy the linter

Allowed files to edit:
- src/foo.py
- src/bar.py
```

This reduces prompt size by ~80% and gives the agent clear, targeted instructions.

---

## Implementation Priority

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Add error_type/failing_paths to last_verification | Low | High |
| P0 | Create stage-specific banners | Low | High |
| P1 | Create minimal fix prompt function | Medium | High |
| P1 | Add granular PromptMode values | Medium | Medium |
| P2 | Implement auto-fix stage | High | High |
| P2 | Smart retry differentiation | Medium | Medium |

---

## Files Modified

1. `src/feature_prd_runner/fsm.py` - Added error_type, stage, and all_failing_paths to last_verification; added `_select_fix_prompt_mode()` helper
2. `src/feature_prd_runner/prompts.py` - Created stage-specific banners with failing file lists; added `_build_minimal_fix_prompt()` and `_build_minimal_allowlist_expansion_prompt()` functions
3. `src/feature_prd_runner/models.py` - Added `PromptMode.FIX_FORMAT`, `FIX_LINT`, `FIX_TYPECHECK`; added `stage` and `all_failing_paths` to `VerificationResult`
4. `src/feature_prd_runner/actions/run_verify.py` - Pass stage and all_failing_paths in VerificationResult
5. `src/feature_prd_runner/actions/run_worker.py` - Use minimal allowlist expansion prompt when appropriate; import new prompt functions

---

## What Was Implemented

### 1. Granular PromptMode Values

Added to `models.py`:
- `PromptMode.FIX_FORMAT` - for formatting issues
- `PromptMode.FIX_LINT` - for lint errors
- `PromptMode.FIX_TYPECHECK` - for type check errors
- Existing `FIX_TESTS` and `FIX_VERIFY` kept for compatibility

### 2. Enhanced VerificationResult Model

Added fields to `VerificationResult`:
- `stage: Optional[str]` - Which stage failed ("format", "lint", "typecheck", "tests")
- `all_failing_paths: list[str]` - All files with issues (not just expansion paths)

### 3. FSM Improvements

In `fsm.py`:
- `last_verification` now stores: `error_type`, `stage`, `all_failing_paths`
- New `_select_fix_prompt_mode(error_type)` function selects the appropriate granular PromptMode
- FSM now sets `FIX_FORMAT`, `FIX_LINT`, `FIX_TYPECHECK`, or `FIX_TESTS` based on error type

### 4. Stage-Specific Prompt Banners

In `_build_phase_prompt()`:
- **Format failures**: "FORMAT CHECK FAILING -- FIX THIS FIRST" + "Fix ONLY the formatting issues"
- **Lint failures**: "LINT CHECK FAILING -- FIX THIS FIRST" + "Fix ONLY the lint errors"
- **Typecheck failures**: "TYPE CHECK FAILING -- FIX THIS FIRST" + "Fix ONLY the type errors"
- **Test failures**: "TESTS ARE FAILING -- FIX THIS FIRST" + "may require code logic changes"

Each banner now includes:
- List of failing files (truncated to 20 with "and N more files" suffix)
- Stage-specific instructions (no code logic changes for format/lint/typecheck)

### 5. Minimal Fix Prompt Function

New `_build_minimal_fix_prompt()` function provides an ultra-focused prompt for simple fixes:
- No PRD reference
- No acceptance criteria
- No implementation plan
- Just: command, error output, failing files, fix instructions

This is available for future use when we want to further reduce token usage for trivial fixes.

### 6. Test Coverage

Added `tests/test_verification_efficiency.py` with 22 comprehensive tests covering:
- PromptMode enum values
- VerificationResult fields and serialization
- FSM `_select_fix_prompt_mode()` function
- FSM storing detailed verification info
- FSM selecting correct granular PromptMode
- Stage-specific prompt banners
- Minimal fix prompts
- File list truncation

---

### 7. CRITICAL FIX: IMPLEMENT Step Now Uses Minimal Fix Prompts

**Problem Discovered:** The `_build_minimal_fix_prompt()` function was created but **never actually used** for the IMPLEMENT step. The code in `run_worker.py` always called `_build_phase_prompt()` regardless of `prompt_mode`, which included the full PRD.

**Evidence from code review:**
```python
# Lines 544-562 in run_worker.py (BEFORE fix)
if spec.type == "codex":
    prompt = _build_phase_prompt(  # ALWAYS called, even for fix_format/fix_lint
        ...
        prompt_mode=task.get("prompt_mode"),  # Passed but not used to select prompt
    )
```

**Solution:** Updated `run_worker.py` to check `prompt_mode` and route to the appropriate prompt builder:
- `fix_format`, `fix_lint`, `fix_typecheck` → `_build_minimal_fix_prompt()` (no PRD)
- `address_review` → `_build_minimal_review_fix_prompt()` (no PRD)
- Everything else → `_build_phase_prompt()` (full context)

**Impact:** Simple verification fixes (format, lint, typecheck) now use ~90% smaller prompts.

---

### 8. Minimal Review Fix Prompt (Additional Fix)

**Problem:** When addressing review blockers (`prompt_mode="address_review"`), the system still used the full phase prompt with PRD, even though the agent only needs:
- The specific review blockers
- The files implicated
- The allowed files to edit

**Solution:** Created `_build_minimal_review_fix_prompt()` function that:
- Lists only the review blockers
- Lists the implicated files
- Shows the allowed files
- Does NOT include full PRD, acceptance criteria, or technical approach

**Impact:** Review blocker fixes now use focused prompts instead of full implementation context.

---

### 9. Minimal Allowlist Expansion Prompt (Additional Fix)

**Problem Discovered:** When verification fails with files outside the allowlist, the FSM routes to `PLAN_IMPL` step (not `IMPLEMENT`). The original `_build_impl_plan_prompt()` always includes the full PRD content, which is wasteful for a simple "add these files to the allowlist" operation.

**Example from real run:**
- Typecheck failed on 10 files outside the current allowlist
- System correctly identified `needs_allowlist_expansion=true`
- But the next prompt included the ENTIRE PRD (500+ lines) just to ask "add these 10 files"

**Solution:** Created `_build_minimal_allowlist_expansion_prompt()` function that:
- Lists only the files that need to be added
- Shows the current allowlist for context
- Includes the verification error type and log excerpt
- Does NOT include full PRD content
- Instructs agent to update the existing plan file

**Code Changes:**
- Added `_build_minimal_allowlist_expansion_prompt()` to `prompts.py`
- Updated `run_worker.py` to use minimal prompt when:
  - `prompt_mode == "expand_allowlist"` AND
  - `plan_expansion_request` is non-empty AND
  - Existing plan file exists (to read current allowlist)

**Result:** Allowlist expansion prompts are now ~90% smaller (from 600+ lines to ~50 lines).

---

---

## Comprehensive Edge Case Review (Second Pass)

A thorough review was conducted to ensure no similar inefficiencies exist. Findings:

### Issues Found and Fixed

| # | Issue | Scope | Status |
|----|-------|-------|--------|
| 1 | `_build_minimal_fix_prompt()` created but never used | IMPLEMENT + fix_* modes | ✅ FIXED |
| 2 | Full PRD in ADDRESS_REVIEW mode | IMPLEMENT + address_review | ✅ FIXED |
| 3 | Full PRD in allowlist expansion (Codex) | PLAN_IMPL + expand_allowlist | ✅ FIXED |

### Known Lower-Priority Issues (Deferred)

| # | Issue | Scope | Impact | Notes |
|----|-------|-------|--------|-------|
| 4 | PRD always read at function entry | All steps | I/O cost | Optimization deferred |
| 5 | prd_markers extracted when not needed | Non-PLAN steps | CPU cost | Minor |
| 6 | Local workers don't receive optimization hints | All local worker steps | Variable | Only affects non-Codex workers |
| 7 | Full PRD in PLAN_IMPL expansion for local workers | PLAN_IMPL expansion (local) | 5000+ tokens | Only affects local workers |
| 8 | Review prompts always include full PRD | REVIEW step | 5000+ tokens | Debatable - review may need PRD context |

### Rationale for Deferred Items

- **Items 4-5**: The I/O and CPU costs are minor compared to token savings. These would require more invasive changes to the function structure.
- **Items 6-7**: Local workers (Ollama, etc.) are less commonly used than Codex workers. The fixes prioritized the main code path.
- **Item 8**: Review step arguably needs PRD context to verify acceptance criteria are met. This would require more analysis.

---

## Future Improvements (Not Yet Implemented)

1. **Auto-fix stage** - Run formatters/fixers automatically before involving agent
2. **Differentiate retry limits** - Format/lint could have fewer retries than tests
3. **Smart escalation** - If same files keep failing, escalate differently
4. **Fix-then-expand strategy** - When verification fails with some files in allowlist and some outside, fix the in-allowlist files first, then expand if needed
5. **Lazy PRD loading** - Only read PRD when the selected prompt builder actually needs it
6. **Local worker optimization** - Add minimal prompt variants for local/Ollama workers
