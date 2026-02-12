# Feature PRD Runner - Roadmap & Upcoming Features

> **Status (2026-02-11):** Legacy roadmap retained for historical context.
> **Current direction:** [`ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](../../ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`README.md`](README.md)

This document tracks planned enhancements to make the Feature PRD Runner more versatile, useful, and robust.

## Status Legend
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete
- 🔵 Under Consideration

---

## Quick Links
- [High Priority Features](#high-priority-features)
- [Foundation Improvements](#phase-1-foundation-weeks-1-2)
- [Observability & UX](#phase-2-observability--ux-weeks-3-4)
- [Flexibility & Extensibility](#phase-3-flexibility--extensibility-weeks-5-7)
- [Scale & Performance](#phase-4-scale--performance-weeks-8-10)
- [Advanced Features](#phase-5-advanced-features-weeks-11)

---

## High Priority Features

### 🟡 1. Active Human-in-the-Loop Controls

**Status**: 🟡 In Progress
**Priority**: P0 (Critical)
**Impact**: Versatility ⭐⭐⭐⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐

**Implementation Status**:
- ✅ Approval gates system with all gate types
- ✅ Message bus for bidirectional communication
- ✅ CLI commands: steer, approve, reject, view-changes, correct, require, breakpoint
- ✅ Web UI approval gates with visual cards
- ✅ Configuration system with timeouts and context display
- ✅ Desktop notifications for approval gates (optional, via plyer)
- ✅ Live collaboration chat in web UI
- ✅ Real-time messaging between human and worker
- ✅ Inline corrections via CLI (correct, require commands)
- ✅ Breakpoint system (set, list, remove, toggle, clear)
- ✅ Conditional breakpoints with expression evaluation
- ✅ File-by-file review UI with diff viewer
- ✅ Per-file approve/reject with comments
- ✅ Review progress tracking
- ❌ Multi-user collaboration (role-based access, multiple approvers)

**Problem**: Current system is too autonomous and hands-off. Hard to steer direction, provide mid-execution guidance, or review/approve changes before they proceed. Once a run starts, you're mostly a passenger until it blocks or completes.

**Solution**:
Comprehensive human-in-the-loop system with approval gates, real-time steering, live feedback, and collaborative controls.

**Key Capabilities**:

**1. Approval Gates**
```yaml
# .prd_runner/config.yaml
approval_gates:
  enabled: true

  gates:
    before_implement:
      enabled: true
      message: "Review implementation plan before proceeding?"
      show_diff: false
      show_plan: true
      timeout: 300  # Auto-approve after 5min

    after_implement:
      enabled: true
      message: "Review code changes before verification?"
      show_diff: true
      allow_edit: true

    before_commit:
      enabled: true
      message: "Review and approve commit?"
      show_diff: true
      show_tests: true
      show_review: true
      required: true  # Cannot skip

    after_review_issues:
      enabled: true
      message: "Review found issues. Continue fixing?"
      show_issues: true
      allow_override: true
```

**2. Interactive Steering During Execution**
```bash
# Real-time control while run is active
feature-prd-runner steer

# Interactive prompt:
# > Run: run-abc123 | Phase: phase-2 | Step: IMPLEMENT
# > Worker is currently: Refactoring auth module...
# >
# > Commands:
# >   [p] Pause execution
# >   [s] Send guidance/steering message
# >   [i] Inject new requirement
# >   [v] View current changes (git diff)
# >   [r] Request explanation of what it's doing
# >   [u] Undo last change
# >   [b] Set breakpoint (pause at next step)
# >   [c] Continue
# >   [q] Quit steering mode
# >
# > Enter command: s
# > Guidance message: Focus on security - ensure input validation
# > Sending to worker... Done!
# > Worker acknowledged: "Adding input validation checks"
```

**3. Live Collaboration Chat**
```python
# src/feature_prd_runner/collaboration.py
class LiveCollaboration:
    """Real-time human-AI collaboration."""

    def send_message(self, message: str, wait_for_response: bool = True):
        """Send message to worker during execution."""
        pass

    def request_clarification(self, question: str) -> str:
        """Ask worker to clarify what it's doing."""
        pass

    def provide_feedback(self, feedback: str, changes: list[str]):
        """Give feedback on specific changes."""
        pass

    def suggest_alternative(self, current_approach: str, alternative: str):
        """Suggest different approach mid-execution."""
        pass

    def request_explanation(self, aspect: str) -> str:
        """Ask worker to explain its reasoning."""
        pass
```

**4. Step-by-Step Approval Mode**
```bash
# Run with step-by-step approval
feature-prd-runner run --prd-file feature.md --interactive

# Or enable for existing run
feature-prd-runner mode --interactive

# Prompts at each step:
# ┌─────────────────────────────────────────────┐
# │ Step: IMPLEMENT (phase-1)                   │
# │ About to: Implement user authentication     │
# ├─────────────────────────────────────────────┤
# │ Planned Changes:                            │
# │   • src/auth/login.py (new)                 │
# │   • src/models/user.py (modify)             │
# │   • tests/test_auth.py (new)                │
# │                                              │
# │ Approach:                                   │
# │   Using JWT tokens with refresh mechanism   │
# │   Password hashing with bcrypt              │
# │   Session management via Redis              │
# └─────────────────────────────────────────────┘
#
# Options:
#   [a] Approve and continue
#   [e] Edit approach (provide guidance)
#   [s] Skip this step
#   [p] Pause and inspect
#   [q] Quit
#
# Choice: e
# Your guidance: Use Argon2 instead of bcrypt, and add rate limiting
# Acknowledged. Proceeding with your guidance...
```

**5. Change Review & Approval UI**
```bash
# Review changes before they're committed
feature-prd-runner review-changes

# Opens interactive UI:
# ╔═══════════════════════════════════════════════╗
# ║ Change Review: phase-1                        ║
# ╠═══════════════════════════════════════════════╣
# ║ Files Changed: 5                              ║
# ║ Lines: +234 / -67                             ║
# ╚═══════════════════════════════════════════════╝
#
# ┌─ src/auth/login.py (new file) ─────────────┐
# │ +  def login(username: str, password: str): │
# │ +      user = User.find_by_username(...)    │
# │ +      if not verify_password(...):         │
# │ +          raise InvalidCredentials()       │
# │ +      return generate_token(user)          │
# └─────────────────────────────────────────────┘
#
# Actions:
#   [✓] Approve this file
#   [✗] Reject this file
#   [e] Edit this file
#   [c] Add comment
#   [n] Next file
#   [a] Approve all
#   [r] Reject all
#
# File 1/5 >
```

**6. Inline Editing & Corrections**
```bash
# Make corrections during execution
feature-prd-runner edit --task phase-1

# Opens editor with current changes
# Make your edits
# On save, changes are applied and worker continues

# Or provide specific corrections
feature-prd-runner correct phase-1 \
  --file src/auth/login.py \
  --issue "Missing rate limiting" \
  --fix "Add @rate_limit(max=5, window=60) decorator"
```

**7. Breakpoints & Inspection**
```bash
# Set breakpoint - pause before specific step
feature-prd-runner breakpoint set --before verify --task phase-1

# Set conditional breakpoint
feature-prd-runner breakpoint set \
  --condition "files_changed > 10" \
  --action pause

# When hit:
# 🔴 BREAKPOINT HIT: phase-1 before VERIFY
#
# Changes so far:
#   • 12 files modified
#   • +456 / -123 lines
#
# Options:
#   [c] Continue
#   [i] Inspect changes (open diff viewer)
#   [e] Edit before continuing
#   [s] Step through verification
#   [a] Abort and rollback
#   [r] Remove breakpoint and continue
```

**8. Guided Corrections**
```python
# src/feature_prd_runner/guidance.py
class GuidanceSystem:
    """Provide structured guidance during execution."""

    def add_requirement(self, requirement: str):
        """Add requirement mid-execution."""
        # "Also ensure all endpoints have authentication"
        pass

    def add_constraint(self, constraint: str):
        """Add constraint on implementation."""
        # "Must avoid network access"
        pass

    def provide_example(self, example: str, context: str):
        """Show example of what you want."""
        # "Like this: <code example>"
        pass

    def course_correct(self, issue: str, correction: str):
        """Correct course mid-implementation."""
        # "The auth approach is too complex, simplify it"
        pass
```

**9. Multi-User Collaboration**
```yaml
# .prd_runner/config.yaml
collaboration:
  enabled: true

  roles:
    owner:
      - approve_changes
      - inject_requirements
      - abort_run

    reviewer:
      - view_changes
      - add_comments
      - suggest_corrections

    observer:
      - view_progress
      - view_changes

  users:
    - email: alice@company.com
      role: owner
    - email: bob@company.com
      role: reviewer
    - email: carol@company.com
      role: observer
```

**10. Annotations & Comments**
```bash
# Add comment to specific change
feature-prd-runner comment add phase-1 \
  --file src/auth/login.py \
  --line 45 \
  --text "This validation logic looks incomplete"

# View comments
feature-prd-runner comments list phase-1

# Reply to comment
feature-prd-runner comment reply comment-123 \
  --text "Good catch, adding missing checks"

# Resolve comment
feature-prd-runner comment resolve comment-123
```

**CLI Commands**:
```bash
# Enter steering mode
feature-prd-runner steer [--task <task-id>]

# Send message to worker
feature-prd-runner message "Add error handling for edge case"

# Request explanation
feature-prd-runner explain "Why did you choose this approach?"

# Provide feedback on changes
feature-prd-runner feedback --file src/auth.py \
  "Good structure, but add more comments"

# Approve changes
feature-prd-runner approve [--all | --files file1 file2]

# Reject changes with reason
feature-prd-runner reject --reason "Doesn't match requirements"

# Set breakpoint
feature-prd-runner breakpoint set --before <step> --task <task-id>

# Enable interactive mode
feature-prd-runner mode --interactive

# Edit current implementation
feature-prd-runner edit [--file <path>]

# Add inline requirement
feature-prd-runner require "Must support OAuth in addition to JWT"

# Course correct
feature-prd-runner steer "Simplify the caching layer, it's over-engineered"
```

**Web UI Integration**:

**Real-Time Collaboration Panel**:
- Live code diff viewer
- Chat interface with AI worker
- Change approval buttons
- Inline commenting on code
- Activity feed (who approved what, when)
- Breakpoint manager
- Guidance injection form

**Approval Workflow UI**:
- Visual approval gates at each step
- Side-by-side diff viewer
- Approve/reject/edit buttons
- Comment threads
- Change history timeline
- Rollback options

**Live Steering Dashboard**:
- Current worker status ("Implementing auth...")
- Real-time file changes streaming
- Chat box for immediate feedback
- Quick action buttons (pause, undo, inject)
- Progress indicator with human touchpoints

**Configuration Options**:
```yaml
# .prd_runner/config.yaml
human_in_the_loop:
  # Approval gates
  approval_gates:
    enabled: true
    require_before_implement: true
    require_before_commit: true
    require_after_review_issues: true
    auto_approve_timeout: 300  # seconds

  # Interactive features
  interactive:
    allow_steering: true
    allow_corrections: true
    allow_inline_edits: true
    allow_requirement_injection: true

  # Collaboration
  collaboration:
    enabled: false
    require_approval_from: 1  # min approvers
    allow_comments: true
    notify_on_changes: true

  # Breakpoints
  breakpoints:
    enabled: true
    pause_on_error: true
    pause_on_large_changes: true
    large_change_threshold: 15  # files

  # Notifications
  notifications:
    desktop: true
    email: false
    slack: false
    sound: true
    request_approval_for:
      - implement
      - commit
      - review_issues
```

**Example Workflows**:

**1. Supervised Implementation**:
```bash
# Start with approvals at each step
feature-prd-runner run --prd-file feature.md \
  --approve-before-implement \
  --approve-before-commit

# At each gate, review and approve/reject
```

**2. Live Steering**:
```bash
# Start run
feature-prd-runner run --prd-file feature.md

# In another terminal, enter steering mode
feature-prd-runner steer

# Send guidance as needed
> "Focus on error handling"
> "Add logging for debugging"
> "Simplify this approach"
```

**3. Collaborative Review**:
```bash
# Dev 1 starts run
feature-prd-runner run --prd-file feature.md --collaborative

# Dev 2 joins to review
feature-prd-runner join run-abc123

# Both can comment, suggest changes, approve
```

**4. Careful Experimentation**:
```bash
# Set breakpoint before risky step
feature-prd-runner breakpoint set --before verify

# Run
feature-prd-runner run --prd-file feature.md

# When breakpoint hits, inspect changes
feature-prd-runner inspect

# If good, continue; if bad, rollback
feature-prd-runner rollback  # or feature-prd-runner continue
```

**Benefits**:
- **Steering Control**: Guide implementation direction in real-time
- **Early Intervention**: Catch issues before they cascade
- **Learning Loop**: Understand AI decision-making through explanations
- **Quality Gates**: Ensure high-quality output through reviews
- **Reduced Rework**: Fix issues immediately rather than after completion
- **Team Collaboration**: Multiple people can oversee and guide runs
- **Confidence**: Feel in control rather than being a passive observer
- **Debugging**: Understand why AI made specific choices
- **Course Correction**: Change direction when needed without restarting
- **Incremental Approval**: Approve changes as they happen, not all at once

**Implementation Details**:

**1. Communication Protocol**:
```python
# src/feature_prd_runner/messaging.py
class MessageBus:
    """Bidirectional communication between human and worker."""

    def send_to_worker(self, message: Message):
        """Send message to running worker (via progress.json)."""
        pass

    def receive_from_worker(self) -> list[Message]:
        """Receive messages from worker."""
        pass

    def request_approval(self, approval_request: ApprovalRequest) -> bool:
        """Block and wait for human approval."""
        pass

# Messages go through progress.json:
{
  "run_id": "...",
  "heartbeat": "...",
  "messages_from_human": [
    {
      "id": "msg-123",
      "type": "guidance",
      "content": "Add rate limiting",
      "timestamp": "..."
    }
  ],
  "messages_to_human": [
    {
      "id": "msg-124",
      "type": "clarification_request",
      "content": "Should rate limiting be per-user or per-IP?",
      "timestamp": "..."
    }
  ],
  "approval_pending": {
    "type": "before_implement",
    "data": {...}
  }
}
```

**2. Approval Gate Implementation**:
```python
# src/feature_prd_runner/approval_gates.py
class ApprovalGate:
    """Pause execution and wait for human approval."""

    def request_approval(
        self,
        gate_type: str,
        context: dict,
        timeout: Optional[int] = None
    ) -> ApprovalResponse:
        """
        Block execution and wait for approval.
        Returns: APPROVED, REJECTED, MODIFIED, TIMEOUT
        """
        # Write approval request to state
        # Poll for response or timeout
        # Return result
        pass
```

**3. Web Socket for Real-Time Updates**:
```python
# src/feature_prd_runner/server/websocket.py
@app.websocket("/ws/steer/{run_id}")
async def steer_websocket(websocket: WebSocket, run_id: str):
    """
    Real-time bidirectional communication for steering.

    Client can:
    - Send guidance messages
    - Request explanations
    - Approve/reject changes
    - Set breakpoints

    Server sends:
    - Current status updates
    - Change notifications
    - Approval requests
    - Worker messages
    """
    await websocket.accept()

    while True:
        # Listen for human input
        data = await websocket.receive_json()

        # Process command
        # Send to worker

        # Send updates back
        await websocket.send_json({...})
```

**Acceptance Criteria**:
- [x] Approval gates system with configurable checkpoints
- [x] `steer` command for real-time interaction
- [x] Live collaboration chat with worker
- [x] Step-by-step approval mode (`--interactive`)
- [x] Change review UI (CLI and Web)
- [x] Inline corrections (`correct`/`require`)
- [x] Breakpoint system with conditions
- [x] Guidance injection API
- [ ] Multi-user collaboration support
- [ ] Comments and annotations system
- [x] WebSocket-based real-time updates
- [ ] Web UI panels for all features
- [x] Notification system for approval requests
- [x] Timeout handling for approvals
- [ ] Rollback on rejection
- [ ] Tests for all approval flows
- [x] Documentation with examples
- [ ] Video tutorials for interactive features

**Related Features**:
- Integrates with #2 (Web UI Dashboard)
- Enhances #3 (Flexible Step Progression)
- Works with #13 (Checkpoints for rollback)
- Complements #5 (Enhanced Errors for better feedback)

---

### 🔴 2. Worker Abstraction & Multi-Provider Support

**Status**: 🔴 Not Started
**Priority**: P0 (Critical)
**Impact**: Versatility ⭐⭐⭐⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Hard-coded dependency on Codex CLI limits flexibility and vendor lock-in.

**Solution**:
Create an abstract worker interface to support multiple AI providers.

**Implementation**:
```python
# src/feature_prd_runner/workers/base.py
class WorkerInterface(ABC):
    """Abstract base for AI workers."""

    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> WorkerResult:
        """Execute a prompt and return structured result."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Verify worker is properly configured."""
        pass

    @abstractmethod
    def estimate_cost(self, prompt: str) -> float:
        """Estimate cost for executing this prompt."""
        pass

# Concrete implementations
class CodexWorker(WorkerInterface):
    """Existing Codex CLI integration."""
    pass

class OpenAIWorker(WorkerInterface):
    """Direct OpenAI API integration."""
    pass

class AnthropicWorker(WorkerInterface):
    """Claude API integration."""
    pass

class AzureOpenAIWorker(WorkerInterface):
    """Azure OpenAI Service integration."""
    pass

class LocalLLMWorker(WorkerInterface):
    """Local model via Ollama/llama.cpp."""
    pass

class CustomWorker(WorkerInterface):
    """User-defined custom worker."""
    pass
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
workers:
  default: codex

  providers:
    codex:
      type: codex
      command: "codex exec -"

    openai:
      type: openai
      api_key_env: "OPENAI_API_KEY"
      model: "gpt-4-turbo"
      max_tokens: 8000
      temperature: 0.2

    claude:
      type: anthropic
      api_key_env: "ANTHROPIC_API_KEY"
      model: "claude-3-opus-20240229"

    local:
      type: local
      endpoint: "http://localhost:11434"
      model: "codellama:34b"

  routing:
    # Route different steps to different workers
    plan: codex
    implement: openai
    review: claude

  fallback:
    enabled: true
    order: ["codex", "openai", "claude"]
```

**CLI**:
```bash
# Use specific worker
feature-prd-runner run --prd-file feature.md --worker openai

# List available workers
feature-prd-runner workers list

# Test worker configuration
feature-prd-runner workers test openai

# Compare worker costs
feature-prd-runner workers estimate --prd-file feature.md
```

**Benefits**:
- Support for multiple AI providers (OpenAI, Anthropic, Azure, local models)
- A/B testing different models for quality/cost optimization
- Fallback to alternative workers on failure
- Cost optimization by routing simple tasks to cheaper models
- Support for local LLMs for sensitive/air-gapped environments
- Future-proof as new models become available

**Acceptance Criteria**:
- [ ] Abstract `WorkerInterface` defined with all required methods
- [ ] At least 3 concrete worker implementations (Codex, OpenAI, Anthropic)
- [ ] Configuration system supports multiple workers with routing
- [ ] CLI commands for worker management
- [ ] Automatic fallback when primary worker fails
- [ ] Cost estimation API for all workers
- [ ] Tests for each worker implementation
- [ ] Documentation with examples for each provider
- [ ] Migration guide from Codex-only to multi-worker setup

---

### 🟡 2. Web UI Dashboard

**Status**: 🟡 In Progress
**Priority**: P0 (Critical)
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Current CLI-only interface makes it hard to monitor long-running tasks, visualize progress, and collaborate with team members.

**Solution**:
Build a modern web-based dashboard for monitoring and controlling runs.

**Implementation**:

**Backend API**:
```python
# src/feature_prd_runner/server/api.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/api/runs")
async def list_runs(project_dir: str):
    """List all runs for a project."""
    pass

@app.get("/api/runs/{run_id}")
async def get_run_details(run_id: str):
    """Get detailed run information."""
    pass

@app.get("/api/runs/{run_id}/phases")
async def get_phases(run_id: str):
    """Get phase breakdown with progress."""
    pass

@app.websocket("/ws/runs/{run_id}")
async def run_updates(websocket: WebSocket, run_id: str):
    """Stream real-time updates for a run."""
    pass

@app.post("/api/runs/{run_id}/control")
async def control_run(run_id: str, action: str):
    """Control run: pause, resume, skip-step, retry."""
    pass

@app.get("/api/metrics")
async def get_metrics(project_dir: str):
    """Get aggregated metrics across runs."""
    pass
```

**Frontend Features**:
```typescript
// web/src/components/
- RunDashboard.tsx      // Main dashboard view
- PhaseTimeline.tsx     // Visual timeline of phases
- LiveLog.tsx           // Streaming log viewer
- MetricsChart.tsx      // Token usage, costs, timing
- DependencyGraph.tsx   // Interactive phase dependency graph
- TaskControl.tsx       // Pause, resume, skip, retry controls
- ReviewPanel.tsx       // Review issues and severity
- AllowlistManager.tsx  // Approve/deny file changes
- ConfigEditor.tsx      // Edit config in UI
```

**Dashboard Views**:

1. **Overview**:
   - Active runs with live status
   - Progress bars for each phase
   - ETA and cost estimates
   - Quick actions (pause, stop, view logs)

2. **Run Detail**:
   - Phase timeline (Gantt chart style)
   - Current step with real-time log streaming
   - File changes diff viewer
   - Test results and verification status
   - Review issues with inline code viewer

3. **Metrics**:
   - Token usage over time (chart)
   - Cost breakdown by phase
   - Performance metrics (time per step)
   - Success/failure rates
   - Historical trends

4. **Control Panel**:
   - Start/stop runs
   - Inject custom prompts
   - Approve allowlist expansions
   - Override review decisions
   - Manual checkpoints

5. **Configuration**:
   - Edit config.yaml in UI
   - Manage workers
   - Set up notifications
   - Configure plugins

**CLI Integration**:
```bash
# Start web server
feature-prd-runner server --port 8080 --project-dir .

# Start with authentication
feature-prd-runner server --auth --users-file users.yaml

# Access at http://localhost:8080
```

**Features**:
- Real-time WebSocket updates
- Multi-project support
- Team collaboration (multiple users viewing same run)
- Mobile-responsive design
- Dark/light theme
- Export reports (PDF, HTML)
- Share run links
- Email/Slack notifications from UI

**Technology Stack**:
- Backend: FastAPI + WebSocket
- Frontend: React + TypeScript
- Charting: Recharts or D3.js
- Real-time: WebSocket + Server-Sent Events
- Styling: Tailwind CSS
- Build: Vite

**Acceptance Criteria**:
- [x] FastAPI backend with all REST endpoints
- [x] WebSocket streaming for real-time updates
- [x] React frontend with all major views
- [x] Live log streaming with search/filter
- [x] Interactive dependency graph visualization
- [x] Metrics charts (tokens, cost, time)
- [x] Run control actions (pause, resume, skip, retry)
- [x] Authentication and authorization system (optional, env-configured)
- [x] Multi-project support
- [x] Mobile-responsive design
- [ ] One-command server startup
- [ ] Documentation and screenshots (in README)

---

### 🟡 3. Flexible Step Progression & Custom Prompts

**Status**: 🟡 In Progress
**Priority**: P0 (Critical)
**Impact**: Versatility ⭐⭐⭐⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Current system is rigid - must follow PLAN → IMPLEMENT → VERIFY → REVIEW → COMMIT flow. Hard to inject one-off tasks or custom workflows.

**Solution**:
Support flexible step progression with custom prompts and ad-hoc tasks.

**Implementation**:

**1. Ad-hoc Custom Prompts** (already partially exists via `--custom-prompt`, but enhance it):
```bash
# Run a one-off task without entering normal cycle
feature-prd-runner exec "Update all copyright headers to 2025"

# Run custom prompt and return to normal flow
feature-prd-runner exec "Refactor auth module for clarity" --then continue

# Run custom prompt in specific context
feature-prd-runner exec "Fix mypy errors" --context phase-2 --files "src/**/*.py"

# Chain multiple custom prompts
feature-prd-runner exec "
  1. Update dependencies in requirements.txt
  2. Run tests and fix any failures
  3. Update CHANGELOG.md
" --chain
```

**2. Custom Step Sequences**:
```yaml
# .prd_runner/workflows/hotfix.yaml
name: hotfix
description: Quick hotfix workflow - skip planning and review
steps:
  - implement
  - verify
  - commit  # Skip review for hotfixes

# .prd_runner/workflows/thorough.yaml
name: thorough
description: Extra thorough review process
steps:
  - plan_impl
  - implement
  - verify
  - review
  - security_scan  # Custom step
  - manual_approval  # Custom step
  - commit

# .prd_runner/workflows/experimental.yaml
name: experimental
description: Experimental feature with checkpoints
steps:
  - plan_impl
  - checkpoint
  - implement
  - checkpoint
  - verify
  - rollback_if_failed  # Custom step
```

**3. Step Injection**:
```bash
# Inject a step before current step
feature-prd-runner inject --before verify "Run benchmark tests"

# Inject after current step
feature-prd-runner inject --after implement "Generate documentation"

# Replace current step
feature-prd-runner inject --replace review "Quick sanity check"
```

**4. Conditional Steps**:
```yaml
# .prd_runner/config.yaml
workflows:
  default:
    steps:
      - plan_impl
      - implement
      - verify
      - review:
          condition: "changed_lines > 100 or files_include('auth/**')"
      - security_scan:
          condition: "files_include('**/*.env') or phase_id == 'security-hardening'"
      - commit
```

**5. Step Hooks**:
```python
# src/feature_prd_runner/hooks.py
class StepHooks:
    """Define hooks for custom logic at each step."""

    @hook('before_implement')
    def backup_current_state(self, phase: dict):
        """Create backup before implementation."""
        pass

    @hook('after_verify')
    def update_coverage_report(self, phase: dict, result: VerifyResult):
        """Update coverage dashboard."""
        pass

    @hook('before_commit')
    def require_ticket_number(self, changes: dict) -> bool:
        """Enforce ticket number in commit message."""
        pass
```

**6. Interactive Mode**:
```bash
# Interactive step-by-step execution
feature-prd-runner interactive --prd-file feature.md

# Prompts:
# > Current step: IMPLEMENT (phase-1)
# > Options:
# >   [c] Continue with step
# >   [s] Skip this step
# >   [r] Retry this step
# >   [e] Execute custom prompt
# >   [j] Jump to different step
# >   [p] Pause and save state
# >   [q] Quit
# > Choice: e
# > Custom prompt: Add debug logging to auth module
# > Executing... Done!
# > Continue with IMPLEMENT? [y/n]: y
```

**7. Parallel Custom Tasks**:
```bash
# Run multiple custom tasks in parallel
feature-prd-runner exec-parallel \
  "Update API documentation" \
  "Regenerate OpenAPI spec" \
  "Run performance benchmarks"
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
workflows:
  enabled: true
  default_workflow: standard

  available:
    - standard  # PLAN → IMPLEMENT → VERIFY → REVIEW → COMMIT
    - hotfix    # IMPLEMENT → VERIFY → COMMIT
    - thorough  # Includes security scans and manual approval
    - experimental  # With checkpoints and rollback

step_hooks:
  enabled: true
  scripts:
    before_implement: "scripts/backup.sh"
    after_verify: "scripts/update_coverage.sh"
    before_commit: "scripts/check_ticket.py"

custom_steps:
  security_scan:
    command: "npm audit"
    timeout: 300

  manual_approval:
    type: interactive
    prompt: "Review changes and approve to continue"
```

**API for Custom Steps**:
```python
# src/feature_prd_runner/custom_steps.py
from feature_prd_runner.steps import CustomStep

class SecurityScanStep(CustomStep):
    """Custom security scanning step."""

    def execute(self, context: StepContext) -> StepResult:
        # Run security scan
        result = subprocess.run(["npm", "audit"], ...)

        if result.returncode != 0:
            return StepResult.failed("Security vulnerabilities found")

        return StepResult.success()

    def should_run(self, context: StepContext) -> bool:
        # Only run for certain phases
        return context.phase_id.startswith("security-")

# Register custom step
register_step("security_scan", SecurityScanStep())
```

**Benefits**:
- One-off tasks without disrupting workflow
- Custom workflows for different scenarios (hotfix, experimental, thorough)
- Step hooks for automation
- Interactive control during execution
- Conditional step execution
- Parallel custom task execution
- Extensible with custom step types

**Acceptance Criteria**:
- [x] `exec` command for one-off custom prompts
- [ ] Custom workflow definitions (YAML)
- [ ] Step injection commands
- [ ] Conditional step execution
- [ ] Step hooks system
- [x] Interactive mode implementation
- [ ] Parallel custom task execution
- [ ] Custom step registration API
- [ ] Built-in workflow presets (standard, hotfix, thorough)
- [ ] Documentation with examples for each feature
- [ ] Tests for workflow engine

---

## Phase 1: Foundation (Weeks 1-2)

### 🔴 4. Configuration Validation & Presets

**Status**: 🔴 Not Started
**Priority**: P1
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐⭐

**Problem**: Manual configuration is error-prone; common issues not detected until runtime.

**Solution**:
Comprehensive configuration validation with auto-fix suggestions and presets.

**Implementation**:
```python
# src/feature_prd_runner/config_validator.py
@dataclass
class ValidationError:
    field: str
    issue: str
    severity: str  # error, warning, info
    suggested_fix: Optional[str]
    auto_fixable: bool

class ConfigValidator:
    """Validate runner configuration."""

    def validate_all(self, config: dict) -> ValidationResult:
        """Run all validations."""
        errors = []
        errors.extend(self.validate_commands(config))
        errors.extend(self.validate_git(config))
        errors.extend(self.validate_worker(config))
        errors.extend(self.validate_paths(config))
        return ValidationResult(errors)

    def validate_commands(self, config: dict) -> list[ValidationError]:
        """Verify test/lint/format commands are valid."""
        # Check command syntax
        # Verify executables exist
        # Test commands with --help
        pass

    def validate_git(self, config: dict) -> list[ValidationError]:
        """Check git repository state."""
        # Verify git repo
        # Check remote connectivity
        # Verify push permissions
        # Check branch naming
        pass

    def validate_worker(self, config: dict) -> list[ValidationError]:
        """Validate worker configuration."""
        # Check Codex/API keys
        # Test worker execution
        # Verify prompt placeholders
        pass

    def suggest_fixes(self, errors: list[ValidationError]) -> list[Fix]:
        """Generate auto-fix suggestions."""
        pass

    def apply_fixes(self, fixes: list[Fix]) -> None:
        """Apply automatic fixes."""
        pass
```

**Presets**:
```python
# src/feature_prd_runner/presets.py
PRESETS = {
    "python-pytest": {
        "verify_profile": "python",
        "test_command": "pytest -v",
        "format_command": "ruff format --check .",
        "lint_command": "ruff check .",
        "typecheck_command": "mypy . --strict",
        "ensure_ruff": "install",
        "ensure_deps": "install",
    },

    "python-unittest": {
        "verify_profile": "python",
        "test_command": "python -m pytest discover",
        "lint_command": "pylint src/",
        "format_command": "black --check .",
        "typecheck_command": "mypy src/",
    },

    "typescript-jest": {
        "test_command": "npm test",
        "lint_command": "npm run lint",
        "typecheck_command": "npm run typecheck",
        "format_command": "npm run format:check",
    },

    "typescript-vitest": {
        "test_command": "vitest run",
        "lint_command": "eslint .",
        "typecheck_command": "tsc --noEmit",
    },

    "go-standard": {
        "test_command": "go test ./... -v",
        "lint_command": "golangci-lint run",
        "format_command": "gofmt -l .",
    },

    "rust-cargo": {
        "test_command": "cargo test",
        "lint_command": "cargo clippy -- -D warnings",
        "format_command": "cargo fmt -- --check",
    },

    "java-maven": {
        "test_command": "mvn test",
        "lint_command": "mvn checkstyle:check",
        "format_command": "mvn spotless:check",
    },

    "ruby-rspec": {
        "test_command": "bundle exec rspec",
        "lint_command": "bundle exec rubocop",
    },
}
```

**CLI**:
```bash
# Initialize with preset
feature-prd-runner init --preset python-pytest --project-dir .

# List available presets
feature-prd-runner presets list

# Validate configuration
feature-prd-runner validate-config --project-dir .

# Validate with auto-fix
feature-prd-runner validate-config --fix --project-dir .

# Check specific aspect
feature-prd-runner validate-config --only git,commands

# Output:
# ✓ Git repository detected
# ✓ Remote 'origin' configured
# ✗ Command 'pytest' not found
#   Fix: pip install pytest
# ⚠ Test command takes >5min (consider timeout)
# ✓ Ruff installed
# ✗ .gitignore missing .prd_runner/
#   Fix: Add '.prd_runner/' to .gitignore [auto-fixable]
#
# Run with --fix to apply 2 automatic fixes
```

**Interactive Setup Wizard**:
```bash
feature-prd-runner init --interactive

# > Welcome to Feature PRD Runner!
# > Let's set up your project.
# >
# > Detected project type: Python (pyproject.toml found)
# > Suggested preset: python-pytest
# > Use this preset? [Y/n]: y
# >
# > Verify configuration:
# >   Test command: pytest -v
# >   Lint command: ruff check .
# >   Format command: ruff format --check .
# >   Typecheck: mypy . --strict
# > Looks good? [Y/n]: y
# >
# > Git configuration:
# >   Remote: origin (git@github.com:user/repo.git)
# >   Default branch: main
# > Correct? [Y/n]: y
# >
# > Worker configuration:
# >   Default worker: codex
# >   Command: codex exec -
# > Test worker? [Y/n]: y
# >   ✓ Worker responding correctly
# >
# > Writing configuration to .prd_runner/config.yaml
# > Adding .prd_runner/ to .gitignore
# > ✓ Setup complete!
# >
# > Next steps:
# >   1. Write a PRD in docs/feature.md
# >   2. Run: feature-prd-runner run --prd-file docs/feature.md
```

**Acceptance Criteria**:
- [ ] `ConfigValidator` class with comprehensive checks
- [ ] At least 8 language/framework presets
- [ ] `init` command with preset selection
- [ ] `validate-config` command with detailed output
- [ ] Auto-fix capability for common issues
- [ ] Interactive setup wizard
- [ ] Preset detection from project structure
- [ ] Tests for all validation rules
- [ ] Documentation for each preset

---

### 🟡 5. Enhanced Error Messages & Debugging

**Status**: 🟡 In Progress
**Priority**: P1
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐⭐

**Problem**: Error messages are often cryptic; hard to debug stuck tasks.

**Solution**:
Rich, actionable error messages with AI-powered debugging assistance.

**Implementation**:
```python
# src/feature_prd_runner/debug.py
class ErrorAnalyzer:
    """Analyze and explain errors."""

    def analyze_error(
        self,
        task_id: str,
        error_type: str,
        error_detail: str,
        context: dict
    ) -> ErrorReport:
        """Generate comprehensive error report."""
        pass

    def suggest_resolution(self, error: ErrorReport) -> list[Action]:
        """AI-powered resolution suggestions."""
        pass

    def explain_blocking(self, task_id: str) -> str:
        """Human-readable blocking explanation."""
        pass

class DebugSession:
    """Interactive debugging session."""

    def replay_task(
        self,
        task_id: str,
        step: TaskStep,
        verbose: bool = True
    ) -> ReplayResult:
        """Replay task execution with detailed logging."""
        pass

    def inspect_state(self, task_id: str) -> StateSnapshot:
        """Inspect full task state."""
        pass

    def trace_history(self, task_id: str) -> list[Event]:
        """Show full event history for task."""
        pass
```

**Enhanced Error Messages**:
```python
# Instead of:
# "Verification failed"

# Provide:
"""
❌ Verification Failed: phase-1 (attempt 3/10)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failed Command:
  pytest -v tests/

Exit Code: 1
Duration: 3.2s

Failed Tests (2):
  📍 tests/test_auth.py::test_login [line 45]
     AssertionError: assert 401 == 200

  📍 tests/test_auth.py::test_logout [line 78]
     AttributeError: 'User' has no attribute 'session'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root Cause Analysis:
  The User model is missing the 'session' field required by
  logout logic. Login test also failing due to incorrect
  status code expectation.

Files Involved:
  • src/models/user.py (missing 'session' field)
  • tests/test_auth.py (2 failing tests)
  • src/auth/logout.py (references user.session)

Suggested Actions:
  1. Add 'session' field to User model
     → File: src/models/user.py:23
     → Add: session: Optional[Session] = None

  2. Update test expectation
     → File: tests/test_auth.py:45
     → Change: assert response.status == 200

  3. Update test fixtures to include session data
     → File: tests/fixtures.py
     → Add session data to user_factory

Quick Fixes:
  [1] Retry verification
      feature-prd-runner retry phase-1

  [2] Skip verification (not recommended)
      feature-prd-runner skip-step phase-1 --step verify

  [3] Debug interactively
      feature-prd-runner debug phase-1

  [4] View full logs
      feature-prd-runner logs phase-1 --step verify

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next Steps:
  The runner will automatically retry this step.
  If it fails again, the task will be blocked for review.

  Attempts remaining: 7/10
"""
```

**CLI Commands**:
```bash
# Debug a task
feature-prd-runner debug phase-1
# Interactive mode with state inspection

# Replay task with verbose logging
feature-prd-runner replay phase-1 --step implement --verbose

# Explain why blocked
feature-prd-runner explain phase-1

# Get AI suggestions
feature-prd-runner suggest-fix phase-1

# Trace full event history
feature-prd-runner trace phase-1

# Inspect task state
feature-prd-runner inspect phase-1

# View detailed logs
feature-prd-runner logs phase-1 --step verify --lines 200

# Compare runs
feature-prd-runner diff run-123 run-456
```

**Acceptance Criteria**:
- [x] Rich error formatting with colors and structure
- [x] Root cause analysis for common errors
- [ ] Actionable suggestions with specific file/line numbers
- [ ] `debug` command with interactive mode
- [ ] `replay` command with verbose logging
- [x] `explain` command for blocked tasks
- [ ] AI-powered `suggest-fix` command
- [x] `trace` and `inspect` commands
- [ ] Error templates for all error types
- [x] Tests for error analysis
- [ ] Documentation with examples

---

## Phase 2: Observability & UX (Weeks 3-4)

### 🟡 6. Comprehensive Metrics & Telemetry

**Status**: 🟡 In Progress
**Priority**: P1
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐

**Problem**: No visibility into resource usage, costs, performance trends.

**Solution**:
Track comprehensive metrics with visualization and alerting.

**Implementation**:
```python
# src/feature_prd_runner/telemetry.py
@dataclass
class RunMetrics:
    """Comprehensive run metrics."""

    # Resource usage
    tokens_used: int = 0
    api_calls: int = 0
    estimated_cost_usd: float = 0.0

    # Timing
    wall_time_seconds: float = 0.0
    worker_time_seconds: float = 0.0
    verification_time_seconds: float = 0.0
    review_time_seconds: float = 0.0

    # Task metrics
    phases_completed: int = 0
    phases_total: int = 0
    test_runs: int = 0
    test_failures: int = 0
    review_iterations: int = 0

    # Code changes
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0

    # Errors
    worker_failures: int = 0
    allowlist_violations: int = 0
    verification_failures: int = 0
    review_blockers: int = 0

    def to_json(self) -> dict: ...
    def pretty_print(self) -> str: ...
    def export_csv(self, path: Path) -> None: ...

class MetricsCollector:
    """Collect and aggregate metrics."""

    def record_event(self, event: Event) -> None: ...
    def get_phase_metrics(self, phase_id: str) -> PhaseMetrics: ...
    def get_run_metrics(self, run_id: str) -> RunMetrics: ...
    def get_historical_metrics(self, days: int = 30) -> list[RunMetrics]: ...
    def export_report(self, format: str = "html") -> str: ...
```

**CLI Commands**:
```bash
# View current run metrics
feature-prd-runner metrics

# Historical metrics
feature-prd-runner metrics --history --days 30

# Cost analysis
feature-prd-runner metrics --cost

# Performance breakdown
feature-prd-runner metrics --timing

# Export report
feature-prd-runner metrics --export report.html
feature-prd-runner metrics --export metrics.csv

# Compare runs
feature-prd-runner metrics --compare run-1 run-2

# Output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Metrics: run-abc123
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Progress:
#   Phases: 3/5 (60%) ████████░░
#   Current: phase-3 (VERIFY)
#   ETA: ~22 minutes
#
# Resource Usage:
#   Tokens: 847,234 / ~1,000,000 (est.)
#   API Calls: 47
#   Cost: $12.34 / ~$15.00 (est.)
#
# Timing:
#   Wall Time: 1h 23m 45s
#   Worker Time: 47m 12s
#   Idle Time: 36m 33s (blocked, waiting)
#
#   Breakdown:
#     PLAN:       8m 23s  (10%)
#     IMPLEMENT: 28m 45s  (61%)
#     VERIFY:     7m 12s  (15%)
#     REVIEW:     2m 52s  (6%)
#     COMMIT:     0m 15s  (1%)
#
# Code Changes:
#   Files: 23 modified, 5 new
#   Lines: +1,234 / -456
#
# Tests:
#   Runs: 12
#   Failures: 3 (25%)
#   Coverage: 87% → 89% (+2%)
#
# Issues:
#   Worker Failures: 2
#   Allowlist Violations: 4 (3 approved)
#   Verification Failures: 3
#   Review Blockers: 1 (resolved)
```

**Web Dashboard Integration**:
- Real-time metrics charts
- Historical trends
- Cost projections
- Performance comparisons
- Alert configuration

**Acceptance Criteria**:
- [x] `RunMetrics` dataclass with all key metrics
- [ ] `MetricsCollector` integration into orchestrator
- [x] `metrics` CLI command with formatting
- [ ] Historical metrics tracking
- [ ] Cost estimation and tracking
- [ ] Performance timing breakdown
- [x] HTML/CSV export formats
- [ ] Metrics comparison tool
- [x] Integration with web dashboard
- [x] Tests for metrics collection
- [ ] Documentation

---

### 🔴 7. Interactive TUI (Terminal UI)

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: CLI output is verbose and hard to follow for long-running tasks.

**Solution**:
Rich terminal UI with real-time updates (like htop for runs).

**Implementation**:
Use `textual` or `rich` for TUI.

```python
# src/feature_prd_runner/tui.py
from textual.app import App
from textual.widgets import Header, Footer, Static, Tree, Log

class RunnerTUI(App):
    """Interactive terminal UI for monitoring runs."""

    def compose(self):
        yield Header()
        yield RunOverview()
        yield PhaseTree()
        yield LiveLog()
        yield MetricsPanel()
        yield Footer()
```

**Features**:
- Real-time progress bars
- Live log streaming
- Phase dependency tree
- Metrics dashboard
- Interactive controls (pause, skip, retry)
- Keyboard shortcuts
- Mouse support

**CLI**:
```bash
# Launch TUI
feature-prd-runner tui --project-dir .

# TUI with specific run
feature-prd-runner tui --run-id abc123
```

**Acceptance Criteria**:
- [ ] TUI implementation using textual
- [ ] Real-time progress updates
- [ ] Live log streaming panel
- [ ] Interactive phase tree
- [ ] Metrics display panel
- [ ] Keyboard controls
- [ ] Mouse support
- [ ] Multiple layout options
- [ ] Tests for TUI components
- [ ] Documentation

---

## Phase 3: Flexibility & Extensibility (Weeks 5-7)

### 🔴 8. Plugin System

**Status**: 🔴 Not Started
**Priority**: P1
**Impact**: Versatility ⭐⭐⭐⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Hard to extend without forking; no community contribution path.

**Solution**:
Comprehensive plugin/hook system for extensibility.

**Implementation**:
```python
# src/feature_prd_runner/plugins/base.py
class PluginHook(ABC):
    """Base class for plugins."""

    name: str
    version: str

    # Lifecycle hooks
    def on_run_start(self, config: dict) -> None: ...
    def on_run_complete(self, metrics: RunMetrics) -> None: ...
    def on_run_failed(self, error: Exception) -> None: ...

    # Phase hooks
    def on_phase_start(self, phase: dict) -> None: ...
    def on_phase_complete(self, phase: dict, result: dict) -> None: ...
    def on_phase_failed(self, phase: dict, error: Exception) -> None: ...

    # Step hooks
    def on_implement_start(self, task: dict) -> None: ...
    def on_implement_complete(self, task: dict, changes: dict) -> None: ...
    def on_verification_failed(self, result: VerifyResult) -> ActionHint: ...
    def on_review_complete(self, review: dict) -> ReviewModification: ...
    def on_commit_ready(self, changes: list[str]) -> CommitMetadata: ...

    # Capability extensions
    def can_provide_worker(self) -> bool: ...
    def get_worker(self) -> WorkerInterface: ...
    def can_provide_parser(self) -> bool: ...
    def get_parser(self, framework: str) -> VerifyParser: ...

# Example plugins
class SlackNotificationPlugin(PluginHook):
    """Send Slack notifications on events."""

    def on_phase_complete(self, phase: dict, result: dict):
        self.send_slack(f"✓ Phase {phase['id']} complete!")

    def on_run_failed(self, error: Exception):
        self.send_slack(f"❌ Run failed: {error}")

class CodeCoveragePlugin(PluginHook):
    """Track and enforce code coverage."""

    def on_verification_complete(self, result: VerifyResult):
        coverage = self.parse_coverage()
        if coverage < self.config['min_coverage']:
            raise CoverageError(f"Coverage {coverage}% < {self.config['min_coverage']}%")

class SecurityScanPlugin(PluginHook):
    """Run security scans before commit."""

    def on_commit_ready(self, changes: list[str]) -> CommitMetadata:
        vulnerabilities = self.scan_dependencies()
        if vulnerabilities:
            raise SecurityError("Vulnerabilities detected")
        return {"security_scan": "passed"}

class JiraIntegrationPlugin(PluginHook):
    """Sync with Jira tickets."""

    def on_phase_start(self, phase: dict):
        self.update_jira_status(phase['id'], "In Progress")

    def on_phase_complete(self, phase: dict, result: dict):
        self.update_jira_status(phase['id'], "Done")
        self.add_jira_comment(f"Phase completed. {result}")
```

**Plugin Discovery**:
```python
# Automatic plugin discovery
# 1. From pypi packages: feature-prd-runner-plugin-*
# 2. From local directory: .prd_runner/plugins/
# 3. From config: .prd_runner/config.yaml

# src/feature_prd_runner/plugins/loader.py
class PluginLoader:
    def discover_plugins(self) -> list[PluginHook]:
        """Auto-discover plugins."""
        pass

    def load_plugin(self, name: str) -> PluginHook:
        """Load specific plugin."""
        pass
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
plugins:
  - name: slack
    enabled: true
    module: feature_prd_runner_plugin_slack
    config:
      webhook_url: "https://hooks.slack.com/..."
      notify_on:
        - phase_complete
        - run_blocked
        - run_failed
      channel: "#ci-notifications"

  - name: coverage
    enabled: true
    module: feature_prd_runner_plugin_coverage
    config:
      min_coverage: 80
      fail_below: true
      report_path: "htmlcov/index.html"

  - name: security-scan
    enabled: true
    module: feature_prd_runner_plugin_security
    config:
      scan_before_commit: true
      tools:
        - npm_audit
        - snyk
        - safety

  - name: jira
    enabled: false
    module: feature_prd_runner_plugin_jira
    config:
      url: "https://company.atlassian.net"
      api_token_env: "JIRA_API_TOKEN"
      project: "PROJ"
```

**CLI**:
```bash
# List plugins
feature-prd-runner plugins list

# Install plugin
feature-prd-runner plugins install slack

# Enable/disable plugin
feature-prd-runner plugins enable slack
feature-prd-runner plugins disable slack

# Configure plugin
feature-prd-runner plugins config slack --webhook-url "..."
```

**Acceptance Criteria**:
- [ ] `PluginHook` base class with all lifecycle hooks
- [ ] Plugin discovery from multiple sources
- [ ] Plugin configuration system
- [ ] At least 3 example plugins (Slack, coverage, security)
- [ ] CLI commands for plugin management
- [ ] Plugin testing framework
- [ ] Documentation for writing plugins
- [ ] Plugin packaging guide
- [ ] Plugin registry/marketplace (future)

---

### 🔴 9. Multi-Language Verification Optimization

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐⭐⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Pytest-optimized parsing; generic for other languages/frameworks.

**Solution**:
Framework-specific parsers for better error extraction.

**Implementation**:
```python
# src/feature_prd_runner/verify_parsers/base.py
class VerifyParser(ABC):
    """Abstract parser for test output."""

    @abstractmethod
    def parse_output(self, output: str) -> VerifyResult:
        """Parse test output into structured result."""
        pass

    @abstractmethod
    def extract_failing_tests(self, output: str) -> list[str]:
        """Extract list of failing test names."""
        pass

    @abstractmethod
    def extract_failing_paths(self, output: str) -> list[str]:
        """Extract file paths that need fixing."""
        pass

    @abstractmethod
    def should_expand_allowlist(self, output: str) -> bool:
        """Determine if allowlist should be expanded."""
        pass

# Concrete implementations
class PytestParser(VerifyParser): ...
class UnittestParser(VerifyParser): ...
class JestParser(VerifyParser): ...
class MochaParser(VerifyParser): ...
class GoTestParser(VerifyParser): ...
class RSpecParser(VerifyParser): ...
class JUnitParser(VerifyParser): ...
class CargoTestParser(VerifyParser): ...
```

**Auto-detection**:
```python
# src/feature_prd_runner/verify_parsers/detector.py
def detect_test_framework(project_dir: Path) -> str:
    """Auto-detect test framework from project files."""
    detectors = {
        "pytest": lambda: (project_dir / "pytest.ini").exists()
                       or (project_dir / "pyproject.toml").exists()
                       or any(project_dir.rglob("conftest.py")),
        "jest": lambda: (project_dir / "jest.config.js").exists()
                     or (project_dir / "jest.config.ts").exists(),
        "mocha": lambda: ".mocharc" in (project_dir / ".mocharc.json").read_text(),
        "go": lambda: (project_dir / "go.mod").exists(),
        "rspec": lambda: (project_dir / ".rspec").exists(),
        "cargo": lambda: (project_dir / "Cargo.toml").exists(),
    }

    for framework, detector in detectors.items():
        if detector():
            return framework

    return "generic"
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
verification:
  parser: auto  # or pytest, jest, go, etc.

  parsers:
    pytest:
      extract_tracebacks: true
      max_traceback_lines: 50

    jest:
      verbose: true
      extract_snapshot_failures: true

    go:
      verbose: true
      extract_panic_traces: true
```

**Acceptance Criteria**:
- [ ] `VerifyParser` abstract base
- [ ] At least 6 framework parsers
- [ ] Auto-detection from project structure
- [ ] Configuration for parser selection
- [ ] Tests for each parser
- [ ] Documentation for adding new parsers

---

### 🔴 10. Smart Allowlist Management

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐

**Problem**: Simple allowlist expansion with arbitrary limit; no context awareness.

**Solution**:
AI-powered allowlist management with pattern learning.

**Implementation**:
```python
# src/feature_prd_runner/allowlist_ai.py
@dataclass
class AllowlistDecision:
    action: str  # APPROVE, DENY, REQUEST_CLARIFICATION
    confidence: float
    reasoning: str
    suggested_patterns: list[str]

class SmartAllowlistManager:
    """Use AI to intelligently expand allowlists."""

    def analyze_violation(
        self,
        disallowed_paths: list[str],
        change_context: dict,
        implementation_plan: dict
    ) -> AllowlistDecision:
        """
        Analyze why changes were made to disallowed files.

        Checks for:
        - Necessary refactoring
        - Transitive dependencies
        - Type definition changes
        - Test updates
        - Common patterns (init files, types, etc.)
        """
        pass

    def learn_from_approval(self, paths: list[str], context: dict):
        """Learn patterns from manual approvals."""
        pass

    def suggest_patterns(self, history: list[dict]) -> list[str]:
        """Suggest allowlist patterns based on history."""
        pass

# Pattern matchers
COMMON_PATTERNS = {
    "test_files": {
        "pattern": "tests/**/*_test.py",
        "reasoning": "Test files for new functionality",
        "auto_approve": True,
    },
    "type_definitions": {
        "pattern": "**/types.py",
        "reasoning": "Type definitions updated for new code",
        "auto_approve": True,
    },
    "package_markers": {
        "pattern": "**/__init__.py",
        "reasoning": "Package initialization",
        "auto_approve": True,
    },
    "config_files": {
        "pattern": "**/*.{yaml,json,toml}",
        "reasoning": "Configuration updates",
        "auto_approve": False,  # Requires review
    },
}
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
allowlist:
  auto_approve_patterns:
    - "tests/**/*_test.py"
    - "tests/**/*_spec.py"
    - "**/__init__.py"
    - "**/types.py"
    - "**/*.d.ts"  # TypeScript type definitions

  require_justification:
    - "**/*.env"
    - "**/config.yaml"
    - "**/secrets.*"

  max_auto_expansions: 10
  max_manual_expansions: 20

  ai_analysis:
    enabled: true
    confidence_threshold: 0.8
    use_history: true
```

**CLI**:
```bash
# Review allowlist violations
feature-prd-runner allowlist review phase-1

# Approve specific paths
feature-prd-runner allowlist approve phase-1 \
  src/models/user.py \
  tests/test_user.py

# Deny with reason
feature-prd-runner allowlist deny phase-1 src/admin/ \
  --reason "Admin changes not in scope"

# View suggestions
feature-prd-runner allowlist suggest phase-1

# Learn from history
feature-prd-runner allowlist train --history 30days
```

**Acceptance Criteria**:
- [ ] `SmartAllowlistManager` with AI analysis
- [ ] Pattern matching system
- [ ] Auto-approval for common patterns
- [ ] Learning from manual approvals
- [ ] CLI commands for allowlist management
- [ ] Configuration for patterns and thresholds
- [ ] Tests for allowlist logic
- [ ] Documentation

---

## Phase 4: Scale & Performance (Weeks 8-10)

### 🟡 11. Parallel Phase Execution

**Status**: 🟡 In Progress
**Priority**: P1
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Sequential execution wastes time when phases are independent.

**Solution**:
Execute independent phases in parallel with dependency resolution.

**Implementation**:
```python
# src/feature_prd_runner/parallel.py
class ParallelExecutor:
    """Execute phases in parallel with topological ordering."""

    def resolve_execution_order(
        self,
        phases: list[dict]
    ) -> list[list[str]]:
        """
        Return batches of phase IDs that can run in parallel.

        Example:
          Input: [A, B(deps=[A]), C(deps=[A]), D(deps=[B,C])]
          Output: [[A], [B, C], [D]]
        """
        pass

    def execute_parallel(
        self,
        phases: list[dict],
        max_workers: int = 3
    ) -> list[PhaseResult]:
        """Execute phases in parallel batches."""
        pass

    def check_circular_deps(self, phases: list[dict]) -> list[str]:
        """Detect circular dependencies."""
        pass
```

**Configuration**:
```yaml
# phase_plan.yaml
phases:
  - id: database-schema
    parallel_group: "backend"
    deps: []

  - id: frontend-components
    parallel_group: "frontend"
    deps: []

  - id: api-endpoints
    parallel_group: "backend"
    deps: ["database-schema"]

  - id: integration
    deps: ["api-endpoints", "frontend-components"]

# .prd_runner/config.yaml
parallel:
  enabled: true
  max_workers: 3
  resource_limits:
    max_memory_mb: 8192
    max_cpu_percent: 80
```

**Execution Flow**:
```
Batch 1 (parallel):
  ├─ database-schema
  └─ frontend-components

Batch 2 (parallel):
  └─ api-endpoints (waits for database-schema)

Batch 3:
  └─ integration (waits for both api-endpoints and frontend-components)
```

**CLI**:
```bash
# Run with parallel execution
feature-prd-runner run --prd-file feature.md --parallel

# Limit workers
feature-prd-runner run --prd-file feature.md --parallel --max-workers 2

# Visualize execution plan
feature-prd-runner plan --prd-file feature.md --show-parallel
```

**Acceptance Criteria**:
- [x] Topological sort for dependency resolution
- [x] Circular dependency detection
- [x] Parallel execution with worker pool
- [ ] Resource limits enforcement
- [ ] Progress tracking for parallel phases
- [ ] Error handling and rollback
- [x] CLI flags for parallel control
- [x] Visualization of execution plan
- [ ] Tests for parallel execution
- [ ] Documentation

---

### 🔴 12. Performance Optimizations

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐

**Problem**: Memory issues with large logs; slow I/O operations.

**Solution**:
Streaming, async I/O, and optimizations for large codebases.

**Implementation**:
```python
# src/feature_prd_runner/streaming.py
class StreamingLogHandler:
    """Stream logs instead of loading into memory."""

    def tail_log(self, log_path: Path, lines: int = 50) -> Iterator[str]:
        """Memory-efficient log tailing."""
        pass

    def search_log(self, log_path: Path, pattern: str) -> list[Match]:
        """Search without loading entire file."""
        pass

    def stream_to_ui(self, log_path: Path) -> AsyncIterator[str]:
        """Stream log lines to UI in real-time."""
        pass

# src/feature_prd_runner/async_verify.py
async def run_verification_parallel(
    commands: dict[str, str]
) -> dict[str, VerifyResult]:
    """Run format, lint, typecheck in parallel."""
    tasks = []
    for name, cmd in commands.items():
        tasks.append(run_verify_async(name, cmd))

    results = await asyncio.gather(*tasks)
    return dict(zip(commands.keys(), results))

# Caching
class ResultCache:
    """Cache verification results for unchanged files."""

    def get_cached(self, file_hash: str, command: str) -> Optional[Result]:
        pass

    def store(self, file_hash: str, command: str, result: Result):
        pass
```

**Optimizations**:
- Streaming log processing (no full load)
- Async/parallel verification commands
- Result caching for unchanged files
- Incremental git operations
- Lazy state loading
- Connection pooling for workers

**Acceptance Criteria**:
- [ ] Streaming log handlers
- [ ] Async verification execution
- [ ] Result caching system
- [ ] Memory profiling tests
- [ ] Performance benchmarks
- [ ] Large codebase tests (>10k files)
- [ ] Documentation

---

### 🔴 13. Checkpoint & Rollback System

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐⭐⭐⭐

**Problem**: No fine-grained rollback; git reset is too coarse.

**Solution**:
Checkpoint system for safe experimentation and fast recovery.

**Implementation**:
```python
# src/feature_prd_runner/checkpoints.py
@dataclass
class Checkpoint:
    id: str
    created_at: str
    task_id: str
    step: TaskStep
    git_ref: str
    state_snapshot: dict
    metadata: dict

class CheckpointManager:
    """Manage fine-grained rollback points."""

    def create_checkpoint(
        self,
        task_id: str,
        step: TaskStep,
        metadata: dict = None,
        auto: bool = False
    ) -> str:
        """
        Create a checkpoint with:
        - Git stash or commit
        - State snapshot (.prd_runner/)
        - Metadata (who, why, when)
        """
        pass

    def rollback_to_checkpoint(self, checkpoint_id: str):
        """
        Restore code and state to checkpoint:
        - Git reset/stash pop
        - Restore .prd_runner/ state
        - Update task queue
        """
        pass

    def list_checkpoints(
        self,
        task_id: Optional[str] = None
    ) -> list[Checkpoint]:
        """List available checkpoints."""
        pass

    def delete_checkpoint(self, checkpoint_id: str):
        """Remove checkpoint."""
        pass

    def auto_checkpoint(self, task_id: str, step: TaskStep):
        """Create automatic checkpoint before risky operations."""
        pass
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
checkpoints:
  enabled: true

  auto_create:
    before_implement: true
    before_verify: false
    before_review: false
    before_commit: true

  retention:
    max_checkpoints: 20
    keep_days: 30

  storage:
    use_git_reflog: true
    compress: true
```

**CLI**:
```bash
# Create manual checkpoint
feature-prd-runner checkpoint create "before-refactor"
feature-prd-runner checkpoint create --task phase-1 --message "Backup"

# List checkpoints
feature-prd-runner checkpoint list
feature-prd-runner checkpoint list --task phase-1

# Rollback
feature-prd-runner checkpoint rollback checkpoint-abc123
feature-prd-runner checkpoint rollback --to "before-refactor"
feature-prd-runner checkpoint rollback --latest

# Delete checkpoint
feature-prd-runner checkpoint delete checkpoint-abc123

# Show checkpoint details
feature-prd-runner checkpoint show checkpoint-abc123
```

**Acceptance Criteria**:
- [ ] `CheckpointManager` with all operations
- [ ] Automatic checkpoints before risky steps
- [ ] Git integration (stash/commit/reflog)
- [ ] State snapshot and restore
- [ ] Retention policy
- [ ] CLI commands
- [ ] Tests for checkpoint operations
- [ ] Documentation

---

## Phase 5: Advanced Features (Weeks 11+)

### 🟡 14. Advanced Dependency Management

**Status**: 🟡 In Progress
**Priority**: P3
**Impact**: Versatility ⭐⭐⭐ | Usefulness ⭐⭐⭐ | Robustness ⭐⭐⭐

**Problem**: Simple task deps; no inter-phase shared state.

**Solution**:
Typed phase outputs and complex dependency graphs.

**Implementation**:
```python
# src/feature_prd_runner/dependencies.py
@dataclass
class PhaseOutput:
    """Typed outputs from phases."""
    phase_id: str
    exports: dict[str, Any]
    artifacts: list[Path]
    metadata: dict

class DependencyGraph:
    """Manage complex dependencies."""

    def resolve_order(self, phases: list[dict]) -> list[list[str]]:
        """Return topologically sorted batches."""
        pass

    def check_circular_deps(self, phases: list[dict]) -> list[str]:
        """Detect circular dependencies."""
        pass

    def get_phase_inputs(self, phase_id: str) -> dict[str, PhaseOutput]:
        """Get outputs from dependency phases."""
        pass

    def export_output(self, phase_id: str, key: str, value: Any):
        """Export value for dependent phases."""
        pass
```

**Configuration**:
```yaml
# phase_plan.yaml
phases:
  - id: database-schema
    exports:
      schema_file: "schema.sql"
      migration_script: "migrations/001.sql"

  - id: backend-api
    deps: ["database-schema"]
    imports:
      schema: "$database-schema.schema_file"
    exports:
      openapi_spec: "openapi.json"
      api_client: "clients/api.ts"

  - id: frontend
    deps: ["backend-api"]
    imports:
      api_spec: "$backend-api.openapi_spec"
      api_client: "$backend-api.api_client"
```

**Acceptance Criteria**:
- [ ] `PhaseOutput` and `DependencyGraph` classes
- [ ] Import/export system in prompts
- [x] Circular dependency detection
- [ ] Tests for dependency resolution
- [ ] Documentation

---

### 🔴 15. Security Enhancements

**Status**: 🔴 Not Started
**Priority**: P2
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐ | Robustness ⭐⭐⭐⭐⭐

**Problem**: Basic allowlist enforcement; no secret detection or security scanning.

**Solution**:
Comprehensive security checks and sandboxing.

**Implementation**:
```python
# src/feature_prd_runner/security.py
class SecurityGuard:
    """Prevent unsafe operations."""

    def scan_for_secrets(self, changes: dict) -> list[SecretMatch]:
        """Detect API keys, passwords in changes."""
        pass

    def validate_dependencies(self, phase: dict) -> SecurityReport:
        """Check for known vulnerabilities."""
        pass

    def sandbox_verification(self, command: str) -> bool:
        """Run verification in isolated environment."""
        pass

    def check_file_permissions(self, paths: list[str]) -> list[Issue]:
        """Check for overly permissive files."""
        pass
```

**Configuration**:
```yaml
# .prd_runner/config.yaml
security:
  secret_detection:
    enabled: true
    patterns:
      - "api[_-]?key"
      - "password"
      - "secret"
      - "token"

  dependency_scanning:
    enabled: true
    tools:
      - npm_audit
      - pip_audit
      - snyk

  file_restrictions:
    max_size_mb: 10
    blocked_extensions:
      - ".exe"
      - ".dll"
      - ".so"

  sandboxing:
    verify_commands: true
    network_isolation: true
```

**Acceptance Criteria**:
- [ ] Secret detection in code changes
- [ ] Dependency vulnerability scanning
- [ ] File permission checks
- [ ] Sandboxed verification (optional)
- [ ] Configuration system
- [ ] Tests
- [ ] Documentation

---

### 🟡 16. Documentation & Onboarding

**Status**: 🟡 In Progress
**Priority**: P2
**Impact**: Versatility ⭐⭐ | Usefulness ⭐⭐⭐⭐ | Robustness ⭐⭐

**Problem**: Documentation could be more discoverable and interactive.

**Solution**:
Interactive onboarding and comprehensive examples.

**Implementation**:
```bash
# Interactive onboarding
feature-prd-runner onboard

# Create example project
feature-prd-runner example --language python --output my-project

# Built-in tutorials
feature-prd-runner tutorial basic
feature-prd-runner tutorial advanced

# Inline help with examples
feature-prd-runner help resume --examples
```

**Content**:
- Getting started guide
- Video tutorials
- Example projects for each language
- Troubleshooting guide
- API documentation
- Plugin development guide
- Architecture overview

**Acceptance Criteria**:
- [ ] Interactive onboarding wizard
- [x] Example project (Python)
- [ ] Video tutorials (basic, advanced)
- [ ] Comprehensive troubleshooting guide
- [ ] API documentation (Sphinx/MkDocs)
- [ ] Plugin development guide
- [ ] Architecture diagrams

---

## Additional Ideas Under Consideration

### 🔵 17. Multi-Repository Support

**Status**: 🔵 Under Consideration
**Priority**: P3

**Idea**: Support features that span multiple repositories (monorepo-style).

**Use Cases**:
- Microservices
- Shared libraries
- Client/server splits

**Challenges**:
- Coordinating commits across repos
- Handling cross-repo dependencies
- Testing across repos

---

### 🔵 18. AI Model Fine-Tuning

**Status**: 🔵 Under Consideration
**Priority**: P3

**Idea**: Fine-tune models on project-specific patterns.

**Benefits**:
- Better code style matching
- Domain-specific knowledge
- Reduced iteration count

**Challenges**:
- Training data collection
- Cost
- Maintenance

---

### 🔵 19. Collaborative Features

**Status**: 🔵 Under Consideration
**Priority**: P3

**Idea**: Multiple developers working on same run.

**Features**:
- Real-time collaboration
- Shared checkpoints
- Code review in UI
- Comments and annotations

---

### 🔵 20. Cost Optimization

**Status**: 🔵 Under Consideration
**Priority**: P3

**Idea**: Automatically optimize costs.

**Features**:
- Model selection per step
- Prompt compression
- Caching aggressive
- Cheaper models for simple tasks

---

## Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Start Week |
|---------|----------|--------|--------|------------|
| Human-in-the-Loop | P0 | High | Very High | 1 |
| Worker Abstraction | P0 | High | Very High | 1 |
| Web UI Dashboard | P0 | Very High | Very High | 1 |
| Flexible Steps | P0 | Medium | Very High | 2 |
| Config Validation | P1 | Medium | High | 1 |
| Enhanced Errors | P1 | Medium | Very High | 2 |
| Metrics & Telemetry | P1 | Medium | Very High | 3 |
| Interactive TUI | P2 | Medium | High | 4 |
| Plugin System | P1 | High | Very High | 5 |
| Multi-Language | P2 | Medium | High | 6 |
| Smart Allowlist | P2 | High | High | 7 |
| Parallel Execution | P1 | High | Very High | 8 |
| Performance | P2 | Medium | High | 9 |
| Checkpoints | P2 | Medium | Very High | 10 |
| Advanced Deps | P3 | Medium | Medium | 11 |
| Security | P2 | Medium | High | 11 |
| Documentation | P2 | High | Medium | 12 |

---

## Success Metrics

### Adoption Metrics
- [ ] 100+ stars on GitHub
- [ ] 10+ community plugins
- [ ] 50+ production users

### Performance Metrics
- [ ] 3-5x speedup with parallel execution
- [ ] 50% reduction in stuck tasks (smart allowlist)
- [ ] 90% reduction in debugging time (better errors)

### Cost Metrics
- [ ] 30% cost reduction (worker optimization)
- [ ] Transparent cost tracking for all users

### Quality Metrics
- [ ] 95% test coverage
- [ ] <5 critical bugs per release
- [ ] Documentation completeness >90%

---

## Contributing

Want to help implement these features? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**How to claim a feature**:
1. Comment on the related GitHub issue
2. Get assignment from maintainers
3. Create feature branch
4. Implement with tests
5. Submit PR with documentation

---

## Feedback

Have ideas not listed here? Open an issue with the `enhancement` label!

**Template**:
```
**Feature**: Brief name
**Problem**: What pain point does this solve?
**Solution**: Proposed approach
**Impact**: Who benefits and how?
**Effort**: Estimated complexity (Low/Medium/High)
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-16 | 1.0 | Initial roadmap created |

---

**Last Updated**: 2025-01-16
**Status**: Living document - Updated as features are implemented
