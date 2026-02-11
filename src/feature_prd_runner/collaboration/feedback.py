"""Structured feedback model — actionable human guidance for agents.

Feedback is *not* free-form text. Each piece of feedback has a type, target,
and action that agents can programmatically incorporate into their prompts.
Feedback persists across task retries so agents don't repeat mistakes.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------

class FeedbackType(str, Enum):
    APPROACH_CHANGE = "approach_change"     # "use X approach instead of Y"
    LIBRARY_SWAP = "library_swap"           # "use library X instead of Y"
    FILE_RESTRICTION = "file_restriction"   # "don't modify file X"
    STYLE_PREFERENCE = "style_preference"   # "prefer X coding style"
    BUG_REPORT = "bug_report"              # "this code has bug X"
    GENERAL = "general"                    # free-form guidance


class FeedbackPriority(str, Enum):
    MUST = "must"          # agent must follow
    SHOULD = "should"      # strong preference
    SUGGESTION = "suggestion"  # nice to have


class FeedbackStatus(str, Enum):
    ACTIVE = "active"
    ADDRESSED = "addressed"
    DISMISSED = "dismissed"


# ---------------------------------------------------------------------------
# Feedback entry
# ---------------------------------------------------------------------------

@dataclass
class Feedback:
    """A single piece of structured feedback."""
    id: str = field(default_factory=lambda: f"fb-{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    feedback_type: FeedbackType = FeedbackType.GENERAL
    priority: FeedbackPriority = FeedbackPriority.SHOULD
    status: FeedbackStatus = FeedbackStatus.ACTIVE

    # Content
    summary: str = ""          # one-line description
    details: str = ""          # full explanation
    target_file: Optional[str] = None    # specific file this applies to
    target_lines: Optional[str] = None   # line range (e.g., "10-25")

    # Action
    action: str = ""           # what the agent should do
    original_value: str = ""   # what to change from (for swaps)
    replacement_value: str = ""  # what to change to (for swaps)

    # Metadata
    created_by: str = ""       # username
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    addressed_at: Optional[str] = None
    agent_response: Optional[str] = None  # how the agent responded to this feedback

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "feedback_type": self.feedback_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "summary": self.summary,
            "details": self.details,
            "target_file": self.target_file,
            "target_lines": self.target_lines,
            "action": self.action,
            "original_value": self.original_value,
            "replacement_value": self.replacement_value,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "addressed_at": self.addressed_at,
            "agent_response": self.agent_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feedback":
        fb = cls()
        for k, v in data.items():
            if k == "feedback_type":
                fb.feedback_type = FeedbackType(v)
            elif k == "priority":
                fb.priority = FeedbackPriority(v)
            elif k == "status":
                fb.status = FeedbackStatus(v)
            elif hasattr(fb, k):
                setattr(fb, k, v)
        return fb

    def to_prompt_instruction(self) -> str:
        """Convert this feedback into a prompt instruction for the agent."""
        parts = [f"[{self.priority.value.upper()}]"]

        if self.feedback_type == FeedbackType.APPROACH_CHANGE:
            parts.append(f"Change approach: {self.summary}")
            if self.action:
                parts.append(f"Action: {self.action}")
        elif self.feedback_type == FeedbackType.LIBRARY_SWAP:
            parts.append(f"Use '{self.replacement_value}' instead of '{self.original_value}'")
        elif self.feedback_type == FeedbackType.FILE_RESTRICTION:
            parts.append(f"Do NOT modify: {self.target_file}")
        elif self.feedback_type == FeedbackType.STYLE_PREFERENCE:
            parts.append(f"Style: {self.summary}")
        elif self.feedback_type == FeedbackType.BUG_REPORT:
            parts.append(f"Fix bug: {self.summary}")
            if self.target_file:
                parts.append(f"in {self.target_file}")
                if self.target_lines:
                    parts.append(f"lines {self.target_lines}")
        else:
            parts.append(self.summary)

        if self.details and self.feedback_type == FeedbackType.GENERAL:
            parts.append(f"Details: {self.details}")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Review comment (line-level)
# ---------------------------------------------------------------------------

@dataclass
class ReviewComment:
    """A line-level comment on a diff, similar to GitHub PR reviews."""
    id: str = field(default_factory=lambda: f"rc-{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    file_path: str = ""
    line_number: int = 0
    line_type: str = "context"   # "added", "removed", "context"

    # Content
    body: str = ""
    resolved: bool = False

    # Threading — parent_id links to another ReviewComment for nested replies
    parent_id: Optional[str] = None

    # Metadata
    author: str = ""             # username or agent_id
    author_type: str = "human"   # "human" or "agent"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_type": self.line_type,
            "body": self.body,
            "resolved": self.resolved,
            "parent_id": self.parent_id,
            "author": self.author,
            "author_type": self.author_type,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


# ---------------------------------------------------------------------------
# Feedback store (in-memory, per-task)
# ---------------------------------------------------------------------------

class FeedbackStore:
    """In-memory store for feedback and review comments, indexed by task_id."""

    def __init__(self) -> None:
        self._feedback: dict[str, list[Feedback]] = defaultdict(list)
        self._comments: dict[str, list[ReviewComment]] = defaultdict(list)

    # -- Feedback CRUD -------------------------------------------------------

    def add_feedback(self, feedback: Feedback) -> None:
        self._feedback[feedback.task_id].append(feedback)

    def get_feedback(
        self,
        task_id: str,
        *,
        active_only: bool = False,
        feedback_type: Optional[FeedbackType] = None,
    ) -> list[Feedback]:
        items = self._feedback.get(task_id, [])
        if active_only:
            items = [f for f in items if f.status == FeedbackStatus.ACTIVE]
        if feedback_type:
            items = [f for f in items if f.feedback_type == feedback_type]
        return items

    def address_feedback(self, feedback_id: str, agent_response: str = "") -> bool:
        for items in self._feedback.values():
            for fb in items:
                if fb.id == feedback_id:
                    fb.status = FeedbackStatus.ADDRESSED
                    fb.addressed_at = datetime.now(timezone.utc).isoformat()
                    fb.agent_response = agent_response
                    return True
        return False

    def dismiss_feedback(self, feedback_id: str) -> bool:
        for items in self._feedback.values():
            for fb in items:
                if fb.id == feedback_id:
                    fb.status = FeedbackStatus.DISMISSED
                    return True
        return False

    def get_prompt_instructions(self, task_id: str, max_chars: int = 4000) -> str:
        """Generate prompt instructions from all active feedback for a task.

        Token-aware: truncates oldest feedback if total exceeds *max_chars*
        (roughly 1000 tokens). Most recent feedback is prioritized.
        """
        active = self.get_feedback(task_id, active_only=True)
        if not active:
            return ""
        # Most recent first — prioritize freshest feedback
        active.sort(key=lambda fb: fb.created_at, reverse=True)
        lines = ["Human feedback to incorporate:"]
        total_len = len(lines[0])
        for fb in active:
            instruction = f"  - {fb.to_prompt_instruction()}"
            if total_len + len(instruction) + 1 > max_chars:
                lines.append("  - [Earlier feedback truncated for context window]")
                break
            lines.append(instruction)
            total_len += len(instruction) + 1
        return "\n".join(lines)

    def get_effectiveness_report(self, task_id: str) -> dict[str, Any]:
        """Return a report of feedback effectiveness for a task.

        Shows how many feedback items were addressed vs still active.
        """
        all_fb = self.get_feedback(task_id)
        total = len(all_fb)
        addressed = sum(1 for fb in all_fb if fb.status == FeedbackStatus.ADDRESSED)
        dismissed = sum(1 for fb in all_fb if fb.status == FeedbackStatus.DISMISSED)
        active = sum(1 for fb in all_fb if fb.status == FeedbackStatus.ACTIVE)
        return {
            "total": total,
            "addressed": addressed,
            "dismissed": dismissed,
            "active": active,
            "addressed_rate": addressed / total if total > 0 else 0.0,
            "unaddressed_items": [
                {"id": fb.id, "summary": fb.summary, "created_at": fb.created_at}
                for fb in all_fb
                if fb.status == FeedbackStatus.ACTIVE
            ],
        }

    # -- Review Comments CRUD ------------------------------------------------

    def add_comment(self, comment: ReviewComment) -> None:
        self._comments[comment.task_id].append(comment)

    def get_comments(
        self,
        task_id: str,
        *,
        file_path: Optional[str] = None,
        unresolved_only: bool = False,
    ) -> list[ReviewComment]:
        items = self._comments.get(task_id, [])
        if file_path:
            items = [c for c in items if c.file_path == file_path]
        if unresolved_only:
            items = [c for c in items if not c.resolved]
        return items

    def resolve_comment(self, comment_id: str) -> bool:
        for items in self._comments.values():
            for c in items:
                if c.id == comment_id:
                    c.resolved = True
                    c.resolved_at = datetime.now(timezone.utc).isoformat()
                    return True
        return False

    def get_replies(self, parent_id: str) -> list[ReviewComment]:
        """Return all replies to a given comment (by parent_id)."""
        replies: list[ReviewComment] = []
        for items in self._comments.values():
            for c in items:
                if c.parent_id == parent_id:
                    replies.append(c)
        replies.sort(key=lambda c: c.created_at)
        return replies

    # -- Cleanup -------------------------------------------------------------

    def clear_task(self, task_id: str) -> None:
        self._feedback.pop(task_id, None)
        self._comments.pop(task_id, None)
