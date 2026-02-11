"""Multi-user management — user store, roles, and request context.

Provides a simple in-memory user store with optional file-based persistence.
Users can have roles (admin, developer, reviewer, viewer) that control access.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class UserRole(str, Enum):
    ADMIN = "admin"           # full access
    DEVELOPER = "developer"   # can run tasks, give feedback
    REVIEWER = "reviewer"     # can review, give feedback, no task launch
    VIEWER = "viewer"         # read-only


# Permissions per role
ROLE_PERMISSIONS: dict[str, set[str]] = {
    UserRole.ADMIN.value: {
        "view", "launch_run", "control_run", "feedback", "review",
        "manage_agents", "manage_users", "configure",
    },
    UserRole.DEVELOPER.value: {
        "view", "launch_run", "control_run", "feedback", "review",
        "manage_agents",
    },
    UserRole.REVIEWER.value: {
        "view", "feedback", "review",
    },
    UserRole.VIEWER.value: {
        "view",
    },
}


@dataclass
class UserProfile:
    """A registered user."""
    id: str = field(default_factory=lambda: f"user-{uuid.uuid4().hex[:8]}")
    username: str = ""
    display_name: str = ""
    role: UserRole = UserRole.DEVELOPER
    avatar_color: str = ""   # CSS color for avatar
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: Optional[str] = None
    active: bool = True

    def has_permission(self, perm: str) -> bool:
        return perm in ROLE_PERMISSIONS.get(self.role.value, set())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name or self.username,
            "role": self.role.value,
            "avatar_color": self.avatar_color,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "active": self.active,
        }


# Default avatar colors for distinguishing users
_AVATAR_COLORS = [
    "#4A90D9", "#D94A4A", "#4AD97B", "#D9A84A",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
]


class UserStore:
    """In-memory store for user profiles.

    Automatically creates a default admin user on init.
    """

    def __init__(self) -> None:
        self._users: dict[str, UserProfile] = {}
        self._by_username: dict[str, str] = {}   # username -> id
        self._color_idx = 0

        # Create default admin user
        self.create_user("admin", role=UserRole.ADMIN, display_name="Admin")

    def _next_color(self) -> str:
        color = _AVATAR_COLORS[self._color_idx % len(_AVATAR_COLORS)]
        self._color_idx += 1
        return color

    def create_user(
        self,
        username: str,
        *,
        role: UserRole = UserRole.DEVELOPER,
        display_name: str = "",
    ) -> UserProfile:
        """Create a new user. Returns existing user if username taken."""
        if username in self._by_username:
            return self._users[self._by_username[username]]

        user = UserProfile(
            username=username,
            display_name=display_name or username,
            role=role,
            avatar_color=self._next_color(),
        )
        self._users[user.id] = user
        self._by_username[username] = user.id
        return user

    def get_by_username(self, username: str) -> Optional[UserProfile]:
        uid = self._by_username.get(username)
        return self._users.get(uid) if uid else None

    def get_by_id(self, user_id: str) -> Optional[UserProfile]:
        return self._users.get(user_id)

    def list_users(self, active_only: bool = True) -> list[UserProfile]:
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.active]
        return users

    def update_last_seen(self, username: str) -> None:
        user = self.get_by_username(username)
        if user:
            user.last_seen = datetime.now(timezone.utc).isoformat()

    def update_role(self, username: str, role: UserRole) -> Optional[UserProfile]:
        user = self.get_by_username(username)
        if user:
            user.role = role
        return user

    def deactivate_user(self, username: str) -> bool:
        user = self.get_by_username(username)
        if user:
            user.active = False
            return True
        return False


class PresenceTracker:
    """Tracks which users are currently online and what they're viewing."""

    def __init__(self) -> None:
        self._presence: dict[str, dict[str, Any]] = {}

    def set_online(self, username: str, context: Optional[dict[str, str]] = None) -> None:
        self._presence[username] = {
            "username": username,
            "online": True,
            "last_active": datetime.now(timezone.utc).isoformat(),
            "viewing": (context or {}).get("viewing"),
            "task_id": (context or {}).get("task_id"),
        }

    def set_offline(self, username: str) -> None:
        if username in self._presence:
            self._presence[username]["online"] = False

    def get_online_users(self) -> list[dict[str, Any]]:
        return [p for p in self._presence.values() if p.get("online")]

    def get_presence(self, username: str) -> Optional[dict[str, Any]]:
        return self._presence.get(username)
