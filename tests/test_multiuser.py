"""Tests for multi-user support: user store, roles, presence, and API endpoints."""

from __future__ import annotations

import pytest


class TestUserStore:
    def test_default_admin(self):
        from feature_prd_runner.server.users import UserStore, UserRole

        store = UserStore()
        admin = store.get_by_username("admin")
        assert admin is not None
        assert admin.role == UserRole.ADMIN
        assert admin.has_permission("manage_users")

    def test_create_user(self):
        from feature_prd_runner.server.users import UserStore, UserRole

        store = UserStore()
        user = store.create_user("alice", role=UserRole.DEVELOPER, display_name="Alice")
        assert user.username == "alice"
        assert user.display_name == "Alice"
        assert user.role == UserRole.DEVELOPER
        assert user.avatar_color != ""

    def test_create_duplicate_returns_existing(self):
        from feature_prd_runner.server.users import UserStore

        store = UserStore()
        u1 = store.create_user("bob")
        u2 = store.create_user("bob")
        assert u1.id == u2.id

    def test_list_users(self):
        from feature_prd_runner.server.users import UserStore

        store = UserStore()
        store.create_user("alice")
        store.create_user("bob")
        users = store.list_users()
        assert len(users) == 3  # admin + alice + bob

    def test_deactivate_user(self):
        from feature_prd_runner.server.users import UserStore

        store = UserStore()
        store.create_user("alice")
        store.deactivate_user("alice")
        active = store.list_users(active_only=True)
        assert not any(u.username == "alice" for u in active)

    def test_update_role(self):
        from feature_prd_runner.server.users import UserStore, UserRole

        store = UserStore()
        store.create_user("alice")
        store.update_role("alice", UserRole.REVIEWER)
        alice = store.get_by_username("alice")
        assert alice.role == UserRole.REVIEWER

    def test_update_last_seen(self):
        from feature_prd_runner.server.users import UserStore

        store = UserStore()
        store.create_user("alice")
        store.update_last_seen("alice")
        alice = store.get_by_username("alice")
        assert alice.last_seen is not None


class TestRolePermissions:
    def test_admin_has_all_permissions(self):
        from feature_prd_runner.server.users import UserProfile, UserRole

        user = UserProfile(username="admin", role=UserRole.ADMIN)
        assert user.has_permission("manage_users")
        assert user.has_permission("launch_run")
        assert user.has_permission("view")

    def test_developer_permissions(self):
        from feature_prd_runner.server.users import UserProfile, UserRole

        user = UserProfile(username="dev", role=UserRole.DEVELOPER)
        assert user.has_permission("launch_run")
        assert user.has_permission("feedback")
        assert not user.has_permission("manage_users")

    def test_reviewer_permissions(self):
        from feature_prd_runner.server.users import UserProfile, UserRole

        user = UserProfile(username="rev", role=UserRole.REVIEWER)
        assert user.has_permission("view")
        assert user.has_permission("feedback")
        assert not user.has_permission("launch_run")

    def test_viewer_permissions(self):
        from feature_prd_runner.server.users import UserProfile, UserRole

        user = UserProfile(username="viewer", role=UserRole.VIEWER)
        assert user.has_permission("view")
        assert not user.has_permission("feedback")
        assert not user.has_permission("launch_run")


class TestPresenceTracker:
    def test_set_online(self):
        from feature_prd_runner.server.users import PresenceTracker

        tracker = PresenceTracker()
        tracker.set_online("alice", {"viewing": "board", "task_id": "t-1"})
        online = tracker.get_online_users()
        assert len(online) == 1
        assert online[0]["username"] == "alice"

    def test_set_offline(self):
        from feature_prd_runner.server.users import PresenceTracker

        tracker = PresenceTracker()
        tracker.set_online("alice")
        tracker.set_offline("alice")
        online = tracker.get_online_users()
        assert len(online) == 0

    def test_multiple_users(self):
        from feature_prd_runner.server.users import PresenceTracker

        tracker = PresenceTracker()
        tracker.set_online("alice")
        tracker.set_online("bob")
        online = tracker.get_online_users()
        assert len(online) == 2

    def test_get_presence(self):
        from feature_prd_runner.server.users import PresenceTracker

        tracker = PresenceTracker()
        tracker.set_online("alice", {"viewing": "dashboard"})
        p = tracker.get_presence("alice")
        assert p is not None
        assert p["viewing"] == "dashboard"

    def test_get_nonexistent_presence(self):
        from feature_prd_runner.server.users import PresenceTracker

        tracker = PresenceTracker()
        assert tracker.get_presence("unknown") is None


class TestUserProfile:
    def test_to_dict(self):
        from feature_prd_runner.server.users import UserProfile, UserRole

        user = UserProfile(
            username="alice",
            display_name="Alice W",
            role=UserRole.DEVELOPER,
            avatar_color="#4A90D9",
        )
        d = user.to_dict()
        assert d["username"] == "alice"
        assert d["display_name"] == "Alice W"
        assert d["role"] == "developer"
        assert d["avatar_color"] == "#4A90D9"
        assert d["active"] is True
