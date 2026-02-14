"""Web server for Agent Orchestrator dashboard."""

from .api import app, create_app

__all__ = ["app", "create_app"]
