"""Multiplexed WebSocket hub for real-time state updates.

This replaces the polling pattern used by frontend components.  A single
WebSocket connection at ``/ws`` supports channel-based subscriptions:

Protocol (client → server):
    {"action": "subscribe", "channels": ["tasks", "agents", "metrics"]}
    {"action": "unsubscribe", "channels": ["metrics"]}
    {"action": "ping"}

Protocol (server → client):
    {"channel": "tasks", "event": "task_created", "data": {...}}
    {"channel": "agents", "event": "agent_progress", "data": {...}}
    {"channel": "system", "event": "pong", "data": {}}

Supported channels:
    tasks        — task CRUD events (created, updated, deleted, transitioned)
    agents       — agent lifecycle events (spawned, progress, completed, error)
    logs         — log streaming (append)
    metrics      — cost/token/timing updates
    notifications — alerts, approvals, budget warnings
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket
from loguru import logger


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------

VALID_CHANNELS = {"tasks", "agents", "logs", "metrics", "notifications", "presence", "status", "phases", "runs", "breakpoints", "approvals", "system"}
HEARTBEAT_INTERVAL = 30.0  # seconds


@dataclass
class _Client:
    ws: WebSocket
    channels: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_event_id: int = 0


class WebSocketHub:
    """Central hub that manages multiplexed WebSocket connections.

    Usage::

        hub = WebSocketHub()

        # In a FastAPI WebSocket endpoint:
        await hub.handle_connection(websocket)

        # From anywhere in the backend:
        await hub.broadcast("tasks", "task_created", {"id": "task-abc"})
    """

    def __init__(self) -> None:
        self._clients: dict[int, _Client] = {}  # id(ws) → client
        self._event_counter = 0
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and run the read/heartbeat loop.

        This method blocks until the client disconnects.
        """
        await websocket.accept()
        client = _Client(ws=websocket)
        cid = id(websocket)
        self._clients[cid] = client
        logger.debug("WS Hub: client connected (total={})", self.client_count)

        try:
            # Send welcome message
            await websocket.send_text(json.dumps({
                "channel": "system",
                "event": "connected",
                "data": {"channels": sorted(VALID_CHANNELS)},
            }))

            # Concurrent read + heartbeat
            await asyncio.gather(
                self._read_loop(client),
                self._heartbeat_loop(client),
            )
        except Exception:
            pass
        finally:
            self._clients.pop(cid, None)
            logger.debug("WS Hub: client disconnected (total={})", self.client_count)

    async def broadcast(
        self,
        channel: str,
        event: str,
        data: Any = None,
        *,
        exclude: Optional[WebSocket] = None,
    ) -> None:
        """Push an event to all clients subscribed to *channel*."""
        self._event_counter += 1
        payload = json.dumps({
            "channel": channel,
            "event": event,
            "data": data or {},
            "event_id": self._event_counter,
        })

        stale: list[int] = []
        for cid, client in self._clients.items():
            if channel not in client.channels and channel != "system":
                continue
            if exclude is not None and client.ws is exclude:
                continue
            try:
                await client.ws.send_text(payload)
                client.last_event_id = self._event_counter
            except Exception:
                stale.append(cid)

        for cid in stale:
            self._clients.pop(cid, None)

    def broadcast_sync(self, channel: str, event: str, data: Any = None) -> None:
        """Fire-and-forget broadcast from synchronous code.

        Schedules the async broadcast on the running event loop (if any).
        Safe to call from non-async contexts like agent pool callbacks.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(channel, event, data))
        except RuntimeError:
            # No running event loop — silently skip
            pass

    async def send_to(self, websocket: WebSocket, channel: str, event: str, data: Any = None) -> None:
        """Send an event to a single client."""
        self._event_counter += 1
        payload = json.dumps({
            "channel": channel,
            "event": event,
            "data": data or {},
            "event_id": self._event_counter,
        })
        try:
            await websocket.send_text(payload)
        except Exception:
            pass

    # -- internals ---------------------------------------------------------

    async def _read_loop(self, client: _Client) -> None:
        """Read and process client messages."""
        while True:
            raw = await client.ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action")
            if action == "subscribe":
                channels = set(msg.get("channels", []))
                valid = channels & VALID_CHANNELS
                client.channels |= valid
                await self.send_to(
                    client.ws, "system", "subscribed",
                    {"channels": sorted(client.channels)},
                )
            elif action == "unsubscribe":
                channels = set(msg.get("channels", []))
                client.channels -= channels
                await self.send_to(
                    client.ws, "system", "unsubscribed",
                    {"channels": sorted(client.channels)},
                )
            elif action == "ping":
                await self.send_to(client.ws, "system", "pong", {})

    async def _heartbeat_loop(self, client: _Client) -> None:
        """Send periodic pings to keep the connection alive."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await client.ws.send_text(json.dumps({
                    "channel": "system",
                    "event": "heartbeat",
                    "data": {"timestamp": time.time()},
                }))
            except Exception:
                break


# ---------------------------------------------------------------------------
# Web Notification Service — broadcasts to frontend via 'notifications' channel
# ---------------------------------------------------------------------------


class WebNotificationService:
    """Pushes notification events through the WebSocket hub.

    Each notification is broadcast on the ``notifications`` channel and picked
    up by the frontend ``NotificationCenter`` component.
    """

    def __init__(self, ws_hub: WebSocketHub) -> None:
        self._hub = ws_hub
        self._counter = 0

    def _push(self, ntype: str, title: str, message: str, severity: str = "info") -> None:
        self._counter += 1
        self._hub.broadcast_sync("notifications", "new", {
            "id": f"notif-{self._counter}",
            "type": ntype,
            "title": title,
            "message": message,
            "severity": severity,
        })

    def task_completed(self, task_id: str, title: str) -> None:
        self._push("success", f"Task completed: {title}", f"Task {task_id} finished successfully.", "success")

    def task_failed(self, task_id: str, error: str) -> None:
        self._push("error", f"Task failed: {task_id}", error[:200], "error")

    def agent_error(self, agent_id: str, error: str) -> None:
        self._push("error", f"Agent error: {agent_id}", error[:200], "error")

    def agent_spawned(self, agent_id: str, role: str) -> None:
        self._push("info", f"Agent spawned: {role}", f"Agent {agent_id} is now active.", "info")

    def approval_needed(self, gate_type: str, task_id: str | None = None) -> None:
        msg = f"Task {task_id}" if task_id else "A task"
        self._push("warning", f"Approval required: {gate_type}", f"{msg} is waiting for approval.", "warning")

    def budget_warning(self, agent_id: str, pct: float) -> None:
        self._push("warning", f"Budget warning: {agent_id}", f"Agent has used {pct:.0f}% of its budget.", "warning")

    def review_requested(self, task_id: str) -> None:
        self._push("info", "Review requested", f"Task {task_id} is ready for review.", "info")

    def mode_changed(self, mode: str) -> None:
        self._push("info", f"HITL mode changed to {mode}", f"Pipeline mode is now '{mode}'.", "info")


# Singleton hub instance
hub = WebSocketHub()
web_notifications = WebNotificationService(hub)
