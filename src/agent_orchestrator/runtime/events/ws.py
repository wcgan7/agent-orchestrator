"""Websocket pub/sub hub for streaming runtime events to clients."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


CHANNELS = {
    "tasks",
    "queue",
    "agents",
    "review",
    "terminal",
    "notifications",
    "system",
}


@dataclass
class _WsClient:
    ws: WebSocket
    channels: set[str] = field(default_factory=set)
    project_ids: set[str] = field(default_factory=set)


class WebSocketHub:
    """Track websocket subscribers and route channel-scoped runtime events."""
    def __init__(self) -> None:
        self._clients: dict[int, _WsClient] = {}
        self._counter = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the event loop used for cross-thread publish scheduling."""
        with self._lock:
            self._loop = loop

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Serve one client connection and process subscribe/unsubscribe traffic."""
        # Remember the active event loop so background threads can publish safely.
        self.attach_loop(asyncio.get_running_loop())
        await websocket.accept()
        client = _WsClient(ws=websocket)
        cid = id(websocket)
        self._clients[cid] = client
        try:
            await websocket.send_text(json.dumps({"channel": "system", "type": "connected", "payload": {"channels": sorted(CHANNELS)}}))
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                action = message.get("action")
                channels = set(message.get("channels", []))
                project_ids = {
                    str(project_id).strip()
                    for project_id in message.get("project_ids", [])
                    if str(project_id).strip()
                }
                single_project_id = str(message.get("project_id") or "").strip()
                if single_project_id:
                    project_ids.add(single_project_id)
                if action == "subscribe":
                    client.channels |= channels & CHANNELS
                    client.project_ids |= project_ids
                    await websocket.send_text(
                        json.dumps(
                            {
                                "channel": "system",
                                "type": "subscribed",
                                "payload": {
                                    "channels": sorted(client.channels),
                                    "project_ids": sorted(client.project_ids),
                                },
                            }
                        )
                    )
                elif action == "unsubscribe":
                    client.channels -= channels
                    client.project_ids -= project_ids
                    await websocket.send_text(
                        json.dumps(
                            {
                                "channel": "system",
                                "type": "unsubscribed",
                                "payload": {
                                    "channels": sorted(client.channels),
                                    "project_ids": sorted(client.project_ids),
                                },
                            }
                        )
                    )
                elif action == "ping":
                    await websocket.send_text(json.dumps({"channel": "system", "type": "pong", "payload": {}}))
        except Exception:
            logger.debug("WebSocket client loop terminated with exception", exc_info=True)
        finally:
            self._clients.pop(cid, None)

    async def publish(self, event: dict[str, Any]) -> None:
        """Send one event to all subscribers matching channel and project filters."""
        self._counter += 1
        payload = json.dumps({**event, "seq": self._counter})
        stale: list[int] = []
        for cid, client in self._clients.items():
            if event.get("channel") not in client.channels and event.get("channel") != "system":
                continue
            if event.get("channel") != "system" and client.project_ids:
                event_project_id = str(event.get("project_id") or "").strip()
                if not event_project_id or event_project_id not in client.project_ids:
                    continue
            try:
                await client.ws.send_text(payload)
            except Exception:
                stale.append(cid)
        for cid in stale:
            self._clients.pop(cid, None)

    def publish_sync(self, event: dict[str, Any]) -> None:
        """Schedule async publish from sync code paths without blocking callers."""
        with self._lock:
            loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.publish(event), loop)
            return
        try:
            loop = asyncio.get_running_loop()
            self.attach_loop(loop)
            loop.create_task(self.publish(event))
        except RuntimeError:
            logger.debug("No running event loop available for publish_sync", exc_info=True)


hub = WebSocketHub()
