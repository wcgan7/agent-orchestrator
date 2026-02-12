from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket


V3_CHANNELS = {
    "tasks",
    "queue",
    "agents",
    "review",
    "quick_actions",
    "notifications",
    "system",
}


@dataclass
class _WsClient:
    ws: WebSocket
    channels: set[str] = field(default_factory=set)


class V3WebSocketHub:
    def __init__(self) -> None:
        self._clients: dict[int, _WsClient] = {}
        self._counter = 0

    async def handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client = _WsClient(ws=websocket)
        cid = id(websocket)
        self._clients[cid] = client
        try:
            await websocket.send_text(json.dumps({"channel": "system", "type": "connected", "payload": {"channels": sorted(V3_CHANNELS)}}))
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                action = message.get("action")
                channels = set(message.get("channels", []))
                if action == "subscribe":
                    client.channels |= channels & V3_CHANNELS
                    await websocket.send_text(json.dumps({"channel": "system", "type": "subscribed", "payload": {"channels": sorted(client.channels)}}))
                elif action == "unsubscribe":
                    client.channels -= channels
                    await websocket.send_text(json.dumps({"channel": "system", "type": "unsubscribed", "payload": {"channels": sorted(client.channels)}}))
                elif action == "ping":
                    await websocket.send_text(json.dumps({"channel": "system", "type": "pong", "payload": {}}))
        except Exception:
            pass
        finally:
            self._clients.pop(cid, None)

    async def publish(self, event: dict[str, Any]) -> None:
        self._counter += 1
        payload = json.dumps({**event, "seq": self._counter})
        stale: list[int] = []
        for cid, client in self._clients.items():
            if event.get("channel") not in client.channels and event.get("channel") != "system":
                continue
            try:
                await client.ws.send_text(payload)
            except Exception:
                stale.append(cid)
        for cid in stale:
            self._clients.pop(cid, None)

    def publish_sync(self, event: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            pass


hub = V3WebSocketHub()
