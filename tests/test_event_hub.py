from __future__ import annotations

import asyncio
import threading

from agent_orchestrator.runtime.events.ws import WebSocketHub


def test_publish_sync_from_background_thread_uses_attached_loop() -> None:
    async def _run() -> None:
        hub = WebSocketHub()
        received: list[dict[str, object]] = []
        done = asyncio.Event()

        async def _fake_publish(event: dict[str, object]) -> None:
            received.append(event)
            done.set()

        hub.publish = _fake_publish  # type: ignore[method-assign]
        hub.attach_loop(asyncio.get_running_loop())

        worker = threading.Thread(target=lambda: hub.publish_sync({"channel": "system", "type": "test"}))
        worker.start()
        worker.join(timeout=2)

        await asyncio.wait_for(done.wait(), timeout=2)
        assert received == [{"channel": "system", "type": "test"}]

    asyncio.run(_run())
