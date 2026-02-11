"""Agent API endpoints for the control center.

Provides real-time agent status, lifecycle management, and handoff operations.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agents.pool import AgentPool
from ..agents.registry import AgentRegistry
from ..agents.scheduler import Scheduler
from ..collaboration.reasoning import ReasoningStore


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpawnAgentRequest(BaseModel):
    role: str
    task_id: Optional[str] = None
    display_name: Optional[str] = None


class AssignTaskRequest(BaseModel):
    task_id: str


class UpdateProgressRequest(BaseModel):
    current_step: Optional[str] = None
    current_file: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    output_line: Optional[str] = None


class ReassignRequest(BaseModel):
    new_role: str
    task_id: Optional[str] = None


class MessageRequest(BaseModel):
    content: str
    sender: str = ""


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_agent_router(
    get_pool: Callable[[], AgentPool],
    get_registry: Callable[[], AgentRegistry],
    get_reasoning_store: Optional[Callable[[], ReasoningStore]] = None,
    get_scheduler: Optional[Callable[[], Scheduler]] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v2/agents", tags=["agents"])

    # -- List & Status -------------------------------------------------------

    @router.get("")
    async def list_agents(status: Optional[str] = None) -> dict[str, Any]:
        pool = get_pool()
        from ..agents.registry import AgentStatus
        filter_status = None
        if status:
            try:
                filter_status = AgentStatus(status)
            except ValueError:
                pass
        agents = pool.list_agents(status=filter_status)
        return {
            "agents": [a.to_dict() for a in agents],
            "active_count": pool.active_count,
            "idle_count": pool.idle_count,
            "capacity": pool.capacity,
        }

    @router.get("/types")
    async def list_agent_types() -> dict[str, Any]:
        registry = get_registry()
        types = registry.list_types()
        return {
            "types": [
                {
                    "role": t.role.value,
                    "display_name": t.display_name,
                    "description": t.description,
                    "task_type_affinity": list(t.task_type_affinity),
                    "allowed_steps": list(t.allowed_steps),
                    "limits": {
                        "max_tokens": t.limits.max_tokens,
                        "max_time_seconds": t.limits.max_time_seconds,
                        "max_cost_usd": t.limits.max_cost_usd,
                        "max_concurrent_files": t.limits.max_concurrent_files,
                    },
                }
                for t in types
            ]
        }

    # -- Scheduler queue -----------------------------------------------------

    @router.get("/scheduler/queue")
    async def get_scheduler_queue() -> dict[str, Any]:
        """Return the current scheduling queue and recent assignments."""
        if get_scheduler is None:
            return {"pending_tasks": [], "recent_assignments": []}
        scheduler = get_scheduler()
        return scheduler.get_queue()

    # -- Single agent --------------------------------------------------------

    @router.get("/{agent_id}")
    async def get_agent(agent_id: str) -> dict[str, Any]:
        pool = get_pool()
        agent = pool.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return agent.to_dict()

    # -- Lifecycle -----------------------------------------------------------

    @router.post("/spawn")
    async def spawn_agent(req: SpawnAgentRequest) -> dict[str, Any]:
        pool = get_pool()
        try:
            overrides: dict[str, Any] = {}
            if req.display_name:
                overrides["display_name"] = req.display_name
            agent = pool.spawn(req.role, task_id=req.task_id, **overrides)
            return agent.to_dict()
        except (KeyError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/{agent_id}/pause")
    async def pause_agent(agent_id: str) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.pause(agent_id)
            return {"status": "paused"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    @router.post("/{agent_id}/resume")
    async def resume_agent(agent_id: str) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.resume(agent_id)
            return {"status": "resumed"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    @router.post("/{agent_id}/terminate")
    async def terminate_agent(agent_id: str) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.terminate(agent_id)
            return {"status": "terminated"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    @router.delete("/{agent_id}")
    async def remove_agent(agent_id: str) -> dict[str, str]:
        pool = get_pool()
        pool.remove(agent_id)
        return {"status": "removed"}

    # -- Task Assignment -----------------------------------------------------

    @router.post("/{agent_id}/assign")
    async def assign_task(agent_id: str, req: AssignTaskRequest) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.assign_task(agent_id, req.task_id)
            return {"status": "assigned", "task_id": req.task_id}
        except (KeyError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/{agent_id}/unassign")
    async def unassign_task(agent_id: str) -> dict[str, Any]:
        pool = get_pool()
        try:
            old_task = pool.unassign_task(agent_id)
            return {"status": "unassigned", "old_task_id": old_task}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # -- Reassign & Message --------------------------------------------------

    @router.post("/{agent_id}/reassign")
    async def reassign_agent(agent_id: str, req: ReassignRequest) -> dict[str, Any]:
        """Terminate the current agent and spawn a replacement with a new role."""
        pool = get_pool()
        agent = pool.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        task_id = req.task_id or agent.task_id
        try:
            pool.terminate(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        try:
            new_agent = pool.spawn(req.new_role, task_id=task_id)
            return {"status": "reassigned", "old_agent_id": agent_id, "new_agent": new_agent.to_dict()}
        except (KeyError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/{agent_id}/message")
    async def send_message_to_agent(agent_id: str, req: MessageRequest) -> dict[str, str]:
        """Send a message/instruction to a running agent.

        Appends the message to the agent's output_tail for the agent to pick up.
        """
        pool = get_pool()
        agent = pool.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        prefix = f"[{req.sender}] " if req.sender else "[system] "
        agent.output_tail.append(f"{prefix}{req.content}")
        return {"status": "sent"}

    # -- Progress Updates ----------------------------------------------------

    @router.post("/{agent_id}/heartbeat")
    async def agent_heartbeat(agent_id: str) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.heartbeat(agent_id)
            return {"status": "ok"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    @router.post("/{agent_id}/progress")
    async def update_agent_progress(agent_id: str, req: UpdateProgressRequest) -> dict[str, str]:
        pool = get_pool()
        try:
            pool.update_progress(
                agent_id,
                current_step=req.current_step,
                current_file=req.current_file,
                tokens_used=req.tokens_used,
                cost_usd=req.cost_usd,
                output_line=req.output_line,
            )
            return {"status": "updated"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # -- Reasoning endpoints -------------------------------------------------

    @router.get("/reasoning/{task_id}")
    async def get_reasoning(task_id: str) -> dict[str, Any]:
        if get_reasoning_store is None:
            return {"reasonings": []}
        store = get_reasoning_store()
        entries = store.get_for_task(task_id)
        return {"reasonings": [r.to_dict() for r in entries]}

    @router.post("/reasoning/{task_id}/step/start")
    async def start_reasoning_step(
        task_id: str,
        agent_id: str,
        agent_role: str = "implementer",
        step_name: str = "",
        reasoning: str = "",
    ) -> dict[str, str]:
        if get_reasoning_store is None:
            raise HTTPException(status_code=501, detail="Reasoning store not available")
        store = get_reasoning_store()
        store.start_step(task_id, agent_id, agent_role, step_name, reasoning)
        return {"status": "started"}

    @router.post("/reasoning/{task_id}/step/complete")
    async def complete_reasoning_step(
        task_id: str,
        agent_id: str,
        step_name: str = "",
        status: str = "completed",
        output: str = "",
    ) -> dict[str, str]:
        if get_reasoning_store is None:
            raise HTTPException(status_code=501, detail="Reasoning store not available")
        store = get_reasoning_store()
        if not store.complete_step(task_id, agent_id, step_name, status, output):
            raise HTTPException(status_code=404, detail="Step not found")
        return {"status": status}

    return router
