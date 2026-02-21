"""Agent route registration for the runtime API."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..domain.models import AgentRecord
from .deps import RouteDeps
from . import router_impl as impl


SpawnAgentRequest = impl.SpawnAgentRequest


def register_agent_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register agent listing, type, and lifecycle routes."""
    @router.get("/agents/types")
    async def get_agent_types(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List supported agent roles and routing affinities.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload describing available agent role definitions.
        """
        container, _, _ = deps.ctx(project_dir)
        cfg = container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        task_role_map = dict(routing.get("task_type_roles") or {})
        role_affinity: dict[str, list[str]] = {}
        for task_type, role in task_role_map.items():
            role_name = str(role or "")
            if not role_name:
                continue
            role_affinity.setdefault(role_name, []).append(str(task_type))
        roles = ["general", "implementer", "reviewer", "researcher", "tester", "planner", "debugger"]
        return {
            "types": [
                {
                    "role": role,
                    "display_name": role.replace("_", " ").title(),
                    "description": f"{role.replace('_', ' ').title()} agent",
                    "task_type_affinity": sorted(role_affinity.get(role, [])),
                    "allowed_steps": ["plan", "implement", "verify", "review"],
                    "limits": {"max_tokens": 0, "max_time_seconds": 0, "max_cost_usd": 0.0},
                }
                for role in roles
            ]
        }


    @router.get("/agents")
    async def list_agents(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List all persisted agent records for the project.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing serialized agent records.
        """
        container, _, _ = deps.ctx(project_dir)
        return {"agents": [agent.to_dict() for agent in container.agents.list()]}

    @router.post("/agents/spawn")
    async def spawn_agent(body: SpawnAgentRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create a new agent record and publish an agent-created event.
        
        Args:
            body: Agent spawn payload with role, capacity, and provider override.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created agent record.
        """
        container, bus, _ = deps.ctx(project_dir)
        agent = AgentRecord(role=body.role, capacity=body.capacity, override_provider=body.override_provider)
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.spawned", entity_id=agent.id, payload=agent.to_dict())
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Pause an existing agent record.
        
        Args:
            agent_id: Identifier of the agent to pause.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "paused"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.paused", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Resume a previously paused agent record.
        
        Args:
            agent_id: Identifier of the agent to resume.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "running"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.resumed", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.post("/agents/{agent_id}/terminate")
    async def terminate_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Mark an agent record as terminated.
        
        Args:
            agent_id: Identifier of the agent to terminate.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated agent record.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        agent = container.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.status = "terminated"
        container.agents.upsert(agent)
        bus.emit(channel="agents", event_type="agent.terminated", entity_id=agent.id, payload={})
        return {"agent": agent.to_dict()}

    @router.delete("/agents/{agent_id}")
    async def remove_agent(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete an agent record by identifier.
        
        Args:
            agent_id: Identifier of the agent to remove.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating successful removal.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}

    @router.post("/agents/{agent_id}/remove")
    async def remove_agent_post(agent_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete an agent record through the POST compatibility endpoint.
        
        Args:
            agent_id: Identifier of the agent to remove.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload indicating successful removal.
        
        Raises:
            HTTPException: If the agent cannot be found.
        """
        container, bus, _ = deps.ctx(project_dir)
        removed = container.agents.delete(agent_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Agent not found")
        bus.emit(channel="agents", event_type="agent.removed", entity_id=agent_id, payload={})
        return {"removed": True}
