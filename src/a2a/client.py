"""A2A Client for calling remote agents."""

import httpx
from typing import Any, Optional
from pydantic import BaseModel


class TaskRequest(BaseModel):
    """Request to send to an A2A agent."""
    task_id: str
    action: str
    payload: dict


class TaskResponse(BaseModel):
    """Response from an A2A agent."""
    task_id: str
    status: str  # "completed", "failed", "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class A2AClient:
    """
    Client for communicating with A2A agents.
    
    Example:
        client = A2AClient()
        result = await client.call_agent(
            "http://localhost:8001",
            action="search",
            payload={"query": "LLM agents"}
        )
    """
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def get_agent_card(self, agent_url: str) -> dict:
        """
        Fetch the agent card from /.well-known/agent.json
        
        Args:
            agent_url: Base URL of the agent
            
        Returns:
            Agent card with capabilities and metadata
        """
        try:
            response = await self._client.get(
                f"{agent_url}/.well-known/agent.json"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def call_agent(
        self,
        agent_url: str,
        action: str,
        payload: dict,
        task_id: Optional[str] = None
    ) -> TaskResponse:
        """
        Call an A2A agent with a task.
        
        Args:
            agent_url: Base URL of the agent
            action: The action/skill to invoke
            payload: Data to send to the agent
            task_id: Optional task ID (auto-generated if not provided)
            
        Returns:
            TaskResponse with result or error
        """
        import uuid
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        request = TaskRequest(
            task_id=task_id,
            action=action,
            payload=payload
        )
        
        try:
            response = await self._client.post(
                f"{agent_url}/tasks",
                json=request.model_dump()
            )
            response.raise_for_status()
            data = response.json()
            
            return TaskResponse(
                task_id=data.get("task_id", task_id),
                status=data.get("status", "completed"),
                result=data.get("result"),
                error=data.get("error")
            )
        except httpx.HTTPStatusError as e:
            return TaskResponse(
                task_id=task_id,
                status="failed",
                error=f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            return TaskResponse(
                task_id=task_id,
                status="failed",
                error=str(e)
            )
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# Singleton client for convenience
_default_client: Optional[A2AClient] = None


def get_a2a_client() -> A2AClient:
    """Get the default A2A client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = A2AClient()
    return _default_client


async def call_agent(agent_url: str, action: str, payload: dict) -> TaskResponse:
    """Convenience function to call an agent."""
    client = get_a2a_client()
    return await client.call_agent(agent_url, action, payload)
