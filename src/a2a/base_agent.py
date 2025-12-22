"""Base class for A2A agents."""

import json
import uvicorn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class TaskRequest(BaseModel):
    """Incoming task request."""
    task_id: str
    action: str
    payload: dict


class TaskResponse(BaseModel):
    """Task response."""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentCard(BaseModel):
    """Agent metadata card (A2A specification)."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    skills: list[str] = []
    endpoint: str = ""
    authentication: Optional[dict] = None


class BaseA2AAgent(ABC):
    """
    Base class for creating A2A-compatible agents.
    
    Subclasses must implement:
        - agent_card: Property returning AgentCard
        - handle_task: Method to process incoming tasks
    
    Example:
        class SearchAgent(BaseA2AAgent):
            @property
            def agent_card(self) -> AgentCard:
                return AgentCard(
                    name="search-agent",
                    skills=["web_search", "arxiv_search"]
                )
            
            async def handle_task(self, action, payload):
                if action == "web_search":
                    return await self.web_search(payload["query"])
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title=self.__class__.__name__)
        self._setup_routes()
    
    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        """Return the agent's metadata card."""
        pass
    
    @abstractmethod
    async def handle_task(self, action: str, payload: dict) -> Any:
        """
        Handle an incoming task.
        
        Args:
            action: The skill/action to perform
            payload: Task data
            
        Returns:
            Task result (any serializable type)
        """
        pass
    
    def _setup_routes(self):
        """Setup FastAPI routes for A2A protocol."""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Return agent metadata card."""
            card = self.agent_card
            card.endpoint = f"http://{self.host}:{self.port}"
            return card.model_dump()
        
        @self.app.post("/tasks")
        async def handle_task_request(request: TaskRequest):
            """Handle incoming task request."""
            try:
                result = await self.handle_task(
                    action=request.action,
                    payload=request.payload
                )
                return TaskResponse(
                    task_id=request.task_id,
                    status="completed",
                    result=result
                ).model_dump()
            except Exception as e:
                return TaskResponse(
                    task_id=request.task_id,
                    status="failed",
                    error=str(e)
                ).model_dump()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "agent": self.agent_card.name}
    
    def run(self):
        """Start the agent server."""
        print(f"🚀 Starting {self.agent_card.name} on http://{self.host}:{self.port}")
        print(f"   Skills: {', '.join(self.agent_card.skills)}")
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    async def run_async(self):
        """Start the agent server asynchronously."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
