"""Shared A2A backend helpers for research orchestration."""

from src.a2a.client import A2AClient
from src.config import get_config

_config = get_config()

SEARCH_AGENT_URL = _config.search_agent_url
PAPER_AGENT_URL = _config.paper_agent_url
SYNTHESIS_AGENT_URL = _config.synthesis_agent_url

_a2a_client = A2AClient(timeout=120.0)


async def call_agent_safe(url: str, action: str, payload: dict) -> dict:
    """Call an A2A agent and normalize failures into an error payload."""
    try:
        response = await _a2a_client.call_agent(url, action, payload)
        if response.status == "completed":
            return response.result or {}
        return {"error": response.error or "Unknown error"}
    except Exception as e:
        return {"error": str(e)}
