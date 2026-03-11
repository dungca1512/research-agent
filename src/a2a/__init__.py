"""A2A (Agent-to-Agent) communication package."""

from src.a2a.client import A2AClient

__all__ = ["A2AClient", "BaseA2AAgent"]


def __getattr__(name: str):
    """Lazily expose BaseA2AAgent to avoid importing server deps too early."""
    if name == "BaseA2AAgent":
        from src.a2a.base_agent import BaseA2AAgent

        return BaseA2AAgent
    raise AttributeError(name)
