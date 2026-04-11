"""Tracker Agent — Persists research sessions to SQLite via A2A protocol."""

from src.a2a.base_agent import BaseA2AAgent, AgentCard
from src.storage.database import (
    init_db,
    save_session,
    list_sessions,
    get_session,
    delete_session,
)


class TrackerAgent(BaseA2AAgent):
    """A2A agent that manages research session persistence."""

    def __init__(self):
        super().__init__(port=8004)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="tracker-agent",
            version="1.0.0",
            description="Persists and retrieves research sessions, papers, and web sources.",
            skills=[
                "save_session",
                "list_sessions",
                "get_session",
                "delete_session",
            ],
        )

    async def handle_task(self, action: str, payload: dict) -> dict:
        # Ensure DB is ready
        await init_db()

        if action == "save_session":
            return await self._save_session(payload)
        elif action == "list_sessions":
            return await self._list_sessions(payload)
        elif action == "get_session":
            return await self._get_session(payload)
        elif action == "delete_session":
            return await self._delete_session(payload)
        else:
            return {"error": f"Unknown action: {action}"}

    async def _save_session(self, payload: dict) -> dict:
        try:
            session_id = await save_session(
                query=payload.get("query", ""),
                report=payload.get("report", ""),
                summary=payload.get("summary", ""),
                papers=payload.get("papers"),
                web_sources=payload.get("web_sources"),
            )
            return {"session_id": session_id}
        except Exception as e:
            return {"error": f"Failed to save session: {e}"}

    async def _list_sessions(self, payload: dict) -> dict:
        try:
            limit = payload.get("limit", 20)
            offset = payload.get("offset", 0)
            sessions = await list_sessions(limit=limit, offset=offset)
            return {"sessions": sessions}
        except Exception as e:
            return {"error": f"Failed to list sessions: {e}"}

    async def _get_session(self, payload: dict) -> dict:
        try:
            session_id = payload.get("session_id")
            if not session_id:
                return {"error": "session_id is required"}
            session = await get_session(int(session_id))
            if not session:
                return {"error": f"Session {session_id} not found"}
            return {"session": session}
        except Exception as e:
            return {"error": f"Failed to get session: {e}"}

    async def _delete_session(self, payload: dict) -> dict:
        try:
            session_id = payload.get("session_id")
            if not session_id:
                return {"error": "session_id is required"}
            deleted = await delete_session(int(session_id))
            return {"deleted": deleted}
        except Exception as e:
            return {"error": f"Failed to delete session: {e}"}


if __name__ == "__main__":
    agent = TrackerAgent()
    agent.run()
