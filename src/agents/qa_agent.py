"""
QA Agent - Interactive Q&A over indexed research sessions using RAG.

Skills:
    - index_session: Index a session's papers and web sources into vector store
    - ask: Answer a question using retrieved context from indexed documents
    - list_indexed: List which sessions are indexed

Run:
    python -m src.agents.qa_agent
"""

from typing import Any

from src.a2a.base_agent import BaseA2AAgent, AgentCard
from src.config import get_config
from src.storage.database import init_db, get_session
from src.storage.vector_store import (
    index_documents,
    query_documents,
    delete_session_vectors,
    get_vector_db,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class QAAgent(BaseA2AAgent):
    """Agent for RAG-based Q&A over research sessions."""

    def __init__(self, port: int = 8005):
        super().__init__(port=port)
        self.config = get_config()
        self.llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self.llm is None:
            if not self.config.has_google_api:
                raise ValueError("GOOGLE_API_KEY is required for QA")
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=self.config.google_api_key,
                temperature=0.2,
            )
        return self.llm

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="qa-agent",
            version="1.0.0",
            description="RAG-based Q&A agent for querying indexed research sessions",
            skills=["index_session", "ask", "list_indexed"],
        )

    async def handle_task(self, action: str, payload: dict) -> Any:
        await init_db()

        if action == "index_session":
            return await self._index_session(payload)
        elif action == "ask":
            return await self._ask(payload)
        elif action == "list_indexed":
            return await self._list_indexed(payload)
        else:
            return {"error": f"Unknown action: {action}"}

    async def _index_session(self, payload: dict) -> dict:
        """Index a research session's content into the vector store."""
        session_id = payload.get("session_id")
        if not session_id:
            return {"error": "session_id is required"}

        session = await get_session(int(session_id))
        if not session:
            return {"error": f"Session {session_id} not found"}

        documents = []

        # Index the report
        if session.get("report"):
            documents.append({
                "text": session["report"],
                "title": f"Research Report: {session['query']}",
                "source": "report",
            })

        # Index papers
        for p in session.get("papers", []):
            text = p.get("abstract", p.get("full_text", ""))
            if text:
                documents.append({
                    "text": text,
                    "title": p.get("title", "Unknown Paper"),
                    "source": f"paper:{p.get('arxiv_id', p.get('id', ''))}",
                })

        # Index web sources
        for w in session.get("web_sources", []):
            text = w.get("snippet", "")
            if text:
                documents.append({
                    "text": text,
                    "title": w.get("title", "Web Source"),
                    "source": f"web:{w.get('url', '')}",
                })

        if not documents:
            return {"error": "No content to index in this session"}

        try:
            chunk_count = await index_documents(int(session_id), documents)
            return {
                "session_id": session_id,
                "documents_indexed": len(documents),
                "chunks_created": chunk_count,
            }
        except Exception as e:
            return {"error": f"Indexing failed: {e}"}

    async def _ask(self, payload: dict) -> dict:
        """Answer a question using RAG over indexed session(s)."""
        question = payload.get("question", "")
        session_id = payload.get("session_id")

        if not question:
            return {"error": "question is required"}
        if not session_id:
            return {"error": "session_id is required"}

        # Retrieve relevant chunks
        try:
            chunks = await query_documents(
                int(session_id), question, top_k=payload.get("top_k", 5)
            )
        except Exception as e:
            return {"error": f"Vector search failed: {e}"}

        if not chunks:
            return {
                "answer": "No relevant information found. Make sure the session is indexed first.",
                "sources": [],
            }

        # Build context
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] ({c['source']}) {c['title']}\n{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the following question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Use inline citations like [1], [2] to reference the context chunks.

## Context
{context}

## Question
{question}

## Answer"""

        try:
            response = await self._get_llm().ainvoke([HumanMessage(content=prompt)])
            return {
                "question": question,
                "answer": response.content,
                "sources": [
                    {"title": c["title"], "source": c["source"]}
                    for c in chunks
                ],
                "chunks_used": len(chunks),
            }
        except Exception as e:
            return {"error": f"LLM generation failed: {e}"}

    async def _list_indexed(self, _payload: dict) -> dict:
        """List which sessions have been indexed in the vector store."""
        db = get_vector_db()
        tables = db.table_names()
        sessions = []
        for t in tables:
            if t.startswith("session_"):
                try:
                    sid = int(t.replace("session_", ""))
                    tbl = db.open_table(t)
                    sessions.append({
                        "session_id": sid,
                        "table": t,
                        "rows": len(tbl),
                    })
                except (ValueError, Exception):
                    continue
        return {"indexed_sessions": sessions}


if __name__ == "__main__":
    agent = QAAgent(port=8005)
    agent.run()
