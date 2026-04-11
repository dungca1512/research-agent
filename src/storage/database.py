"""Async SQLite database for persisting research sessions, papers, and knowledge graph."""

import json
import aiosqlite
from pathlib import Path
from datetime import datetime

# Default database path
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "research.db"


async def get_db(db_path: Path | None = None) -> aiosqlite.Connection:
    """Get a database connection with WAL mode enabled."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db(db_path: Path | None = None) -> None:
    """Initialize database schema. Safe to call multiple times."""
    db = await get_db(db_path)
    try:
        await db.executescript("""
            -- Research sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                summary TEXT,
                report TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Papers collected during research
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                arxiv_id TEXT,
                doi TEXT,
                abstract TEXT,
                pdf_url TEXT,
                full_text TEXT,
                source TEXT DEFAULT 'arxiv',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Web sources collected during research
            CREATE TABLE IF NOT EXISTS web_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                url TEXT NOT NULL,
                title TEXT,
                snippet TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Knowledge graph nodes
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                node_type TEXT NOT NULL,  -- paper, author, concept, method, dataset
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Knowledge graph edges
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
                source_node_id INTEGER NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
                target_node_id INTEGER NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,  -- cites, uses, proposes, authored_by, etc.
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_papers_session ON papers(session_id);
            CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_web_sources_session ON web_sources(session_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_session ON knowledge_nodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_knowledge_edges_session ON knowledge_edges(session_id);
        """)
        await db.commit()
    finally:
        await db.close()


# ─── Session CRUD ───────────────────────────────────────────────────────────

async def save_session(
    query: str,
    report: str = "",
    summary: str = "",
    papers: list[dict] | None = None,
    web_sources: list[dict] | None = None,
) -> int:
    """Save a research session and return its ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO sessions (query, report, summary) VALUES (?, ?, ?)",
            (query, report, summary),
        )
        session_id = cursor.lastrowid

        if papers:
            await db.executemany(
                """INSERT INTO papers
                   (session_id, title, authors, year, arxiv_id, abstract, pdf_url, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        session_id,
                        p.get("title", ""),
                        json.dumps(p["authors"]) if isinstance(p.get("authors"), list) else p.get("authors", ""),
                        p.get("year"),
                        p.get("arxiv_id", p.get("id", "")),
                        p.get("abstract", p.get("summary", "")),
                        p.get("pdf_url", ""),
                        p.get("source", "arxiv"),
                    )
                    for p in papers
                ],
            )

        if web_sources:
            await db.executemany(
                """INSERT INTO web_sources (session_id, url, title, snippet)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        session_id,
                        s.get("url", ""),
                        s.get("title", ""),
                        s.get("snippet", s.get("content", "")),
                    )
                    for s in web_sources
                ],
            )

        await db.commit()
        return session_id
    finally:
        await db.close()


async def list_sessions(limit: int = 20, offset: int = 0) -> list[dict]:
    """List research sessions, most recent first."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT s.id, s.query, s.summary, s.created_at,
                      COUNT(DISTINCT p.id) as paper_count,
                      COUNT(DISTINCT w.id) as web_count
               FROM sessions s
               LEFT JOIN papers p ON p.session_id = s.id
               LEFT JOIN web_sources w ON w.session_id = s.id
               GROUP BY s.id
               ORDER BY s.created_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_session(session_id: int) -> dict | None:
    """Get full session details including papers and web sources."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        session = await cursor.fetchone()
        if not session:
            return None

        session_dict = dict(session)

        cursor = await db.execute(
            "SELECT * FROM papers WHERE session_id = ? ORDER BY year DESC",
            (session_id,),
        )
        session_dict["papers"] = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT * FROM web_sources WHERE session_id = ?", (session_id,)
        )
        session_dict["web_sources"] = [dict(r) for r in await cursor.fetchall()]

        return session_dict
    finally:
        await db.close()


async def delete_session(session_id: int) -> bool:
    """Delete a research session and all related data."""
    db = await get_db()
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ─── Knowledge Graph CRUD ───────────────────────────────────────────────────

async def save_knowledge_node(
    session_id: int, label: str, node_type: str, metadata: str = ""
) -> int:
    """Save a knowledge graph node and return its ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO knowledge_nodes (session_id, label, node_type, metadata)
               VALUES (?, ?, ?, ?)""",
            (session_id, label, node_type, metadata),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def save_knowledge_edge(
    session_id: int, source_id: int, target_id: int, relation: str, weight: float = 1.0
) -> int:
    """Save a knowledge graph edge and return its ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO knowledge_edges (session_id, source_node_id, target_node_id, relation, weight)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, source_id, target_id, relation, weight),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_knowledge_graph(session_id: int) -> dict:
    """Get all nodes and edges for a session's knowledge graph."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM knowledge_nodes WHERE session_id = ?", (session_id,)
        )
        nodes = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT * FROM knowledge_edges WHERE session_id = ?", (session_id,)
        )
        edges = [dict(r) for r in await cursor.fetchall()]

        return {"nodes": nodes, "edges": edges}
    finally:
        await db.close()
