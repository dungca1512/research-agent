"""LanceDB vector store for RAG — index and query research documents."""

import lancedb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_config

# Default vector store path
VECTOR_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "vectors"

# Text splitter for chunking documents
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get Google embedding model."""
    config = get_config()
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=config.google_api_key,
    )


def get_vector_db(db_path: Path | None = None) -> lancedb.DBConnection:
    """Get a LanceDB connection."""
    path = db_path or VECTOR_DB_PATH
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


async def index_documents(
    session_id: int,
    documents: list[dict],
    table_name: str | None = None,
) -> int:
    """
    Chunk and index documents into the vector store.

    Args:
        session_id: Research session ID
        documents: List of dicts with 'text', 'title', 'source' keys
        table_name: Optional table name (default: session_{id})

    Returns:
        Number of chunks indexed
    """
    import asyncio

    table = table_name or f"session_{session_id}"
    embeddings = _get_embeddings()

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue
        chunks = _splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "session_id": session_id,
                "chunk_index": i,
            })

    if not all_chunks:
        return 0

    # Embed chunks
    texts = [c["text"] for c in all_chunks]
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

    # Prepare records for LanceDB
    records = []
    for chunk, vector in zip(all_chunks, vectors):
        records.append({
            "vector": vector,
            "text": chunk["text"],
            "title": chunk["title"],
            "source": chunk["source"],
            "session_id": chunk["session_id"],
            "chunk_index": chunk["chunk_index"],
        })

    # Store in LanceDB
    db = get_vector_db()
    if table in db.table_names():
        tbl = db.open_table(table)
        tbl.add(records)
    else:
        db.create_table(table, records)

    return len(records)


async def query_documents(
    session_id: int,
    question: str,
    top_k: int = 5,
    table_name: str | None = None,
) -> list[dict]:
    """
    Query the vector store for relevant document chunks.

    Args:
        session_id: Research session ID
        question: Query string
        top_k: Number of results to return
        table_name: Optional table name

    Returns:
        List of matching chunks with text, title, source, and score
    """
    import asyncio

    table = table_name or f"session_{session_id}"
    db = get_vector_db()

    if table not in db.table_names():
        return []

    embeddings = _get_embeddings()
    query_vector = await asyncio.to_thread(embeddings.embed_query, question)

    tbl = db.open_table(table)
    results = (
        tbl.search(query_vector)
        .limit(top_k)
        .to_pandas()
    )

    return [
        {
            "text": row["text"],
            "title": row["title"],
            "source": row["source"],
            "score": row.get("_distance", 0.0),
        }
        for _, row in results.iterrows()
    ]


async def delete_session_vectors(session_id: int, table_name: str | None = None) -> bool:
    """Delete all vectors for a session."""
    table = table_name or f"session_{session_id}"
    db = get_vector_db()

    if table in db.table_names():
        db.drop_table(table)
        return True
    return False
