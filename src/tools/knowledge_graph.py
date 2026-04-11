"""Knowledge Graph builder using NetworkX + pyvis for interactive visualization."""

import json
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

from src.storage.database import (
    get_session,
    save_knowledge_node,
    save_knowledge_edge,
    get_knowledge_graph,
)


# ─── Graph Construction ────────────────────────────────────────────────────

async def build_graph_from_session(session_id: int) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from a research session's data.

    Creates nodes for papers, authors, concepts (from web sources),
    and edges for authorship, citation context, and topical relationships.

    Args:
        session_id: Research session ID

    Returns:
        NetworkX DiGraph
    """
    session = await get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    G = nx.DiGraph()

    # Add query node as central hub
    query_label = session["query"][:80]
    G.add_node(
        f"query_{session_id}",
        label=query_label,
        node_type="query",
        color="#e74c3c",
        size=30,
    )

    # Add paper nodes + author nodes
    for p in session.get("papers", []):
        paper_id = f"paper_{p.get('arxiv_id', p['id'])}"
        G.add_node(
            paper_id,
            label=p["title"][:60],
            node_type="paper",
            color="#3498db",
            size=20,
            title=p["title"],  # tooltip
        )
        G.add_edge(f"query_{session_id}", paper_id, relation="researched")

        # Author nodes
        authors = p.get("authors", "")
        if isinstance(authors, str) and authors:
            author_list = [a.strip() for a in authors.split(",")]
        elif isinstance(authors, list):
            author_list = authors
        else:
            author_list = []

        for author in author_list[:5]:
            author_id = f"author_{author.replace(' ', '_')}"
            if not G.has_node(author_id):
                G.add_node(
                    author_id,
                    label=author,
                    node_type="author",
                    color="#2ecc71",
                    size=12,
                )
            G.add_edge(author_id, paper_id, relation="authored")

    # Add web source nodes
    for w in session.get("web_sources", []):
        source_id = f"web_{w['id']}"
        G.add_node(
            source_id,
            label=w.get("title", w["url"])[:50],
            node_type="web",
            color="#f39c12",
            size=14,
            title=w.get("url", ""),
        )
        G.add_edge(f"query_{session_id}", source_id, relation="referenced")

    return G


async def build_graph_from_db(session_id: int) -> nx.DiGraph:
    """
    Build a NetworkX graph from knowledge_nodes / knowledge_edges tables.

    Args:
        session_id: Research session ID

    Returns:
        NetworkX DiGraph
    """
    data = await get_knowledge_graph(session_id)
    G = nx.DiGraph()

    color_map = {
        "paper": "#3498db",
        "author": "#2ecc71",
        "concept": "#9b59b6",
        "method": "#e67e22",
        "dataset": "#1abc9c",
    }

    for node in data["nodes"]:
        meta = {}
        if node.get("metadata"):
            try:
                meta = json.loads(node["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        G.add_node(
            str(node["id"]),
            label=node["label"],
            node_type=node["node_type"],
            color=color_map.get(node["node_type"], "#95a5a6"),
            size=meta.get("size", 15),
            title=meta.get("title", node["label"]),
        )

    for edge in data["edges"]:
        G.add_edge(
            str(edge["source_node_id"]),
            str(edge["target_node_id"]),
            relation=edge["relation"],
            weight=edge.get("weight", 1.0),
        )

    return G


# ─── Persist Graph to DB ───────────────────────────────────────────────────

async def persist_graph(session_id: int, G: nx.DiGraph) -> dict:
    """
    Save a NetworkX graph to the knowledge_nodes / knowledge_edges tables.

    Returns:
        Summary with node_count and edge_count
    """
    id_map: dict[str, int] = {}

    for node_id, attrs in G.nodes(data=True):
        meta = json.dumps({
            k: v for k, v in attrs.items()
            if k not in ("label", "node_type")
        })
        db_id = await save_knowledge_node(
            session_id=session_id,
            label=attrs.get("label", str(node_id)),
            node_type=attrs.get("node_type", "concept"),
            metadata=meta,
        )
        id_map[node_id] = db_id

    edge_count = 0
    for src, dst, attrs in G.edges(data=True):
        if src in id_map and dst in id_map:
            await save_knowledge_edge(
                session_id=session_id,
                source_id=id_map[src],
                target_id=id_map[dst],
                relation=attrs.get("relation", "related"),
                weight=attrs.get("weight", 1.0),
            )
            edge_count += 1

    return {"node_count": len(id_map), "edge_count": edge_count}


# ─── Visualization ─────────────────────────────────────────────────────────

def render_graph_html(
    G: nx.DiGraph,
    output_path: str | Path = "data/knowledge_graph.html",
    height: str = "700px",
    width: str = "100%",
) -> str:
    """
    Render a NetworkX graph as an interactive HTML file using pyvis.

    Args:
        G: NetworkX graph
        output_path: Where to save the HTML file
        height: Canvas height
        width: Canvas width

    Returns:
        Absolute path to the generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)

    # Add nodes
    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs.get("label", str(node_id)),
            color=attrs.get("color", "#95a5a6"),
            size=attrs.get("size", 15),
            title=attrs.get("title", attrs.get("label", str(node_id))),
            shape="dot",
        )

    # Add edges
    for src, dst, attrs in G.edges(data=True):
        net.add_edge(
            src,
            dst,
            title=attrs.get("relation", ""),
            width=attrs.get("weight", 1.0),
        )

    net.save_graph(str(output_path))
    return str(output_path.resolve())


def get_graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Return basic statistics about a knowledge graph."""
    type_counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": type_counts,
        "density": round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0,
    }
