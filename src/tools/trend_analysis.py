"""Trend analysis for research topics using pandas + plotly."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.storage.database import list_sessions, get_db
from src.tools.semantic_scholar import search_semantic_scholar


# ─── Session-based Trends ──────────────────────────────────────────────────

async def session_trends() -> dict[str, Any]:
    """
    Analyze trends across all research sessions.

    Returns:
        Dict with summary stats and paths to generated charts.
    """
    sessions = await list_sessions(limit=500)
    if not sessions:
        return {"error": "No research sessions found"}

    df = pd.DataFrame(sessions)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date

    # Sessions over time
    daily = df.groupby("date").size().reset_index(name="count")

    out_dir = Path("data/trends")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = px.bar(
        daily, x="date", y="count",
        title="Research Sessions Over Time",
        labels={"date": "Date", "count": "Sessions"},
    )
    sessions_chart = str(out_dir / "sessions_over_time.html")
    fig.write_html(sessions_chart)

    # Papers per session
    fig2 = px.bar(
        df, x="id", y="paper_count",
        title="Papers Collected Per Session",
        labels={"id": "Session #", "paper_count": "Papers"},
    )
    papers_chart = str(out_dir / "papers_per_session.html")
    fig2.write_html(papers_chart)

    return {
        "total_sessions": len(df),
        "total_papers": int(df["paper_count"].sum()),
        "total_web_sources": int(df["web_count"].sum()),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "charts": [sessions_chart, papers_chart],
    }


# ─── Publication Trends (Semantic Scholar) ─────────────────────────────────

async def publication_trends(
    query: str,
    max_papers: int = 50,
) -> dict[str, Any]:
    """
    Analyze publication trends for a topic using Semantic Scholar data.

    Args:
        query: Research topic to analyze
        max_papers: Number of papers to fetch (max 100)

    Returns:
        Dict with trend data and chart paths.
    """
    papers = await search_semantic_scholar(query, min(max_papers, 100))
    if not papers:
        return {"error": f"No papers found for: {query}"}

    df = pd.DataFrame(papers)

    # Filter to papers with year data
    df = df[df["year"].notna() & (df["year"] > 0)]
    if df.empty:
        return {"error": "No papers with valid year data"}

    df["year"] = df["year"].astype(int)

    out_dir = Path("data/trends")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Publications per year
    yearly = df.groupby("year").size().reset_index(name="count")
    fig = px.bar(
        yearly, x="year", y="count",
        title=f"Publications Per Year: {query}",
        labels={"year": "Year", "count": "Papers"},
    )
    yearly_chart = str(out_dir / "publications_per_year.html")
    fig.write_html(yearly_chart)

    # Citation distribution
    if "citation_count" in df.columns:
        df_sorted = df.nlargest(20, "citation_count")
        fig2 = px.bar(
            df_sorted,
            x="title",
            y="citation_count",
            title=f"Top Cited Papers: {query}",
            labels={"title": "Paper", "citation_count": "Citations"},
        )
        fig2.update_xaxes(tickangle=45)
        citations_chart = str(out_dir / "top_cited.html")
        fig2.write_html(citations_chart)
    else:
        citations_chart = None

    charts = [yearly_chart]
    if citations_chart:
        charts.append(citations_chart)

    # Summary stats
    stats = {
        "query": query,
        "papers_analyzed": len(df),
        "year_range": f"{int(df['year'].min())} - {int(df['year'].max())}",
        "peak_year": int(yearly.loc[yearly["count"].idxmax(), "year"]),
        "peak_count": int(yearly["count"].max()),
        "charts": charts,
    }

    if "citation_count" in df.columns:
        stats["total_citations"] = int(df["citation_count"].sum())
        stats["avg_citations"] = round(df["citation_count"].mean(), 1)
        top = df.nlargest(3, "citation_count")
        stats["top_papers"] = [
            {"title": r["title"], "citations": int(r["citation_count"]), "year": int(r["year"])}
            for _, r in top.iterrows()
        ]

    return stats


def format_trends_markdown(result: dict) -> str:
    """Format trend analysis result as Markdown."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = []

    if "query" in result:
        lines.append(f"# Publication Trends: {result['query']}\n")
        lines.append(f"- **Papers analyzed:** {result['papers_analyzed']}")
        lines.append(f"- **Year range:** {result['year_range']}")
        lines.append(f"- **Peak year:** {result['peak_year']} ({result['peak_count']} papers)")

        if "total_citations" in result:
            lines.append(f"- **Total citations:** {result['total_citations']}")
            lines.append(f"- **Avg citations:** {result['avg_citations']}")

        if "top_papers" in result:
            lines.append("\n## Top Cited Papers\n")
            for i, p in enumerate(result["top_papers"], 1):
                lines.append(f"{i}. **{p['title']}** ({p['year']}) — {p['citations']} citations")
    else:
        lines.append("# Research Session Trends\n")
        lines.append(f"- **Total sessions:** {result['total_sessions']}")
        lines.append(f"- **Total papers:** {result['total_papers']}")
        lines.append(f"- **Total web sources:** {result['total_web_sources']}")
        lines.append(f"- **Date range:** {result['date_range']}")

    if result.get("charts"):
        lines.append("\n## Charts\n")
        for chart in result["charts"]:
            lines.append(f"- `{chart}`")

    return "\n".join(lines)
