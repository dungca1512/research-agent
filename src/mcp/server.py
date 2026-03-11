"""
Research Agent MCP Server with A2A Integration

Exposes research tools via MCP that orchestrate multiple A2A agents.

Usage:
    # Run MCP Gateway (after starting agents)
    python main.py mcp
    
Architecture:
    Claude → MCP Gateway → A2A → [Search Agent, Paper Agent, Synthesis Agent]
"""

from fastmcp import FastMCP
from src.research.backend import (
    PAPER_AGENT_URL,
    SEARCH_AGENT_URL,
    SYNTHESIS_AGENT_URL,
    call_agent_safe,
)
from src.research.deepagents_runner import run_deep_research

# Create MCP server
mcp = FastMCP("research-agent")


# ============= Web Search Tools =============

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information on any topic.
    Returns recent articles, blog posts, and web content.
    
    Args:
        query: Search query to look up
        max_results: Maximum number of results (default: 5)
    
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    result = await call_agent_safe(
        SEARCH_AGENT_URL,
        "web_search",
        {"query": query, "max_results": max_results}
    )
    
    if "error" in result:
        return f"Search failed: {result['error']}"
    
    output = []
    for i, r in enumerate(result.get("results", []), 1):
        output.append(f"{i}. **{r.get('title', 'N/A')}**")
        output.append(f"   URL: {r.get('url', 'N/A')}")
        output.append(f"   {r.get('content', '')[:200]}...")
        output.append("")
    
    return "\n".join(output) or "No results found"


# ============= ArXiv Tools =============

@mcp.tool()
async def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers and research publications.
    
    Args:
        query: Search query (e.g., "transformer attention mechanism")
        max_results: Maximum papers to return (default: 5)
    
    Returns:
        List of papers with titles, authors, and abstracts
    """
    result = await call_agent_safe(
        SEARCH_AGENT_URL,
        "arxiv_search",
        {"query": query, "max_results": max_results}
    )
    
    if "error" in result:
        return f"Search failed: {result['error']}"
    
    output = []
    for i, p in enumerate(result.get("papers", []), 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        
        output.append(f"{i}. **{p.get('title', 'N/A')}**")
        output.append(f"   ArXiv: {p.get('arxiv_id', 'N/A')} | {p.get('published', 'N/A')}")
        output.append(f"   Authors: {authors}")
        output.append(f"   Abstract: {p.get('summary', '')[:300]}...")
        output.append("")
    
    return "\n".join(output) or "No papers found"


@mcp.tool()
async def arxiv_get_paper(arxiv_id: str) -> str:
    """
    Get detailed information about a specific arXiv paper.
    
    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.00001")
    
    Returns:
        Detailed paper information including full abstract
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "get_paper",
        {"arxiv_id": arxiv_id}
    )
    
    if "error" in result:
        return f"Failed to get paper: {result['error']}"
    
    paper = result.get("paper", {})
    if not paper:
        return f"Paper not found: {arxiv_id}"
    
    authors = ", ".join(paper.get("authors", []))
    
    return f"""# {paper.get('title', 'N/A')}

**ArXiv ID:** {paper.get('arxiv_id', arxiv_id)}
**Published:** {paper.get('published', 'N/A')}
**Authors:** {authors}
**Categories:** {', '.join(paper.get('categories', []))}

## Abstract
{paper.get('summary', 'N/A')}

**PDF:** {paper.get('pdf_url', 'N/A')}
"""


# ============= Paper Parser Tools =============

@mcp.tool()
async def read_paper_pdf(pdf_url: str) -> str:
    """
    Download and extract text from a research paper PDF.
    
    Args:
        pdf_url: Direct URL to the PDF file
    
    Returns:
        Extracted text content from the paper
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "parse_pdf",
        {"pdf_url": pdf_url, "max_pages": 10}
    )
    
    if not result.get("success", False):
        return f"Failed to parse PDF: {result.get('error', 'Unknown error')}"
    
    content = result.get("content", "")[:8000]
    
    output = f"## Paper Content\n**Source:** {pdf_url}\n\n{content}"
    
    if len(result.get("content", "")) > 8000:
        output += "\n\n[... Content truncated ...]"
    
    return output


# ============= Deep Research Tool =============

@mcp.tool()
async def deep_research(query: str) -> str:
    """
    Run comprehensive research using multiple specialized agents.
    
    This orchestrates:
    1. Search Agent - finds web content and papers
    2. Paper Agent - analyzes papers (Semantic Scholar)
    3. Synthesis Agent - generates detailed report with citations
    
    Best for complex research questions. Generates reports with:
    - Executive Summary
    - Key Findings with inline citations
    - Detailed Analysis
    - References section
    
    Args:
        query: Research question or topic
    
    Returns:
        Comprehensive research report in Markdown (1000+ words)
    """
    return await run_deep_research(query)


# ============= Agent Status Tool =============

@mcp.tool()
async def check_agents_status() -> str:
    """
    Check the health status of all A2A agents.
    
    Returns:
        Status of each agent (healthy/offline)
    """
    import httpx
    
    agents = [
        ("Search Agent", SEARCH_AGENT_URL),
        ("Paper Agent", PAPER_AGENT_URL),
        ("Synthesis Agent", SYNTHESIS_AGENT_URL),
    ]
    
    output = ["## Agent Status\n"]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in agents:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    output.append(f"✅ **{name}** ({url}): Healthy")
                else:
                    output.append(f"⚠️ **{name}** ({url}): Unhealthy")
            except Exception:
                output.append(f"❌ **{name}** ({url}): Offline")
    
    return "\n".join(output)


# ============= Semantic Scholar Tools =============

@mcp.tool()
async def semantic_search(query: str, max_results: int = 10) -> str:
    """
    Search Semantic Scholar for papers with citation metrics.
    Better for finding influential papers with citation counts.
    
    Args:
        query: Search query
        max_results: Maximum papers (default: 10)
    
    Returns:
        Papers with citation counts and impact metrics
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "semantic_search",
        {"query": query, "max_results": max_results}
    )
    
    if "error" in result:
        return f"Search failed: {result['error']}"
    
    output = [f"## Semantic Scholar Results for: {query}\n"]
    for i, p in enumerate(result.get("papers", []), 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        
        output.append(f"### {i}. {p.get('title', 'N/A')}")
        output.append(f"**ID:** `{p.get('paper_id', 'N/A')}`")
        output.append(f"**Year:** {p.get('year', 'N/A')} | **Citations:** {p.get('citation_count', 0)} | **Influential:** {p.get('influential_citations', 0)}")
        output.append(f"**Authors:** {authors}")
        if p.get('fields'):
            output.append(f"**Fields:** {', '.join(p.get('fields', []))}")
        if p.get('abstract'):
            output.append(f"**Abstract:** {p.get('abstract', '')[:300]}...")
        if p.get('pdf_url'):
            output.append(f"**PDF:** {p.get('pdf_url')}")
        output.append("")
    
    return "\n".join(output) or "No papers found"


@mcp.tool()
async def get_paper_citations(paper_id: str, limit: int = 20) -> str:
    """
    Get papers that cite a specific paper (who cited this work).
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum citations to return
    
    Returns:
        List of citing papers with metadata
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "get_citations",
        {"paper_id": paper_id, "limit": limit}
    )
    
    if "error" in result:
        return f"Failed: {result['error']}"
    
    citations = result.get("citations", [])
    output = [f"## Papers Citing: {paper_id}\n**Total:** {len(citations)} citations\n"]
    
    for i, p in enumerate(citations, 1):
        authors = ", ".join(p.get("authors", [])[:2])
        output.append(f"{i}. **{p.get('title', 'N/A')}** ({p.get('year', 'N/A')})")
        output.append(f"   Authors: {authors} | Citations: {p.get('citation_count', 0)}")
    
    return "\n".join(output)


@mcp.tool()
async def get_paper_references(paper_id: str, limit: int = 20) -> str:
    """
    Get papers that a specific paper references (what this work cites).
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum references to return
    
    Returns:
        List of referenced papers
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "get_references",
        {"paper_id": paper_id, "limit": limit}
    )
    
    if "error" in result:
        return f"Failed: {result['error']}"
    
    references = result.get("references", [])
    output = [f"## References in Paper: {paper_id}\n**Total:** {len(references)} references\n"]
    
    for i, p in enumerate(references, 1):
        authors = ", ".join(p.get("authors", [])[:2])
        output.append(f"{i}. **{p.get('title', 'N/A')}** ({p.get('year', 'N/A')})")
        output.append(f"   Authors: {authors} | Citations: {p.get('citation_count', 0)}")
    
    return "\n".join(output)


# ============= Section Extraction Tools =============

@mcp.tool()
async def extract_paper_sections(pdf_url: str) -> str:
    """
    Extract specific sections from a paper PDF.
    Returns Abstract, Introduction, Methodology, Results, Conclusion.
    
    Args:
        pdf_url: URL to the PDF file
    
    Returns:
        Extracted sections in Markdown format
    """
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "extract_sections",
        {"pdf_url": pdf_url}
    )
    
    if "error" in result:
        return f"Failed: {result['error']}"
    
    output = ["## Extracted Sections\n"]
    
    if result.get("abstract"):
        output.append("### Abstract")
        output.append(result["abstract"][:1000])
        output.append("")
    
    if result.get("methodology"):
        output.append("### Methodology")
        output.append(result["methodology"][:2000])
        output.append("")
    
    if result.get("results"):
        output.append("### Results")
        output.append(result["results"][:2000])
        output.append("")
    
    if result.get("conclusion"):
        output.append("### Conclusion")
        output.append(result["conclusion"][:1000])
        output.append("")
    
    refs = result.get("references", [])
    if refs:
        output.append(f"### References ({len(refs)} found)")
        for i, ref in enumerate(refs[:10], 1):
            output.append(f"{i}. {ref[:200]}")
    
    return "\n".join(output)


@mcp.tool()
async def format_citation(
    title: str,
    authors: str,
    year: str,
    style: str = "apa"
) -> str:
    """
    Format a paper citation in different styles.
    
    Args:
        title: Paper title
        authors: Comma-separated author names
        year: Publication year
        style: Citation style - apa, mla, or bibtex
    
    Returns:
        Formatted citation string
    """
    paper = {
        "title": title,
        "authors": [a.strip() for a in authors.split(",")],
        "year": year
    }
    
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "format_citation",
        {"paper": paper, "style": style}
    )
    
    if "error" in result:
        return f"Failed: {result['error']}"
    
    return f"**{style.upper()} Citation:**\n```\n{result.get('citation', '')}\n```"


# ============= Resources =============

@mcp.resource("research://capabilities")
def get_capabilities() -> str:
    """List available research capabilities."""
    return """# Research Agent Capabilities (Multi-Agent A2A)

## Architecture
Claude → MCP Gateway → A2A Protocol → Specialized Agents

## Agents:
1. **Search Agent** (port 8001)
   - web_search, arxiv_search, decompose_query

2. **Paper Agent** (port 8002)
   - get_paper, parse_pdf, semantic_search
   - get_citations, get_references, extract_sections, format_citation

3. **Synthesis Agent** (port 8003)
   - synthesize, generate_report, compare_papers

## MCP Tools:
### Basic Search
- web_search, arxiv_search, arxiv_get_paper

### Semantic Scholar (NEW)
- semantic_search - Search with citation metrics
- get_paper_citations - Who cites this paper
- get_paper_references - What this paper cites

### Paper Analysis (NEW)
- extract_paper_sections - Get Abstract, Methods, Results, Conclusion
- format_citation - Format in APA, MLA, BibTeX

### Deep Research
- deep_research - DeepAgents-powered research orchestration
- read_paper_pdf, check_agents_status
"""


# Entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Agent MCP Server")
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default) or sse"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3000,
        help="Port for SSE mode (default: 3000)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE mode (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "sse":
        print(f"🚀 Starting MCP server in SSE mode on http://{args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # STDIO mode (default) - no print to avoid corrupting the protocol
        mcp.run(transport="stdio")
