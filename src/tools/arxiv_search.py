"""ArXiv paper search tool."""

from typing import Optional
from langchain_core.tools import tool
import arxiv
from src.config import get_config


def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> list[dict]:
    """
    Search arXiv for academic papers.
    
    Args:
        query: Search query (supports arXiv query syntax)
        max_results: Maximum number of papers to return
        sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
        
    Returns:
        List of paper metadata dictionaries
    """
    config = get_config()
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=min(max_results, config.max_papers),
            sort_by=sort_by
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "arxiv_id": result.entry_id.split("/")[-1],
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "primary_category": result.primary_category,
            })
        
        return papers
        
    except Exception as e:
        print(f"ArXiv search failed: {e}")
        return []


def get_paper_by_id(arxiv_id: str) -> Optional[dict]:
    """
    Get a specific paper by its arXiv ID.
    
    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.00001")
        
    Returns:
        Paper metadata dictionary or None if not found
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        
        for result in client.results(search):
            return {
                "title": result.title,
                "arxiv_id": arxiv_id,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            }
        
        return None
        
    except Exception as e:
        print(f"Failed to get paper {arxiv_id}: {e}")
        return None


@tool
def arxiv_search_tool(query: str) -> str:
    """
    Search arXiv for academic papers and research publications.
    Use this tool to find peer-reviewed papers, preprints, and scholarly research.
    
    Args:
        query: The search query for finding papers (e.g., "transformer attention mechanism" or "LLM agents")
        
    Returns:
        A formatted string containing paper titles, authors, and abstracts
    """
    papers = search_arxiv(query)
    
    if not papers:
        return f"No papers found for query: {query}"
    
    output = f"## ArXiv Papers for: {query}\n\n"
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += " et al."
        
        output += f"### {i}. {paper['title']}\n"
        output += f"**ArXiv ID:** {paper['arxiv_id']} | **Published:** {paper['published']}\n"
        output += f"**Authors:** {authors_str}\n"
        output += f"**Categories:** {', '.join(paper['categories'][:3])}\n\n"
        output += f"**Abstract:** {paper['summary'][:500]}...\n\n"
        output += f"**PDF:** {paper['pdf_url']}\n\n"
        output += "---\n\n"
    
    return output
