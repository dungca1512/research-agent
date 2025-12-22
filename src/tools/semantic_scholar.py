"""Semantic Scholar API integration for academic search."""

import httpx
from typing import Optional
from langchain_core.tools import tool


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


async def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    fields: list[str] = None
) -> list[dict]:
    """
    Search Semantic Scholar for academic papers.
    
    Args:
        query: Search query
        max_results: Maximum papers to return
        fields: Paper fields to include
        
    Returns:
        List of paper metadata
    """
    if fields is None:
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "influentialCitationCount", 
            "openAccessPdf", "fieldsOfStudy", "publicationVenue"
        ]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": ",".join(fields)
                }
            )
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for paper in data.get("data", []):
                papers.append({
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "year": paper.get("year"),
                    "authors": [a.get("name") for a in paper.get("authors", [])],
                    "citation_count": paper.get("citationCount", 0),
                    "influential_citations": paper.get("influentialCitationCount", 0),
                    "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                    "fields": paper.get("fieldsOfStudy", []),
                    "venue": paper.get("publicationVenue", {}).get("name") if paper.get("publicationVenue") else None,
                })
            return papers
            
    except Exception as e:
        print(f"Semantic Scholar search failed: {e}")
        return []


async def get_paper_details(paper_id: str) -> Optional[dict]:
    """
    Get detailed information about a paper by Semantic Scholar ID.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Detailed paper information
    """
    fields = [
        "paperId", "title", "abstract", "year", "authors",
        "citationCount", "influentialCitationCount", "references",
        "citations", "openAccessPdf", "fieldsOfStudy", "tldr"
    ]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                params={"fields": ",".join(fields)}
            )
            response.raise_for_status()
            paper = response.json()
            
            return {
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "citation_count": paper.get("citationCount", 0),
                "influential_citations": paper.get("influentialCitationCount", 0),
                "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                "fields": paper.get("fieldsOfStudy", []),
                "tldr": paper.get("tldr", {}).get("text") if paper.get("tldr") else None,
                "reference_count": len(paper.get("references", [])),
                "citation_papers": len(paper.get("citations", [])),
            }
            
    except Exception as e:
        print(f"Get paper details failed: {e}")
        return None


async def get_paper_citations(paper_id: str, limit: int = 20) -> list[dict]:
    """
    Get papers that cite this paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum citations to return
        
    Returns:
        List of citing papers
    """
    fields = "title,authors,year,citationCount"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                params={"fields": fields, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            
            citations = []
            for item in data.get("data", []):
                paper = item.get("citingPaper", {})
                citations.append({
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": [a.get("name") for a in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                })
            return citations
            
    except Exception as e:
        print(f"Get citations failed: {e}")
        return []


async def get_paper_references(paper_id: str, limit: int = 20) -> list[dict]:
    """
    Get papers that this paper references.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum references to return
        
    Returns:
        List of referenced papers
    """
    fields = "title,authors,year,citationCount"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                params={"fields": fields, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            
            references = []
            for item in data.get("data", []):
                paper = item.get("citedPaper", {})
                if paper.get("title"):  # Skip null references
                    references.append({
                        "paper_id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "authors": [a.get("name") for a in paper.get("authors", [])],
                        "year": paper.get("year"),
                        "citation_count": paper.get("citationCount", 0),
                    })
            return references
            
    except Exception as e:
        print(f"Get references failed: {e}")
        return []


def format_citation(paper: dict, style: str = "apa") -> str:
    """
    Format a paper as a citation.
    
    Args:
        paper: Paper metadata dict
        style: Citation style (apa, mla, bibtex)
        
    Returns:
        Formatted citation string
    """
    authors = paper.get("authors", [])
    title = paper.get("title", "Untitled")
    year = paper.get("year", "n.d.")
    
    if style == "apa":
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{authors[0]} et al."
        else:
            author_str = "Unknown Author"
        return f"{author_str} ({year}). {title}."
    
    elif style == "mla":
        if authors:
            author_str = authors[0]
            if len(authors) > 1:
                author_str += ", et al."
        else:
            author_str = "Unknown Author"
        return f'{author_str}. "{title}." {year}.'
    
    elif style == "bibtex":
        first_author = authors[0].split()[-1].lower() if authors else "unknown"
        key = f"{first_author}{year}"
        author_str = " and ".join(authors) if authors else "Unknown"
        return f"""@article{{{key},
  title={{{title}}},
  author={{{author_str}}},
  year={{{year}}}
}}"""
    
    return f"{', '.join(authors[:3])}. ({year}). {title}."
