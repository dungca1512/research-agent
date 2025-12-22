"""Web search tool using Tavily and DuckDuckGo."""

from typing import Optional
from langchain_core.tools import tool
from src.config import get_config


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily API with DuckDuckGo fallback.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and content
    """
    config = get_config()
    
    # Try Tavily first
    if config.has_tavily_api:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.tavily_api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True
            )
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "source": "tavily"
                })
            return results
        except Exception as e:
            print(f"Tavily search failed: {e}, falling back to DuckDuckGo")
    
    # Fallback to DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "source": "duckduckgo"
                })
            return results
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return []


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information on a given topic.
    Use this tool to find recent articles, blog posts, and general web content.
    
    Args:
        query: The search query to look up on the web
        
    Returns:
        A formatted string containing the search results
    """
    results = search_web(query)
    
    if not results:
        return f"No results found for query: {query}"
    
    output = f"## Web Search Results for: {query}\n\n"
    for i, result in enumerate(results, 1):
        output += f"### {i}. {result['title']}\n"
        output += f"**URL:** {result['url']}\n"
        output += f"{result['content']}\n\n"
    
    return output
