"""
Search Agent - Specialized A2A agent for web and academic search.

Skills:
    - web_search: Search the web using Tavily/DuckDuckGo
    - arxiv_search: Search arXiv for academic papers
    - decompose_query: Break down complex queries

Run:
    python -m src.agents.search_agent
"""

from typing import Any
from src.a2a.base_agent import BaseA2AAgent, AgentCard
from src.tools.web_search import search_web
from src.tools.arxiv_search import search_arxiv
from src.config import get_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class SearchAgent(BaseA2AAgent):
    """Agent specialized in searching web and academic sources."""
    
    def __init__(self, port: int = 8001):
        super().__init__(port=port)
        config = get_config()
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=config.google_api_key,
            temperature=0.1
        )
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="search-agent",
            version="1.0.0",
            description="Specialized agent for web and academic paper search",
            skills=["web_search", "arxiv_search", "decompose_query"]
        )
    
    async def handle_task(self, action: str, payload: dict) -> Any:
        """Route task to appropriate skill."""
        
        if action == "web_search":
            return await self._web_search(payload)
        elif action == "arxiv_search":
            return await self._arxiv_search(payload)
        elif action == "decompose_query":
            return await self._decompose_query(payload)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _web_search(self, payload: dict) -> dict:
        """Search the web."""
        query = payload.get("query", "")
        max_results = payload.get("max_results", 5)
        
        results = search_web(query, max_results)
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    async def _arxiv_search(self, payload: dict) -> dict:
        """Search arXiv papers."""
        query = payload.get("query", "")
        max_results = payload.get("max_results", 5)
        
        papers = search_arxiv(query, max_results)
        
        return {
            "query": query,
            "papers": papers,
            "count": len(papers)
        }
    
    async def _decompose_query(self, payload: dict) -> dict:
        """Decompose a complex query into sub-queries."""
        query = payload.get("query", "")
        
        prompt = f"""Break down this research query into 2-3 focused search queries.
Consider different aspects, academic vs practical, general vs specific.

Query: {query}

Return ONLY the queries, one per line."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        
        # Always include original
        if query not in queries:
            queries.insert(0, query)
        
        return {
            "original_query": query,
            "sub_queries": queries
        }


# Entry point
if __name__ == "__main__":
    agent = SearchAgent(port=8001)
    agent.run()
