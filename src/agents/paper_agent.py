"""
Paper Agent - Specialized A2A agent for paper analysis.

Skills:
    - get_paper: Get paper details by arXiv ID
    - parse_pdf: Extract text from PDF
    - extract_key_info: Extract key findings from paper text

Run:
    python -m src.agents.paper_agent
"""

from typing import Any
from src.a2a.base_agent import BaseA2AAgent, AgentCard
from src.tools.arxiv_search import get_paper_by_id
from src.tools.paper_parser import parse_paper
from src.tools.semantic_scholar import (
    search_semantic_scholar, get_paper_details, 
    get_paper_citations, get_paper_references, format_citation
)
from src.tools.section_extractor import (
    extract_paper_sections, extract_methodology,
    extract_results, extract_conclusion, extract_references_list
)
from src.tools.paper_comparison import extract_paper_structured
from src.config import get_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class PaperAgent(BaseA2AAgent):
    """Agent specialized in paper analysis and parsing."""

    def __init__(self, port: int = 8002):
        super().__init__(port=port)
        self.config = get_config()
        self.llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Create the LLM lazily so non-LLM paper tools can still run."""
        if self.llm is None:
            if not self.config.has_google_api:
                raise ValueError("GOOGLE_API_KEY is required for extract_key_info")
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=self.config.google_api_key,
                temperature=0.1
            )
        return self.llm
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="paper-agent",
            version="2.0.0",
            description="Specialized agent for paper analysis, citations, and PDF parsing",
            skills=[
                "get_paper", "parse_pdf", "extract_key_info",
                "semantic_search", "get_citations", "get_references",
                "extract_sections", "format_citation", "extract_structured"
            ]
        )
    
    async def handle_task(self, action: str, payload: dict) -> Any:
        """Route task to appropriate skill."""
        
        if action == "get_paper":
            return await self._get_paper(payload)
        elif action == "parse_pdf":
            return await self._parse_pdf(payload)
        elif action == "extract_key_info":
            return await self._extract_key_info(payload)
        elif action == "semantic_search":
            return await self._semantic_search(payload)
        elif action == "get_citations":
            return await self._get_citations(payload)
        elif action == "get_references":
            return await self._get_references(payload)
        elif action == "extract_sections":
            return await self._extract_sections(payload)
        elif action == "format_citation":
            return await self._format_citation(payload)
        elif action == "extract_structured":
            return await self._extract_structured(payload)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _get_paper(self, payload: dict) -> dict:
        """Get paper details by arXiv ID."""
        arxiv_id = payload.get("arxiv_id", "")
        paper = get_paper_by_id(arxiv_id)
        if not paper:
            return {"error": f"Paper not found: {arxiv_id}"}
        return {"paper": paper}
    
    async def _parse_pdf(self, payload: dict) -> dict:
        """Parse PDF and extract text."""
        pdf_url = payload.get("pdf_url", "")
        max_pages = payload.get("max_pages", 10)
        result = parse_paper(pdf_url, max_pages)
        return result
    
    async def _semantic_search(self, payload: dict) -> dict:
        """Search Semantic Scholar."""
        query = payload.get("query", "")
        max_results = payload.get("max_results", 10)
        papers = await search_semantic_scholar(query, max_results)
        return {"query": query, "papers": papers, "count": len(papers)}
    
    async def _get_citations(self, payload: dict) -> dict:
        """Get papers that cite this paper."""
        paper_id = payload.get("paper_id", "")
        limit = payload.get("limit", 20)
        citations = await get_paper_citations(paper_id, limit)
        return {"paper_id": paper_id, "citations": citations, "count": len(citations)}
    
    async def _get_references(self, payload: dict) -> dict:
        """Get papers that this paper references."""
        paper_id = payload.get("paper_id", "")
        limit = payload.get("limit", 20)
        references = await get_paper_references(paper_id, limit)
        return {"paper_id": paper_id, "references": references, "count": len(references)}
    
    async def _extract_sections(self, payload: dict) -> dict:
        """Extract sections from PDF."""
        pdf_url = payload.get("pdf_url", "")
        sections = await extract_paper_sections(pdf_url)
        return sections
    
    async def _format_citation(self, payload: dict) -> dict:
        """Format paper as citation."""
        paper = payload.get("paper", {})
        style = payload.get("style", "apa")
        citation = format_citation(paper, style)
        return {"style": style, "citation": citation}
    
    async def _extract_structured(self, payload: dict) -> dict:
        """Extract structured paper info using Pydantic schema."""
        title = payload.get("title", "Unknown")
        content = payload.get("content", payload.get("summary", ""))
        if not content:
            return {"error": "content or summary is required"}
        try:
            return await extract_paper_structured(title, content)
        except Exception as e:
            return {"error": f"Structured extraction failed: {e}"}

    async def _extract_key_info(self, payload: dict) -> dict:
        """Extract key findings from paper content."""
        content = payload.get("content", "")
        paper_title = payload.get("title", "Unknown")
        
        prompt = f"""Analyze this academic paper and extract key information.

Paper Title: {paper_title}

Content:
{content[:6000]}

Extract and return:
1. Main contribution/thesis
2. Key methodology
3. Important findings (3-5 bullet points)
4. Limitations mentioned
5. Future work suggested

Be concise and factual."""
        
        response = await self._get_llm().ainvoke([HumanMessage(content=prompt)])
        
        return {
            "title": paper_title,
            "analysis": response.content
        }


# Entry point
if __name__ == "__main__":
    agent = PaperAgent(port=8002)
    agent.run()
