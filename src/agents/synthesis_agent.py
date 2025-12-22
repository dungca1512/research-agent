"""
Synthesis Agent - Specialized A2A agent for report generation.

Skills:
    - synthesize: Combine findings from multiple sources
    - generate_report: Create structured research report
    - compare_papers: Compare multiple papers

Run:
    python -m src.agents.synthesis_agent
"""

from typing import Any
from src.a2a.base_agent import BaseA2AAgent, AgentCard
from src.config import get_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class SynthesisAgent(BaseA2AAgent):
    """Agent specialized in synthesizing research and generating reports."""
    
    def __init__(self, port: int = 8003):
        super().__init__(port=port)
        config = get_config()
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=config.google_api_key,
            temperature=0.3
        )
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="synthesis-agent",
            version="2.0.0",
            description="Specialized agent for comprehensive research synthesis and detailed report generation",
            skills=["synthesize", "generate_report", "compare_papers"]
        )
    
    async def handle_task(self, action: str, payload: dict) -> Any:
        """Route task to appropriate skill."""
        
        if action == "synthesize":
            return await self._synthesize(payload)
        elif action == "generate_report":
            return await self._generate_report(payload)
        elif action == "compare_papers":
            return await self._compare_papers(payload)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _synthesize(self, payload: dict) -> dict:
        """Synthesize findings from multiple sources."""
        query = payload.get("query", "")
        web_results = payload.get("web_results", [])
        papers = payload.get("papers", [])
        
        # Build numbered source list for citations
        all_sources = []
        sources_text = "# Available Sources\n\n"
        
        sources_text += "## Web Sources\n"
        for i, r in enumerate(web_results[:8], 1):
            source_id = len(all_sources) + 1
            all_sources.append({
                "id": source_id,
                "type": "web",
                "title": r.get('title', 'N/A'),
                "url": r.get('url', ''),
                "content": r.get('content', '')
            })
            sources_text += f"[{source_id}] **{r.get('title', 'N/A')}**\n"
            sources_text += f"    URL: {r.get('url', 'N/A')}\n"
            sources_text += f"    Content: {r.get('content', '')[:400]}\n\n"
        
        sources_text += "\n## Academic Papers\n"
        for i, p in enumerate(papers[:8], 1):
            source_id = len(all_sources) + 1
            all_sources.append({
                "id": source_id,
                "type": "paper",
                "title": p.get('title', 'N/A'),
                "authors": p.get('authors', []),
                "year": p.get('published', p.get('year', '')),
                "arxiv_id": p.get('arxiv_id', ''),
                "summary": p.get('summary', '')
            })
            authors = ", ".join(p.get('authors', [])[:3])
            if len(p.get('authors', [])) > 3:
                authors += " et al."
            sources_text += f"[{source_id}] **{p.get('title', 'N/A')}**\n"
            sources_text += f"    Authors: {authors}\n"
            sources_text += f"    Year: {p.get('published', p.get('year', 'N/A'))}\n"
            sources_text += f"    Abstract: {p.get('summary', '')[:500]}\n\n"
        
        prompt = f"""You are a research analyst creating a comprehensive synthesis of findings.

# Research Topic
{query}

{sources_text}

# Task
Create a detailed synthesis that:
1. Identifies ALL key themes and concepts across sources
2. Notes areas of consensus between sources
3. Highlights any conflicting information or debates
4. Identifies gaps in current research
5. Provides context and background

# Requirements
- Write in a scholarly but accessible style
- Use inline citations like [1], [2], [3] to reference sources
- Be thorough and comprehensive (aim for 500-800 words)
- Include specific details, statistics, and quotes where available
- Organize by themes, not by source

Write the synthesis now:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "query": query,
            "synthesis": response.content,
            "sources": all_sources,
            "source_count": len(all_sources)
        }
    
    async def _generate_report(self, payload: dict) -> dict:
        """Generate a comprehensive research report with proper citations."""
        query = payload.get("query", "")
        synthesis = payload.get("synthesis", "")
        sources = payload.get("sources", [])
        
        # Build references section
        references = "\n## References\n\n"
        for s in sources:
            if s.get("type") == "web":
                references += f"[{s['id']}] {s['title']}. Available at: {s.get('url', 'N/A')}\n\n"
            else:
                authors = ", ".join(s.get('authors', [])[:3])
                if len(s.get('authors', [])) > 3:
                    authors += " et al."
                year = s.get('year', 'N/A')
                if isinstance(year, str) and len(year) > 4:
                    year = year[:4]
                arxiv = f" arXiv:{s.get('arxiv_id')}" if s.get('arxiv_id') else ""
                references += f"[{s['id']}] {authors}. ({year}). {s['title']}.{arxiv}\n\n"
        
        prompt = f"""You are a senior research analyst creating a comprehensive research report.

# Research Question
{query}

# Research Synthesis
{synthesis}

# Task
Create a detailed, professional research report with the following structure:

## 1. Executive Summary
- 3-5 sentences summarizing the key findings
- Highlight the most important insights

## 2. Introduction
- Background and context of the research topic
- Why this topic matters
- Scope of this report

## 3. Key Findings
- Organize findings by theme (not by source)
- Use bullet points and sub-bullets for clarity
- Include specific details, numbers, and quotes
- Use inline citations [1], [2], etc.

## 4. Detailed Analysis
- Deep dive into each major theme
- Discuss implications and significance
- Note any limitations or caveats
- Compare different perspectives

## 5. Research Gaps & Future Directions
- What questions remain unanswered?
- What areas need more research?
- Potential next steps

## 6. Conclusion
- Synthesize the main takeaways
- Provide actionable insights if applicable

# Requirements
- Write 1000-1500 words minimum
- Use proper Markdown formatting with headers
- Include inline citations [1], [2], [3] throughout
- Be thorough, detailed, and analytical
- Maintain academic rigor while being accessible

Write the complete report now:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Append references
        full_report = response.content + "\n\n---\n" + references
        
        return {
            "query": query,
            "report": full_report,
            "source_count": len(sources)
        }
    
    async def _compare_papers(self, payload: dict) -> dict:
        """Compare multiple papers in detail."""
        papers = payload.get("papers", [])
        
        if len(papers) < 2:
            return {"error": "Need at least 2 papers to compare"}
        
        papers_text = ""
        for i, p in enumerate(papers[:5], 1):
            authors = ", ".join(p.get('authors', [])[:3])
            papers_text += f"\n### Paper {i}: {p.get('title', 'N/A')}\n"
            papers_text += f"**Authors:** {authors}\n"
            papers_text += f"**Year:** {p.get('year', p.get('published', 'N/A'))}\n"
            papers_text += f"**Summary:** {p.get('summary', 'N/A')[:800]}\n"
        
        prompt = f"""Compare the following research papers in detail:

{papers_text}

Provide a comprehensive comparison including:

## 1. Overview
Brief description of each paper's main contribution

## 2. Methodology Comparison
- What approaches does each paper use?
- How do they differ in their methods?

## 3. Key Findings Comparison
- What are the main results of each?
- Where do they agree or disagree?

## 4. Strengths & Limitations
For each paper, note:
- Key strengths
- Potential limitations

## 5. Complementary Aspects
- How do these papers build on each other?
- What gaps does one fill that another doesn't?

## 6. Recommendation
- Which paper is best for what use case?
- How should a researcher use these together?

Be analytical and specific."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "paper_count": len(papers),
            "comparison": response.content
        }


# Entry point
if __name__ == "__main__":
    agent = SynthesisAgent(port=8003)
    agent.run()
