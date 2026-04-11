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
from src.tools.paper_comparison import (
    compare_papers_structured,
    format_comparison_markdown,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class SynthesisAgent(BaseA2AAgent):
    """Agent specialized in synthesizing research and generating reports."""

    def __init__(self, port: int = 8003):
        super().__init__(port=port)
        self.config = get_config()
        self.llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Create the LLM lazily and fail with a clear message if unavailable."""
        if self.llm is None:
            if not self.config.has_google_api:
                raise ValueError("GOOGLE_API_KEY is required for synthesis tasks")
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=self.config.google_api_key,
                temperature=0.3
            )
        return self.llm
    
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
        
        prompt = f"""Bạn là một nhà phân tích nghiên cứu tạo bản tổng hợp toàn diện từ các nguồn tài liệu.

# Chủ đề nghiên cứu
{query}

{sources_text}

# Nhiệm vụ
Tạo một bản tổng hợp chi tiết:
1. Xác định TẤT CẢ các chủ đề và khái niệm chính từ các nguồn
2. Ghi nhận các điểm đồng thuận giữa các nguồn
3. Làm nổi bật các thông tin mâu thuẫn hoặc tranh luận
4. Xác định các khoảng trống trong nghiên cứu hiện tại
5. Cung cấp bối cảnh và nền tảng kiến thức

# Yêu cầu
- Viết theo phong cách học thuật nhưng dễ hiểu
- Sử dụng trích dẫn nội tuyến như [1], [2], [3] để tham chiếu nguồn
- Đầy đủ và toàn diện (khoảng 500-800 từ)
- Bao gồm các chi tiết, số liệu thống kê cụ thể khi có
- Tổ chức theo chủ đề, không theo từng nguồn
- QUAN TRỌNG: Viết toàn bộ bằng tiếng Việt

Viết bản tổng hợp ngay bây giờ:"""
        
        response = await self._get_llm().ainvoke([HumanMessage(content=prompt)])
        
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
        
        prompt = f"""Bạn là một nhà phân tích nghiên cứu cấp cao đang tạo một báo cáo nghiên cứu toàn diện.

# Câu hỏi nghiên cứu
{query}

# Bản tổng hợp nghiên cứu
{synthesis}

# Nhiệm vụ
Tạo một báo cáo nghiên cứu chi tiết, chuyên nghiệp với cấu trúc sau:

## 1. Tóm tắt tổng quan
- 3-5 câu tóm tắt các phát hiện chính
- Làm nổi bật những thông tin quan trọng nhất

## 2. Giới thiệu
- Bối cảnh và nền tảng của chủ đề nghiên cứu
- Tại sao chủ đề này quan trọng
- Phạm vi của báo cáo này

## 3. Các phát hiện chính
- Tổ chức theo chủ đề (không theo nguồn)
- Dùng bullet points và sub-bullets cho rõ ràng
- Bao gồm chi tiết, số liệu và trích dẫn cụ thể
- Sử dụng trích dẫn nội tuyến [1], [2], v.v.

## 4. Phân tích chi tiết
- Đi sâu vào từng chủ đề chính
- Thảo luận về ý nghĩa và tầm quan trọng
- Ghi chú các hạn chế hoặc lưu ý
- So sánh các quan điểm khác nhau

## 5. Khoảng trống nghiên cứu & Hướng phát triển
- Những câu hỏi nào còn chưa được trả lời?
- Những lĩnh vực nào cần nghiên cứu thêm?
- Các bước tiếp theo có thể thực hiện

## 6. Kết luận
- Tổng hợp các điểm chính
- Cung cấp các thông tin hữu ích nếu có thể áp dụng

# Yêu cầu
- Tối thiểu 1000-1500 từ
- Sử dụng định dạng Markdown đúng với các tiêu đề rõ ràng
- Bao gồm trích dẫn nội tuyến [1], [2], [3] xuyên suốt báo cáo
- Đầy đủ, chi tiết và có tính phân tích cao
- Duy trì tính nghiêm túc học thuật trong khi vẫn dễ tiếp cận
- QUAN TRỌNG: Viết toàn bộ báo cáo bằng tiếng Việt

Viết báo cáo đầy đủ ngay bây giờ:"""
        
        response = await self._get_llm().ainvoke([HumanMessage(content=prompt)])
        
        # Append references
        full_report = response.content + "\n\n---\n" + references
        
        return {
            "query": query,
            "report": full_report,
            "source_count": len(sources)
        }
    
    async def _compare_papers(self, payload: dict) -> dict:
        """Compare multiple papers using structured extraction."""
        papers = payload.get("papers", [])

        if len(papers) < 2:
            return {"error": "Need at least 2 papers to compare"}

        try:
            matrix = await compare_papers_structured(papers)
            markdown = format_comparison_markdown(matrix)
            return {
                "paper_count": len(papers),
                "comparison": markdown,
                "matrix": matrix,
            }
        except Exception:
            # Fallback to free-form comparison if structured extraction fails
            return await self._compare_papers_fallback(papers)

    async def _compare_papers_fallback(self, papers: list[dict]) -> dict:
        """Free-form LLM comparison as fallback."""
        papers_text = ""
        for i, p in enumerate(papers[:5], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            papers_text += f"\n### Paper {i}: {p.get('title', 'N/A')}\n"
            papers_text += f"**Authors:** {authors}\n"
            papers_text += f"**Year:** {p.get('year', p.get('published', 'N/A'))}\n"
            papers_text += f"**Summary:** {p.get('summary', 'N/A')[:800]}\n"

        prompt = f"""Compare the following research papers in detail:

{papers_text}

Provide a comparison covering: overview, methodology, key findings, strengths & limitations, complementary aspects, and recommendation.

Be analytical and specific."""

        response = await self._get_llm().ainvoke([HumanMessage(content=prompt)])

        return {
            "paper_count": len(papers),
            "comparison": response.content,
        }


# Entry point
if __name__ == "__main__":
    agent = SynthesisAgent(port=8003)
    agent.run()
