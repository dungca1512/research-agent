"""DeepAgents-powered runner behind the MCP deep_research tool."""

from functools import lru_cache
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_config
from src.research.backend import (
    PAPER_AGENT_URL,
    SEARCH_AGENT_URL,
    SYNTHESIS_AGENT_URL,
    call_agent_safe,
)

DEEP_RESEARCH_SYSTEM_PROMPT = """You are the deep research engine behind an MCP tool called `deep_research`.

Your job is to produce a rigorous Markdown research report for the user's query.

Working rules:
- Use `write_todos` for any non-trivial query so the research plan stays explicit.
- Prefer a mix of web sources and academic sources when available.
- Use the search and paper tools to gather evidence, not guesses.
- If a paper looks central to the question, inspect it more deeply with the paper tools.
- Use the filesystem tools when notes get large; keep concise research notes and source tracking.
- When you have enough evidence, call `compile_research_report` exactly once with the curated result lists.
- Your final response must be only the `report` returned by `compile_research_report`, with no preamble or extra commentary.
- The final report must start with `# Research Report:`.
- Do not mention hidden reasoning, internal tools, A2A, or DeepAgents in the final answer.
- Avoid the `execute` tool. This backend does not support shell execution.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    """Build the model instance used by DeepAgents."""
    config = get_config()
    if not config.has_google_api:
        raise RuntimeError(
            "GOOGLE_API_KEY is required to run the DeepAgents-powered deep_research tool."
        )
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        google_api_key=config.google_api_key,
        temperature=config.llm_temperature,
    )


async def decompose_query(query: str) -> list[str]:
    """Break the research question into focused sub-queries."""
    result = await call_agent_safe(
        SEARCH_AGENT_URL,
        "decompose_query",
        {"query": query},
    )
    return result.get("sub_queries", [query])


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search through the Search Agent."""
    result = await call_agent_safe(
        SEARCH_AGENT_URL,
        "web_search",
        {"query": query, "max_results": max_results},
    )
    return result.get("results", [])


async def search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    """Run an arXiv search through the Search Agent."""
    result = await call_agent_safe(
        SEARCH_AGENT_URL,
        "arxiv_search",
        {"query": query, "max_results": max_results},
    )
    return result.get("papers", [])


async def search_semantic_scholar(query: str, max_results: int = 5) -> list[dict]:
    """Run a Semantic Scholar search through the Paper Agent."""
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "semantic_search",
        {"query": query, "max_results": max_results},
    )
    return result.get("papers", [])


async def get_paper(arxiv_id: str) -> dict:
    """Fetch paper metadata by arXiv identifier."""
    return await call_agent_safe(
        PAPER_AGENT_URL,
        "get_paper",
        {"arxiv_id": arxiv_id},
    )


async def read_paper_pdf(pdf_url: str, max_pages: int = 10) -> dict:
    """Download and extract text from a paper PDF."""
    return await call_agent_safe(
        PAPER_AGENT_URL,
        "parse_pdf",
        {"pdf_url": pdf_url, "max_pages": max_pages},
    )


async def extract_paper_sections(pdf_url: str) -> dict:
    """Extract major sections from a paper PDF."""
    return await call_agent_safe(
        PAPER_AGENT_URL,
        "extract_sections",
        {"pdf_url": pdf_url},
    )


async def get_paper_citations(paper_id: str, limit: int = 20) -> list[dict]:
    """Find papers that cite a specific Semantic Scholar paper."""
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "get_citations",
        {"paper_id": paper_id, "limit": limit},
    )
    return result.get("citations", [])


async def get_paper_references(paper_id: str, limit: int = 20) -> list[dict]:
    """Find references cited by a specific Semantic Scholar paper."""
    result = await call_agent_safe(
        PAPER_AGENT_URL,
        "get_references",
        {"paper_id": paper_id, "limit": limit},
    )
    return result.get("references", [])


async def synthesize_findings(query: str, web_results: list[dict], papers: list[dict]) -> dict:
    """Synthesize collected evidence into a coherent research summary."""
    return await call_agent_safe(
        SYNTHESIS_AGENT_URL,
        "synthesize",
        {
            "query": query,
            "web_results": web_results,
            "papers": papers,
        },
    )


async def generate_report(query: str, synthesis: str, sources: list[dict]) -> dict:
    """Generate the final report from synthesis output and numbered sources."""
    return await call_agent_safe(
        SYNTHESIS_AGENT_URL,
        "generate_report",
        {
            "query": query,
            "synthesis": synthesis,
            "sources": sources,
        },
    )


def merge_papers_by_title(primary_papers: list[dict], extra_papers: list[dict]) -> list[dict]:
    """Append non-duplicate papers using case-insensitive title matching."""
    merged = list(primary_papers)
    seen_titles = {paper.get("title", "").lower() for paper in merged}

    for paper in extra_papers:
        title = paper.get("title", "").lower()
        if title and title not in seen_titles:
            merged.append(paper)
            seen_titles.add(title)

    return merged


def dedupe_web_results(results: list[dict]) -> list[dict]:
    """Drop duplicate web results by URL while preserving order."""
    deduped = []
    seen_urls = set()

    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            deduped.append(result)
            seen_urls.add(url)

    return deduped


def ensure_report_header(query: str, report: str, source_count: int) -> str:
    """Add the standard report header if the generated report does not include it."""
    if report.lstrip().startswith("# Research Report:"):
        return report

    header = f"""# Research Report: {query}

> Sources analyzed: {source_count} (Web articles + Academic papers)
> Generated by: DeepAgents + A2A research backend

---

"""
    return header + report.lstrip()


def _message_text(content: Any) -> str:
    """Extract human-readable text from a LangChain message content payload."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


@tool
async def decompose_query_tool(query: str) -> list[str]:
    """Generate 2-3 focused research sub-queries for the user's topic."""
    return await decompose_query(query)


@tool
async def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return result objects with title, url, content, and source."""
    return await search_web(query, max_results=max_results)


@tool
async def arxiv_search_tool(query: str, max_results: int = 8) -> list[dict]:
    """Search arXiv and return paper objects with title, authors, summary, ids, and metadata."""
    return await search_arxiv(query, max_results=max_results)


@tool
async def semantic_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search Semantic Scholar and return papers with citations, abstracts, and paper IDs."""
    return await search_semantic_scholar(query, max_results=max_results)


@tool
async def get_paper_tool(arxiv_id: str) -> dict:
    """Get detailed metadata for a specific arXiv paper by ID."""
    return await get_paper(arxiv_id)


@tool
async def read_paper_pdf_tool(pdf_url: str, max_pages: int = 10) -> dict:
    """Read a research paper PDF and return extracted text content."""
    return await read_paper_pdf(pdf_url, max_pages=max_pages)


@tool
async def extract_paper_sections_tool(pdf_url: str) -> dict:
    """Extract abstract, introduction, methodology, results, conclusion, and references from a PDF."""
    return await extract_paper_sections(pdf_url)


@tool
async def get_paper_citations_tool(paper_id: str, limit: int = 20) -> list[dict]:
    """List papers that cite a Semantic Scholar paper."""
    return await get_paper_citations(paper_id, limit=limit)


@tool
async def get_paper_references_tool(paper_id: str, limit: int = 20) -> list[dict]:
    """List references cited by a Semantic Scholar paper."""
    return await get_paper_references(paper_id, limit=limit)


@tool
async def compile_research_report(
    query: str,
    web_results: list[dict],
    papers: list[dict],
) -> dict:
    """Compile a final report from curated web results and paper metadata.

    Call this only after you have collected and curated the evidence you want
    included in the final report.
    """
    deduped_web_results = dedupe_web_results(web_results)
    deduped_papers = merge_papers_by_title([], papers)

    synthesis_result = await synthesize_findings(query, deduped_web_results, deduped_papers)
    synthesis = synthesis_result.get("synthesis", "")
    sources = synthesis_result.get("sources", [])

    report_result = await generate_report(query, synthesis, sources)
    source_count = report_result.get("source_count", len(sources))
    report = ensure_report_header(
        query,
        report_result.get("report", "Failed to generate report"),
        source_count,
    )
    return {
        "report": report,
        "source_count": source_count,
    }


@lru_cache(maxsize=1)
def _build_deep_research_agent():
    """Create the shared DeepAgents graph used by the MCP tool."""
    return create_deep_agent(
        model=_get_llm(),
        tools=[
            decompose_query_tool,
            web_search_tool,
            arxiv_search_tool,
            semantic_search_tool,
            get_paper_tool,
            read_paper_pdf_tool,
            extract_paper_sections_tool,
            get_paper_citations_tool,
            get_paper_references_tool,
            compile_research_report,
        ],
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        name="deep-research-agent",
    )


async def run_deep_research(query: str) -> str:
    """Run the DeepAgents-powered research workflow behind the MCP tool."""
    agent = _build_deep_research_agent()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": f"deep-research-{uuid4().hex}"}},
    )

    messages = result.get("messages", [])
    for message in reversed(messages):
        text = _message_text(getattr(message, "content", ""))
        if text:
            return text

    raise RuntimeError("Deep research agent completed without a final report.")
