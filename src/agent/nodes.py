"""Node functions for the Research Agent workflow - A2A Version.

This version communicates with A2A agents instead of calling tools directly.
Both LangGraph CLI and Claude Desktop now share the same A2A backend.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.state import ResearchState
from src.a2a.client import A2AClient
from src.config import get_config

# Agent endpoints (same as MCP server)
SEARCH_AGENT_URL = "http://localhost:8001"
PAPER_AGENT_URL = "http://localhost:8002"
SYNTHESIS_AGENT_URL = "http://localhost:8003"

# A2A client
a2a_client = A2AClient(timeout=60.0)

# Prompts
QUERY_DECOMPOSITION_PROMPT = """You are a research assistant helping to decompose a research query into effective search queries.

Given the original research query, generate 2-3 focused search queries that will help find relevant information.
Consider:
- Different aspects of the topic
- Both general and specific terms
- Academic vs practical perspectives

Original Query: {query}

Return ONLY the search queries, one per line. No numbering or explanations."""

SYNTHESIS_PROMPT = """You are a research assistant synthesizing information from multiple sources.

Original Research Query: {query}

Web Search Results:
{web_results}

ArXiv Papers Found:
{arxiv_results}

Based on the above information, provide a comprehensive synthesis that:
1. Identifies key themes and findings
2. Notes any conflicting information
3. Highlights gaps that may need further research

Be concise but thorough. Focus on factual information with proper attribution."""

REPORT_PROMPT = """You are a research assistant creating a final research report.

Original Query: {query}

Synthesized Findings:
{synthesis}

Create a well-structured research report in Markdown format with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (organized by theme)
4. Sources & References
5. Further Research Suggestions

The report should be professional, accurate, and cite sources where applicable."""


def get_llm():
    """Get the configured LLM instance."""
    config = get_config()
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        google_api_key=config.google_api_key,
        temperature=config.llm_temperature,
    )


async def call_agent_safe(url: str, action: str, payload: dict) -> dict:
    """Safely call an A2A agent, return error dict on failure."""
    try:
        response = await a2a_client.call_agent(url, action, payload)
        if response.status == "completed":
            return response.result or {}
        else:
            return {"error": response.error or "Unknown error"}
    except Exception as e:
        return {"error": str(e)}


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def query_decomposition_node(state: ResearchState) -> dict:
    """
    Decompose the original query into search sub-queries.
    Uses Search Agent's decompose_query action.
    """
    async def _decompose():
        result = await call_agent_safe(
            SEARCH_AGENT_URL,
            "decompose_query",
            {"query": state["query"]}
        )
        return result.get("sub_queries", [state["query"]])
    
    queries = run_async(_decompose())
    
    # Always include the original query
    if state["query"] not in queries:
        queries.insert(0, state["query"])
    
    return {
        "search_queries": queries,
        "messages": [AIMessage(content=f"Generated {len(queries)} search queries via A2A")]
    }


def web_search_node(state: ResearchState) -> dict:
    """
    Perform web searches via Search Agent (A2A).
    """
    async def _search():
        all_results = []
        for query in state["search_queries"][:3]:
            result = await call_agent_safe(
                SEARCH_AGENT_URL,
                "web_search",
                {"query": query, "max_results": 3}
            )
            if "error" not in result:
                all_results.extend(result.get("results", []))
        return all_results
    
    all_results = run_async(_search())
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    return {
        "web_results": unique_results,
        "messages": [AIMessage(content=f"Found {len(unique_results)} web results via A2A")]
    }


def arxiv_search_node(state: ResearchState) -> dict:
    """
    Search arXiv via Search Agent (A2A).
    """
    async def _search():
        all_papers = []
        for query in state["search_queries"][:2]:
            result = await call_agent_safe(
                SEARCH_AGENT_URL,
                "arxiv_search",
                {"query": query, "max_results": 3}
            )
            if "error" not in result:
                all_papers.extend(result.get("papers", []))
        return all_papers
    
    all_papers = run_async(_search())
    
    # Deduplicate by arxiv_id
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        arxiv_id = p.get("arxiv_id", "")
        if arxiv_id and arxiv_id not in seen_ids:
            seen_ids.add(arxiv_id)
            unique_papers.append(p)
    
    return {
        "arxiv_papers": unique_papers,
        "messages": [AIMessage(content=f"Found {len(unique_papers)} arXiv papers via A2A")]
    }


def synthesis_node(state: ResearchState) -> dict:
    """
    Synthesize information via Synthesis Agent (A2A).
    """
    async def _synthesize():
        result = await call_agent_safe(
            SYNTHESIS_AGENT_URL,
            "synthesize",
            {
                "query": state["query"],
                "web_results": state["web_results"][:10],
                "papers": state["arxiv_papers"][:5]
            }
        )
        return result
    
    result = run_async(_synthesize())
    
    if "error" in result:
        # Fallback to local LLM if Synthesis Agent fails
        llm = get_llm()
        
        # Format web results
        web_results_text = ""
        for i, r in enumerate(state["web_results"][:10], 1):
            web_results_text += f"{i}. **{r.get('title', 'N/A')}**\n   URL: {r.get('url', '')}\n   {r.get('content', '')[:300]}...\n\n"
        
        if not web_results_text:
            web_results_text = "No web results found."
        
        # Format arXiv results
        arxiv_text = ""
        for i, p in enumerate(state["arxiv_papers"][:5], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            arxiv_text += f"{i}. **{p.get('title', 'N/A')}** ({p.get('arxiv_id', '')})\n"
            arxiv_text += f"   Authors: {authors}\n"
            arxiv_text += f"   Abstract: {p.get('summary', '')[:400]}...\n\n"
        
        if not arxiv_text:
            arxiv_text = "No arXiv papers found."
        
        prompt = SYNTHESIS_PROMPT.format(
            query=state["query"],
            web_results=web_results_text,
            arxiv_results=arxiv_text
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        synthesis = response.content
    else:
        synthesis = result.get("synthesis", "")
    
    return {
        "synthesis": synthesis,
        "iteration": state["iteration"] + 1,
        "messages": [AIMessage(content="Completed synthesis via A2A")]
    }


def report_node(state: ResearchState) -> dict:
    """
    Generate the final research report via Synthesis Agent (A2A).
    """
    async def _generate_report():
        result = await call_agent_safe(
            SYNTHESIS_AGENT_URL,
            "generate_report",
            {
                "query": state["query"],
                "synthesis": state["synthesis"],
                "sources": state["web_results"][:5] + state["arxiv_papers"][:5]
            }
        )
        return result
    
    result = run_async(_generate_report())
    
    if "error" in result:
        # Fallback to local LLM if Synthesis Agent fails
        llm = get_llm()
        
        prompt = REPORT_PROMPT.format(
            query=state["query"],
            synthesis=state["synthesis"]
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        report = response.content
    else:
        report = result.get("report", "Failed to generate report")
    
    return {
        "final_report": report,
        "should_continue": False,
        "messages": [AIMessage(content="Generated final research report via A2A")]
    }


def should_continue_research(state: ResearchState) -> str:
    """
    Determine if we should continue research or generate report.
    
    Returns:
        "continue" to do more research, "report" to generate final report
    """
    config = get_config()
    
    # Stop if we've reached max iterations
    if state["iteration"] >= config.max_iterations:
        return "report"
    
    # Stop if we have enough results
    if len(state["web_results"]) >= 5 and len(state["arxiv_papers"]) >= 3:
        return "report"
    
    # Continue if we don't have enough data
    if len(state["web_results"]) == 0 and len(state["arxiv_papers"]) == 0:
        return "continue"
    
    # Default to generating report after first good iteration
    return "report"
