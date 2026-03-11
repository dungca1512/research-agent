"""Tool to summarize an arXiv paper given its ID."""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import get_config
from src.tools.arxiv_search import get_paper_by_id

SUMMARIZE_PROMPT = """You are a research assistant. Given the title and abstract of an academic paper, write a concise summary in 3-5 sentences.

Focus on:
- The main problem or research question
- The proposed approach or method
- The key results or contributions

Paper Title: {title}

Abstract:
{abstract}

Write a clear, accessible summary:"""


def summarize_paper(arxiv_id: str) -> str:
    """
    Fetch an arXiv paper and generate a short LLM-based summary.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.00001")

    Returns:
        A short summary string, or an error message if the paper is not found.
    """
    paper = get_paper_by_id(arxiv_id)
    if not paper:
        return f"Paper not found: {arxiv_id}"

    config = get_config()
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model,
        google_api_key=config.google_api_key,
        temperature=0.3,
    )

    prompt = SUMMARIZE_PROMPT.format(
        title=paper["title"],
        abstract=paper["summary"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    summary_text = response.content.strip()

    authors = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors += " et al."

    return (
        f"## {paper['title']}\n\n"
        f"**ArXiv ID:** {arxiv_id} | **Published:** {paper['published']}\n"
        f"**Authors:** {authors}\n\n"
        f"### Summary\n{summary_text}\n\n"
        f"**PDF:** {paper['pdf_url']}\n"
    )


@tool
def summarize_paper_tool(arxiv_id: str) -> str:
    """
    Fetch an arXiv paper by ID and return a short, LLM-generated summary.
    Use this tool when you need a quick overview of a specific paper.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.00001" or "2106.09685")

    Returns:
        A formatted string with paper metadata and a concise 3-5 sentence summary
    """
    return summarize_paper(arxiv_id)
