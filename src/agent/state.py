"""State definition for the Research Agent workflow."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class ResearchState(TypedDict):
    """
    State for the research agent workflow.
    
    Attributes:
        messages: Conversation history with the LLM
        query: Original research query from user
        search_queries: Generated sub-queries for searching
        web_results: Results from web searches
        arxiv_papers: Papers found from arXiv
        paper_contents: Extracted text from papers
        synthesis: Synthesized information from all sources
        final_report: Final research report
        iteration: Current iteration count
        should_continue: Whether to continue researching
    """
    
    # Message handling with append-only behavior
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Research state
    query: str
    search_queries: list[str]
    web_results: list[dict]
    arxiv_papers: list[dict]
    paper_contents: list[str]
    
    # Output state
    synthesis: str
    final_report: str
    
    # Control state
    iteration: int
    should_continue: bool


def create_initial_state(query: str) -> ResearchState:
    """
    Create initial state for a new research session.
    
    Args:
        query: The user's research query
        
    Returns:
        Initialized ResearchState
    """
    return {
        "messages": [],
        "query": query,
        "search_queries": [],
        "web_results": [],
        "arxiv_papers": [],
        "paper_contents": [],
        "synthesis": "",
        "final_report": "",
        "iteration": 0,
        "should_continue": True,
    }
