"""LangGraph workflow definition for Research Agent."""

from typing import Callable, Optional
from langgraph.graph import StateGraph, END
from src.agent.state import ResearchState, create_initial_state
from src.agent.nodes import (
    query_decomposition_node,
    web_search_node,
    arxiv_search_node,
    synthesis_node,
    report_node,
    should_continue_research,
)

# Node display names for streaming
NODE_NAMES = {
    "decompose": "🔍 Decomposing query",
    "web_search": "🌐 Searching web (A2A → Search Agent)",
    "arxiv_search": "📚 Searching arXiv (A2A → Search Agent)",
    "synthesize": "🧠 Synthesizing results (A2A → Synthesis Agent)",
    "report": "📝 Generating report (A2A → Synthesis Agent)",
}


def create_research_agent():
    """
    Create and compile the research agent workflow.
    
    The workflow follows this pattern:
    1. Query Decomposition - Break down the research query
    2. Parallel Search - Web + ArXiv searches
    3. Synthesis - Combine and analyze results
    4. Decision - Continue or generate report
    5. Report - Generate final research report
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("decompose", query_decomposition_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("arxiv_search", arxiv_search_node)
    workflow.add_node("synthesize", synthesis_node)
    workflow.add_node("report", report_node)
    
    # Define edges
    workflow.set_entry_point("decompose")
    
    # After decomposition, do both searches
    workflow.add_edge("decompose", "web_search")
    workflow.add_edge("web_search", "arxiv_search")
    workflow.add_edge("arxiv_search", "synthesize")
    
    # After synthesis, decide whether to continue or report
    workflow.add_conditional_edges(
        "synthesize",
        should_continue_research,
        {
            "continue": "decompose",  # Loop back for more research
            "report": "report",       # Generate final report
        }
    )
    
    # Report is the final node
    workflow.add_edge("report", END)
    
    # Compile the graph
    return workflow.compile()


async def run_research(query: str) -> str:
    """
    Run the research agent on a query.
    
    Args:
        query: The research query to investigate
        
    Returns:
        The final research report
    """
    agent = create_research_agent()
    initial_state = create_initial_state(query)
    
    # Run the agent
    final_state = await agent.ainvoke(initial_state)
    
    return final_state["final_report"]


def run_research_sync(query: str) -> str:
    """
    Run the research agent synchronously.
    
    Args:
        query: The research query to investigate
        
    Returns:
        The final research report
    """
    agent = create_research_agent()
    initial_state = create_initial_state(query)
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    return final_state["final_report"]


def run_research_stream(
    query: str,
    on_node_start: Optional[Callable[[str, str], None]] = None,
    on_node_end: Optional[Callable[[str, dict], None]] = None,
) -> str:
    """
    Run the research agent with streaming - shows step-by-step execution.
    
    Args:
        query: The research query to investigate
        on_node_start: Callback(node_name, display_name) when node starts
        on_node_end: Callback(node_name, result) when node completes
        
    Returns:
        The final research report
    """
    agent = create_research_agent()
    initial_state = create_initial_state(query)
    
    final_report = ""
    
    # Stream through the graph execution
    for event in agent.stream(initial_state):
        for node_name, node_output in event.items():
            # Callback when node starts
            if on_node_start:
                display_name = NODE_NAMES.get(node_name, node_name)
                on_node_start(node_name, display_name)
            
            # Extract useful info from output
            result_info = {}
            if isinstance(node_output, dict):
                if "search_queries" in node_output:
                    result_info["queries"] = node_output["search_queries"]
                if "web_results" in node_output:
                    result_info["web_count"] = len(node_output["web_results"])
                if "arxiv_papers" in node_output:
                    result_info["paper_count"] = len(node_output["arxiv_papers"])
                if "synthesis" in node_output:
                    result_info["synthesis_length"] = len(node_output["synthesis"])
                if "final_report" in node_output:
                    final_report = node_output["final_report"]
                    result_info["report_length"] = len(final_report)
            
            # Callback when node ends
            if on_node_end:
                on_node_end(node_name, result_info)
    
    return final_report

