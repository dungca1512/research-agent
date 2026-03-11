"""Chainlit chat UI for Research Agent."""

import chainlit as cl
from src.agent.graph import create_research_agent, NODE_NAMES
from src.agent.state import create_initial_state
from src.config import get_config


@cl.on_chat_start
async def on_chat_start():
    config = get_config()
    if not config.has_google_api:
        await cl.Message(
            content="**Error:** `GOOGLE_API_KEY` is not set. Please configure your environment."
        ).send()
        return

    await cl.Message(
        content=(
            "**Research Agent** is ready.\n\n"
            "Type any research question and I'll search the web, arXiv, "
            "and synthesize a report for you."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return

    agent = create_research_agent()
    initial_state = create_initial_state(query)

    final_report = ""

    async for event in agent.astream(initial_state):
        for node_name, node_output in event.items():
            display_name = NODE_NAMES.get(node_name, node_name)

            # Build a short summary of what this node produced
            details = _summarize_node_output(node_name, node_output)

            async with cl.Step(name=display_name) as step:
                step.output = details

            if isinstance(node_output, dict) and "final_report" in node_output:
                final_report = node_output["final_report"]

    if final_report:
        await cl.Message(content=final_report).send()
    else:
        await cl.Message(content="No report was generated. Please try again.").send()


def _summarize_node_output(node_name: str, output: dict) -> str:  # noqa: ARG001
    if not isinstance(output, dict):
        return ""

    parts = []
    if "search_queries" in output:
        queries = output["search_queries"]
        parts.append(f"Generated {len(queries)} sub-queries:\n" + "\n".join(f"- {q}" for q in queries))
    if "web_results" in output:
        parts.append(f"Found {len(output['web_results'])} web results.")
    if "arxiv_papers" in output:
        parts.append(f"Found {len(output['arxiv_papers'])} arXiv papers.")
    if "synthesis" in output:
        snippet = output["synthesis"][:300]
        parts.append(f"Synthesis preview:\n{snippet}...")
    if "final_report" in output:
        parts.append(f"Report generated ({len(output['final_report'])} chars).")

    return "\n\n".join(parts) or "Done."
