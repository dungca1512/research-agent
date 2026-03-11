---
name: research-agent
description: >
  Multi-agent AI research system using LangGraph, A2A protocol, and MCP.
  Use when working with agent nodes, A2A agents, MCP server tools,
  LangGraph state or graph, web search, arxiv tools, or the research pipeline.
---

# Research Agent Skill

## Architecture Overview

```
User Query
    │
    ▼
main.py (CLI: research / chat / mcp / test-tools)
    │
    ▼
src/agent/graph.py  ←── LangGraph StateGraph
    │   nodes: decompose → web_search → arxiv_search → synthesize → report
    │
    ▼  (via A2A HTTP calls)
src/research/backend.py  ←── call_agent_safe()
    │
    ├── Search Agent  (port 8001)  src/agents/search_agent.py
    ├── Paper Agent   (port 8002)  src/agents/paper_agent.py
    └── Synthesis Agent (port 8003) src/agents/synthesis_agent.py
    │
    ▼  (exposed via MCP)
src/mcp/server.py  ←── FastMCP tools for Claude Desktop / Cursor
```

## Key Conventions

### Adding a new A2A skill to an existing agent
1. Add action name to `agent_card.skills` list
2. Add `elif action == "new_action":` branch in `handle_task()`
3. Implement `async def _new_action(self, payload: dict) -> dict`
4. Return typed dict — never raise, always return `{"error": "..."}` on failure

```python
# Pattern: every A2A handler
async def _my_action(self, payload: dict) -> dict:
    try:
        result = do_something(payload.get("key", ""))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

### Adding a new MCP tool
1. Open `src/mcp/server.py`
2. Decorate with `@mcp.tool()`
3. Call the relevant A2A agent via `call_agent_safe(AGENT_URL, "action", payload)`
4. Return formatted Markdown string (MCP tools return `str`)

```python
@mcp.tool()
async def my_new_tool(query: str) -> str:
    """Docstring = tool description shown to Claude."""
    result = await call_agent_safe(SEARCH_AGENT_URL, "web_search", {"query": query})
    if "error" in result:
        return f"Failed: {result['error']}"
    return format_results(result)
```

### Adding a new LangGraph node
1. Write `def my_node(state: ResearchState) -> dict:` in `src/agent/nodes.py`
2. Return only the state keys you want to update
3. Register in `src/agent/graph.py`: `workflow.add_node("my_node", my_node)`
4. Wire edges: `workflow.add_edge("prev_node", "my_node")`
5. Add display name to `NODE_NAMES` dict for streaming UI

```python
def my_node(state: ResearchState) -> dict:
    # state is read-only input, return dict of updates
    return {
        "web_results": [...],
        "messages": [AIMessage(content="done")]
    }
```

### Extending ResearchState
Edit `src/agent/state.py`. Use `Annotated[list, operator.add]` for append-only fields (like `messages`), plain types for replace fields.

```python
class ResearchState(TypedDict):
    my_new_field: list[str]  # replace on update
    messages: Annotated[Sequence[BaseMessage], operator.add]  # append-only
```

### Adding a new standalone A2A agent
1. Create `src/agents/my_agent.py`
2. Subclass `BaseA2AAgent` from `src/a2a/base_agent.py`
3. Implement `agent_card` property and `handle_task()` method
4. Add port + URL to `src/config.py` and `.env.example`
5. Add health check target in `config.agent_health_targets()`
6. Start in `start_agents.sh`

## Running the System

```bash
# 1. Start all A2A agents (Terminal 1)
./start_agents.sh

# 2a. Run CLI research
python main.py research "your query here"

# 2b. Or start MCP server (Terminal 2)
python main.py mcp

# 2c. Or interactive chat
python main.py chat
```

## File Map

| Path | Purpose |
|------|---------|
| `src/config.py` | All env vars, agent URLs, limits |
| `src/agent/state.py` | LangGraph state schema |
| `src/agent/graph.py` | Graph wiring, `run_research_*` entrypoints |
| `src/agent/nodes.py` | Node functions, A2A calls, fallback LLM |
| `src/agents/*.py` | A2A agent servers (FastAPI via BaseA2AAgent) |
| `src/a2a/client.py` | `A2AClient` — HTTP calls to agents |
| `src/a2a/base_agent.py` | `BaseA2AAgent` — FastAPI + route setup |
| `src/mcp/server.py` | MCP tool definitions (FastMCP) |
| `src/research/backend.py` | `call_agent_safe()` shared helper |
| `src/research/deepagents_runner.py` | DeepAgents-powered deep_research tool |
| `src/tools/` | Raw tool functions (web search, arxiv, PDF) |

## Common Pitfalls

- **Never print in stdio MCP mode** — corrupts the MCP protocol. Use stderr or skip logging.
- **A2A handlers must be async** — `BaseA2AAgent.handle_task()` is awaited.
- **LangGraph nodes must be sync** — use `run_async()` helper in `nodes.py` to call async A2A code from sync node functions.
- **`call_agent_safe` never raises** — always check `if "error" in result` before using result data.
- **State updates are merges** — returning `{"web_results": [...]}` replaces only that key; other keys stay unchanged.
- **`_index.md` must stay updated** — Codex uses it for skill discovery; add entries when adding new skills.

## Environment Variables

```bash
GOOGLE_API_KEY=...        # Required — Gemini LLM
TAVILY_API_KEY=...        # Optional — web search (falls back to DuckDuckGo)
LLM_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.1
SEARCH_AGENT_URL=http://localhost:8001   # Override for remote agents
PAPER_AGENT_URL=http://localhost:8002
SYNTHESIS_AGENT_URL=http://localhost:8003
```