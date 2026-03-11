# Claude Agent Instructions — Research Agent

## Skill System
Before starting any task, check `.claude/skills/_index.md` for available skills.
If a relevant skill exists, read its `SKILL.md` before writing any code.

## Project Conventions
- Python async/await throughout — A2A handlers are always `async def`
- LangGraph nodes are sync — use `run_async()` in `src/agent/nodes.py` to bridge
- All A2A calls go through `call_agent_safe()` — never call `A2AClient` directly in nodes
- MCP tools return `str` (Markdown) — A2A handlers return `dict`
- Never raise exceptions in A2A handlers — return `{"error": "..."}` instead

## Code Style
- Type hints required on all public functions
- Pydantic models for request/response schemas (see `base_agent.py`)
- Config always via `get_config()` — never hardcode URLs or keys
- New agents inherit `BaseA2AAgent`, new tools use `@tool` decorator from LangChain

## Testing Changes
```bash
# Quick check: does the agent start?
python -m src.agents.search_agent &
sleep 2
curl http://localhost:8001/health

# Run a single research query end-to-end
python main.py research "test query" --verbose
```
