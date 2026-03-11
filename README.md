# Research Agent

AI-powered research assistant built with **LangGraph**, **A2A protocol**, and **MCP**. Searches the web and arXiv, synthesizes findings, and generates structured reports.

## Features

- **Multi-source search** — Web (Tavily/DuckDuckGo) + arXiv papers
- **Multi-agent pipeline** — 3 specialized A2A agents (Search, Paper, Synthesis)
- **Paper summarization** — Fetch any arXiv paper and get an LLM-generated summary
- **MCP gateway** — Expose all tools to Claude Desktop / Cursor
- **Chat UI** — Chainlit web interface with step-by-step streaming
- **Docker Compose** — One command to run the entire system

## Architecture

```
User Query
    │
    ▼
main.py / chainlit_app.py (CLI or Web UI)
    │
    ▼
src/agent/graph.py  ←── LangGraph StateGraph
    │   decompose → web_search → arxiv_search → synthesize → report
    │
    ▼  A2A HTTP calls
src/research/backend.py  ←── call_agent_safe()
    │
    ├── Search Agent    :8001   web search, arxiv search
    ├── Paper Agent     :8002   PDF parsing, Semantic Scholar
    └── Synthesis Agent :8003   report generation
    │
    ▼  MCP (SSE)
src/mcp/server.py  ←── Claude Desktop / Cursor
```

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Configure API keys
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY and TAVILY_API_KEY

# 2. Start everything
docker compose up --build
```

| Service | URL | Purpose |
|---------|-----|---------|
| Chat UI | http://localhost:8080 | Chainlit web chat |
| MCP server | http://localhost:3000/sse | Claude Desktop / Cursor |
| Search Agent | http://localhost:8001/docs | Swagger UI |
| Paper Agent | http://localhost:8002/docs | Swagger UI |
| Synthesis Agent | http://localhost:8003/docs | Swagger UI |

### Option B — Local (manual)

```bash
pip install -r requirements.txt
cp .env.example .env

# Terminal 1 — start A2A agents
./start_agents.sh

# Terminal 2 — chat UI
chainlit run chainlit_app.py

# Or CLI
python main.py research "What are LLM agents?"
python main.py chat
```

## CLI Commands

```bash
python main.py research "query"          # one-shot research, prints report
python main.py research "query" -o out.md  # save to file
python main.py chat                      # interactive terminal chat
python main.py mcp                       # MCP server (stdio)
python main.py mcp --transport sse --port 3000  # MCP server (SSE)
python main.py test-tools                # verify all tools work
python main.py info                      # show config
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web |
| `arxiv_search` | Search arXiv papers |
| `arxiv_get_paper` | Get full details by arXiv ID |
| `summarize_paper` | Fetch paper + LLM-generated summary |
| `read_paper_pdf` | Extract text from PDF URL |
| `extract_paper_sections` | Get Abstract / Methods / Results / Conclusion |
| `semantic_search` | Search with citation metrics (Semantic Scholar) |
| `get_paper_citations` | Who cites a paper |
| `get_paper_references` | What a paper cites |
| `format_citation` | Format in APA / MLA / BibTeX |
| `deep_research` | Full multi-agent research pipeline |
| `check_agents_status` | Health check all A2A agents |

## Connect Claude Desktop

**Local:**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/research-agent",
      "env": {
        "PYTHONPATH": "/absolute/path/to/research-agent",
        "GOOGLE_API_KEY": "your_key_here",
        "TAVILY_API_KEY": "your_key_here"
      }
    }
  }
}
```

**Docker:**

```json
{
  "mcpServers": {
    "research-agent": {
      "url": "http://localhost:3000/sse"
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Gemini LLM |
| `TAVILY_API_KEY` | No | — | Web search (falls back to DuckDuckGo) |
| `LLM_MODEL` | No | `gemini-2.5-flash` | Model name |
| `LLM_TEMPERATURE` | No | `0.1` | Sampling temperature |
| `SEARCH_AGENT_URL` | No | `http://localhost:8001` | Override for remote agents |
| `PAPER_AGENT_URL` | No | `http://localhost:8002` | |
| `SYNTHESIS_AGENT_URL` | No | `http://localhost:8003` | |

## Project Structure

```
research-agent/
├── chainlit_app.py         # Chainlit web UI
├── main.py                 # CLI entry point
├── docker-compose.yml      # All services
├── Dockerfile
├── start_agents.sh         # Local agent startup
├── mcp_config.json         # MCP config examples
├── requirements.txt
└── src/
    ├── config.py
    ├── a2a/
    │   ├── base_agent.py   # BaseA2AAgent class
    │   └── client.py       # A2A HTTP client
    ├── agent/
    │   ├── graph.py        # LangGraph workflow
    │   ├── nodes.py        # Node functions
    │   └── state.py        # ResearchState schema
    ├── agents/
    │   ├── search_agent.py    # :8001
    │   ├── paper_agent.py     # :8002
    │   └── synthesis_agent.py # :8003
    ├── mcp/
    │   └── server.py       # FastMCP tool definitions
    ├── research/
    │   ├── backend.py      # call_agent_safe()
    │   └── deepagents_runner.py
    └── tools/              # Raw tool functions
        ├── arxiv_search.py
        ├── summarize_paper.py
        ├── web_search.py
        ├── paper_parser.py
        ├── section_extractor.py
        └── semantic_scholar.py
```

## License

MIT
