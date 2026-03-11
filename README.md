# 🔬 Research Agent

AI-powered research assistant built with **LangChain** and **LangGraph** for automated literature research, paper discovery, and report generation.

## Features

- **Multi-source Search**: Web (Tavily/DuckDuckGo) + ArXiv papers
- **Iterative Research**: Automatically expands search based on findings
- **Paper Summarization**: Extract key information from academic papers
- **Structured Reports**: Generate comprehensive Markdown reports with citations

## Architecture

```
Query → Decompose → Web Search → ArXiv Search → Synthesize → Report
              ↑                                   |
              └───────── (if more info needed) ───┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd /absolute/path/to/research-agent
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required:
- `GOOGLE_API_KEY` - Google AI API key for Gemini

Optional:
- `TAVILY_API_KEY` - For advanced web search (falls back to DuckDuckGo)

### 3. Run Research

```bash
# Basic research query
python main.py research "What are the latest advances in LLM agents?"

# Save report to file
python main.py research "transformer attention mechanisms" -o report.md

# Test tools
python main.py test-tools

# Show config info
python main.py info
```

## Project Structure

```
research-agent/
├── main.py                 # CLI entry point
├── start_agents.sh         # Start all A2A agents
├── requirements.txt        # Dependencies
└── src/
    ├── config.py           # Configuration
    ├── a2a/                 # A2A Protocol
    │   ├── client.py       # A2A client
    │   └── base_agent.py   # Base agent class
    ├── agents/             # Specialized agents
    │   ├── search_agent.py # Port 8001
    │   ├── paper_agent.py  # Port 8002
    │   └── synthesis_agent.py # Port 8003
    ├── tools/              # Research tools
    └── mcp/                # MCP Gateway
```

## Multi-Agent Architecture

```
Claude → MCP Gateway → A2A Protocol → [Search | Paper | Synthesis] Agents
```

### Start Multi-Agent System

```bash
# Terminal 1: Start all agents
./start_agents.sh

# Terminal 2: Start MCP Gateway
python main.py mcp
```

### Or start agents individually

```bash
python -m src.agents.search_agent    # Port 8001
python -m src.agents.paper_agent     # Port 8002
python -m src.agents.synthesis_agent # Port 8003
```

## Example Output

```markdown
# Research Report: LLM Agents

## Executive Summary
Recent advances in LLM-based agents focus on...

## Key Findings
- ReAct pattern for reasoning and acting
- Multi-agent collaboration systems
- Tool use and function calling

## Sources
1. [Paper Title] - arXiv:2301.xxxxx
2. [Blog Post] - https://...
```

## MCP Server (Model Context Protocol)

This agent can be used as an MCP server, allowing AI assistants like Claude Desktop, Cursor, or other MCP clients to use its research tools.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for articles and content |
| `arxiv_search` | Find academic papers on arXiv |
| `arxiv_get_paper` | Get details of a specific paper by ID |
| `read_paper_pdf` | Extract text from paper PDFs |
| `deep_research` | DeepAgents-powered multi-source research |

### Run as MCP Server

```bash
# Using FastMCP CLI
fastmcp run src/mcp/server.py

# Or directly
python -m src.mcp.server
```

### Configure Claude Desktop

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
        "GOOGLE_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Configure Cursor / VS Code

Add to your settings or MCP config:

```json
{
  "research-agent": {
    "command": "python",
    "args": ["-m", "src.mcp.server"],
    "cwd": "/absolute/path/to/research-agent"
  }
}
```

## License

MIT
