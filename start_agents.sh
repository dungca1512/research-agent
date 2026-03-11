#!/bin/bash
# Start all A2A agents for the Research Agent system
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "🚀 Starting Research Agent Multi-Agent System..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Optionally activate a conda environment if requested.
if [[ -n "${RESEARCH_AGENT_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found but RESEARCH_AGENT_CONDA_ENV was set."
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${RESEARCH_AGENT_CONDA_ENV}"
fi

cd "${SCRIPT_DIR}"

echo -e "${BLUE}Starting Search Agent on port 8001...${NC}"
"${PYTHON_BIN}" -m src.agents.search_agent &
SEARCH_PID=$!
sleep 2

echo -e "${BLUE}Starting Paper Agent on port 8002...${NC}"
"${PYTHON_BIN}" -m src.agents.paper_agent &
PAPER_PID=$!
sleep 2

echo -e "${BLUE}Starting Synthesis Agent on port 8003...${NC}"
"${PYTHON_BIN}" -m src.agents.synthesis_agent &
SYNTHESIS_PID=$!
sleep 2

echo ""
echo -e "${GREEN}✅ All agents started!${NC}"
echo ""
echo "Agent PIDs:"
echo "  Search Agent:    $SEARCH_PID"
echo "  Paper Agent:     $PAPER_PID"
echo "  Synthesis Agent: $SYNTHESIS_PID"
echo ""
echo "To stop all agents: kill $SEARCH_PID $PAPER_PID $SYNTHESIS_PID"
echo ""
echo "Now you can run the MCP Gateway:"
echo "  ${PYTHON_BIN} main.py mcp"
echo ""

# Wait for all background processes
wait
