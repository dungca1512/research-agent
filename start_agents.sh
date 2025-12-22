#!/bin/bash
# Start all A2A agents for the Research Agent system

echo "🚀 Starting Research Agent Multi-Agent System..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate agent

cd /Users/dungca/agent

echo -e "${BLUE}Starting Search Agent on port 8001...${NC}"
python -m src.agents.search_agent &
SEARCH_PID=$!
sleep 2

echo -e "${BLUE}Starting Paper Agent on port 8002...${NC}"
python -m src.agents.paper_agent &
PAPER_PID=$!
sleep 2

echo -e "${BLUE}Starting Synthesis Agent on port 8003...${NC}"
python -m src.agents.synthesis_agent &
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
echo "  python main.py mcp"
echo ""

# Wait for all background processes
wait
