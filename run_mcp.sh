#!/bin/bash
# MCP Gateway for Claude Desktop
# Note: Start agents separately before using this
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" main.py mcp "$@"
