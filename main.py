#!/usr/bin/env python3
"""
Research Agent - AI-powered research assistant using LangGraph.

Usage:
    python main.py research "What are the latest advances in LLM agents?"
    python main.py chat   # Interactive mode
    python main.py --help
"""

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

from src.config import get_config

app = typer.Typer(
    name="research-agent",
    help="AI-powered research assistant using LangGraph",
    add_completion=False,
)
console = Console()



@app.command()
def research(
    query: str = typer.Argument(..., help="Research query to investigate"),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Save report to file (markdown)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed progress"
    ),
):
    """
    Run the research agent on a query and generate a comprehensive report.
    
    Example:
        python main.py research "What are the latest advances in transformer architectures?"
    """
    from src.agent.graph import run_research_sync

    config = get_config()
    
    # Validate configuration
    if not config.has_google_api:
        console.print("[red]Error:[/red] GOOGLE_API_KEY not set in environment")
        console.print("Copy .env.example to .env and add your API key")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold blue]Research Agent[/bold blue]\n\n"
        f"Query: [green]{query}[/green]\n"
        f"Model: {config.llm_model}",
        title="🔬 Starting Research"
    ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Researching...", total=None)
            
            # Run the research agent
            report = run_research_sync(query)
            
            progress.update(task, description="[green]Complete![/green]")
        
        # Display the report
        console.print("\n")
        console.print(Panel(
            Markdown(report),
            title="📋 Research Report",
            border_style="green"
        ))
        
        # Save to file if requested
        if output:
            output.write_text(report)
            console.print(f"\n[green]Report saved to:[/green] {output}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def test_tools():
    """Test individual tools to verify they work correctly."""
    from src.tools.web_search import search_web
    from src.tools.arxiv_search import search_arxiv
    
    console.print("[bold]Testing Tools...[/bold]\n")
    
    # Test web search
    console.print("[blue]1. Testing Web Search (DuckDuckGo)...[/blue]")
    try:
        results = search_web("LangGraph agent tutorial", max_results=2)
        console.print(f"   [green]✓[/green] Found {len(results)} web results")
        for r in results[:2]:
            console.print(f"     - {r['title'][:50]}...")
    except Exception as e:
        console.print(f"   [red]✗[/red] Web search failed: {e}")
    
    # Test arXiv search
    console.print("\n[blue]2. Testing ArXiv Search...[/blue]")
    try:
        papers = search_arxiv("LLM agents", max_results=2)
        console.print(f"   [green]✓[/green] Found {len(papers)} papers")
        for p in papers[:2]:
            console.print(f"     - {p['title'][:50]}...")
    except Exception as e:
        console.print(f"   [red]✗[/red] ArXiv search failed: {e}")
    
    console.print("\n[green]Tool testing complete![/green]")


@app.command()
def info():
    """Show configuration and environment info."""
    config = get_config()
    
    console.print(Panel.fit(
        f"[bold]Configuration[/bold]\n\n"
        f"LLM Model: {config.llm_model}\n"
        f"Temperature: {config.llm_temperature}\n"
        f"Max Iterations: {config.max_iterations}\n"
        f"Max Papers: {config.max_papers}\n\n"
        f"[bold]API Keys[/bold]\n"
        f"Google API: {'[green]✓ Set[/green]' if config.has_google_api else '[red]✗ Not set[/red]'}\n"
        f"Tavily API: {'[green]✓ Set[/green]' if config.has_tavily_api else '[yellow]○ Not set (using DuckDuckGo)[/yellow]'}",
        title="ℹ️ Research Agent Info"
    ))


@app.command()
def history(
    session_id: int = typer.Argument(
        None,
        help="View details of a specific session (omit to list all)"
    ),
    delete: bool = typer.Option(
        False,
        "--delete", "-d",
        help="Delete the specified session"
    ),
):
    """
    View research history — list past sessions or view details.

    Examples:
        python main.py history          # List all sessions
        python main.py history 3        # View session #3 details
        python main.py history 3 -d     # Delete session #3
    """
    import asyncio
    from src.storage.database import init_db, list_sessions, get_session, delete_session
    from rich.table import Table

    async def _run():
        await init_db()

        if session_id and delete:
            deleted = await delete_session(session_id)
            if deleted:
                console.print(f"[green]Session #{session_id} deleted.[/green]")
            else:
                console.print(f"[red]Session #{session_id} not found.[/red]")
            return

        if session_id:
            session = await get_session(session_id)
            if not session:
                console.print(f"[red]Session #{session_id} not found.[/red]")
                return

            console.print(Panel.fit(
                f"[bold]Query:[/bold] {session['query']}\n"
                f"[bold]Created:[/bold] {session['created_at']}\n"
                f"[bold]Papers:[/bold] {len(session['papers'])}  |  "
                f"[bold]Web Sources:[/bold] {len(session['web_sources'])}",
                title=f"Session #{session_id}"
            ))

            if session['papers']:
                console.print("\n[bold]Papers:[/bold]")
                for p in session['papers']:
                    year = f" ({p['year']})" if p.get('year') else ""
                    console.print(f"  - {p['title']}{year}")

            if session['web_sources']:
                console.print("\n[bold]Web Sources:[/bold]")
                for w in session['web_sources']:
                    console.print(f"  - {w['title']}  [dim]{w['url']}[/dim]")

            if session.get('report'):
                console.print("\n")
                console.print(Panel(
                    Markdown(session['report']),
                    title="Report",
                    border_style="green"
                ))
            return

        # List all sessions
        sessions = await list_sessions()
        if not sessions:
            console.print("[dim]No research sessions yet. Run 'python main.py research \"query\"' first.[/dim]")
            return

        table = Table(title="Research History")
        table.add_column("#", style="bold", width=4)
        table.add_column("Query", max_width=50)
        table.add_column("Papers", justify="center", width=7)
        table.add_column("Web", justify="center", width=5)
        table.add_column("Date", width=19)

        for s in sessions:
            table.add_row(
                str(s['id']),
                s['query'][:50],
                str(s['paper_count']),
                str(s['web_count']),
                str(s['created_at'])[:19],
            )

        console.print(table)

    asyncio.run(_run())


@app.command(name="knowledge-graph")
def knowledge_graph(
    session_id: int = typer.Argument(..., help="Session ID to build graph from"),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Output HTML file path (default: data/kg_session_{id}.html)"
    ),
    persist: bool = typer.Option(
        False, "--persist", "-p",
        help="Save graph to database knowledge tables"
    ),
):
    """
    Build and visualize a knowledge graph from a research session.

    Example:
        python main.py knowledge-graph 3
        python main.py knowledge-graph 3 --persist
    """
    import asyncio
    from src.tools.knowledge_graph import (
        build_graph_from_session, render_graph_html,
        get_graph_stats, persist_graph,
    )
    from src.storage.database import init_db

    async def _run():
        await init_db()
        G = await build_graph_from_session(session_id)
        stats = get_graph_stats(G)

        console.print(Panel.fit(
            f"[bold]Nodes:[/bold] {stats['nodes']}  |  "
            f"[bold]Edges:[/bold] {stats['edges']}  |  "
            f"[bold]Density:[/bold] {stats['density']}\n"
            f"[bold]Types:[/bold] {', '.join(f'{k}: {v}' for k, v in stats['node_types'].items())}",
            title=f"Knowledge Graph — Session #{session_id}"
        ))

        out_path = output or Path(f"data/kg_session_{session_id}.html")
        html_path = render_graph_html(G, out_path)
        console.print(f"[green]Graph saved to:[/green] {html_path}")

        if persist:
            result = await persist_graph(session_id, G)
            console.print(
                f"[green]Persisted:[/green] {result['node_count']} nodes, "
                f"{result['edge_count']} edges"
            )

    asyncio.run(_run())


@app.command()
def trends(
    query: str = typer.Argument(
        None,
        help="Topic to analyze publication trends for (omit for session trends)"
    ),
    max_papers: int = typer.Option(50, "--max", "-m", help="Max papers to fetch"),
):
    """
    Analyze research trends — publication timelines and citation data.

    Examples:
        python main.py trends                          # Session trends
        python main.py trends "transformer attention"  # Publication trends
    """
    import asyncio
    from src.tools.trend_analysis import (
        session_trends, publication_trends, format_trends_markdown,
    )

    async def _run():
        if query:
            return await publication_trends(query, max_papers)
        return await session_trends()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing trends...", total=None)
        result = asyncio.run(_run())
        progress.update(task, description="[green]Complete![/green]")

    md = format_trends_markdown(result)
    console.print("\n")
    console.print(Panel(Markdown(md), title="Trend Analysis", border_style="cyan"))

    for chart in result.get("charts", []):
        console.print(f"[green]Chart saved:[/green] {chart}")


@app.command(name="export")
def export_report(
    session_id: int = typer.Argument(..., help="Session ID to export"),
    fmt: str = typer.Option(
        "docx", "--format", "-f",
        help="Export format: docx, pptx, or latex"
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """
    Export a research session report to DOCX, PPTX, or LaTeX.

    Examples:
        python main.py export 3 --format docx
        python main.py export 3 -f pptx -o slides.pptx
        python main.py export 3 -f latex
    """
    import asyncio
    from src.storage.database import init_db, get_session
    from src.tools.export import export_docx, export_pptx, export_latex

    async def _get():
        await init_db()
        return await get_session(session_id)

    session = asyncio.run(_get())
    if not session:
        console.print(f"[red]Session #{session_id} not found.[/red]")
        raise typer.Exit(1)

    report = session.get("report", "")
    if not report:
        console.print(f"[red]Session #{session_id} has no report.[/red]")
        raise typer.Exit(1)

    title = f"Research: {session['query'][:60]}"
    ext_map = {"docx": ".docx", "pptx": ".pptx", "latex": ".tex"}
    ext = ext_map.get(fmt, ".docx")
    out = output or Path(f"output/session_{session_id}{ext}")

    exporters = {"docx": export_docx, "pptx": export_pptx, "latex": export_latex}
    exporter = exporters.get(fmt)
    if not exporter:
        console.print(f"[red]Unknown format: {fmt}. Use docx, pptx, or latex.[/red]")
        raise typer.Exit(1)

    path = exporter(report, out, title)
    console.print(f"[green]Exported to:[/green] {path}")


@app.command()
def compare(
    arxiv_ids: str = typer.Argument(
        ...,
        help="Comma-separated arXiv IDs (e.g., '2301.00001,2305.12345')"
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save comparison to file (markdown)"
    ),
):
    """
    Compare multiple arXiv papers side-by-side with structured extraction.

    Example:
        python main.py compare "2301.00001,2305.12345"
    """
    import asyncio
    from src.tools.arxiv_search import get_paper_by_id
    from src.tools.paper_comparison import compare_papers_structured, format_comparison_markdown

    ids = [aid.strip() for aid in arxiv_ids.split(",") if aid.strip()]
    if len(ids) < 2:
        console.print("[red]Need at least 2 arXiv IDs separated by commas.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Fetching {len(ids)} papers...[/bold]")
    papers = []
    for aid in ids[:5]:
        paper = get_paper_by_id(aid)
        if paper:
            console.print(f"  [green]✓[/green] {paper['title'][:60]}...")
            papers.append(paper)
        else:
            console.print(f"  [red]✗[/red] {aid} not found")

    if len(papers) < 2:
        console.print("[red]Need at least 2 valid papers to compare.[/red]")
        raise typer.Exit(1)

    async def _compare():
        return await compare_papers_structured(papers)

    console.print("\n[bold]Running structured comparison...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Comparing papers...", total=None)
        try:
            matrix = asyncio.run(_compare())
            progress.update(task, description="[green]Complete![/green]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    markdown = format_comparison_markdown(matrix)

    console.print("\n")
    console.print(Panel(
        Markdown(markdown),
        title="Paper Comparison Matrix",
        border_style="cyan"
    ))

    if output:
        output.write_text(markdown)
        console.print(f"\n[green]Comparison saved to:[/green] {output}")


@app.command()
def mcp(
    transport: str = typer.Option(
        "stdio",
        "--transport", "-t",
        help="Transport type: stdio (for Claude Desktop) or sse (for web)"
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port for SSE transport"
    ),
):
    """
    Run the Research Agent as an MCP server.
    
    This exposes the research tools via Model Context Protocol,
    allowing AI assistants like Claude Desktop to use them.
    
    Examples:
        # Run for Claude Desktop (stdio)
        python main.py mcp
        
        # Run as web server (SSE)
        python main.py mcp --transport sse --port 8000
    """
    from src.mcp.server import mcp as mcp_server
    
    # Only print info for SSE mode (stdio must be clean for MCP protocol)
    if transport == "sse":
        console.print(Panel.fit(
            f"[bold blue]Research Agent MCP Server[/bold blue]\n\n"
            f"Transport: [green]{transport}[/green]\n"
            f"Port: {port}\n\n"
            f"[bold]Available Tools:[/bold]\n"
            f"  • web_search - Search the web\n"
            f"  • arxiv_search - Search arXiv papers\n"
            f"  • arxiv_get_paper - Get paper details\n"
            f"  • read_paper_pdf - Parse PDF papers\n"
            f"  • deep_research - Full research workflow",
            title="🔌 Starting MCP Server"
        ))
    
    try:
        if transport == "sse":
            mcp_server.run(transport="sse", port=port)
        else:
            # stdio mode - no output allowed!
            mcp_server.run()
    except KeyboardInterrupt:
        if transport == "sse":
            console.print("\n[yellow]MCP Server stopped.[/yellow]")
    except Exception as e:
        if transport == "sse":
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def chat(
    verbose: bool = typer.Option(
        True,
        "--verbose", "-v",
        help="Show step-by-step execution flow"
    ),
):
    """
    Interactive chat mode - research multiple queries in a session.
    
    Connects to A2A agents and allows continuous conversation.
    Shows real-time execution flow with --verbose (default: on).
    Type 'quit' or 'exit' to stop.
    
    Example:
        python main.py chat
        python main.py chat --no-verbose
    """
    from src.agent.graph import run_research_stream, run_research_sync
    
    config = get_config()
    
    # Validate configuration
    if not config.has_google_api:
        console.print("[red]Error:[/red] GOOGLE_API_KEY not set in environment")
        console.print("Copy .env.example to .env and add your API key")
        raise typer.Exit(1)
    
    # Check A2A agents health
    console.print("[dim]Checking A2A agents...[/dim]")
    import httpx
    agents = config.agent_health_targets()
    
    all_healthy = True
    for name, url in agents:
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            if response.status_code == 200:
                console.print(f"  [green]✓[/green] {name}")
            else:
                console.print(f"  [red]✗[/red] {name} (unhealthy)")
                all_healthy = False
        except Exception:
            console.print(f"  [red]✗[/red] {name} (offline)")
            all_healthy = False
    
    if not all_healthy:
        console.print("\n[yellow]Warning:[/yellow] Some agents are offline.")
        console.print("Run [bold]./start_agents.sh[/bold] first.\n")
    
    mode_text = "[green]verbose[/green]" if verbose else "[dim]quiet[/dim]"
    console.print(Panel.fit(
        f"[bold blue]Research Agent - Interactive Mode[/bold blue]\n\n"
        f"Type your research query and press Enter.\n"
        f"Streaming mode: {mode_text}\n"
        f"Commands: [dim]quit, exit, clear, help, verbose[/dim]",
        title="🔬 Chat Mode"
    ))
    
    history = []
    
    while True:
        try:
            # Get user input
            console.print()
            query = console.input("[bold green]You>[/bold green] ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if query.lower() == "clear":
                console.clear()
                continue
            
            if query.lower() == "verbose":
                verbose = not verbose
                status = "[green]ON[/green]" if verbose else "[red]OFF[/red]"
                console.print(f"Streaming mode: {status}")
                continue
            
            if query.lower() == "help":
                console.print("""
[bold]Commands:[/bold]
  quit, exit, q  - Exit chat mode
  clear          - Clear screen
  verbose        - Toggle streaming mode on/off
  help           - Show this help

[bold]Workflow Flow:[/bold]
  1. 🔍 Decompose query → sub-queries
  2. 🌐 Web search (A2A → Search Agent)
  3. 📚 ArXiv search (A2A → Search Agent)  
  4. 🧠 Synthesize (A2A → Synthesis Agent)
  5. 📝 Generate report (A2A → Synthesis Agent)
""")
                continue
            
            # Run research with or without streaming
            console.print()
            
            if verbose:
                # Streaming mode - show each step
                from rich.live import Live
                from rich.table import Table
                
                steps = []
                current_step = {"name": "", "status": ""}
                
                def on_node_start(node_name: str, display_name: str):
                    current_step["name"] = display_name
                    current_step["status"] = "[yellow]running...[/yellow]"
                    console.print(f"  {display_name} [yellow]...[/yellow]")
                
                def on_node_end(node_name: str, result: dict):
                    # Show result summary
                    info_parts = []
                    if "queries" in result:
                        info_parts.append(f"{len(result['queries'])} queries")
                    if "web_count" in result:
                        info_parts.append(f"{result['web_count']} web results")
                    if "paper_count" in result:
                        info_parts.append(f"{result['paper_count']} papers")
                    if "synthesis_length" in result:
                        info_parts.append(f"{result['synthesis_length']} chars")
                    if "report_length" in result:
                        info_parts.append(f"{result['report_length']} chars")
                    
                    if info_parts:
                        console.print(f"    └─ [dim]{', '.join(info_parts)}[/dim]")
                
                try:
                    console.print("[bold]Execution Flow:[/bold]")
                    report = run_research_stream(
                        query,
                        on_node_start=on_node_start,
                        on_node_end=on_node_end
                    )
                    console.print("  [green]✓ Complete![/green]")
                except Exception as e:
                    console.print(f"  [red]✗ Error: {e}[/red]")
                    continue
            else:
                # Quiet mode - just spinner
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Researching via A2A agents...", total=None)
                    
                    try:
                        report = run_research_sync(query)
                        progress.update(task, description="[green]Complete![/green]")
                    except Exception as e:
                        progress.update(task, description=f"[red]Error: {e}[/red]")
                        console.print(f"[red]Failed:[/red] {e}")
                        continue
            
            # Display result
            console.print()
            console.print(Panel(
                Markdown(report),
                title="📋 Research Report",
                border_style="green"
            ))
            
            # Save to history
            history.append({"query": query, "report": report})
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit.[/yellow]")
            continue
        except EOFError:
            break
    
    console.print(f"\n[dim]Session ended. {len(history)} queries researched.[/dim]")


if __name__ == "__main__":
    app()
