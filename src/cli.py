"""
cli.py — Interactive CLI for the RAG Customer Support Assistant
================================================================
A beautiful Rich-powered terminal interface for testing the system.

Usage:
    python -m src.cli
    python -m src.cli --query "What is your refund policy?"
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from src.logger import logger

console = Console()


BANNER = """
██████╗  █████╗  ██████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║██║  ███╗
██╔══██╗██╔══██║██║   ██║
██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
RAG Customer Support Assistant v1.0
"""


def print_result(result: dict) -> None:
    """Pretty-print a query result to the terminal."""
    escalated = result.get("escalated", False)
    confidence = result.get("confidence", 0.0)
    intent = result.get("intent", "Unknown")
    answer = result.get("answer", "")
    ticket_id = result.get("ticket_id")
    latency = result.get("latency_ms", 0)

    # Status badge
    if escalated:
        status = "[bold yellow]⚡ ESCALATED TO HUMAN AGENT[/bold yellow]"
        border_color = "yellow"
    elif confidence >= 0.75:
        status = "[bold green]✓ HIGH CONFIDENCE[/bold green]"
        border_color = "green"
    else:
        status = "[bold cyan]~ MODERATE CONFIDENCE[/bold cyan]"
        border_color = "cyan"

    # Metadata row
    meta = (
        f"[dim]Intent:[/dim] [bold]{intent}[/bold]  |  "
        f"[dim]Confidence:[/dim] [bold]{confidence:.1%}[/bold]  |  "
        f"[dim]Latency:[/dim] [bold]{latency}ms[/bold]"
    )
    if ticket_id:
        meta += f"  |  [dim]Ticket:[/dim] [bold magenta]{ticket_id[:8]}…[/bold magenta]"

    console.print()
    console.print(Panel(
        f"{status}\n\n{meta}\n\n---\n\n{answer}",
        title="[bold blue]💬 Assistant Response[/bold blue]",
        border_style=border_color,
        padding=(1, 2),
    ))
    console.print()


def show_tickets() -> None:
    """Display all pending HITL tickets in a table."""
    from src.hitl import hitl_queue
    tickets = hitl_queue.get_pending()

    if not tickets:
        console.print("[dim]No pending tickets.[/dim]")
        return

    table = Table(title="📋 Pending HITL Tickets", box=box.ROUNDED, show_lines=True)
    table.add_column("Ticket ID", style="magenta", width=12)
    table.add_column("Query", style="white", max_width=50)
    table.add_column("Reason", style="yellow")
    table.add_column("Created", style="dim")

    for t in tickets:
        tid = t["id"][:8] + "…"
        created = t["created_at"][:19] if t["created_at"] else "—"
        table.add_row(tid, t["query"][:80], t["reason"], created)

    console.print(table)


def interactive_mode() -> None:
    """Run the interactive CLI loop."""
    from src.rag_workflow import RAGAssistant

    console.print(f"[bold blue]{BANNER}[/bold blue]")
    console.print("[dim]Type your question and press Enter. Commands: :quit :tickets :help[/dim]\n")

    try:
        assistant = RAGAssistant()
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load assistant: {e}[/bold red]")
        console.print("[yellow]Run ingestion first:[/yellow] [bold]python -m src.ingest --pdf <file.pdf>[/bold]")
        sys.exit(1)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in (":quit", ":exit", ":q"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif user_input.lower() == ":tickets":
            show_tickets()
            continue
        elif user_input.lower() == ":help":
            console.print(Panel(
                ":quit — Exit\n:tickets — Show pending HITL tickets\n:help — This message",
                title="Commands", border_style="dim"
            ))
            continue

        # Process query
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            result = assistant.query(user_input)

        print_result(result)


def single_query_mode(query: str) -> None:
    """Run a single query and exit."""
    from src.rag_workflow import RAGAssistant

    console.print(f"[bold blue]RAG Customer Support Assistant[/bold blue]\n")

    try:
        assistant = RAGAssistant()
    except Exception as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        sys.exit(1)

    with console.status("[bold green]Processing...[/bold green]", spinner="dots"):
        result = assistant.query(query)

    print_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Customer Support CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query mode")
    args = parser.parse_args()

    if args.query:
        single_query_mode(args.query)
    else:
        interactive_mode()
