#!/usr/bin/env python3
# query.py — Interactive Q&A loop with memory retention. Run after ingest.py.

from logging import config
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent))

import time
from google.genai.errors import ServerError

from src.pipeline.index_builder import load_index, configure_settings
from src.agent.qa import ask
from src.agent.memory import get_memory


def call_with_retry(question, index, memory, max_retries=3):
    """Call ask() with exponential backoff retry on ServerError."""
    delays = [1, 2, 4]
    
    for attempt in range(max_retries):
        try:
            return ask(question, index, memory=memory)
        except ServerError as e:
            if getattr(config, "DEBUG_LLM", False):
                print(f"[query] ServerError on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = delays[attempt]
                console.print(f"[yellow]Service unavailable. Retrying in {delay}s... ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(delay)
            else:
                raise

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]📚 Document Q&A Agent[/bold cyan]\n"
        "[dim]Google GenAI + ChromaDB + Memory[/dim]",
        border_style="cyan"
    ))

    console.print("\n[dim]Loading index from ChromaDB...[/dim]")
    try:
        configure_settings()
        index = load_index()
        console.print("[green]✓ Index loaded successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to load index: {e}[/red]")
        console.print("[yellow]Tip: Run `python ingestion.py` first to build the index.[/yellow]")
        return

    # Initialize memory
    console.print("[dim]Initializing conversation memory...[/dim]")
    try:
        memory = get_memory()
        memory.initialize()
        console.print("[green]✓ Memory initialized[/green]\n")
    except Exception as e:
        console.print(f"[yellow]⚠ Memory init warning: {e}[/yellow]")
        memory = None

    console.print("\n[dim]Type your question and press Enter. Type 'quit' to exit.[/dim]\n")

    while True:
        try:
            question = console.input("[bold yellow]🔍 Question:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        console.print("\n[dim]Retrieving relevant chunks...[/dim]")

        try:
            result = call_with_retry(question, index, memory=memory)
        except ServerError:
            console.print(Panel(
                "[yellow]The service is temporarily unavailable due to high demand. "
                "Please try again in a moment.[/yellow]",
                title="[bold red]⚠ Service Unavailable[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))
            console.print()
            continue
        except Exception as e:
            console.print(Panel(
                "[yellow]Something went wrong. Please try again.[/yellow]",
                title="[bold red]⚠ Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))
            console.print()
            continue

        # Print answer
        console.print(Panel(
            result["answer"],
            title="[bold green]💬 Answer[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        # console.print(result["answer"] + "\n")

        # # Print sources
        # if result["sources"]:
        #     sources_text = Text()
        #     for i, src in enumerate(result["sources"], 1):
        #         # We use markup=True here because qa.py is returning rich [link=...] syntax
        #         sources_text.append(Text.from_markup(f"  {i}. {src}\n", style="dim"))
        #     console.print(Panel(
        #         sources_text,
        #         title="[bold blue]📎 Sources[/bold blue]",
        #         border_style="blue",
        #         padding=(0, 1)
        #     ))

        console.print()


    console.print("\n[dim]Bye! 👋[/dim]\n")


if __name__ == "__main__":
    main()
