# backend/rag_system/cli/commands.py
import asyncio
import json
import logging
import click
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from pathlib import Path
from typing import Optional, List, Dict


from rag_system.config.settings import get_settings, AppSettings
from rag_system.models.schemas import (
    IngestionRequest,
    QueryRequest,
    LLMProvider,
    StatusEnum,
)
from rag_system.services.rag_service import RAGService
from rag_system.core.embeddings import EmbeddingService
from rag_system.core.vector_store import VectorStoreService
from rag_system.services.llm_service import LLMService
from rag_system.utils.exceptions import BaseRAGError

logger = logging.getLogger(__name__) # Will be configured by main.py
console = Console()


# Helper to initialize services for CLI context
# This is synchronous for CLI, but RAGService methods are async
def get_cli_rag_service(settings_override: Optional[AppSettings] = None) -> RAGService:
    """Initializes and returns RAGService for CLI use."""
    # This function is called by each command, so services are re-initialized.
    # For a long-running CLI tool or daemon, you'd initialize once.
    # Here, it's acceptable as CLI commands are typically short-lived.
    try:
        current_settings = settings_override or get_settings()
        embedding_service = EmbeddingService(settings=current_settings)
        vector_store_service = VectorStoreService(
            settings=current_settings, embedding_service=embedding_service
        )
        llm_service = LLMService(settings=current_settings)
        rag_service = RAGService(
            settings=current_settings,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            llm_service=llm_service,
        )
        return rag_service
    except BaseRAGError as e:
        console.print(f"[bold red]Error initializing RAG services for CLI: {e.message}[/bold red]")
        console.print(f"Detail: {e.detail}")
        console.print("Please check your .env configuration and service availability.")
        raise click.Abort() # Aborts the CLI command
    except Exception as e:
        console.print(f"[bold red]Unexpected error initializing RAG services for CLI: {str(e)}[/bold red]")
        console.print("Please check logs for more details.")
        raise click.Abort()


@click.command(name="ingest", help="Ingest documents into the RAG system.")
@click.argument(
    "documents_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    "--patterns",
    "-p",
    "file_patterns_str", # Store as string, parse later
    help='Glob patterns for files to ingest (e.g., "*.txt,*.md"). Comma-separated.',
    default=None, # Will use settings default
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    default=None, # Will use settings default
    help="Search for files in subdirectories.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output."
)
def ingest_documents_cli(documents_path: str, file_patterns_str: Optional[str], recursive: Optional[bool], verbose: bool) -> None:
    """
    CLI command to ingest documents from a specified path.
    """
    settings = get_settings()
    if verbose:
        logger.parent.setLevel(logging.DEBUG) # Set root logger to DEBUG for verbosity
        console.print(f"[dim]Verbose mode enabled. Log level set to DEBUG.[/dim]")

    console.print(
        Panel(f"[bold cyan]Starting Document Ingestion[/bold cyan]\nSource: {Path(documents_path).resolve()}", expand=False)
    )

    # Prepare IngestionRequest
    ingest_req_data = {"source_directory": documents_path}
    if file_patterns_str:
        ingest_req_data["file_patterns"] = [p.strip() for p in file_patterns_str.split(',')]
    # recursive flag default is handled by Pydantic model or RAGService using settings
    if recursive is not None:
        ingest_req_data["recursive"] = recursive

    try:
        ingest_request = IngestionRequest(**ingest_req_data)
    except Exception as e: # Pydantic validation error
        console.print(f"[bold red]Error in ingestion parameters: {e}[/bold red]")
        return

    rag_service = get_cli_rag_service(settings)

    try:
        with console.status("[bold green]Processing documents...", spinner="dots") as status:
            response = asyncio.run(rag_service.ingest_documents(ingest_request))
            status.stop()

        console.print("\n[bold green]Ingestion Complete![/bold green]")
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Message", response.message)
        table.add_row("Documents Processed", str(response.documents_processed))
        table.add_row("Chunks Added/Updated", str(response.chunks_added))
        if response.errors:
            table.add_row("Errors", f"[bold red]{len(response.errors)}[/bold red]")
            console.print(table)
            console.print("\n[bold red]Errors during ingestion:[/bold red]")
            for err in response.errors:
                console.print(f"- {err}")
        else:
            console.print(table)

    except BaseRAGError as e:
        console.print(f"[bold red]Ingestion failed: {e.message}[/bold red]")
        if e.detail:
            console.print(f"Details: {e.detail}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during ingestion: {e}[/bold red]",)
        logger.error("Unexpected CLI ingestion error", exc_info=True)


@click.command(name="query", help="Query the RAG system with a natural language question.")
@click.argument("user_query", type=str)
@click.option(
    "--provider",
    type=click.Choice([p.value for p in LLMProvider], case_sensitive=False),
    default=None, # Will use settings default
    help="LLM provider to use.",
)
@click.option(
    "--model",
    "llm_model_name",
    type=str,
    default=None,
    help="Specific LLM model name to use (overrides provider's default)."
)
@click.option(
    "--top-k",
    type=int,
    default=None, # Will use settings default
    help="Number of relevant chunks to retrieve.",
)
@click.option(
    "--threshold",
    "similarity_threshold",
    type=float,
    default=None, # Will use settings default
    help="Minimum similarity score for retrieved chunks (0.0 to 1.0).",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print raw JSON response."
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output for debugging."
)
def query_system_cli(
    user_query: str,
    provider: Optional[str],
    llm_model_name: Optional[str],
    top_k: Optional[int],
    similarity_threshold: Optional[float],
    raw: bool,
    verbose: bool,
) -> None:
    """
    CLI command to query the RAG system.
    """
    settings = get_settings()
    if verbose:
        logger.parent.setLevel(logging.DEBUG)
        console.print(f"[dim]Verbose mode enabled. Log level set to DEBUG.[/dim]")

    console.print(Panel(f"[bold cyan]Querying RAG System[/bold cyan]\nUser Query: \"{user_query}\"", expand=False))

    query_req_data = {"query_text": user_query}
    if provider:
        try:
            query_req_data["llm_provider"] = LLMProvider(provider.lower())
        except ValueError:
            console.print(f"[bold red]Invalid LLM provider: {provider}. Valid choices: {[p.value for p in LLMProvider]}[/bold red]")
            return
    if llm_model_name:
        query_req_data["llm_model_name"] = llm_model_name
    if top_k is not None:
        query_req_data["top_k"] = top_k
    if similarity_threshold is not None:
        query_req_data["similarity_threshold"] = similarity_threshold

    try:
        query_request = QueryRequest(**query_req_data)
    except Exception as e: # Pydantic validation error
        console.print(f"[bold red]Error in query parameters: {e}[/bold red]")
        return

    rag_service = get_cli_rag_service(settings)

    try:
        with console.status("[bold green]Thinking...", spinner="earth") as status:
            response = asyncio.run(rag_service.query(query_request))
            status.stop()

        if raw:
            json_output = response.model_dump_json(indent=2)
            console.print(Syntax(json_output, "json", theme="native", line_numbers=True))
            return

        console.print("\n[bold green]LLM Answer:[/bold green]")
        console.print(Panel(response.llm_answer, title="Synthesized Answer", border_style="green", expand=True))

        console.print(f"\n[dim]LLM Used: {response.llm_provider_used.value} ({response.llm_model_used})[/dim]")

        if response.retrieved_chunks:
            console.print("\n[bold yellow]Retrieved Chunks (Sources):[/bold yellow]")
            table = Table(title="Retrieved Chunks", show_lines=True)
            table.add_column("ID", style="cyan", max_width=20, overflow="fold")
            table.add_column("Score", style="magenta", justify="right")
            table.add_column("Source", style="green", max_width=30, overflow="fold")
            table.add_column("Content Snippet", style="white", max_width=60, overflow="fold")

            for chunk in response.retrieved_chunks:
                source_display = chunk.metadata.get("filename") or chunk.metadata.get("source_id", "N/A")
                table.add_row(
                    chunk.id,
                    f"{chunk.score:.4f}",
                    source_display,
                    chunk.content.replace("\n", " ")[:150] + "...",
                )
            console.print(table)
        else:
            console.print("\n[yellow]No relevant chunks were retrieved for this query.[/yellow]")

    except BaseRAGError as e:
        console.print(f"[bold red]Query failed: {e.message}[/bold red]")
        if e.detail:
            console.print(f"Details: {e.detail}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during query: {e}[/bold red]")
        logger.error("Unexpected CLI query error", exc_info=True)


@click.command(name="status", help="Check the status of the RAG system and its components.")
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print raw JSON response."
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output for debugging."
)
def system_status_cli(raw: bool, verbose: bool) -> None:
    """
    CLI command to check the system status.
    """
    settings = get_settings()
    if verbose:
        logger.parent.setLevel(logging.DEBUG)
        console.print(f"[dim]Verbose mode enabled. Log level set to DEBUG.[/dim]")

    console.print(Panel("[bold cyan]Checking RAG System Status[/bold cyan]", expand=False))
    rag_service = get_cli_rag_service(settings)

    try:
        with console.status("[bold green]Fetching status...", spinner="dots") as status_spinner:
            status_response = asyncio.run(rag_service.get_system_status())
            status_spinner.stop()

        if raw:
            json_output = status_response.model_dump_json(indent=2)
            console.print(Syntax(json_output, "json", theme="native", line_numbers=True))
            return

        console.print(f"\n[bold]Overall System Status: ", end="")
        status_color = "green"
        if status_response.system_status == StatusEnum.ERROR:
            status_color = "red"
        elif status_response.system_status == StatusEnum.DEGRADED:
            status_color = "yellow"
        console.print(f"[{status_color}]{status_response.system_status.value}[/{status_color}]")
        console.print(f"[dim]App: {status_response.app_name or settings.APP_NAME}, Env: {status_response.environment or settings.ENVIRONMENT}, Version: {status_response.version or 'N/A'}[/dim]")
        console.print(f"[dim]Timestamp: {status_response.timestamp.isoformat()}[/dim]")


        if status_response.components:
            console.print("\n[bold yellow]Component Status:[/bold yellow]")
            table = Table(title="Components", show_lines=True)
            table.add_column("Component Name", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Message", style="white", overflow="fold")
            table.add_column("Details", style="dim", overflow="fold")

            for comp in status_response.components:
                comp_status_color = "green"
                if comp.status == StatusEnum.ERROR:
                    comp_status_color = "red"
                elif comp.status == StatusEnum.DEGRADED:
                    comp_status_color = "yellow"

                details_str = json.dumps(comp.details, indent=2) if comp.details else "N/A"
                table.add_row(
                    comp.name,
                    f"[{comp_status_color}]{comp.status.value}[/{comp_status_color}]",
                    comp.message or "N/A",
                    details_str if verbose else (details_str[:100]+"..." if len(details_str) > 100 else details_str)
                )
            console.print(table)
        else:
            console.print("\n[yellow]No component status details available.[/yellow]")

    except BaseRAGError as e:
        console.print(f"[bold red]Failed to get system status: {e.message}[/bold red]")
        if e.detail:
            console.print(f"Details: {e.detail}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred while fetching status: {e}[/bold red]")
        logger.error("Unexpected CLI status error", exc_info=True)

