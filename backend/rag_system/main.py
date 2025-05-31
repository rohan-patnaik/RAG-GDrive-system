#!/usr/bin/env python3
# backend/rag_system/main.py
"""
Main entry point for the RAG System CLI.
This module provides the command-line interface for the RAG system.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

import click
import structlog
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Setup rich traceback handling
install_rich_traceback(show_locals=True)

# Import after rich setup for better error display
from rag_system.config.logging_config import setup_logging
from rag_system.config.settings import get_settings, AppSettings
from rag_system.cli.commands import ingest_documents_cli, query_system_cli, system_status_cli
from rag_system import __version__

console = Console()
logger = structlog.get_logger(__name__)


def validate_python_version() -> None:
    """Ensure we're running on a supported Python version."""
    if sys.version_info < (3, 9):
        console.print(
            "[bold red]Error:[/bold red] RAG System requires Python 3.9 or higher. "
            f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.",
            style="red"
        )
        sys.exit(1)


def setup_app_logging(verbose: bool, debug: bool, settings: AppSettings) -> None:
    """Setup application logging based on CLI flags and settings."""
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = settings.LOG_LEVEL
    
    setup_logging(log_level, settings.LOG_FILE_PATH)
    logger.info("RAG System CLI started", version=__version__, log_level=log_level)


@click.group()
@click.version_option(version=__version__, prog_name="RAG System")
@click.option(
    '--verbose', '-v', 
    is_flag=True, 
    help='Enable verbose output (INFO level logging)'
)
@click.option(
    '--debug', '-d',
    is_flag=True,
    help='Enable debug output (DEBUG level logging)'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help='Path to custom .env configuration file'
)
@click.pass_context
def cli(
    ctx: click.Context, 
    verbose: bool, 
    debug: bool, 
    config_file: Optional[str]
) -> None:
    """
    RAG Document System - Production-Ready Retrieval-Augmented Generation
    
    A comprehensive system for document ingestion, embedding, and LLM-powered querying
    with support for OpenAI, Anthropic, and Google Gemini.
    
    Examples:
        rag-system ingest ./documents --recursive
        rag-system query "What is artificial intelligence?"
        rag-system status
    """
    validate_python_version()
    
    # Load settings (optionally from custom config file)
    try:
        if config_file:
            # Override default .env path
            import os
            os.environ["RAG_CONFIG_FILE"] = config_file
        
        settings = get_settings()
        setup_app_logging(verbose, debug, settings)
        
        # Store settings in context for subcommands
        ctx.ensure_object(dict)
        ctx.obj['settings'] = settings
        ctx.obj['verbose'] = verbose
        ctx.obj['debug'] = debug
        
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}", style="red")
        logger.error("Failed to load configuration", error=str(e))
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show detailed version information."""
    settings: AppSettings = ctx.obj['settings']
    
    console.print(f"[bold]RAG System[/bold] version [cyan]{__version__}[/cyan]")
    console.print(f"Python: [cyan]{sys.version.split()[0]}[/cyan]")
    console.print(f"Environment: [cyan]{settings.ENVIRONMENT}[/cyan]")
    console.print(f"Config Path: [cyan]{settings.LOG_FILE_PATH}[/cyan]")


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Run system diagnostics and health checks."""
    console.print("[bold]Running RAG System Diagnostics...[/bold]")
    
    settings: AppSettings = ctx.obj['settings']
    issues_found = []
    
    # Check Python version
    console.print("✓ Python version check passed", style="green")
    
    # Check required directories
    vector_store_path = Path(settings.VECTOR_STORE_PATH)
    if not vector_store_path.parent.exists():
        issues_found.append(f"Vector store parent directory missing: {vector_store_path.parent}")
    else:
        console.print("✓ Vector store directory accessible", style="green")
    
    # Check API keys
    api_keys = {
        "OpenAI": settings.OPENAI_API_KEY,
        "Anthropic": settings.ANTHROPIC_API_KEY,
        "Google": settings.GOOGLE_API_KEY,
    }
    
    for provider, key in api_keys.items():
        if key and len(key) > 10:
            console.print(f"✓ {provider} API key configured", style="green")
        else:
            console.print(f"⚠ {provider} API key missing or invalid", style="yellow")
    
    # Check if at least one API key is valid
    if not any(key and len(key) > 10 for key in api_keys.values()):
        issues_found.append("No valid LLM API keys found")
    
    if issues_found:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for issue in issues_found:
            console.print(f"  • {issue}", style="red")
        sys.exit(1)
    else:
        console.print("\n[bold green]All diagnostics passed![/bold green]")


# Add subcommands
cli.add_command(ingest_documents_cli, name="ingest")
cli.add_command(query_system_cli, name="query")
cli.add_command(system_status_cli, name="status")


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error in CLI", error=str(e), exc_info=True)
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}", style="red")
        console.print("[dim]Check logs for more details.[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
