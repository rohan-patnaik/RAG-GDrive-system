# backend/rag_system/cli/utils.py
import logging
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def display_error(message: str, details: str = None) -> None:
    """Helper function to display errors in the CLI."""
    console.print(f"[bold red]ERROR: {message}[/bold red]")
    if details:
        console.print(f"[dim]Details: {details}[/dim]")


def display_success(message: str) -> None:
    """Helper function to display success messages."""
    console.print(f"[bold green]SUCCESS: {message}[/bold green]")


def display_warning(message: str) -> None:
    """Helper function to display warning messages."""
    console.print(f"[bold yellow]WARNING: {message}[/bold yellow]")


# Add any other CLI-specific utility functions here.
# For example, formatting outputs, handling user confirmations, etc.

if __name__ == "__main__":
    # Example usage
    display_error("Something went wrong.", "Check the logs for more information.")
    display_success("Operation completed successfully.")
    display_warning("This is a warning message.")
