"""Helper functions for the CLI."""

from rich.table import Table


def create_table(
    key_title: str = "", value_title: str = "", show_header: bool = True
) -> Table:
    """Create a table for displaying model status."""
    table = Table(show_header=show_header, header_style="bold magenta")
    table.add_column(key_title, style="dim")
    table.add_column(value_title)
    return table
