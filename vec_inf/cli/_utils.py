"""Helper functions for the CLI.

This module provides utility functions for creating consistent table displays
in the command-line interface.
"""

from rich.table import Table


def create_table(
    key_title: str = "", value_title: str = "", show_header: bool = True
) -> Table:
    """Create a table for displaying model status.

    Creates a two-column Rich table with consistent styling for displaying
    key-value pairs in the CLI.

    Parameters
    ----------
    key_title : str, default=""
        Title for the key column
    value_title : str, default=""
        Title for the value column
    show_header : bool, default=True
        Whether to display column headers

    Returns
    -------
    Table
        Rich Table instance with configured styling:
        - Headers in bold magenta
        - Key column in dim style
        - Value column in default style
    """
    table = Table(show_header=show_header, header_style="bold magenta")
    table.add_column(key_title, style="dim")
    table.add_column(value_title)
    return table
