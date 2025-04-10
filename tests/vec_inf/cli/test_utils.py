"""Tests for the utils functions in the vec-inf cli."""

from vec_inf.cli._utils import create_table


def test_create_table_with_header():
    """Test that create_table creates a table with the correct header."""
    table = create_table("Key", "Value")
    assert table.columns[0].header == "Key"
    assert table.columns[1].header == "Value"
    assert table.show_header is True


def test_create_table_without_header():
    """Test create_table without header."""
    table = create_table(show_header=False)
    assert table.show_header is False
