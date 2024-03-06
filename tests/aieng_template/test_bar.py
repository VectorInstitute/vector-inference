"""Test for bar module."""

from aieng_template.bar import bar as barfn


def test_barfn() -> None:
    """Test bar function."""
    assert barfn("foo") == "barfoo"
