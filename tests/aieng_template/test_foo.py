"""Test for foo module."""

from aieng_template.foo import foo as foofn


def test_foofn() -> None:
    """Test foofn function."""
    assert foofn("bar") == "foobar"
