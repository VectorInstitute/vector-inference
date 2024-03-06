"""Integration test example."""

import pytest

from aieng_template.bar import bar as barfn
from aieng_template.foo import foo as foofn


@pytest.mark.integration_test()
def test_foofn_barfn() -> None:
    """Test foo and bar."""
    foobar = foofn("bar") + " " + barfn("foo")
    assert foobar == "foobar barfoo"
