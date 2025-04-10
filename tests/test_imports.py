"""Test the imports of the vec_inf package."""

import unittest

import pytest


class TestVecInfImports(unittest.TestCase):
    """Test the imports of the vec_inf package."""

    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            # CLI imports
            import vec_inf.cli
            import vec_inf.cli._cli
            import vec_inf.cli._helper

            # Client imports
            import vec_inf.client
            import vec_inf.client._config
            import vec_inf.client._exceptions
            import vec_inf.client._helper
            import vec_inf.client._models
            import vec_inf.client._utils
            import vec_inf.client._vars  # noqa: F401

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
