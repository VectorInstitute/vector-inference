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

            # API imports
            import vec_inf.api
            import vec_inf.api.client
            import vec_inf.api._helper
            import vec_inf.api._models

            # Shared imports
            import vec_inf.shared
            import vec_inf.shared._config
            import vec_inf.shared._exceptions
            import vec_inf.shared._helper
            import vec_inf.shared._models
            import vec_inf.shared._utils
            import vec_inf.shared._vars

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
