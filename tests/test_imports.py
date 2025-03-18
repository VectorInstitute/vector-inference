"""Test the imports of the vec_inf package."""

import unittest


class TestVecInfImports(unittest.TestCase):
    """Test the imports of the vec_inf package."""

    def test_import_cli_modules(self):
        """Test the imports of the vec_inf.cli modules."""
        try:
            import vec_inf.cli._cli
            import vec_inf.cli._config
            import vec_inf.cli._helper
            import vec_inf.cli._utils  # noqa: F401
        except ImportError as e:
            self.fail(f"Import failed: {e}")
