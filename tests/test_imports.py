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
            import vec_inf.client._client_vars  # noqa: F401
            import vec_inf.client._config
            import vec_inf.client._exceptions
            import vec_inf.client._helper
            import vec_inf.client._slurm_script_generator
            import vec_inf.client._utils
            import vec_inf.client.api
            import vec_inf.client.models
            import vec_inf.client.slurm_vars

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
