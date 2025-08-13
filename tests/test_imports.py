"""Test the imports of the vec_inf package."""

import unittest

import pytest


class TestVecInfImports(unittest.TestCase):
    """Test the imports of the vec_inf package."""

    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            # CLI imports
            import vec_inf.cli  # noqa: PLC0415
            import vec_inf.cli._cli  # noqa: PLC0415
            import vec_inf.cli._helper  # noqa: PLC0415

            # Client imports
            import vec_inf.client  # noqa: PLC0415
            import vec_inf.client._client_vars  # noqa: F401, PLC0415
            import vec_inf.client._exceptions  # noqa: PLC0415
            import vec_inf.client._helper  # noqa: PLC0415
            import vec_inf.client._slurm_script_generator  # noqa: PLC0415
            import vec_inf.client._slurm_templates  # noqa: PLC0415
            import vec_inf.client._slurm_vars  # noqa: PLC0415
            import vec_inf.client._utils  # noqa: PLC0415
            import vec_inf.client.api  # noqa: PLC0415
            import vec_inf.client.config  # noqa: PLC0415
            import vec_inf.client.models  # noqa: F401, PLC0415

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
