# Import test for vec_inf modules
# This script tests the importability of all modules in the vec_inf package using only core dependencies

import unittest

class TestVecInfImports(unittest.TestCase):
    def test_import_cli_modules(self):
        try:
            import vec_inf.cli._cli
            import vec_inf.cli._config
            import vec_inf.cli._helper
            import vec_inf.cli._utils
        except ImportError as e:
            self.fail(f"Import failed: {e}")

if __name__ == "__main__":
    unittest.main()
