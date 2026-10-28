"""Static contract for the local launcher regression gate.

Avoid executing PowerShell, installing packages or launching Voila in tests.
"""
from pathlib import Path
import unittest


class LocalStartupGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (Path(__file__).resolve().parents[1]
                      / 'start_application_template.ps1').read_text(encoding='utf-8')

    def test_full_unittest_discovery_is_in_launcher(self):
        self.assertIn("-m unittest discover -s tests -p 'test_*.py' -v", self.script)
        self.assertIn("Join-Path $DevRoot 'src'", self.script)
        self.assertIn("-m pytest tests/test_config.py tests/test_meta.py -q", self.script)

    def test_gate_is_after_dependencies_and_before_voila(self):
        self.assertLess(self.script.index('& $Pip install -r $VenvRequirements'),
                        self.script.index('Running local Python regression suite'))
        self.assertLess(self.script.index('Running local Python regression suite'),
                        self.script.index('# --- Launch Voila ---'))
        self.assertLess(self.script.index('Python regression tests failed.'),
                        self.script.index('& $Voila .'+chr(92)+'$Notebook'))

    def test_fails_closed_and_restores_pythonpath(self):
        self.assertIn("if ($LASTEXITCODE -ne 0)", self.script)
        self.assertIn("throw 'Python regression tests failed. Voila startup was cancelled.'", self.script)
        self.assertIn('$env:PYTHONPATH = $PreviousPythonPath', self.script)
        self.assertIn('Remove-Item Env:PYTHONPATH', self.script)
        self.assertIn("throw 'Pytest regression tests failed. Voila startup was cancelled.'", self.script)


if __name__ == '__main__':
    unittest.main()
