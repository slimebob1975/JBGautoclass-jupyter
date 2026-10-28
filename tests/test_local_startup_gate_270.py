"""Revision 272: static contract for fast and full local regression gates."""
from pathlib import Path
import unittest


class LocalStartupGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.launcher = (root / 'start_application_template.ps1').read_text(encoding='utf-8')
        cls.runner = (root / 'scripts/run_regression.ps1').read_text(encoding='utf-8')

    def test_launcher_invokes_shared_fast_gate_before_voila(self):
        self.assertIn("scripts\\run_regression.ps1') -PythonExecutable $Py", self.launcher)
        self.assertLess(self.launcher.index('run_regression.ps1'), self.launcher.index('# --- Launch Voila ---'))
        self.assertNotIn("-m unittest discover -s tests -p 'test_*.py'", self.launcher)

    def test_full_suite_is_explicit_and_includes_pytest(self):
        self.assertIn('[switch]$Full', self.runner)
        self.assertIn("-m unittest discover -s tests -p 'test_*.py' -v", self.runner)
        self.assertIn('-m pytest tests/test_config.py tests/test_meta.py -q', self.runner)

    def test_fast_suite_is_explicit_and_missing_files_fail_closed(self):
        for name in ('test_classifier_source_boundary_212.py',
                     'test_csv_full_classifier_run_259.py',
                     'test_csv_saved_prediction_gui_265.py',
                     'test_csv_dataset_cascade_269c.py'):
            self.assertIn(name, self.runner)
        self.assertIn('Required fast regression module is missing:', self.runner)
        self.assertIn('Fast regression tests failed:', self.runner)

    def test_fail_closed_and_pythonpath_restoration(self):
        self.assertIn("throw 'Full unittest regression tests failed.'", self.runner)
        self.assertIn("throw 'Pytest regression tests failed.'", self.runner)
        self.assertIn('if ($LASTEXITCODE -ne 0)', self.runner)
        self.assertIn('$env:PYTHONPATH = $PreviousPythonPath', self.runner)
        self.assertIn('Remove-Item Env:PYTHONPATH', self.runner)


if __name__ == '__main__':
    unittest.main()
