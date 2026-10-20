"""Non-Windows safety contracts for the operator-driven rev160 cleanup script."""
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "scripts" / "cleanup_voila_kernels.ps1").read_text(encoding="utf-8")


class CleanupSafety160(unittest.TestCase):
    def test_opt_in_and_explicit_ids_required(self):
        self.assertIn('if ($Apply -and $requested.Count -eq 0)', SOURCE)
        self.assertIn("if (-not $Apply)", SOURCE)
        self.assertIn('return', SOURCE)

    def test_requires_stopped_server_and_session_scoping(self):
        self.assertIn('Server launcher PID', SOURCE)
        self.assertIn('Storing connection files in', SOURCE)
        self.assertIn("$sessionDir + '\\kernel-'", SOURCE)
        self.assertIn('$rejected.ContainsKey($id)', SOURCE)
        self.assertIn('$started.ContainsKey($id)', SOURCE)

    def test_never_weakens_csrf_or_kills_unrelated_python(self):
        self.assertNotIn('disable_check_xsrf', SOURCE)
        self.assertNotIn('xsrf_cookies=False', SOURCE)
        self.assertNotIn('Stop-Process -Name', SOURCE)
        self.assertIn('Stop-Process -Id $candidate.PID', SOURCE)


if __name__ == '__main__':
    unittest.main()
