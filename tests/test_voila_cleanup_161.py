"""Revision 161 source-level invariants (Windows network checks need live tests)."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "cleanup_voila_kernels.ps1").read_text(encoding="utf-8")
BACKLOG = (ROOT / "BACKLOG.md").read_text(encoding="utf-8")

class CleanupSafety161(unittest.TestCase):
    def test_checks_listener_and_fails_closed(self):
        self.assertIn('Get-NetTCPConnection -State Listen -ErrorAction Stop', SCRIPT)
        self.assertIn('Cannot check TCP listeners', SCRIPT)
        self.assertIn('Get-NetTCPConnection unavailable', SCRIPT)
        self.assertIn('TCP port $Port is still listening', SCRIPT)

    def test_port_check_applies_only_to_destructive_mode(self):
        self.assertIn('if ($Apply) { Assert-VoilaPortClosed -Port $voilaPort }', SCRIPT)
        self.assertIn('if (-not $Apply)', SCRIPT)
        self.assertGreaterEqual(SCRIPT.count('Assert-VoilaPortClosed -Port $voilaPort'), 3)
        self.assertIn('if ($Apply -and $requested.Count -eq 0)', SCRIPT)

    def test_refuses_any_live_launcher_pid(self):
        self.assertIn('if ($null -ne $serverProcess)', SCRIPT)
        self.assertIn('Server launcher PID is active', SCRIPT)
        self.assertIn('Stop-Process -Id $candidate.PID', SCRIPT)

    def test_gui_spike_backlog(self):
        self.assertIn('P2 — GUI framework architecture spike (Panel / NiceGUI)', BACKLOG)
        self.assertIn('No full rewrite', BACKLOG)

if __name__ == '__main__':
    unittest.main()
