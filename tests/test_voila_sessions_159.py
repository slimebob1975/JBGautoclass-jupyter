"""Regression checks for safe kernel lifecycle reporting in revision 159."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_voila_sessions import summarize

A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class VoilaSessions159Tests(unittest.TestCase):
    def test_pairs_kernel_ids_and_deduplicates_repeated_shutdown_entries(self):
        sample = (f"[INFO] Kernel started: {A}\n"
                  f"[INFO] 403 POST /voila/api/shutdown/{A}: '_xsrf' argument missing\n"
                  f"[INFO] 403 POST /voila/api/shutdown/{A} 2.00ms\n"
                  f"[INFO] Kernel started: {B}\n")
        report = summarize(sample)
        self.assertEqual(report['started'], 2)
        self.assertEqual(report['rejected_requests'], 2)
        self.assertEqual(report['rejected_kernels'], 1)
        self.assertEqual(report['rejected_shutdowns'][A], 2)
        self.assertEqual(report['started_without_rejected_shutdown'], [B])

    def test_no_kernel_shutdown_actions_or_csrf_bypass(self):
        script = (ROOT / 'scripts/diagnose_voila_sessions.py').read_text(encoding='utf-8')
        launcher = (ROOT / 'start_application_template.ps1').read_text(encoding='utf-8')
        self.assertNotIn('disable_check_xsrf', script + launcher)
        self.assertNotIn('xsrf_cookies=False', script + launcher)
        self.assertNotIn('shutdown_kernel(', script)
        self.assertIn('diagnose_voila_sessions.py', launcher)

    def test_empty_log(self):
        self.assertEqual(summarize('')['rejected_requests'], 0)
        self.assertEqual(summarize('')['started'], 0)


if __name__ == '__main__':
    unittest.main()
