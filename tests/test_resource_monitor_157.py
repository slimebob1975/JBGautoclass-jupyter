import time
import unittest
from unittest.mock import patch

from JBGResourceMonitor157 import safe_grid_workers, training_heartbeat


class _Log:
    def __init__(self):
        self.messages = []

    def write_message(self, message, level="INFO"):
        self.messages.append((level, message))


class ResourceMonitor157Tests(unittest.TestCase):
    def test_workers_unchanged_with_ample_memory(self):
        self.assertEqual(safe_grid_workers(4, available=8*1024**3, total=16*1024**3), (4, None))

    def test_low_ram_reduces_workers(self):
        workers, notice = safe_grid_workers(4, available=512*1024**2, total=16*1024**3)
        self.assertEqual(workers, 1)
        self.assertIn("reducing workers", notice)

    def test_does_not_restrict_single_worker(self):
        self.assertEqual(safe_grid_workers(1, available=100, total=1024), (1, None))

    def test_heartbeat_reports_completion(self):
        log = _Log()
        with training_heartbeat(log, "test", interval=0.01):
            time.sleep(0.025)
        values = [message for _, message in log.messages]
        self.assertTrue(any("Training started:" in m for m in values))
        self.assertTrue(any("Training finished:" in m for m in values))
        self.assertTrue(any("Training running:" in m for m in values))

    def test_heartbeat_preserves_original_exception(self):
        log = _Log()
        with self.assertRaisesRegex(ValueError, "training error"):
            with training_heartbeat(log, "test", interval=0.01):
                raise ValueError("training error")
        self.assertTrue(any("Training failed:" in m for _, m in log.messages))

    def test_monitor_failure_does_not_block_training(self):
        log = _Log()
        with patch("JBGResourceMonitor157._memory_snapshot", side_effect=OSError("mock")):
            with training_heartbeat(log, "test", interval=0.01):
                pass
        self.assertEqual(log.messages, [])


if __name__ == "__main__":
    unittest.main()
