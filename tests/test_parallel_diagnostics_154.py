"""Revision 154: process-batch diagnostics do not alter scoring or commit order."""
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
import pandas as pd
from JBGFeatureSearch import Search


class FakeExecutor:
    def __init__(self, failure=None):
        self.failure = failure
        self.futures = []

    def submit(self, fn, *args):
        future = Future()
        if self.failure is not None and len(self.futures) == self.failure:
            future.set_exception(RuntimeError('simulated worker failure'))
        else:
            columns = args[-2]
            future.set_result((dict(columns=columns, mean_mcc=0.1, error=''), 2))
        self.futures.append(future)
        return future


def make_search(executor):
    search = Search.__new__(Search)
    search.X = pd.DataFrame(np.ones((4, 3)), columns=list('abc'))
    search.y = pd.Series([0, 1, 0, 1])
    search.target = 1
    search.options = SimpleNamespace(parallel_workers=2, max_evaluations=8, model_type='Linear')
    search.text_options = None
    search.groups = None
    search.scope = 'development'
    search.cache = {}
    search.fit_count = 0
    search.folds = [(np.array([0, 1]), np.array([2, 3]))] * 2
    search.pause_event = None
    search.executor = executor
    search.progress = None
    search.events = []
    search.rows = []
    search.on_event = search.events.append
    search.sink = search.rows.append
    return search


class ParallelDiagnostics154(unittest.TestCase):
    def test_successful_batch_is_committed_in_order(self):
        search = make_search(FakeExecutor())
        rows = search.evaluate_batch([(('a',), 'random'), (('b',), 'random')])
        self.assertEqual([r['columns'] for r in rows], [('a',), ('b',)])
        self.assertEqual(len(search.rows), 2)
        self.assertEqual(search.fit_count, 4)
        self.assertEqual([e['event'] for e in search.events if e['event'].startswith('parallel_batch')],
                         ['parallel_batch_started', 'parallel_batch_completed'])

    def test_failed_worker_preserves_error_and_partial_checkpoint(self):
        executor = FakeExecutor(failure=1)
        search = make_search(executor)
        with self.assertRaisesRegex(RuntimeError, 'simulated worker failure') as caught:
            search.evaluate_batch([(('a',), 'random'), (('b',), 'random')])
        self.assertEqual(len(search.rows), 1)
        self.assertEqual(len(search.cache), 1)
        self.assertTrue(any(e['event'] == 'parallel_batch_failed' for e in search.events))
        self.assertTrue(any('scope=development' in n for n in caught.exception.__notes__))


if __name__ == '__main__':
    unittest.main()
