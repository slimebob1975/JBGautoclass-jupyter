"""Rev140: bounded deterministic process-parallel candidate search."""
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from JBGFeatureSearch import InvestigationPaused, SearchOptions, run_experiment


class ParallelSearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(301)
        self.X = pd.DataFrame(rng.normal(size=(90, 8)), columns=[f'x{i}' for i in range(8)])
        self.y = pd.Series(np.where(self.X.x0 + self.X.x1 * 0.5 > 0, 'yes', 'no'))
        self.options = SearchOptions(trials=5, sizes=(2, 4, 6), inner_folds=2, outer_folds=2,
                                     max_evaluations=10, top_sets=1, refine_rounds=1,
                                     max_iterations=100)

    def test_serial_and_parallel_match(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            serial = run_experiment(self.X, self.y, 'yes', root / 'serial',
                                    options=self.options, progress=None)
            parallel = run_experiment(self.X, self.y, 'yes', root / 'parallel',
                                      options=replace(self.options, parallel_workers=2), progress=None)
            self.assertEqual(serial['recommendation'], parallel['recommendation'])
            self.assertEqual(serial['nested_validation'], parallel['nested_validation'])
            self.assertEqual(serial['evaluated_subsets'], parallel['evaluated_subsets'])

    def test_parallel_pause_resume_matches_uninterrupted(self):
        options = replace(self.options, parallel_workers=2)
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            expected = run_experiment(self.X, self.y, 'yes', root / 'complete',
                                      options=options, progress=None)
            event = threading.Event()
            def pause_on_candidate(entry):
                if entry['event'] == 'candidate_completed' and entry['scope'] == 'outer_1' and entry['candidate'] >= 3:
                    event.set()
            with self.assertRaises(InvestigationPaused):
                run_experiment(self.X, self.y, 'yes', root / 'paused', options=options,
                               progress=None, on_event=pause_on_candidate, pause_event=event)
            recovered = run_experiment(self.X, self.y, 'yes', root / 'paused',
                                       options=options, resume=True, progress=None)
            self.assertEqual(expected['recommendation'], recovered['recommendation'])
            self.assertEqual(expected['nested_validation'], recovered['nested_validation'])

    def test_grouped_parallel_folds(self):
        options = replace(self.options, parallel_workers=2)
        groups = pd.Series(np.repeat(np.arange(30), 3))
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            expected = run_experiment(self.X, self.y, 'yes', root / 'serial',
                                      options=replace(options, parallel_workers=1), groups=groups, progress=None)
            actual = run_experiment(self.X, self.y, 'yes', root / 'parallel',
                                    options=options, groups=groups, progress=None)
            self.assertEqual(expected['nested_validation'], actual['nested_validation'])

class ParallelWorkerFailureRegressionTests(unittest.TestCase):
    """Rev179: worker failure must not silently complete an uncommitted batch."""

    def test_failed_second_future_keeps_only_first_durable_commit(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        from JBGFeatureSearch import Search

        rng = np.random.default_rng(179)
        X = pd.DataFrame(rng.normal(size=(32, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(['yes', 'no'] * 16)
        events, commits = [], []
        options = replace(SearchOptions(inner_folds=2, outer_folds=2,
                                        max_evaluations=3, parallel_workers=2),
                          max_iterations=100)
        search = Search(X, y, 'yes', options, {}, 'outer_1', commits.append,
                        on_event=events.append)
        good = Future()
        good.set_result((dict(columns=('a',), error='', mean_mcc=0.5), 2))
        failed = Future()
        failed.set_exception(RuntimeError('synthetic worker interruption'))
        executor = Mock()
        executor.submit.side_effect = [good, failed]
        search.executor = executor

        with self.assertRaisesRegex(RuntimeError, 'synthetic worker interruption'):
            search.evaluate_batch([(('a',), 'random'), (('b',), 'random')])
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0]['columns'], ('a',))
        self.assertEqual(set(search.cache), {('a',)})
        self.assertEqual(search.fit_count, 2)
        failed_events = [event for event in events if event['event'] == 'parallel_batch_failed']
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0]['error_type'], 'RuntimeError')
        self.assertFalse(any(event['event'] == 'parallel_batch_completed' for event in events))
        self.assertEqual(executor.submit.call_count, 2)
