"""Rev139: durable checkpoint, safe matching, deterministic restart."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from JBGFeatureSearch import InvestigationPaused, SearchOptions, run_experiment


class ResumeRegressionTests(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(91)
        X = pd.DataFrame(rng.normal(size=(96, 6)), columns=[f'f{i}' for i in range(6)])
        y = pd.Series(np.where(X.f0 + X.f1 > 0, 'Ja', 'Nej'))
        return X, y

    @staticmethod
    def options():
        return SearchOptions(trials=3, sizes=(2, 3), inner_folds=2, outer_folds=2,
                             max_evaluations=4, top_sets=1, refine_rounds=0,
                             model_type='Linear', max_iterations=100)

    def test_restart_reproduces_uninterrupted_result_and_repairs_csv(self):
        X, y = self.data()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = run_experiment(X, y, 'Ja', root / 'expected', options=self.options(), progress=None)
            pause = threading.Event()
            def stop_after_second_candidate(event):
                if event['event'] == 'candidate_completed' and event['scope'] == 'outer_1' and event['candidate'] == 2:
                    pause.set()
            with self.assertRaises(InvestigationPaused):
                run_experiment(X, y, 'Ja', root / 'restart', options=self.options(),
                               progress=None, on_event=stop_after_second_candidate,
                               pause_event=pause)
            run_dir = root / 'restart'
            state = json.loads((run_dir / 'summary.json').read_text())
            self.assertEqual(state['status'], 'paused')
            self.assertEqual(len(list((run_dir / '.checkpoints').glob('candidate_*.json'))), 2)
            # Abrupt shutdown while appending report CSV must not invalidate commits.
            (run_dir / 'subsets.csv').write_text('torn trailing CSV data', encoding='utf-8')
            actual = run_experiment(X, y, 'Ja', run_dir, options=self.options(),
                                    progress=None, resume=True)
            self.assertEqual(actual['study_id'], state['study_id'])
            self.assertEqual(expected['nested_validation'], actual['nested_validation'])
            self.assertEqual(expected['recommendation'], actual['recommendation'])
            self.assertEqual(actual['status'], 'completed')
            self.assertEqual(len(list((run_dir / '.checkpoints').glob('candidate_*.json'))),
                             actual['evaluated_subsets'])

    def test_partial_outer_validation_is_not_repeated(self):
        X, y = self.data()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = run_experiment(X, y, 'Ja', root / 'expected',
                                      options=self.options(), progress=None)
            pause = threading.Event()
            def pause_after_first_outer_fit(event):
                if event['event'] == 'outer_validation_completed' and event.get('fold') == 1:
                    pause.set()
            with self.assertRaises(InvestigationPaused):
                run_experiment(X, y, 'Ja', root / 'restart', options=self.options(),
                               progress=None, pause_event=pause, on_event=pause_after_first_outer_fit)
            files = list((root / 'restart' / '.checkpoints').glob('outer_*.json'))
            self.assertEqual(len(files), 2)
            resumed = run_experiment(X, y, 'Ja', root / 'restart',
                                     options=self.options(), progress=None, resume=True)
            self.assertEqual(expected['nested_validation'], resumed['nested_validation'])
            self.assertEqual(len(list((root / 'restart' / '.checkpoints').glob('outer_*.json'))), 4)

    def test_resume_mid_refinement_preserves_budget_neighbourhood(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(size=(120, 12)), columns=[f'c{i}' for i in range(12)])
        y = pd.Series(np.where(X.c1 + X.c2 > 0, 'Ja', 'Nej'))
        opts = SearchOptions(model_type='Linear', trials=4, sizes=(3, 6), inner_folds=2,
                             outer_folds=2, max_evaluations=9, top_sets=1,
                             refine_rounds=1, max_iterations=100)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = run_experiment(X, y, 'Ja', root / 'expected', options=opts, progress=None)
            pause = threading.Event()
            def stop_during_refinement(event):
                if event['event'] == 'candidate_completed' and event['scope'] == 'outer_1' and event['candidate'] == 7:
                    pause.set()
            with self.assertRaises(InvestigationPaused):
                run_experiment(X, y, 'Ja', root / 'restart', options=opts,
                               progress=None, pause_event=pause, on_event=stop_during_refinement)
            self.assertTrue(list((root / 'restart' / '.checkpoints').glob('refinement_plan_*.json')))
            actual = run_experiment(X, y, 'Ja', root / 'restart',
                                    options=opts, progress=None, resume=True)
            self.assertEqual(expected['nested_validation'], actual['nested_validation'])
            self.assertEqual(expected['recommendation'], actual['recommendation'])

    def test_resume_rejects_changed_feature_values_and_settings(self):
        X, y = self.data()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'search'
            pause = threading.Event()
            def pause_after_one(event):
                if event['event'] == 'candidate_completed':
                    pause.set()
            with self.assertRaises(InvestigationPaused):
                run_experiment(X, y, 'Ja', path, options=self.options(), progress=None,
                               pause_event=pause, on_event=pause_after_one)
            X_changed = X.copy()
            X_changed.iloc[0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'does not match'):
                run_experiment(X_changed, y, 'Ja', path, options=self.options(),
                               progress=None, resume=True)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                run_experiment(X, y, 'Ja', path, options=SearchOptions(**{
                    **self.options().__dict__, 'seed': 99}), progress=None, resume=True)


if __name__ == '__main__':
    unittest.main()

class SavedRecoveryServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        cls._module_patch = patch.dict(sys.modules, {
            'JBGHandler': ModuleType('JBGHandler'),
            'SQLDataLayer': ModuleType('SQLDataLayer'),
        })
        cls._module_patch.start()
        sys.modules['JBGHandler'].JBGHandler = object
        sys.modules['SQLDataLayer'].DataLayer = object

    @classmethod
    def tearDownClass(cls):
        import sys
        sys.modules.pop('JBGFeatureSearchService', None)
        cls._module_patch.stop()

    def test_resume_uses_snapshot_without_gui_config(self):
        import hashlib
        import pickle
        from unittest.mock import patch
        from JBGFeatureSearchService import run_training_feature_search
        X = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y = pd.Series(['yes', 'no', 'yes', 'no'])
        settings = {'text_columns': ('feature',), 'ngram_range': (1, 1)}
        options = SearchOptions(trials=3, sizes=(2, 3), inner_folds=2,
                                outer_folds=2, max_evaluations=4)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            record = dict(X=X, y=y, holdout_signature='holdout',
                          context={'source_tables': ['old.database.table']},
                          text_options=settings, target='yes')
            payload = pickle.dumps(record)
            (folder / 'development_snapshot.pkl').write_bytes(payload)
            (folder / 'development_snapshot.sha256').write_text(hashlib.sha256(payload).hexdigest())
            from dataclasses import asdict
            (folder / 'summary.json').write_text(json.dumps({'options': asdict(options)}))
            captured = {}
            def fake_run(x, labels, target, output, **kwargs):
                captured.update(kwargs)
                self.assertTrue(x.equals(X))
                self.assertTrue(labels.equals(y))
                self.assertEqual(target, 'yes')
                return {'status': 'completed'}
            with patch('JBGFeatureSearchService.run_experiment', fake_run):
                output = run_training_feature_search(None, None, options=None,
                                                     output=folder, resume=True)
            self.assertTrue(captured['resume'])
            self.assertEqual(captured['options'], options)
            self.assertEqual(captured['text_options'], settings)
            self.assertEqual(output['output_directory'], str(folder))

    def test_corrupted_snapshot_is_rejected_before_unpickling(self):
        from JBGFeatureSearchService import run_training_feature_search
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'development_snapshot.pkl').write_bytes(b'altered')
            (path / 'development_snapshot.sha256').write_text('bad checksum')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                run_training_feature_search(None, None, options=None, output=path, resume=True)
