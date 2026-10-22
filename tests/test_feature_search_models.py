"""Investigative families and the reusable GUI-facing search contract.

Run with pytest, or PYTHONPATH=src python -m unittest discover -s tests
-p test_feature_search_models.py to avoid unrelated application bootstrap.
"""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

import JBGFeatureSearch as fs


def cli():
    spec = importlib.util.spec_from_file_location('feature_search_cli',
        Path(__file__).parents[1] / 'scripts/feature_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureSearchModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        values, labels = make_classification(n_samples=90, n_features=4,
            n_informative=2, n_redundant=1, weights=[0.7, 0.3], random_state=14)
        self.X = pd.DataFrame(values, columns=['a', 'b', 'c', 'd'], index=np.arange(1, 91))
        self.y = pd.Series(np.where(labels, 'Inkorrekt', 'Korrekt'), index=self.X.index)
        self.options = fs.SearchOptions(trials=2, sizes=(1, 2), inner_folds=3,
            outer_folds=3, refine_rounds=0, max_evaluations=3,
            non_linear_max_iterations=40, non_linear_hidden_layers=(5,), tree_n_estimators=7)
        self.threads = threadpool_limits(limits=1)
        self.addCleanup(self.threads.restore_original_limits)

    def test_factories_preserve_linear_defaults_and_separate_budgets(self):
        name, model = fs.make_estimator(fs.SearchOptions())
        self.assertEqual(name, 'logistic')
        reference = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, random_state=42)
        self.assertEqual(model.get_params(), reference.get_params())
        options = replace(self.options, max_iterations=123, C=0.2,
            non_linear_max_iterations=17, non_linear_alpha=0.01,
            non_linear_hidden_layers=(7, 3), non_linear_learning_rate=0.02,
            tree_n_estimators=9, tree_max_depth=4, tree_min_samples_leaf=2)
        name, mlp = fs.make_estimator(replace(options, model_type='Non-linear'))
        self.assertIsInstance(mlp, MLPClassifier)
        self.assertEqual((mlp.max_iter, mlp.alpha, mlp.hidden_layer_sizes, mlp.learning_rate_init),
                         (17, 0.01, (7, 3), 0.02))
        self.assertFalse(mlp.early_stopping)
        name, tree = fs.make_estimator(replace(options, model_type='Tree'))
        self.assertIsInstance(tree, ExtraTreesClassifier)
        self.assertEqual((tree.n_estimators, tree.max_depth, tree.min_samples_leaf, tree.n_jobs),
                         (9, 4, 2, 1))
        self.assertFalse(tree.warm_start)
        # Retained inactive controls do not alter the active estimator.
        self.assertEqual(tree.get_params(), fs.make_estimator(replace(options,
            model_type='Tree', C=5., max_iterations=777, non_linear_alpha=0.5))[1].get_params())

    def test_combination_identity_serializes_nested_estimators(self):
        options = replace(self.options, model_type='Combination')
        identity = fs.model_identity(options)
        encoded = json.loads(json.dumps(identity, ensure_ascii=False))
        self.assertEqual(encoded['model_type'], 'Combination')
        self.assertEqual(encoded['parameters']['weights'], [1, 2, 1])
        members = encoded['parameters']['estimators']
        self.assertEqual([member[0] for member in members], ['mlpc', 'rfcl', 'abc'])
        self.assertIn('MLPClassifier', members[0][1]['estimator'])
        self.assertIn('RandomForestClassifier', members[1][1]['estimator'])
        self.assertIn('AdaBoostClassifier', members[2][1]['estimator'])
        self.assertEqual(members[0][1]['parameters']['random_state'], options.seed)
        self.assertEqual(members[1][1]['parameters']['random_state'], options.seed)

    def test_invalid_family_settings_fail_before_outputs(self):
        invalid = [dict(model_type='Other'), dict(non_linear_max_iterations=0),
            dict(non_linear_max_iterations=True), dict(non_linear_alpha=-1),
            dict(non_linear_alpha=np.nan), dict(non_linear_alpha=True),
            dict(non_linear_alpha='invalid'), dict(C=True), dict(min_gain=False),
            dict(non_linear_hidden_layers=()),
            dict(non_linear_hidden_layers=(1, 0)), dict(non_linear_hidden_layers=[4]),
            dict(non_linear_learning_rate=0), dict(non_linear_learning_rate=np.inf),
            dict(tree_n_estimators=0), dict(tree_n_estimators=True),
            dict(tree_max_depth=0), dict(tree_max_depth=True), dict(tree_min_samples_leaf=0)]
        for index, settings in enumerate(invalid):
            with self.subTest(settings=settings):
                output = self.root / str(index)
                with self.assertRaises(ValueError):
                    fs.run_experiment(self.X, self.y, 'Inkorrekt', output,
                        replace(self.options, **settings), progress=None)
                self.assertFalse(output.exists())

    def test_all_families_fit_sparse_fold_inputs_and_are_cloneable(self):
        X = self.X.astype(pd.SparseDtype('float64', 0.))
        for family in ('Linear', 'Non-linear', 'Tree'):
            for scaler in ('STA', 'MAX'):
                with self.subTest(family=family, scaler=scaler):
                    model = clone(fs.make_pipeline(('a', 'b'),
                        replace(self.options, model_type=family, scaler=scaler), {}))
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', fs.ConvergenceWarning)
                        model.fit(X.iloc[:60], self.y.iloc[:60])
                    self.assertTrue(sparse.issparse(model.named_steps['columns'].transform(X.iloc[60:])))
                    self.assertEqual(model.predict(X.iloc[60:]).shape, (30,))
                    scaler_model = model.named_steps['scale']
                    if scaler == 'STA':
                        np.testing.assert_allclose(scaler_model.mean_, self.X.iloc[:60][['a', 'b']].mean())
                    else:
                        np.testing.assert_allclose(scaler_model.max_abs_, self.X.iloc[:60][['a', 'b']].abs().max())

    def test_nested_validation_reproducibility_metrics_and_progress_for_each_family(self):
        original_fit = fs.FoldColumnEncoder.fit
        for family in ('Linear', 'Non-linear', 'Tree'):
            with self.subTest(family=family):
                options = replace(self.options, model_type=family)
                seen, events = [], []
                def fit(encoder, X, y=None):
                    seen.append(set(X.index))
                    return original_fit(encoder, X, y)
                output = self.root / family
                with patch.object(fs.FoldColumnEncoder, 'fit', fit):
                    result = fs.run_experiment(self.X, self.y, 'Inkorrekt', output,
                        options, progress=None, on_event=events.append,
                        reserved_holdout_signature='untouched')
                repeats = fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root / (family + '_repeat'),
                    options, progress=None)
                self.assertEqual(result['recommendation'], repeats['recommendation'])
                self.assertEqual(result['nested_validation'], repeats['nested_validation'])
                self.assertEqual(result['attempted_fits'], len(seen))
                expected_fits = options.planned_fit_upper_bound + 2 * options.outer_folds
                self.assertEqual(result['attempted_fits'], expected_fits)
                started = [e for e in events if e['event'] == 'fit_started']
                self.assertEqual([e['attempted_fits'] for e in started], list(range(1, expected_fits + 1)))
                self.assertEqual(events[0]['event'], 'study_started')
                self.assertEqual(events[-1]['event'], 'study_completed')
                self.assertTrue(all(e['model_type'] == family for e in events if 'model_type' in e))
                self.assertEqual(result['signatures']['reserved_holdout'], 'untouched')
                scores, counts = [], Counter()
                for fold, (train, test) in enumerate(fs.folds_for(self.X, self.y, 3, 42)):
                    # Nine inner fits and two outer fits per outer search.
                    outer_train = set(self.X.iloc[train].index)
                    for indices in seen[fold * 11:(fold + 1) * 11]:
                        self.assertTrue(indices <= outer_train)
                        self.assertFalse(indices & set(self.X.iloc[test].index))
                    baseline = fs.make_pipeline(tuple(self.X.columns), options, {})
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', fs.ConvergenceWarning)
                        baseline.fit(self.X.iloc[train], self.y.iloc[train])
                    predicted = baseline.predict(self.X.iloc[test])
                    scores.append(matthews_corrcoef(self.y.iloc[test], predicted))
                    counts.update(fs.classification_counts(self.y.iloc[test], predicted, 'Inkorrekt'))
                self.assertAlmostEqual(result['nested_validation']['full_baseline']['mean_mcc'], np.mean(scores))
                for key, value in fs.metrics(counts).items():
                    self.assertAlmostEqual(result['nested_validation']['full_baseline'][key], value)
                exported = pd.read_csv(output / 'subsets.csv', sep=';')
                self.assertFalse(exported.duplicated(['scope', 'columns']).any())
                self.assertEqual(set(exported.model_type), {family})
                self.assertEqual(result['recommendation']['validation_status'], 'candidate')
                json.dumps(result)

    def test_duplicate_subset_emits_no_extra_fit_events(self):
        events = []
        search = fs.Search(self.X, self.y, 'Inkorrekt', self.options, {}, 'test',
                           lambda row: None, on_event=events.append)
        row = search.evaluate(('a', 'b'), 'random')
        count = len(events)
        self.assertIs(search.evaluate(('b', 'a'), 'refine'), row)
        self.assertEqual(len(events), count)
        self.assertEqual(search.fit_count, 3)

    def test_provenance_tracks_values_types_context_and_rejects_credentials(self):
        context = dict(project='monthly', source_kind='sql', source_tables=['db.dbo.first'],
                       class_column='label', id_column='id', excluded_columns=['PaymentId'])
        sql_schema = {name: 'float' for name in self.X.columns}
        first = fs.study_provenance(self.X, context, source_sql_schema=sql_schema)
        self.assertEqual(first, fs.study_provenance(self.X.copy(), context, source_sql_schema=sql_schema))
        changed = self.X.copy()
        changed.iloc[0, 0] += 1
        self.assertNotEqual(first['development_feature_sha256'],
                            fs.study_provenance(changed, context, source_sql_schema=sql_schema)['development_feature_sha256'])
        self.assertEqual(first['original_column_schema'][0], {'name': 'a', 'dtype': 'float64'})
        self.assertEqual(first['source_sql_schema'][0], {'name': 'a', 'sql_type': 'float', 'family': 'numeric'})
        self.assertEqual({entry['name'] for entry in first['source_sql_schema']}, set(self.X.columns))
        with self.assertRaisesRegex(ValueError, 'missing 1 investigated feature'):
            fs.study_provenance(self.X, context, source_sql_schema={name: 'float' for name in self.X.columns[:-1]})
        self.assertEqual(first['context'], context)
        self.assertEqual(first['feature_search_revision'], '130')
        self.assertEqual(len(first['backend_source_sha256']), 64)
        with self.assertRaises(ValueError):
            fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root / 'secret',
                self.options, study_context=dict(context, sql_password='not-for-export'))
        self.assertFalse((self.root / 'secret').exists())

    def test_interrupt_preserves_accurate_attempted_fits_and_terminal_event(self):
        events = []
        def callback(event):
            events.append(event)
            if event['event'] == 'candidate_completed':
                raise KeyboardInterrupt
        output = self.root / 'interrupted'
        with self.assertRaises(KeyboardInterrupt):
            fs.run_experiment(self.X, self.y, 'Inkorrekt', output, self.options,
                              progress=None, on_event=callback)
        result = json.loads((output / 'summary.json').read_text())
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['attempted_fits'], 3)
        self.assertEqual(result['evaluated_subsets'], 1)
        self.assertEqual(len(pd.read_csv(output / 'subsets.csv', sep=';')), 1)
        self.assertEqual(events[-1]['event'], 'study_interrupted')

    def test_outer_fit_failure_retains_count_and_original_exception(self):
        original = fs.make_pipeline
        calls = []
        def pipeline(columns, options, text_options):
            model = original(columns, options, text_options)
            calls.append(model)
            if len(calls) == 10:  # Nine inner fits, then the first outer fit.
                model.fit = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('outer failed'))
            return model
        with patch.object(fs, 'make_pipeline', pipeline):
            with self.assertRaisesRegex(RuntimeError, 'outer failed'):
                fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root / 'failed',
                                  self.options, progress=None)
        result = json.loads((self.root / 'failed' / 'summary.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['attempted_fits'], 10)

    def test_optional_callback_failure_does_not_discard_study(self):
        with warnings.catch_warnings(record=True) as recorded:
            result = fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root / 'report',
                self.options, progress=None, on_event=lambda event: 1 / 0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['progress_reporting_error'], 'ZeroDivisionError')
        self.assertEqual(result['attempted_fits'],
                         self.options.planned_fit_upper_bound + 2 * self.options.outer_folds)
        self.assertEqual(sum('callback failed' in str(w.message) for w in recorded), 1)

    def test_cli_legacy_aliases_and_family_settings(self):
        parser = cli().parser()
        legacy = parser.parse_args(['--max-iterations', '123'])
        named = parser.parse_args(['--linear-max-iterations', '123'])
        self.assertEqual(legacy.max_iterations, named.max_iterations)
        self.assertEqual(legacy.model_type, 'Linear')
        args = parser.parse_args(['--model-type', 'Non-linear', '--non-linear-max-iterations', '45',
            '--non-linear-hidden-layers', '8', '4', '--tree-n-estimators', '12'])
        self.assertEqual((args.non_linear_max_iterations, args.non_linear_hidden_layers,
                          args.tree_n_estimators), (45, [8, 4], 12))

    def test_cli_all_models_write_provenance_and_reserve_holdout(self):
        for index, family in enumerate(('Linear', 'Non-linear', 'Tree')):
            with self.subTest(family=family), redirect_stdout(io.StringIO()):
                output = self.root / f'cli_{index}'
                code = cli().main(['--demo', '--model-type', family, '--trials', '1', '--sizes', '2',
                    '--inner-folds', '2', '--outer-folds', '2', '--refine-rounds', '0',
                    '--max-evaluations', '2', '--non-linear-max-iterations', '10',
                    '--non-linear-hidden-layers', '4', '--tree-n-estimators', '5',
                    '--exclude', 'feature_1', '--output', str(output)])
                self.assertEqual(code, 0)
                result = json.loads((output / 'summary.json').read_text())
                self.assertEqual(result['development_rows'], 256)
                self.assertEqual(result['investigative_model']['model_type'], family)
                self.assertEqual(result['provenance']['context']['excluded_columns'], ['feature_1'])
                self.assertEqual(result['provenance']['context']['source_kind'], 'synthetic')
                self.assertTrue(result['signatures']['reserved_holdout'])
                self.assertNotIn('feature_1', result['original_columns'])

    def test_grouped_cv_uses_same_groups_and_records_signature(self):
        groups = pd.Series(np.arange(len(self.X)) // 2, index=self.X.index)
        result = fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root / 'groups',
            replace(self.options, model_type='Tree'), groups=groups, progress=None)
        self.assertTrue(result['grouped_cv'])
        self.assertEqual(len(result['signatures']['development_groups']), 64)
        for train, test in fs.folds_for(self.X, self.y, 3, 42, groups):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_refinement_checks_complete_neighbourhood_for_all_families(self):
        for family in ('Linear', 'Non-linear', 'Tree'):
            with self.subTest(family=family):
                options = replace(self.options, model_type=family, trials=1,
                    sizes=(1,), top_sets=1, refine_rounds=2, max_evaluations=20)
                search = fs.Search(self.X, self.y, 'Inkorrekt', options, {}, 'test', lambda row: None)
                scores = {tuple(self.X.columns): -0.1, ('a',): 0.1, ('a', 'b'): 0.5}
                def evaluate(columns, stage):
                    columns = tuple(c for c in self.X if c in columns)
                    if columns not in search.cache:
                        search.cache[columns] = dict(columns=columns, n_columns=len(columns),
                            mean_mcc=scores.get(columns, 0.), std_mcc=0., error='')
                    return search.cache[columns]
                with patch.object(fs, 'random_subsets', return_value=[('a',)]), \
                        patch.object(search, 'evaluate', evaluate):
                    best = search.run()
                self.assertEqual(best['columns'], ('a', 'b'))
                self.assertEqual(search.moves[0]['operation'], 'add')
                self.assertTrue({('a',), ('b',), ('a', 'b', 'c'), ('a', 'b', 'd')} <= set(search.cache))

    def test_failure_warnings_and_target_counts_remain_visible(self):
        original_fit = fs.FoldColumnEncoder.fit
        def fit(encoder, X, y=None):
            if encoder.columns == ('a',):
                raise ValueError('unusable source column')
            warnings.warn('intentional convergence warning', fs.ConvergenceWarning)
            return original_fit(encoder, X, y)
        for family in ('Linear', 'Non-linear', 'Tree'):
            with self.subTest(family=family), patch.object(fs.FoldColumnEncoder, 'fit', fit):
                search = fs.Search(self.X, self.y, 'Inkorrekt',
                    replace(self.options, model_type=family), {}, 'test', lambda row: None)
                failed = search.evaluate(('a',), 'random')
                successful = search.evaluate(('b',), 'random')
                self.assertTrue(np.isnan(failed['mean_mcc']))
                self.assertIn('unusable source column', failed['error'])
                self.assertEqual(failed['columns'], ('a',))
                self.assertEqual(successful['rows'], len(self.y))
                self.assertGreaterEqual(successful['convergence_warnings'], 3)
                self.assertEqual(successful['tp'] + successful['fn'], (self.y == 'Inkorrekt').sum())

    def test_full_neighbourhood_budget_and_existing_output_guards(self):
        search = fs.Search(self.X, self.y, 'Inkorrekt',
            replace(self.options, refine_rounds=10), {}, 'test', lambda row: None)
        search.run()
        self.assertEqual(len(search.cache), 3)
        self.assertEqual(search.moves, [])
        self.assertEqual(search.fit_count, 9)
        (self.root / 'keep.txt').write_text('unchanged')
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            fs.run_experiment(self.X, self.y, 'Inkorrekt', self.root, self.options, progress=None)
        self.assertEqual((self.root / 'keep.txt').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
