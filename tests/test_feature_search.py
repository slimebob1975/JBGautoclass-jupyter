"""Experimental search: fold isolation, budgets, metrics and CLI contracts."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import matthews_corrcoef

import JBGFeatureSearch as fs


def dataset(n=120, p=6):
    values, labels = make_classification(n_samples=n, n_features=p, n_informative=2,
        n_redundant=1, weights=[0.75, 0.25], flip_y=0, random_state=14)
    X = pd.DataFrame(values, columns=[f'f{i}' for i in range(p)], index=np.arange(1, n+1))
    return X, pd.Series(np.where(labels, 'Inkorrekt', 'Korrekt'), index=X.index)


def options(**kwargs):
    values = dict(trials=4, sizes=(1, 2, 3), inner_folds=3, outer_folds=3,
                  top_sets=1, refine_rounds=1, max_evaluations=15)
    values.update(kwargs)
    return fs.SearchOptions(**values)


def cli():
    spec = importlib.util.spec_from_file_location('feature_search_cli', Path(__file__).parents[1] / 'scripts/feature_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kwargs', [dict(trials=0), dict(inner_folds=1), dict(outer_folds=1),
    dict(sizes=()), dict(sizes=(0,)), dict(max_evaluations=4), dict(min_gain=-1),
    dict(C=0), dict(C=np.nan), dict(scaler='NOS'), dict(refine_rounds=-1), dict(seed=-1)])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        options(**kwargs).validate()


def test_random_sets_unique_reproducible_and_cover_requested_sizes():
    cols = tuple(f'f{i}' for i in range(10))
    opts = options(trials=30, max_evaluations=31, sizes=(1, 3, 5))
    first = fs.random_subsets(cols, opts)
    assert first == fs.random_subsets(cols, opts)
    assert len(first) == len(set(first)) == 30
    assert {len(s) for s in first} == {1, 3, 5}
    assert cols not in first
    assert all(tuple(c for c in cols if c in subset) == subset for subset in first)
    assert len(fs.random_subsets(('a', 'b'), options(trials=20, max_evaluations=21, sizes=(1, 2)))) == 2


def test_metrics_match_independent_target_counts():
    y = ['I', 'I', 'I', 'K', 'K', 'K', 'K']
    predicted = ['I', 'I', 'K', 'I', 'K', 'K', 'K']
    result = fs.metrics(fs.classification_counts(y, predicted, 'I'))
    assert (result['tp'], result['fp'], result['fn'], result['tn']) == (2, 1, 1, 3)
    assert result['precision'] == result['recall'] == result['f1'] == 2/3
    assert result['predicted_positives'] == result['target_support'] == 3
    assert fs.metrics(Counter())['f1'] == 0


def test_adapter_clone_and_selected_column_vocab_are_fold_local():
    X = pd.DataFrame({'num': [1., np.nan, 3., 4.], 'words': ['apple', 'banana', 'apple', 'validationonly'],
                      'unused': ['leak'] * 4}, index=[1, 2, 3, 4])
    settings = dict(text_columns=('words', 'unused'), stop_words=False, use_encryption=False,
                    use_categorization=False)
    adapter = fs.FoldColumnEncoder(('num', 'words'), settings)
    fitted = clone(adapter).fit(X.iloc[:3])
    assert fitted.converter_.text_columns_ == ['words']
    assert 'validationonly' not in fitted.converter_.tfidvectorizer_.vocabulary_
    assert 'leak' not in fitted.converter_.tfidvectorizer_.vocabulary_
    assert fitted.transform(X.iloc[3:]).shape[0] == 1
    assert adapter.text_options == settings
    assert adapter.columns == ('num', 'words')
    numeric = clone(fs.FoldColumnEncoder(('num',), settings)).fit(X.iloc[:3])
    assert numeric.converter_ is None


def test_candidate_counts_match_reference_cv_and_cache_adds_no_fits():
    X, y = dataset()
    opts = options()
    sink = []
    study = fs.Search(X, y, 'Inkorrekt', opts, {}, 'test', sink.append)
    row = study.evaluate(('f0', 'f2'), 'random')
    scores, expected = [], Counter()
    for train, test in study.folds:
        model = fs.make_pipeline(('f0', 'f2'), opts, {})
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[test])
        scores.append(matthews_corrcoef(y.iloc[test], pred))
        expected.update(fs.classification_counts(y.iloc[test], pred, 'Inkorrekt'))
    assert row['mean_mcc'] == pytest.approx(np.mean(scores))
    for key, value in fs.metrics(expected).items():
        assert row[key] == pytest.approx(value)
    assert study.evaluate(('f2', 'f0'), 'refine') is row
    assert study.fit_count == opts.inner_folds
    assert len(sink) == 1


def test_budget_and_failed_candidates_do_not_silently_change_columns(monkeypatch):
    X, y = dataset()
    original = fs.FoldColumnEncoder.fit
    def fit(self, X, y=None):
        if 'f0' in self.columns:
            raise ValueError('intentional unusable column')
        return original(self, X, y)
    monkeypatch.setattr(fs.FoldColumnEncoder, 'fit', fit)
    rows = []
    opts = options(trials=3, max_evaluations=4, refine_rounds=5)
    study = fs.Search(X, y, 'Inkorrekt', opts, {}, 'test', rows.append)
    bad = study.evaluate(('f0',), 'random')
    good = study.evaluate(('f1',), 'random')
    assert 'intentional' in bad['error'] and np.isnan(bad['mean_mcc'])
    assert good['error'] == ''
    assert bad['columns'] == ('f0',)
    study.run()
    assert len(rows) <= opts.max_evaluations
    assert study.fit_count <= opts.max_evaluations * opts.inner_folds


def test_nested_outer_isolation_reference_baseline_and_exports(tmp_path, monkeypatch):
    X, y = dataset()
    original_run = fs.Search.run
    seen = {}
    def run(self):
        seen[self.scope] = set(self.X.index)
        return original_run(self)
    monkeypatch.setattr(fs.Search, 'run', run)
    opts = options()
    result = fs.run_experiment(X, y, 'Inkorrekt', tmp_path/'run', opts,
                               reserved_holdout_signature='reserved', progress=None)
    assert result['status'] == 'completed'
    assert result['signatures']['reserved_holdout'] == 'reserved'
    independent, counts = [], Counter()
    for fold, (train, test) in enumerate(fs.folds_for(X, y, 3, opts.seed), 1):
        assert seen[f'outer_{fold}'] == set(X.iloc[train].index)
        assert not seen[f'outer_{fold}'] & set(X.iloc[test].index)
        baseline = fs.make_pipeline(tuple(X.columns), opts, {})
        baseline.fit(X.iloc[train], y.iloc[train])
        predicted = baseline.predict(X.iloc[test])
        independent.append(matthews_corrcoef(y.iloc[test], predicted))
        counts.update(fs.classification_counts(y.iloc[test], predicted, 'Inkorrekt'))
    assert result['nested_validation']['full_baseline']['mean_mcc'] == pytest.approx(np.mean(independent))
    assert result['nested_validation']['full_baseline']['rows'] == len(X)
    for key, value in fs.metrics(counts).items():
        assert result['nested_validation']['full_baseline'][key] == pytest.approx(value)
    subsets = pd.read_csv(tmp_path/'run/subsets.csv', sep=';')
    assert set(subsets.scope) == {'outer_1', 'outer_2', 'outer_3', 'development'}
    assert all(len(rows) <= opts.max_evaluations for _, rows in subsets.groupby('scope'))
    assert len(subsets) == result['evaluated_subsets']
    assert subsets.groupby('scope')['columns'].nunique().equals(subsets.groupby('scope').size())
    selection = json.loads((tmp_path/'run/selected_features.json').read_text())
    assert set(selection['selected_columns']) | set(selection['excluded_columns']) == set(X.columns)
    assert result['attempted_fits'] <= result['planned_fit_upper_bound']
    assert (tmp_path/'run/outer_validation.csv').exists()
    assert (tmp_path/'run/selection_stability.csv').exists()
    assert (tmp_path/'run/subset_comparisons.csv').exists()
    assert set(Path(tmp_path/'run').glob('*.sav')) == set()


def test_repeated_studies_select_same_columns_and_scores(tmp_path):
    X, y = dataset(n=90, p=4)
    opts = options(trials=2, sizes=(1, 2), refine_rounds=0, max_evaluations=3)
    a = fs.run_experiment(X, y, 'Inkorrekt', tmp_path/'a', opts, progress=None)
    b = fs.run_experiment(X, y, 'Inkorrekt', tmp_path/'b', opts, progress=None)
    assert a['recommendation'] == b['recommendation']
    assert a['nested_validation'] == b['nested_validation']


def test_group_boundaries_and_missing_group_validation():
    X, y = dataset(n=120)
    groups = pd.Series(np.arange(120) // 3, index=X.index)
    for train, test in fs.folds_for(X, y, 3, 42, groups):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
    groups.iloc[0] = np.nan
    with pytest.raises(ValueError, match='nonmissing'):
        fs.folds_for(X, y, 3, 42, groups)


@pytest.mark.parametrize('change', ['target', 'index', 'duplicate', 'labels', 'groups'])
def test_invalid_data_fails_before_output(tmp_path, change):
    X, y = dataset()
    target = 'Inkorrekt'
    groups = None
    if change == 'target': target = 'absent'
    if change == 'index': y.index = y.index + 1
    if change == 'duplicate': X.columns = ['same'] * X.shape[1]
    if change == 'labels': y.iloc[0] = np.nan
    if change == 'groups': groups = pd.Series(range(len(X)), index=X.index+1)
    with pytest.raises(ValueError):
        fs.run_experiment(X, y, target, tmp_path/'bad', options(), groups=groups, progress=None)
    assert not (tmp_path/'bad').exists()


def test_preexisting_output_is_not_overwritten(tmp_path):
    X, y = dataset()
    (tmp_path/'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='never overwritten'):
        fs.run_experiment(X, y, 'Inkorrekt', tmp_path, options(), progress=None)
    assert (tmp_path/'keep.txt').read_text() == 'keep'


def test_interruption_retains_completed_rows_and_status(tmp_path, monkeypatch):
    X, y = dataset()
    real = fs.Search.evaluate
    def evaluate(self, cols, stage):
        if len(self.cache) == 2:
            raise KeyboardInterrupt
        return real(self, cols, stage)
    monkeypatch.setattr(fs.Search, 'evaluate', evaluate)
    with pytest.raises(KeyboardInterrupt):
        fs.run_experiment(X, y, 'Inkorrekt', tmp_path/'partial', options(), progress=None)
    summary = json.loads((tmp_path/'partial/summary.json').read_text())
    assert summary['status'] == 'interrupted' and summary['evaluated_subsets'] == 2
    assert len(pd.read_csv(tmp_path/'partial/subsets.csv', sep=';')) == 2


def test_cli_demo_smoke_and_no_reserved_rows_enter_search(tmp_path, monkeypatch):
    module = cli()
    original = module.load_development
    loaded = []
    def load(args):
        result = original(args)
        loaded.append(set(result[0].index))
        return result
    monkeypatch.setattr(module, 'load_development', load)
    code = module.main(['--demo', '--trials', '2', '--sizes', '2', '4', '--inner-folds', '3',
                        '--outer-folds', '3', '--refine-rounds', '0', '--max-evaluations', '3',
                        '--output', str(tmp_path/'demo')])
    assert code == 0 and len(loaded[0]) == 256
    summary = json.loads((tmp_path/'demo/summary.json').read_text())
    assert summary['development_rows'] == 256
    assert summary['nested_validation']['selected']['rows'] == 256


def test_refinement_checks_all_additions_and_removals_before_accepting(monkeypatch):
    X, y = dataset(p=4)
    opts = options(trials=1, sizes=(1,), refine_rounds=2, max_evaluations=20)
    monkeypatch.setattr(fs, 'random_subsets', lambda cols, opts: [('f0',)])
    study = fs.Search(X, y, 'Inkorrekt', opts, {}, 'test', lambda row: None)
    scores = {tuple(X.columns): -0.1, ('f0',): 0.1, ('f0', 'f1'): 0.5}
    def evaluate(cols, stage):
        cols = tuple(c for c in X if c in cols)
        if cols not in study.cache:
            study.cache[cols] = dict(columns=cols, n_columns=len(cols), mean_mcc=scores.get(cols, 0.0),
                                    std_mcc=0.0, error='')
        return study.cache[cols]
    monkeypatch.setattr(study, 'evaluate', evaluate)
    best = study.run()
    assert best['columns'] == ('f0', 'f1')
    assert study.moves[0]['operation'] == 'add' and study.moves[0]['feature'] == 'f1'
    assert study.moves[0]['gain'] == pytest.approx(0.4)
    # The next round tested dropping each retained column and adding either remaining one.
    assert {('f0',), ('f1',), ('f0', 'f1', 'f2'), ('f0', 'f1', 'f3')} <= set(study.cache)
    assert len(study.moves) == 1


def snapshot_config(tmp_path):
    from Config import Config
    from GUIHandler import GUIHandler
    config = Config()
    config.connection.data_text_columns = ['code']
    config.connection.data_numerical_columns = ['a', 'b']
    config.connection.class_column = 'label'
    config.connection.id_column = 'id'
    config.connection.trusted_connection = True
    config.mode.dark_number_target = 'Inkorrekt'
    config.mode.test_size = 0.2
    params = {name: getattr(config, name) for name in ('connection', 'mode', 'io', 'debug', 'mail', 'name', 'save')}
    path = tmp_path / 'last.json'
    path.write_text(json.dumps(GUIHandler._config_params_to_last_run_snapshot(params)))
    return path, config


def test_csv_last_run_reuses_jbg_split_and_preserves_settings(tmp_path, monkeypatch):
    import JBGLogger
    from Config import Config
    from unittest.mock import Mock
    monkeypatch.setattr(Config, 'get_username', classmethod(lambda cls: 'test-user'))
    monkeypatch.setattr(JBGLogger, 'JBGLogger', lambda **kwargs: Mock())
    module = cli()
    path, config = snapshot_config(tmp_path)
    before = path.read_bytes()
    n = 90
    raw = pd.DataFrame(dict(code=['01'] * n, a=np.arange(n), b=np.arange(n) * 2,
        id=np.arange(1, n+1), label=['Korrekt'] * 72 + ['Inkorrekt'] * 18))
    csv = tmp_path / 'raw.csv'
    raw.sample(frac=1, random_state=1).to_csv(csv, sep=';', index=False)
    args = module.parser().parse_args(['--last-run', str(path), '--csv', str(csv)])
    X, y, target, settings, reserved = module.load_development(args)
    assert len(X) == len(y) == 72 and target == 'Inkorrekt'
    assert set(X.columns) == {'code', 'a', 'b'}
    assert X.code.unique().tolist() == ['01']
    assert settings['text_columns'] == ('code',)
    assert reserved and path.read_bytes() == before
    raw.sample(frac=1, random_state=2).to_csv(csv, sep=';', index=False)
    X2, y2, _, _, signature2 = module.load_development(args)
    assert X.index.equals(X2.index) and y.equals(y2) and reserved == signature2


def test_config_path_loads_absolute_file_and_sql_route_only_fetches(tmp_path, monkeypatch):
    import JBGLogger
    import SQLDataLayer
    from Config import Config
    from unittest.mock import Mock, call
    monkeypatch.setattr(Config, 'get_username', classmethod(lambda cls: 'test-user'))
    monkeypatch.setattr(JBGLogger, 'JBGLogger', lambda **kwargs: Mock())
    module = cli()
    _, config = snapshot_config(tmp_path)
    config_path = tmp_path / 'config.py'
    config.save_to_file(config_path)
    n = 90
    # get_column_names follows text, numeric, ID, class order.
    raw = [['01', i, i * 2, i + 1, 'Inkorrekt' if i < 18 else 'Korrekt'] for i in range(n)]
    fake = Mock()
    fake.get_dataset.return_value = raw
    monkeypatch.setattr(SQLDataLayer, 'DataLayer', lambda config, logger: fake)
    args = module.parser().parse_args(['--config', str(config_path)])
    X, y, _, _, _ = module.load_development(args)
    assert len(X) == len(y) == 72
    assert fake.method_calls == [call.get_dataset()]


def test_failed_nested_validation_marks_result_and_retains_candidates(tmp_path, monkeypatch):
    X, y = dataset()
    original = fs.make_pipeline
    def pipeline(cols, opts, settings):
        model = original(cols, opts, settings)
        real_fit = model.fit
        def fit(features, labels):
            # Outer train is 80 rows; inner train is 53/54 rows.
            if len(features) == 80:
                raise ValueError('outer fit intentionally failed')
            return real_fit(features, labels)
        model.fit = fit
        return model
    monkeypatch.setattr(fs, 'make_pipeline', pipeline)
    with pytest.raises(ValueError, match='intentionally failed'):
        fs.run_experiment(X, y, 'Inkorrekt', tmp_path/'failed', options(), progress=None)
    summary = json.loads((tmp_path/'failed/summary.json').read_text())
    assert summary['status'] == 'failed'
    assert len(pd.read_csv(tmp_path/'failed/subsets.csv', sep=';')) > 0


def test_sta_scaler_fitted_on_training_values_and_empty_exports_have_headers(tmp_path):
    X = pd.DataFrame({'number': [1., 2., 3., 9000.]})
    y = pd.Series(['Korrekt', 'Korrekt', 'Inkorrekt', 'Inkorrekt'])
    pipeline = fs.make_pipeline(('number',), options(), {})
    pipeline.fit(X.iloc[:3], y.iloc[:3])
    assert pipeline.named_steps['scale'].mean_[0] == 2.
    pipeline.predict(X.iloc[3:])
    assert pipeline.named_steps['scale'].mean_[0] == 2.
    fs.export_frame([], tmp_path/'empty.csv', columns=['scope', 'gain'])
    assert list(pd.read_csv(tmp_path/'empty.csv', sep=';').columns) == ['scope', 'gain']


def test_csv_duplicate_headers_fail_before_any_model_fit(tmp_path, monkeypatch):
    from Config import Config
    monkeypatch.setattr(Config, 'get_username', classmethod(lambda cls: 'test-user'))
    path, _ = snapshot_config(tmp_path)
    csv_file = tmp_path/'duplicate.csv'
    csv_file.write_text('code;a;a;b;id;label\n01;1;2;3;1;Korrekt\n')
    args = cli().parser().parse_args(['--last-run', str(path), '--csv', str(csv_file)])
    # JBGLogger uses an unavailable terminal backend in the local harness.
    import JBGLogger
    from unittest.mock import Mock
    monkeypatch.setattr(JBGLogger, 'JBGLogger', lambda **kwargs: Mock())
    with pytest.raises(ValueError, match='duplicate'):
        cli().load_development(args)


def test_cli_group_key_is_excluded_and_grouped_nested_search_completes(tmp_path, monkeypatch):
    module = cli()
    X, y = dataset()
    X['person'] = np.arange(len(X)) // 3
    monkeypatch.setattr(module, 'load_development', lambda args: (X, y, 'Inkorrekt', {}, 'reserved'))
    code = module.main(['--demo', '--group-column', 'person', '--trials', '2', '--sizes', '1', '2',
        '--inner-folds', '3', '--outer-folds', '3', '--refine-rounds', '0', '--max-evaluations', '3',
        '--output', str(tmp_path/'grouped')])
    assert code == 0
    summary = json.loads((tmp_path/'grouped/summary.json').read_text())
    assert summary['grouped_cv'] is True
    assert 'person' not in summary['original_columns']
    assert 'person' not in summary['recommendation']['selected_columns']
