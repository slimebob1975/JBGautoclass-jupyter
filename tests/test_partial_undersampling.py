"""Fold-local partial profiles through real sampling, CV and saved pipelines."""
from collections import Counter
import pickle
from pathlib import Path

import ipywidgets
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from Config import Config
from GUI.Widgets import Widgets
from GUIHandler import GUIHandler
from Helpers import build_model_performance_matrix
from JBGMeta import Algorithm, Oversampling, Preprocess, Reduction, Undersampling
from JBGSampling import partial_undersampling_counts
from test_sampling_search import handler, data

PROFILES = [('R20', 20), ('R10', 10), ('R05', 5), ('R02', 2)]
CHOICES = ['NUG', 'R20', 'R10', 'R05', 'R02', 'RND']


@pytest.mark.parametrize('name,multiple', PROFILES)
@pytest.mark.parametrize('as_sparse', [False, True])
def test_actual_resampling_keeps_all_minority_and_expected_majority(name, multiple, as_sparse):
    y = np.array(['Korrekt'] * 250 + ['Inkorrekt'] * 5)
    X = np.arange(len(y) * 2).reshape(-1, 2)
    if as_sparse:
        X = sparse.csr_matrix(X)
    sampler = Undersampling[name].call_undersampler().set_params(random_state=42)
    Xr, yr = sampler.fit_resample(X, y)
    assert Counter(yr) == {'Korrekt': 5 * multiple, 'Inkorrekt': 5}
    assert set(range(250, 255)) <= set(sampler.sample_indices_)
    assert sparse.issparse(Xr) == as_sparse
    restored = pickle.loads(pickle.dumps(clone(sampler)))
    restored.fit_resample(X, y)
    np.testing.assert_array_equal(restored.sample_indices_, sampler.sample_indices_)


@pytest.mark.parametrize('name,multiple', PROFILES)
@pytest.mark.parametrize('counts', [(8, 5), (5, 5), (5, 8)])
def test_already_balanced_fold_does_not_fail_or_drop_rows(name, multiple, counts):
    y = np.array([0] * counts[0] + [1] * counts[1])
    sampler = Undersampling[name].call_undersampler()
    _, yr = sampler.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
    assert Counter(yr) == Counter(y)
    assert sorted(sampler.sample_indices_) == list(range(len(y)))


def test_multiclass_caps_each_class_and_legacy_defaults_are_unchanged():
    y = ['a'] * 51 + ['b'] * 23 + ['c'] * 3
    assert partial_undersampling_counts(y, 5) == {'a': 15, 'b': 15, 'c': 3}
    _, yr = Undersampling.R05.call_undersampler().fit_resample(np.ones((77, 2)), y)
    assert Counter(yr) == {'a': 15, 'b': 15, 'c': 3}
    assert Undersampling.RND.call_undersampler().sampling_strategy == 'auto'
    assert Undersampling.defaultUndersampler() is Undersampling.NUG


@pytest.mark.parametrize('multiple', [0, -1, 2.5, True])
def test_invalid_cap_rejected(multiple):
    with pytest.raises(ValueError):
        partial_undersampling_counts([0, 1], multiple)


def test_config_snapshot_generated_file_and_real_widget_roundtrip(tmp_path):
    config = Config(mode=Config.Mode(undersamplers=CHOICES, undersampler=Undersampling.R05))
    path = tmp_path / 'config.json'
    config.save_to_file(path)
    loaded = Config.load_config_from_json(path)
    assert loaded.get_undersampler_abbreviations() == CHOICES
    assert loaded.mode.undersampler is Undersampling.R05
    params = {name: getattr(config, name) for name in ('connection', 'mode', 'io', 'debug', 'mail', 'name', 'save')}
    snapshot = GUIHandler._config_params_to_last_run_snapshot(params)
    restored = GUIHandler._last_run_snapshot_to_config_params(snapshot, 'user', 'password')
    assert restored['mode'].undersamplers == CHOICES
    widgets = Widgets.__new__(Widgets)
    import json
    settings = json.loads((Path(__file__).parents[1] / 'src/GUI/default_settings.json').read_text())
    widgets.default_widgets = settings['widgets']
    widgets.default_widgets['undersampler_dropdown']['params']['value'] = CHOICES
    widgets.widgets = {}
    widgets._migrate_sampling_widgets()
    assert isinstance(widgets.undersampler_dropdown, ipywidgets.SelectMultiple)
    assert widgets.undersampler_dropdown.value == tuple(CHOICES)
    assert set(CHOICES) <= {value for _, value in widgets.undersampler_dropdown.options}


def test_fold_counts_scaler_seed_worker_pickle_and_fresh_retraining():
    dh = data(n=240, minority=12)
    mh = handler(over=('NOG',), under=('R05',))
    pipe = mh.get_pipeline(Reduction.NOR, None, Algorithm.LRN, LogisticRegression(max_iter=300),
                           Preprocess.STA, Preprocess.STA.call_preprocess(),
                           Oversampling.NOG, Undersampling.R05, 5)
    folds = StratifiedKFold(3, shuffle=True, random_state=42)
    result = cross_validate(pipe, dh.X_train, dh.Y_train, cv=folds,
                            scoring='matthews_corrcoef', return_estimator=True,
                            n_jobs=2, error_score='raise')
    assert np.isfinite(result['test_score']).all()
    for estimator, (train, validation) in zip(result['estimator'], folds.split(dh.X_train, dh.Y_train)):
        sampler = estimator.named_steps['R05']
        counts = Counter(dh.Y_train.iloc[train])
        assert sampler.sampling_strategy_[0] == counts[1] * 5
        assert len(sampler.sample_indices_) == counts[1] * 6
        assert estimator.named_steps['STA'].n_samples_seen_ == len(train)
        assert sampler.random_state == 42
    fitted = pipe.fit(dh.X_train, dh.Y_train)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(dh.X_train), fitted.predict(dh.X_train))
    retrained = mh.retrain_picked_model(restored, dh.X_train, dh.Y_train)
    assert retrained is not restored
    assert retrained.named_steps['R05'].sampling_strategy_[0] == 60
    np.testing.assert_array_equal(retrained.named_steps['R05'].sample_indices_, fitted.named_steps['R05'].sample_indices_)


def test_six_way_search_export_and_selected_profile_final_search(tmp_path, monkeypatch):
    from JBGMeta import AlgorithmGridSearchParams
    from JBGModelPersistence import load_model_artifact, save_model_artifact
    mh = handler(over=('NOG',), under=CHOICES)
    from JBGLogger import JBGLogger
    mh.handler.logger.print_test_performance.side_effect = lambda results, path: JBGLogger.print_test_performance(mh.handler.logger, results, path)
    dh = data(n=240, minority=12)
    path = tmp_path / 'cv.csv'
    selected = mh.spot_check_machine_learning_models(dh, str(path), k=3)
    table = build_model_performance_matrix(mh.handler.logger.print_test_performance.call_args.args[0])
    assert set(table.Undersampling) == set(CHOICES)
    assert len(table) == 6
    assert table['Mean cv'].notna().all()
    assert set(pd.read_csv(path, sep=';').Undersampling) == set(CHOICES)
    # Exercise final tuning and persistence even if the real six-way winner is NUG.
    mh = handler(over=('NOG',), under=('R10',))
    monkeypatch.setitem(AlgorithmGridSearchParams.LRN.value, 'parameters', {'C': [0.1, 1.0]})
    selected = mh.get_model_from(dh, str(tmp_path / 'final.csv'))
    assert selected.undersampler is Undersampling.R10
    assert selected.pipeline.named_steps['R10'].sampling_strategy_[0] == 120
    saved = tmp_path / 'model.sav'
    metadata = (selected.oversampler, selected.undersampler, selected.preprocess, selected.reduction, selected.algorithm)
    save_model_artifact(saved, mh.handler.config, None, metadata, selected.pipeline, None, 5, compatibility={"schema_version": 1})
    config, _, stored_metadata, pipeline, _, _ = load_model_artifact(saved)
    assert config.mode.undersampler is Undersampling.R10
    assert stored_metadata[1] is Undersampling.R10
    assert config.get_undersampler_abbreviations() == ['R10']
    np.testing.assert_array_equal(pipeline.predict(dh.X_train), selected.pipeline.predict(dh.X_train))


@pytest.mark.parametrize('name,multiple', PROFILES)
def test_real_mlpc_max_pipeline_and_prediction_do_not_resample(name, multiple):
    from sklearn.neural_network import MLPClassifier
    dh = data(n=120, minority=4)
    mh = handler(over=('NOG',), under=(name,))
    pipe = mh.get_pipeline(Reduction.NOR, None, Algorithm.MLPC,
                           MLPClassifier(hidden_layer_sizes=(8,), max_iter=60),
                           Preprocess.MAX, Preprocess.MAX.call_preprocess(),
                           Oversampling.NOG, Undersampling[name], 5)
    pipe.fit(dh.X_train, dh.Y_train)
    sampler = pipe.named_steps[name]
    assert sampler.sampling_strategy_[0] == min(116, 4 * multiple)
    indices = sampler.sample_indices_.copy()
    predictions = pipe.predict(dh.X_train.iloc[:17])
    assert len(predictions) == 17
    np.testing.assert_array_equal(sampler.sample_indices_, indices)
