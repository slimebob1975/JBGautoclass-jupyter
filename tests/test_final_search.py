"""Keep the actual Extra Trees baseline eligible and report training-only tuning."""
import json
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from Config import Config
from JBGHandler import ModelHandler
from JBGMeta import Algorithm, AlgorithmGridSearchParams, ScoreMetric
from JBGFinalSearch import extra_trees_search_parameters, include_extra_trees_baseline
import Helpers


def handler():
    config = Config(mode=Config.Mode(scoring=ScoreMetric.mcc))
    config.io.verbose = False
    return ModelHandler(SimpleNamespace(config=config, logger=Mock(), STANDARD_DESIRED_N_JOBS=1))


def data():
    X, y = make_classification(n_samples=160, n_features=5, n_redundant=0,
                               weights=[0.9, 0.1], flip_y=0, class_sep=1.5, random_state=4)
    return pd.DataFrame(X), pd.Series(y)


def pipe(**kwargs):
    return Pipeline([('STA', StandardScaler()), ('ETC', ExtraTreesClassifier(random_state=42, **kwargs))])


def prefixed(grid):
    return {'ETC__' + key: value for key, value in grid.items()}


def test_bounded_runtime_grid_retains_factory_defaults_and_legacy_enum_value():
    enum = AlgorithmGridSearchParams.ETC
    candidates = list(ParameterGrid(enum.parameters))
    assert len(candidates) == 24
    defaults = Algorithm.ETC.call_algorithm(max_iterations=100, size=160).get_params()
    assert any(all(defaults[key] == value for key, value in candidate.items()) for candidate in candidates)
    assert len(ParameterGrid(enum.value['parameters'])) == 972
    assert pickle.loads(pickle.dumps(enum)) is enum
    assert enum.parameters == extra_trees_search_parameters()
    assert enum.parameters['max_depth'] == (None, 12, 24)


def test_actual_baseline_is_already_in_default_grid_without_duplication():
    model = pipe()
    grid = prefixed(extra_trees_search_parameters())
    selected, baseline = include_extra_trees_baseline(model, grid)
    assert selected is grid and len(ParameterGrid(selected)) == 24
    assert baseline['ETC__max_depth'] is None
    assert baseline['ETC__n_estimators'] == 100
    assert baseline['ETC__class_weight'] is None
    assert model.named_steps['ETC'].random_state == 42


@pytest.mark.parametrize('grid', [
    {'ETC__max_depth': [1]},
    [{'ETC__max_depth': [1], 'ETC__n_estimators': [10]}, {'ETC__max_depth': [2]}],
])
def test_restricted_grid_adds_exact_custom_baseline_without_mutating_input(grid):
    import copy
    before = copy.deepcopy(grid)
    model = pipe(n_estimators=7, criterion='entropy', max_depth=17, class_weight='balanced')
    selected, baseline = include_extra_trees_baseline(model, grid)
    assert grid == before
    candidates = list(ParameterGrid(selected))
    assert len(candidates) == len(ParameterGrid(grid)) + 1
    assert candidates[-1] == baseline
    candidate = model.set_params(**candidates[-1]).named_steps['ETC']
    assert (candidate.n_estimators, candidate.max_depth, candidate.criterion, candidate.class_weight) == (7, 17, 'entropy', 'balanced')


def test_other_model_searches_keep_their_exact_grid():
    model = Pipeline([('LRN', LogisticRegression())])
    grid = {'LRN__C': [0.1, 10]}
    selected, baseline = include_extra_trees_baseline(model, grid)
    assert selected is grid and baseline is None


def test_real_search_scores_match_reference_no_extra_fits_and_export_roundtrip(monkeypatch, tmp_path):
    X, y = data()
    model = pipe()
    grid = prefixed(extra_trees_search_parameters())
    fits = []
    original_fit = ExtraTreesClassifier.fit
    def tracked(estimator, X, y, *args, **kwargs):
        fits.append(len(y))
        return original_fit(estimator, X, y, *args, **kwargs)
    monkeypatch.setattr(ExtraTreesClassifier, 'fit', tracked)
    mh = handler()
    mh._selected_cv_score = (model, 0.123)
    best, results = mh.model_parameter_grid_search(model, grid, 3, X, y)
    assert len(fits) == 24 * 3 + 1
    summary = mh._last_grid_search_summary
    assert len(results) == 24
    assert results.is_baseline.sum() == 1 and results.is_selected.sum() == 1
    baseline_row = results.loc[results.is_baseline].iloc[0]
    assert summary['baseline_cv_mean'] == baseline_row.mean_test_score
    assert summary['best_cv_mean'] == results.loc[results.is_selected, 'mean_test_score'].iloc[0]
    assert summary['best_cv_mean'] >= summary['baseline_cv_mean']
    assert summary['spot_check_cv_mean'] == 0.123
    assert summary['metric'] == 'Matthews Corr. Coefficient'
    assert 'holdout excluded' in summary['data_scope']
    assert not hasattr(model.named_steps['ETC'], 'classes_')
    mh._export_final_search_results(results, tmp_path/'crossval.csv')
    saved = json.loads((tmp_path/'crossval_grid_search_summary.json').read_text())
    table = pd.read_csv(tmp_path/'crossval_grid_search.csv', sep=';', decimal=',')
    assert saved == summary and len(table) == 24
    assert table.is_baseline.sum() == 1 and table.is_selected.sum() == 1
    assert len(fits) == 73  # Export/reporting does not retrain.
    assert 'data:application/json;base64,' in mh.handler.logger.print_code.call_args.args[1]
    reference = GridSearchCV(pipe(), grid, scoring='matthews_corrcoef',
        cv=StratifiedKFold(3, shuffle=True, random_state=1), error_score='raise').fit(X, y)
    assert summary['best_cv_mean'] == reference.best_score_
    assert summary['best_params'] == reference.best_params_
    np.testing.assert_array_equal(best.predict(X), reference.predict(X))
    messages = ' '.join(str(call.args[0]) for call in mh.handler.logger.print_info.call_args_list)
    assert 'Final GridSearch CV' in messages and 'best-minus-baseline=' in messages
    assert mh.handler.logger.print_code.call_args_list[-3].args[0] == 'GridSearch selected parameters'


def test_custom_baseline_is_scored_in_the_same_search():
    X, y = data()
    mh = handler()
    best, results = mh.model_parameter_grid_search(pipe(n_estimators=7), {'ETC__max_depth': [1]}, 3, X, y)
    assert len(results) == 2
    assert results.is_baseline.sum() == 1
    assert mh._last_grid_search_summary['best_minus_baseline'] >= 0
    assert results.loc[results.is_baseline, 'params'].iloc[0]['ETC__max_depth'] is None
    assert best.named_steps['ETC'].n_estimators == 7


def test_stale_spot_check_score_and_failed_search_summary_are_not_reused(monkeypatch):
    X, y = data()
    mh = handler()
    mh._selected_cv_score = (pipe(), 0.9)
    mh.model_parameter_grid_search(pipe(n_estimators=7), {'ETC__max_depth': [1]}, 3, X, y)
    assert mh._last_grid_search_summary['spot_check_cv_mean'] is None
    monkeypatch.setattr(GridSearchCV, 'fit', Mock(side_effect=ValueError('fit failed')))
    from JBGExceptions import ModelException
    with pytest.raises(ModelException):
        mh.model_parameter_grid_search(pipe(), {'ETC__max_depth': [1]}, 3, X, y)
    assert mh._last_grid_search_summary is None


def test_optional_csv_failure_keeps_fitted_model_and_json(monkeypatch, tmp_path):
    X, y = data()
    mh = handler()
    best, results = mh.model_parameter_grid_search(pipe(n_estimators=7), {'ETC__max_depth': [1]}, 3, X, y)
    expected = best.predict(X)
    monkeypatch.setattr(Helpers, 'save_matrix_as_csv', Mock(side_effect=OSError('CSV unavailable')))
    mh._export_final_search_results(results, tmp_path/'crossval.csv')
    assert (tmp_path/'crossval_grid_search_summary.json').exists()
    np.testing.assert_array_equal(expected, best.predict(X))
    assert mh.handler.logger.print_warning.called


def test_training_entry_point_exports_search_and_never_reads_holdout(tmp_path):
    from JBGHandler import Model
    from JBGMeta import AlgorithmTuple, PreprocessTuple, ReductionTuple
    class TrainingOnly(SimpleNamespace):
        @property
        def X_validation(self):
            raise AssertionError('holdout access during model selection')
        @property
        def Y_validation(self):
            raise AssertionError('holdout access during model selection')
    X, y = data()
    mh = handler()
    mh.model = Model()
    mh.STANDARD_K_FOLDS = 3
    mode = mh.handler.config.mode
    mode.algorithm = AlgorithmTuple(('ETC',))
    mode.preprocessor = PreprocessTuple(('MAX',))
    mode.feature_selection = ReductionTuple(('NOR',))
    dh = TrainingOnly(X=X, Y=y, X_train=X.copy(), Y_train=y.copy())
    model = mh.get_model_from(dh, str(tmp_path/'crossval.csv'))
    summary = json.loads((tmp_path/'crossval_grid_search_summary.json').read_text())
    assert summary['candidate_count'] == 24 and summary['cv_folds'] == 3
    assert summary['spot_check_cv_mean'] is not None
    assert summary['best_cv_mean'] >= summary['baseline_cv_mean']
    assert (tmp_path/'crossval_grid_search.csv').exists()
    assert model.algorithm is Algorithm.ETC
    assert model.pipeline.predict(X[:5]).shape == (5,)
