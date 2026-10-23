"""Real sampling/CV checks for the selected Cartesian product and winning model."""
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from Config import Config
from Helpers import build_model_performance_matrix
from JBGHandler import Model, ModelHandler
from JBGMeta import (Algorithm, AlgorithmGridSearchParams, AlgorithmTuple, Oversampling,
                     Preprocess, PreprocessTuple, Reduction, ReductionTuple, Undersampling)
from JBGExceptions import ModelException, PipelineException


class TrainingOnly(SimpleNamespace):
    @property
    def X_validation(self):
        raise AssertionError('Holdout must not be accessed during candidate search')

    @property
    def Y_validation(self):
        raise AssertionError('Holdout must not be accessed during candidate search')


def data(n=140, minority=None):
    X, y = make_classification(n_samples=n, n_features=5, n_redundant=0,
                               weights=[0.8, 0.2], flip_y=0, random_state=11)
    if minority is not None:
        y[:] = 0
        y[-minority:] = 1
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return TrainingOnly(X=X, Y=y, X_train=X.copy(), Y_train=y.copy())


def handler(over=('NOG', 'RND'), under=('NUG', 'RND'), algorithms=('LRN',)):
    config = Config(mode=Config.Mode(
        oversamplers=list(over), undersamplers=list(under),
        algorithm=AlgorithmTuple(algorithms), preprocessor=PreprocessTuple(('STA',)),
        feature_selection=ReductionTuple(('NOR',)), max_iterations=300,
    ))
    from JBGMeta import ScoreMetric
    config.mode.scoring = ScoreMetric.mcc
    mh = ModelHandler(SimpleNamespace(config=config, logger=Mock(),
                                      STANDARD_DESIRED_N_JOBS=1, DEFAULT_N_JOBS_DESIRED=1))
    mh.model = Model()
    return mh


def test_all_pairs_share_folds_and_report_real_cv_scores(tmp_path):
    mh = handler(over=('NOG', 'RND', 'SME'), under=('NUG', 'RND', 'TKL'), algorithms=('LRN', 'DTC'))
    mh.handler.config.mode.preprocessor = PreprocessTuple(('NOS', 'STA'))
    dh = data()
    observed = []
    original = mh.get_cross_val_score
    def track(pipeline, dh, kfold, algorithm):
        observed.append(tuple((tuple(a), tuple(b)) for a, b in kfold.split(dh.X_train, dh.Y_train)))
        return original(pipeline, dh, kfold, algorithm)
    mh.get_cross_val_score = track
    selected = mh.spot_check_machine_learning_models(dh, str(tmp_path/'cv.csv'), k=3)
    rows = mh.handler.logger.print_test_performance.call_args.args[0]
    table = build_model_performance_matrix(rows)
    assert len(table) == 3 * 3 * 2 * 2
    assert table['Exception'].eq('').all(), table[['Oversampling', 'Undersampling', 'Exception']]
    assert np.isfinite(table['Mean cv']).all()
    assert len(observed) == len(table) and all(folds == observed[0] for folds in observed)
    assert set(zip(table.Oversampling, table.Undersampling)) == {
        (o, u) for o in ('NOG', 'RND', 'SME') for u in ('NUG', 'RND', 'TKL')}
    assert selected.oversampler.name == table.iloc[0].Oversampling
    assert selected.undersampler.name == table.iloc[0].Undersampling
    assert mh.handler.config.mode.oversampler is selected.oversampler
    assert mh.handler.config.mode.undersampler is selected.undersampler
    assert mh.handler.config.get_oversampler_abbreviations() == ['NOG', 'RND', 'SME']
    assert mh.handler.logger.start_inline_progress.call_args.args[2] == len(table)
    assert 'Holdout' not in ' '.join(table.columns)


@pytest.mark.parametrize('over,under', [
    (Oversampling.NOG, Undersampling.NUG), (Oversampling.RND, Undersampling.NUG),
    (Oversampling.NOG, Undersampling.RND), (Oversampling.NOG, Undersampling.TKL),
    (Oversampling.RND, Undersampling.RND), (Oversampling.SME, Undersampling.RND),
])
def test_original_row_scaling_prediction_and_unique_names(over, under):
    dh = data()
    mh = handler()
    estimator = LogisticRegression(max_iter=300)
    pipe = mh.get_pipeline(Reduction.NOR, None, Algorithm.LRN, estimator,
                           Preprocess.STA, Preprocess.STA.call_preprocess(), over, under, 5)
    original_names = [name for name, _ in pipe.steps]
    assert len(original_names) == len(set(original_names))
    fitted = pipe.fit(dh.X_train, dh.Y_train)
    assert fitted.named_steps['STA'].n_samples_seen_ == len(dh.Y_train)
    assert original_names.index('STA') < original_names.index(over.name)
    assert estimator.random_state is None
    assert fitted.named_steps['LRN'].random_state == 42
    assert fitted.predict(dh.X_train.iloc[:7]).shape == (7,)
    restored = pickle.loads(pickle.dumps(fitted))
    assert [name for name, _ in restored.steps] == original_names
    np.testing.assert_array_equal(restored.predict(dh.X_train), fitted.predict(dh.X_train))


def test_random_sampling_is_reproducible_and_does_not_mutate_factory_estimators():
    dh = data()
    mh = handler()
    args = (Reduction.NOR, None, Algorithm.LRN, LogisticRegression(max_iter=300),
            Preprocess.STA, Preprocess.STA.call_preprocess(), Oversampling.RND, Undersampling.RND, 5)
    first = mh.get_pipeline(*args).fit(dh.X_train, dh.Y_train)
    second = mh.get_pipeline(*args).fit(dh.X_train, dh.Y_train)
    np.testing.assert_array_equal(first.named_steps['RND'].sample_indices_, second.named_steps['RND'].sample_indices_)
    np.testing.assert_array_equal(first.named_steps['UNDER_RND'].sample_indices_, second.named_steps['UNDER_RND'].sample_indices_)
    np.testing.assert_array_equal(first.predict(dh.X_train), second.predict(dh.X_train))


def test_small_class_caps_neighbors_and_failed_pair_does_not_disable_sampling(tmp_path):
    mh = handler(over=('NOG', 'RND', 'SME'), under=('NUG',))
    dh = data(minority=7)
    selected = mh.spot_check_machine_learning_models(dh, str(tmp_path/'cv.csv'), k=3)
    pipe = mh._build_spot_check_pipeline(Reduction.NOR, Algorithm.LRN, Preprocess.STA,
        None, LogisticRegression(max_iter=300), Preprocess.STA.call_preprocess(),
        Oversampling.SME, Undersampling.NUG, StratifiedKFold(3), dh, 5)
    assert pipe.named_steps['SME'].k_neighbors == 3
    scores = cross_validate(pipe, dh.X_train, dh.Y_train, cv=StratifiedKFold(3),
                            scoring='matthews_corrcoef', error_score='raise')['test_score']
    assert np.isfinite(scores).all()
    tiny = data(minority=2)
    mh = handler(over=('NOG', 'SME'), under=('NUG',))
    mh.spot_check_machine_learning_models(tiny, str(tmp_path/'tiny.csv'), k=2)
    table = build_model_performance_matrix(mh.handler.logger.print_test_performance.call_args.args[0])
    assert table.loc[table.Oversampling.eq('NOG'), 'Mean cv'].notna().all()
    assert table.loc[table.Oversampling.eq('SME'), 'Exception'].str.contains('at least two minority').all()
    assert mh.handler.config.get_oversampler_abbreviations() == ['NOG', 'SME']


def test_final_search_and_retraining_use_winning_pair_and_actual_train_support(monkeypatch, tmp_path):
    mh = handler(over=('NOG', 'RND'), under=('NUG', 'RND'))
    dh = data(minority=7)
    # Full-data support is deliberately different: it must not drive fold count.
    dh.Y = pd.Series([0]*200 + [1]*70)
    monkeypatch.setitem(AlgorithmGridSearchParams.LRN.value, 'parameters', {'C': [0.1, 1.0]})
    model = mh.get_model_from(dh, str(tmp_path/'cv.csv'))
    assert mh.handler.logger.start_inline_progress.call_args.args[2] == 4
    assert model.pipeline.named_steps['STA'].n_samples_seen_ == len(dh.Y_train)
    assert model.pipeline.named_steps['LRN'].C in (0.1, 1.0)
    retrained = mh.retrain_picked_model(model.pipeline, dh.X_train, dh.Y_train)
    assert retrained is not model.pipeline
    assert [name for name, _ in retrained.steps] == [name for name, _ in model.pipeline.steps]
    assert retrained.named_steps[model.oversampler.name].__class__ == model.pipeline.named_steps[model.oversampler.name].__class__


def test_no_usable_pair_raises_and_negative_mcc_can_still_win(tmp_path):
    mh = handler(over=('SME',), under=('NUG',))
    with pytest.raises(ModelException, match='No model candidate completed'):
        mh.spot_check_machine_learning_models(data(minority=2), str(tmp_path/'failed.csv'), k=2)
    mh = handler(over=('NOG',), under=('NUG',))
    mh.get_cross_val_score = lambda **kwargs: np.array([-0.2, -0.1, -0.3])
    model = mh.spot_check_machine_learning_models(data(), str(tmp_path/'negative.csv'), k=3)
    assert model.pipeline is not None


def test_saved_artifact_keeps_winning_pair_and_requested_candidates(tmp_path):
    from JBGModelPersistence import load_model_artifact, save_model_artifact
    mh = handler(over=('NOG', 'RND'), under=('NUG', 'RND'))
    dh = data()
    selected = mh.spot_check_machine_learning_models(dh, str(tmp_path/'cv.csv'), k=3)
    selected.pipeline = mh.train_picked_model(selected.pipeline, dh.X_train, dh.Y_train)
    path = tmp_path / 'selected.sav'
    metadata = (selected.oversampler, selected.undersampler, selected.preprocess,
                selected.reduction, selected.algorithm)
    save_model_artifact(path, mh.handler.config, None, metadata, selected.pipeline, None, 5, compatibility={"schema_version": 1})
    saved_config, _, saved_metadata, saved_pipeline, _, _ = load_model_artifact(path)
    assert saved_metadata[:2] == metadata[:2]
    assert saved_config.mode.oversampler is selected.oversampler
    assert saved_config.mode.undersampler is selected.undersampler
    assert saved_config.get_oversampler_abbreviations() == ['NOG', 'RND']
    np.testing.assert_array_equal(saved_pipeline.predict(dh.X_train), selected.pipeline.predict(dh.X_train))
