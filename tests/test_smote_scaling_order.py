"""Scale numerical SMOTE neighborhoods using original training-fold rows."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from scipy.sparse import csr_matrix, issparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from JBGExceptions import PipelineException
from JBGHandler import ModelHandler
from JBGMeta import Algorithm, Oversampling, Preprocess, Reduction, Undersampling
from JBGModelPersistence import save_model_artifact


def data():
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=3, n_redundant=0,
        weights=[0.8, 0.2], flip_y=0, class_sep=1.5, random_state=23,
    )
    X[:, 0] *= 1000
    return X, y


def handler():
    config = SimpleNamespace(mode=SimpleNamespace(decision_policy="default", probability_threshold=0.5))
    return ModelHandler(SimpleNamespace(logger=Mock(), config=config))


def pipeline(sampler=Oversampling.SME, preprocessor=Preprocess.STA,
             reduction=Reduction.NOR, undersampler=Undersampling.NUG,
             categorical_feature_indices=None):
    reducer = reduction.get_PCA(120, 5, 3) if reduction is Reduction.PCA else None
    result = handler().get_pipeline(
        reduction, reducer, Algorithm.LRN, LogisticRegression(max_iter=500),
        preprocessor, preprocessor.call_preprocess(), sampler, undersampler, 5,
        rfe_features=3, categorical_feature_indices=categorical_feature_indices,
    )
    if sampler is Oversampling.SME:
        result.set_params(SME__random_state=23)
    return result


@pytest.mark.parametrize("sampler", [
    Oversampling.SME, Oversampling.ADA, Oversampling.BRD,
    Oversampling.KMS, Oversampling.SVM,
])
def test_numerical_neighbor_samplers_place_scaling_before_both_samplers(sampler):
    names = [name for name, _ in pipeline(sampler).steps]
    assert names == ["IMP", "FLT", "STA", sampler.name, "NUG", "LRN"]
    assert sampler.uses_scaled_distances()


@pytest.mark.parametrize("sampler", [Oversampling.NOG, Oversampling.RND])
def test_numerical_random_paths_scale_first(sampler):
    names = [name for name, _ in pipeline(sampler).steps]
    assert names == ["IMP", "STA", sampler.name, "NUG", "LRN"]
    assert not sampler.uses_scaled_distances()


def test_smote_nc_requires_explicit_categorical_feature_contract():
    with pytest.raises(PipelineException, match="SMOTE-NC requires fitted categorical feature roles"):
        pipeline(Oversampling.SNC)


def test_smote_n_requires_all_converted_features_to_be_categorical():
    with pytest.raises(PipelineException, match="SMOTE-N requires every converted predictor"):
        pipeline(Oversampling.SNN, categorical_feature_indices=(0, 1))


def test_categorical_sampler_pipeline_order_preserves_category_codes_until_sampling():
    smotenc = pipeline(Oversampling.SNC, categorical_feature_indices=(4,))
    smoten = pipeline(Oversampling.SNN, categorical_feature_indices=tuple(range(5)))
    assert [name for name, _ in smotenc.steps] == ["IMP", "FLT", "SNC", "NUG", "STA", "LRN"]
    assert [name for name, _ in smoten.steps] == ["IMP", "SNN", "NUG", "STA", "LRN"]
    assert smotenc.named_steps["SNC"].categorical_features == [4]
    assert not Oversampling.SNC.uses_scaled_distances()
    assert not Oversampling.SNN.uses_scaled_distances()


@pytest.mark.parametrize("sampler", [
    Oversampling.SME, Oversampling.ADA, Oversampling.BRD,
    Oversampling.KMS, Oversampling.SVM,
])
def test_numerical_sampler_families_fit_with_scaler_on_original_rows(sampler):
    X, y = data()
    model = pipeline(sampler)
    model.set_params(**{f"{sampler.name}__random_state": 23})
    if sampler is Oversampling.KMS:
        model.set_params(KMS__cluster_balance_threshold=0)
    model.fit(X, y)
    assert model.named_steps["STA"].n_samples_seen_ == len(X)
    assert model.predict_proba(X[:9]).shape == (9, 2)


def test_rfe_and_neighbor_undersampling_stay_after_scaled_smote():
    X, y = data()
    fitted = pipeline(reduction=Reduction.RFE, undersampler=Undersampling.TKL).fit(X, y)
    assert [name for name, _ in fitted.steps] == ["IMP", "FLT", "STA", "SME", "TKL", "RFE", "LRN"]
    assert fitted.named_steps["STA"].n_samples_seen_ == len(X)
    assert fitted.named_steps["RFE"].n_features_ == 3
    assert fitted.predict(X[:9]).shape == (9,)


def test_actual_smote_input_and_synthetic_samples_are_invariant_to_feature_units(monkeypatch):
    X, y = data()
    seen, synthetic = [], []
    original = SMOTE._fit_resample

    def observe(self, X, y):
        seen.append(X.copy())
        result = original(self, X, y)
        synthetic.append(result[0].copy())
        return result

    monkeypatch.setattr(SMOTE, "_fit_resample", observe)
    first = pipeline().fit(X, y)
    different_units = X.copy()
    different_units[:, 0] *= 1000
    second = pipeline().fit(different_units, y)
    assert len(seen) == 2
    np.testing.assert_allclose(np.std(seen[0], axis=0), 1)
    np.testing.assert_allclose(seen[0], seen[1], atol=1e-12)
    np.testing.assert_allclose(synthetic[0], synthetic[1], atol=1e-12)
    assert len(synthetic[0]) > len(X)
    assert first.named_steps["STA"].n_samples_seen_ == len(X)
    assert second.named_steps["STA"].n_samples_seen_ == len(X)
    np.testing.assert_allclose(first.named_steps["STA"].mean_, X.mean(axis=0))


@pytest.mark.parametrize("preprocessor", [Preprocess.NOS, Preprocess.STA, Preprocess.MAX, Preprocess.MIX])
def test_imputation_float_conversion_and_selected_scaler_run_before_smote(preprocessor, monkeypatch):
    X, y = data()
    X = np.round(X).astype(float)
    X[0, 0] = np.nan
    seen = []
    original = SMOTE._fit_resample

    def observe(self, X, y):
        seen.append(X.copy())
        return original(self, X, y)

    monkeypatch.setattr(SMOTE, "_fit_resample", observe)
    fitted = pipeline(preprocessor=preprocessor).fit(X, y)
    assert len(seen) == 1 and seen[0].dtype == np.float64
    assert np.isfinite(seen[0]).all()
    expected = np.nan_to_num(X, nan=0)
    expected = clone(preprocessor.call_preprocess()).fit_transform(expected)
    np.testing.assert_allclose(seen[0], expected)
    assert fitted.predict(X[:10]).shape == (10,)


def test_sparse_integer_smote_stays_sparse_and_uses_floating_point(monkeypatch):
    X, y = data()
    X = csr_matrix(np.round(X).astype(np.int64))
    original = SMOTE._fit_resample
    seen = []

    def observe(self, X, y):
        assert issparse(X) and X.dtype == np.float64
        result = original(self, X, y)
        assert issparse(result[0]) and result[0].dtype == np.float64
        seen.append(X.shape)
        return result

    monkeypatch.setattr(SMOTE, "_fit_resample", observe)
    fitted = pipeline().fit(X, y)
    assert seen == [X.shape]
    assert fitted.predict(X[:11]).shape == (11,)


@pytest.mark.parametrize("reduction", [Reduction.NOR, Reduction.PCA])
def test_prediction_skips_resampling_and_preserves_query_rows(reduction, monkeypatch):
    X, y = data()
    fitted = pipeline(reduction=reduction).fit(X, y)
    scale = fitted.named_steps["STA"].scale_.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("prediction must not fit or resample")

    monkeypatch.setattr(SMOTE, "fit_resample", forbidden)
    monkeypatch.setattr(type(fitted.named_steps["STA"]), "fit", forbidden)
    assert fitted.predict(X[:13]).shape == (13,)
    assert fitted.predict_proba(X[:13]).shape == (13, 2)
    np.testing.assert_array_equal(fitted.named_steps["STA"].scale_, scale)


@pytest.mark.parametrize("workers", [1, 2])
def test_real_cv_fits_scaler_on_original_fold_only(workers):
    X, y = data()
    folds = StratifiedKFold(3, shuffle=True, random_state=7)
    results = cross_validate(
        pipeline(), X, y, cv=folds, n_jobs=workers,
        scoring="matthews_corrcoef", return_estimator=True, error_score="raise",
    )
    assert np.isfinite(results["test_score"]).all()
    for fitted, (train, test) in zip(results["estimator"], folds.split(X, y)):
        scaler = fitted.named_steps["STA"]
        assert scaler.n_samples_seen_ == len(train)
        np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
        assert fitted.predict(X[test]).shape == (len(test),)


def test_grid_search_and_full_data_retraining_keep_correct_order():
    X, y = data()
    search = GridSearchCV(
        pipeline(), {"LRN__C": [0.1, 1]}, cv=3,
        scoring="matthews_corrcoef", error_score="raise",
    ).fit(X[:90], y[:90])
    fitted = search.best_estimator_
    assert fitted.named_steps["STA"].n_samples_seen_ == 90
    retrained = handler().retrain_picked_model(fitted, X, y)
    assert retrained is not fitted
    assert retrained.named_steps["STA"].n_samples_seen_ == 120
    assert fitted.named_steps["STA"].n_samples_seen_ == 90
    assert retrained.predict_proba(X[:7]).shape == (7, 2)


def test_new_and_legacy_saved_pipelines_reload_in_fresh_process_without_reordering(tmp_path):
    X, y = data()
    current = pipeline()
    old = pipeline()
    steps = old.named_steps
    old.steps = [(name, steps[name]) for name in ["IMP", "FLT", "SME", "NUG", "STA", "LRN"]]
    np.save(tmp_path / "X.npy", X)
    for name, model in [("current", current), ("legacy", old)]:
        model.fit(X, y)
        np.save(tmp_path / f"{name}.npy", model.predict(X))
        payload = ({}, None, ("SME", "NUG", "STA", "NOR", "LRN"), model, None, 5)
        save_model_artifact(tmp_path / f"{name}.sav", *payload)
    code = """
import sys, numpy as np
from pathlib import Path
from JBGModelPersistence import load_model_artifact
root = Path(sys.argv[1]); X = np.load(root/'X.npy')
for name in ('current', 'legacy'):
    model = load_model_artifact(root/f'{name}.sav')[3]
    np.testing.assert_array_equal(model.predict(X), np.load(root/f'{name}.npy'))
    order = [step for step, _ in model.steps]
    assert (order.index('STA') < order.index('SME')) == (name == 'current')
"""
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, env=env,
                   capture_output=True, text=True)


def categorical_data():
    rng = np.random.RandomState(11)
    majority = 30
    minority = 10
    X = np.column_stack([
        np.r_[rng.normal(0.0, 1.0, majority), rng.normal(2.0, 1.0, minority)],
        np.r_[rng.normal(1.0, 0.5, majority), rng.normal(3.0, 0.5, minority)],
        np.r_[np.tile([0.0, 1.0, 2.0], 10), np.tile([1.0, 2.0], 5)],
    ])
    y = np.array([0] * majority + [1] * minority)
    return X, y


def test_smote_nc_keeps_declared_category_values_discrete(monkeypatch):
    X, y = categorical_data()
    observed = []
    original = SMOTENC._fit_resample

    def inspect(self, X_input, y_input):
        result = original(self, X_input, y_input)
        observed.append(result[0])
        return result

    monkeypatch.setattr(SMOTENC, "_fit_resample", inspect)
    fitted = pipeline(
        Oversampling.SNC,
        preprocessor=Preprocess.NOS,
        categorical_feature_indices=(2,),
    ).fit(X, y)
    assert fitted.named_steps["SNC"].categorical_features == [2]
    assert len(observed) == 1
    synthetic_categories = np.asarray(observed[0])[:, 2]
    assert set(np.unique(synthetic_categories)).issubset(set(np.unique(X[:, 2])))
    assert fitted.predict(X[:7]).shape == (7,)


def test_smote_n_keeps_all_feature_values_discrete(monkeypatch):
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2],
        [2, 0], [2, 1], [2, 2], [0, 0], [1, 1], [2, 2],
        [0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1],
        [2, 2], [2, 1], [1, 2], [2, 0], [1, 1], [2, 2],
    ], dtype=float)
    y = np.array([0] * 18 + [1] * 6)
    observed = []
    original = SMOTEN._fit_resample

    def inspect(self, X_input, y_input):
        result = original(self, X_input, y_input)
        observed.append(result[0])
        return result

    monkeypatch.setattr(SMOTEN, "_fit_resample", inspect)
    fitted = handler().get_pipeline(
        Reduction.NOR, None, Algorithm.LRN, LogisticRegression(max_iter=500),
        Preprocess.NOS, Preprocess.NOS.call_preprocess(), Oversampling.SNN,
        Undersampling.NUG, 2, categorical_feature_indices=(0, 1),
    ).fit(X, y)
    assert len(observed) == 1
    result = np.asarray(observed[0])
    for column in range(X.shape[1]):
        assert set(np.unique(result[:, column])).issubset(set(np.unique(X[:, column])))
    assert fitted.predict(X[:5]).shape == (5,)


def test_spot_check_derives_smote_nc_indices_from_fitted_converter_roles():
    X, y = categorical_data()
    X = pd.DataFrame(X, columns=["numeric_a", "numeric_b", "category"])
    mh = handler()
    mh.model = SimpleNamespace(text_converter=SimpleNamespace(category_columns_=["category"]))
    dh = SimpleNamespace(X_train=X, Y_train=pd.Series(y))
    pipe = mh._build_spot_check_pipeline(
        Reduction.NOR, Algorithm.LRN, Preprocess.NOS, None,
        LogisticRegression(max_iter=500), Preprocess.NOS.call_preprocess(),
        Oversampling.SNC, Undersampling.NUG,
        StratifiedKFold(3, shuffle=True, random_state=7), dh, X.shape[1],
    )
    assert pipe.named_steps["SNC"].categorical_features == [2]
    fitted = pipe.fit(X, y)
    assert fitted.predict(X.iloc[:5]).shape == (5,)
