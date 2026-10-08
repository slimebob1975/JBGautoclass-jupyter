"""Scale numerical SMOTE neighborhoods using original training-fold rows."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix, issparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

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
    return ModelHandler(SimpleNamespace(logger=Mock()))


def pipeline(sampler=Oversampling.SME, preprocessor=Preprocess.STA,
             reduction=Reduction.NOR, undersampler=Undersampling.NUG):
    reducer = reduction.get_PCA(120, 5, 3) if reduction is Reduction.PCA else None
    result = handler().get_pipeline(
        reduction, reducer, Algorithm.LRN, LogisticRegression(max_iter=500),
        preprocessor, preprocessor.call_preprocess(), sampler, undersampler, 5,
        rfe_features=3,
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


@pytest.mark.parametrize("sampler", [Oversampling.NOG, Oversampling.RND, Oversampling.SNC, Oversampling.SNN])
def test_other_sampler_families_keep_existing_order(sampler):
    names = [name for name, _ in pipeline(sampler).steps]
    float_step = ["FLT"] if sampler.requires_float_input() else []
    assert names == ["IMP", *float_step, sampler.name, "NUG", "STA", "LRN"]
    assert not sampler.uses_scaled_distances()


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
