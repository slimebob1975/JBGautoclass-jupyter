"""Build estimator pipelines without depending on ModelHandler or the GUI.

The ordering and validation rules are preserved from ModelHandler.get_pipeline.
"""
from __future__ import annotations

from typing import Union
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline

from JBGDecisionPolicy import DecisionPipeline, validate_decision_policy
from JBGNativeCatBoost import RawCatBoostColumns
from JBGMeta import Algorithm, Preprocess, Reduction, Oversampling, Undersampling, Estimator, Transform
from JBGExceptions import PipelineException
from JBGSampling import validate_categorical_oversampling
import Helpers


def build_model_pipeline(
    reduction: Reduction, feature_reducer: Transform, algorithm: Algorithm,
    estimator: Estimator, preprocessor: Preprocess, scaler: Transform,
    oversampler: Oversampling, undersampler: Undersampling, max_features: int,
    rfe_features: int = None, max_k_neighbors: int = None,
    categorical_feature_indices: tuple[int, ...] = None,
    *, decision_policy: str = "default", probability_threshold: float = 0.5,
    positive_label=None, catboost_schema_initialized: bool = False,
    catboost_original_columns=None, catboost_categorical_columns=None,
) -> Union[Pipeline, Estimator]:
    policy, threshold = validate_decision_policy(
        decision_policy,
        probability_threshold)
    if policy == "fixed" and not hasattr(estimator, "predict_proba"):
        raise PipelineException(f"{algorithm.name} does not expose predict_proba()")
    pipeline_type = DecisionPipeline if policy == "fixed" else Pipeline
    decision_args = ({"decision_policy": policy, "probability_threshold": threshold,
                      "positive_label": positive_label}
                     if policy == "fixed" else {})
    if algorithm == Algorithm.CBST:
        if (preprocessor != Preprocess.NOS or reduction != Reduction.NOR or
            oversampler != Oversampling.NOG or undersampler != Undersampling.NUG):
            raise PipelineException("Native CatBoost requires NOS/NOR/NOG/NUG (no scaling, reduction or sampling)")
        if not catboost_schema_initialized:
            raise PipelineException("Native CatBoost schema was not initialized")
        return pipeline_type(steps=[
            ("native_columns", RawCatBoostColumns(
                catboost_original_columns, catboost_categorical_columns)),
            (algorithm.name, clone(estimator)),
        ], verbose=False, **decision_args)
    try:
        # First, put estimator/algorithm in pipeline
        # CV clones share fixed randomness without mutating enum factory objects.
        estimator = clone(estimator)
        seed_params = {name: 42 for name, value in estimator.get_params(deep=True).items()
                       if (name == "random_state" or name.endswith("__random_state")) and value is None}
        if seed_params:
            estimator.set_params(**seed_params)
        steps = [(algorithm.name, estimator)]

        # Secondly, add the other steps to the pipeline BEFORE the algorithm
        steps.insert(0, (preprocessor.name, scaler))

        # RFE object must unfortunately be updated with the correct estimator at this point
        if reduction == Reduction.RFE:
            the_feature_reducer = RFE(estimator=estimator, n_features_to_select=rfe_features)
        else:
            the_feature_reducer = feature_reducer

        # Then add the feature reduction part of the pipeline between the preprocessor and the estimator
        steps.insert(1, (reduction.name, the_feature_reducer))

        # Keep over before under and disambiguate the shared RND enum name.
        over = oversampler.get_callable_oversampler()
        under = undersampler.get_callable_undersampler()

        categorical_feature_indices = tuple(categorical_feature_indices or ())
        if oversampler in (Oversampling.SNC, Oversampling.SNN):
            try:
                categorical_feature_indices = validate_categorical_oversampling(
                    oversampler.name, categorical_feature_indices, int(max_features)
                )
            except ValueError as ex:
                raise PipelineException(str(ex)) from ex

        if oversampler == Oversampling.SNC:
            over.set_params(categorical_features=list(categorical_feature_indices))

        for sampler in (over, under):
            if "random_state" in sampler.get_params(deep=False):
                sampler.set_params(random_state=42)
        if max_k_neighbors is not None:
            params = over.get_params(deep=False)
            for key in ("k_neighbors", "n_neighbors"):
                if key in params and isinstance(params[key], int):
                    over.set_params(**{key: min(params[key], max_k_neighbors)})
        under_name = undersampler.name if undersampler.name != oversampler.name else f"UNDER_{undersampler.name}"
        steps.insert(0, (oversampler.name, over))
        steps.insert(1, (under_name, under))

        # Every numerical sampling alternative fits scaling on original rows.
        # This also scales neighbor undersampling when no SMOTE is selected.
        # Categorical SMOTE paths retain their historical separate contract.
        if oversampler not in (Oversampling.SNC, Oversampling.SNN):
            steps.insert(0, steps.pop(2))

        # Interpolation-based SMOTE-family samplers must receive floating-point
        # features.  With integer input imbalanced-learn preserves the integer dtype
        # for generated samples, which truncates interpolation and can surface later
        # as float64-to-int64 failures (notably in MLPC GridSearchCV runs).
        if oversampler.requires_float_input():
            float_transformer = FunctionTransformer(
                Helpers.ensure_float64, validate=False, accept_sparse=True
            )
            steps.insert(0, ("FLT", float_transformer))

        # The oversampling or undersampling techniques may require some imputation of NaN values
        if oversampler or undersampler:
            imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
            steps.insert(0, ("IMP", imputer))

        # Sparse integer matrices must be converted before SimpleImputer:
        # sklearn cannot cast a floating-point fill value back to int64.
        # Keep SciPy sparse input sparse and preserve the original order
        # for random and all-categorical sampling alternatives.
        if oversampler.requires_float_input():
            steps.insert(0, steps.pop(1))  # FLT, IMP, scaler, sampler

        # Enforce the sampling-order contract at runtime. Numerical neighbor
        # interpolation must see the selected fold-fitted scaler first; categorical
        # samplers must see ordinal category codes before generic scaling.
        step_names = [name for name, _ in steps]
        scaler_index = step_names.index(preprocessor.name)
        oversampler_index = step_names.index(oversampler.name)
        if oversampler.uses_scaled_distances() and scaler_index > oversampler_index:
            raise PipelineException(
                "Numerical neighbor oversampling must run after fold-local scaling."
            )
        if oversampler in (Oversampling.SNC, Oversampling.SNN) and oversampler_index > scaler_index:
            raise PipelineException(
                "Categorical oversampling must run before generic scaling so category codes are not "
                "treated as continuous values during resampling."
            )

        # Check that all steps in pipeline implements "fit"
        steps = [step for step in steps if hasattr(step[1] , "fit") and callable(getattr(step[1] , "fit"))]

    except Exception as ex:
        raise PipelineException(f"Could not build Pipeline correctly: {str(ex)}") from ex

    return pipeline_type(steps=steps, verbose=False, **decision_args)
