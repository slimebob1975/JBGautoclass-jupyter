"""Construct the legacy model-artifact payload without changing its schema.

The source-boundary and semantic compatibility checks remain at the call site.
This module only owns the positional payload layout consumed by
``JBGModelPersistence.save_model_artifact``.
"""
from __future__ import annotations

from typing import Any


def build_model_artifact_payload(
    save_config: Any, model: Any, pipeline: Any, keras_name: str | None
) -> list[Any]:
    """Return the established six-element payload, preserving object identity."""
    return [
        save_config,
        model.text_converter,
        (
            model.oversampler,
            model.undersampler,
            model.preprocess,
            model.reduction,
            model.algorithm,
        ),
        pipeline,
        keras_name,
        model.n_features_out,
    ]


def restore_model_from_artifact_payload(artifact: Any, model_factory: Any) -> Any:
    """Reconstruct a model using the established six-field artifact layout.

    Call only after source-boundary and semantic compatibility checks. Keras
    sidecar reattachment, when required, is performed by the caller first.
    Keep the model class at its original import location for pickle stability.
    """
    _config, text_converter, (oversampler, undersampler, preprocess, reduction, algorithm), pipeline, _keras_name, n_features = artifact
    return model_factory(
        text_converter=text_converter,
        oversampler=oversampler,
        undersampler=undersampler,
        preprocess=preprocess,
        reduction=reduction,
        algorithm=algorithm,
        pipeline=pipeline,
        n_features_out=n_features,
    )
