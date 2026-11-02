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
