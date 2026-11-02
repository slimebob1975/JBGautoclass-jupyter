"""Coordinate legacy model-artifact loading without GUI dependencies.

Callers retain public error handling and prediction configuration updates.
The explicit dependencies preserve existing JBGHandler patch points and avoid
importing the handler/model classes into this module (and import cycles).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def load_checked_model_artifact(
    filename: Path,
    current_config: Any,
    logger: Any,
    model_factory: Callable[..., Any],
    *,
    load_artifact: Callable[..., Any],
    review_source: Callable[..., Any],
    review_semantics: Callable[..., Any],
    resolve_sidecar: Callable[..., Any],
    load_keras: Callable[..., Any],
    attach_keras: Callable[..., Any],
    restore_model: Callable[..., Any],
    initialization_error: type[Exception],
) -> tuple[Any, Any]:
    """Validate before sidecar access and reconstruction; return model, feature count.

    Failure handling remains in ``ModelHandler.load_model_from_file``.
    """
    artifact, compatibility = load_artifact(filename)
    config, text_converter, (_oversampler, _undersampler, _preprocess,
                             _reduction, _algorithm), pipeline, keras_name, n_features = artifact

    source_review = review_source(current_config, config, compatibility)
    if source_review.issues:
        raise initialization_error('; '.join(source_review.issues))

    compatibility_review = review_semantics(config, text_converter, compatibility)
    if compatibility_review.issues:
        raise initialization_error(
            'Saved model compatibility metadata is inconsistent: '
            + '; '.join(compatibility_review.issues)
        )
    for warning in compatibility_review.warnings:
        logger.print_warning(warning)

    if keras_name is not None:
        keras_sidecar = resolve_sidecar(filename, keras_name)
        keras_model = load_keras(str(keras_sidecar))
        if not attach_keras(pipeline, keras_name, keras_model):
            raise initialization_error(
                'Current Keras artifact lacks fitted SciKeras wrapper metadata; '
                'retrain and save a new model with the current JBG version.'
            )

    return restore_model(artifact, model_factory), n_features


def save_checked_model_artifact(
    filename: Path, save_config: Any, model: Any, compatibility: Any,
    *, keras_library: Any, detach_keras: Callable[..., Any],
    build_payload: Callable[..., Any], save_artifact: Callable[..., Any],
    sidecar_path: Callable[..., Any],
) -> None:
    """Persist the established artifact and optional Keras sidecar in legacy order.

    Source metadata creation and SQL/CSV-specific exception handling remain
    with ModelHandler. Dependencies are injected to preserve patch points.
    """
    if model.algorithm.lib == keras_library:
        keras_name = model.pipeline.steps[-1][0]
        persisted_pipeline = detach_keras(model.pipeline, keras_name)
        data = build_payload(save_config, model, persisted_pipeline, keras_name)
        save_artifact(filename, *data, compatibility=compatibility)
        keras_sidecar = sidecar_path(filename, keras_name)
        model.pipeline.steps[-1][-1].model_.save(str(keras_sidecar))
    else:
        data = build_payload(save_config, model, model.pipeline, None)
        save_artifact(filename, *data, compatibility=compatibility)
