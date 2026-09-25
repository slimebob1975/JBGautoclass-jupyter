"""Versioned serialization contract for JBG model artifacts.

The project still uses dill for compatibility with existing model files, but the
on-disk object now has an explicit format marker and schema version.  Legacy
six-item list/tuple artifacts remain readable so existing saved models continue
to work.

Serialized model artifacts are trusted-input-only.  Loading dill/pickle data can
execute code and must never be used for untrusted files.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import dill

MODEL_ARTIFACT_FORMAT = "JBGAutoClassification.model"
MODEL_ARTIFACT_VERSION = 1
KERAS_MODEL_FILE_EXTENSION = ".keras"
MODEL_ARTIFACT_FIELDS = (
    "config",
    "text_converter",
    "model_components",
    "pipeline",
    "keras_name",
    "n_features_out",
)


def get_keras_model_sidecar_path(filename: str | Path, keras_name: str) -> Path:
    """Return the Keras 3 native sidecar path for a persisted JBG model artifact."""
    return Path(f"{filename}.{keras_name}{KERAS_MODEL_FILE_EXTENSION}")


def get_legacy_keras_model_sidecar_path(filename: str | Path, keras_name: str) -> Path:
    """Return the pre-Keras-3 sidecar path used by historical JBG model artifacts."""
    return Path(f"{filename}.{keras_name}")


def resolve_keras_model_sidecar_path(filename: str | Path, keras_name: str) -> Path:
    """Prefer the native `.keras` sidecar while retaining legacy path discovery."""
    preferred = get_keras_model_sidecar_path(filename, keras_name)
    if preferred.exists():
        return preferred

    legacy = get_legacy_keras_model_sidecar_path(filename, keras_name)
    if legacy.exists():
        return legacy

    return preferred


def detach_keras_model_from_pipeline(pipeline, keras_name: str):
    """Return a persistable pipeline that keeps fitted SciKeras metadata but not model_."""
    if pipeline is None or not getattr(pipeline, "steps", None):
        raise ValueError("Cannot persist Keras metadata from an empty pipeline.")

    step_name, estimator = pipeline.steps[-1]
    if step_name != keras_name:
        raise ValueError(
            f"Expected final pipeline step {keras_name!r}, found {step_name!r}."
        )

    persisted_pipeline = copy.copy(pipeline)
    persisted_pipeline.steps = list(pipeline.steps)
    estimator_stub = copy.copy(estimator)
    if hasattr(estimator_stub, "model_"):
        estimator_stub.model_ = None
    persisted_pipeline.steps[-1] = (step_name, estimator_stub)
    return persisted_pipeline


def attach_keras_model_to_pipeline(pipeline, keras_name: str, keras_model) -> bool:
    """Attach a loaded Keras model to a persisted fitted SciKeras wrapper, if present."""
    if pipeline is None or not getattr(pipeline, "steps", None):
        return False

    step_name, estimator = pipeline.steps[-1]
    if step_name != keras_name or not hasattr(estimator, "model_"):
        return False

    estimator.model_ = keras_model
    return True


class ModelArtifactError(ValueError):
    """Raised when a serialized model does not satisfy the supported contract."""


def build_model_artifact(
    config,
    text_converter,
    model_components,
    pipeline,
    keras_name,
    n_features_out,
) -> dict[str, Any]:
    """Build the current versioned model artifact envelope."""
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "version": MODEL_ARTIFACT_VERSION,
        "serializer": "dill",
        "payload": {
            "config": config,
            "text_converter": text_converter,
            "model_components": model_components,
            "pipeline": pipeline,
            "keras_name": keras_name,
            "n_features_out": n_features_out,
        },
    }


def unpack_model_artifact(serialized: Any) -> tuple:
    """Normalize a current or legacy serialized artifact to the six-field contract."""
    if isinstance(serialized, Mapping) and serialized.get("format") == MODEL_ARTIFACT_FORMAT:
        version = serialized.get("version")
        if version != MODEL_ARTIFACT_VERSION:
            raise ModelArtifactError(
                f"Unsupported model artifact version {version!r}; "
                f"supported version is {MODEL_ARTIFACT_VERSION}."
            )

        payload = serialized.get("payload")
        if not isinstance(payload, Mapping):
            raise ModelArtifactError("Model artifact payload is missing or is not a mapping.")

        missing = [field for field in MODEL_ARTIFACT_FIELDS if field not in payload]
        if missing:
            raise ModelArtifactError(
                "Model artifact payload is missing required field(s): " + ", ".join(missing)
            )

        model_components = payload["model_components"]
        if not isinstance(model_components, (list, tuple)) or len(model_components) != 5:
            raise ModelArtifactError(
                "Model artifact field 'model_components' must contain exactly five entries: "
                "oversampler, undersampler, preprocess, reduction, algorithm."
            )

        return tuple(payload[field] for field in MODEL_ARTIFACT_FIELDS)

    # Backwards compatibility with the unversioned format used before patch 055.
    if isinstance(serialized, (list, tuple)) and len(serialized) == len(MODEL_ARTIFACT_FIELDS):
        model_components = serialized[2]
        if not isinstance(model_components, (list, tuple)) or len(model_components) != 5:
            raise ModelArtifactError(
                "Legacy model artifact has an invalid model-component tuple; expected five entries."
            )
        return tuple(serialized)

    raise ModelArtifactError(
        "Unrecognized model artifact format. Expected a versioned JBG model artifact "
        "or a legacy six-item model payload."
    )


def load_model_artifact(filename: str | Path) -> tuple:
    """Load and validate a trusted model artifact from disk."""
    with open(filename, "rb") as infile:
        return unpack_model_artifact(dill.load(infile))


def load_model_config(filename: str | Path):
    """Load persisted configuration while preserving legacy config-only compatibility."""
    with open(filename, "rb") as infile:
        serialized = dill.load(infile)

    if isinstance(serialized, Mapping) and serialized.get("format") == MODEL_ARTIFACT_FORMAT:
        return unpack_model_artifact(serialized)[0]

    # Older config-loading code only required item zero and some locally generated
    # config fixtures did not contain a complete six-field model payload. Keep that
    # narrow compatibility here without weakening full model loading validation.
    if isinstance(serialized, (list, tuple)) and serialized:
        return serialized[0]

    raise ModelArtifactError(
        "Unrecognized model artifact format while loading persisted configuration."
    )


def save_model_artifact(
    filename: str | Path,
    config,
    text_converter,
    model_components,
    pipeline,
    keras_name,
    n_features_out,
) -> None:
    """Atomically write the current model artifact format with dill."""
    filename = Path(filename)
    temporary = filename.with_name(filename.name + ".tmp")
    artifact = build_model_artifact(
        config=config,
        text_converter=text_converter,
        model_components=model_components,
        pipeline=pipeline,
        keras_name=keras_name,
        n_features_out=n_features_out,
    )

    try:
        with open(temporary, "wb") as outfile:
            dill.dump(artifact, outfile)
        os.replace(temporary, filename)
    finally:
        if temporary.exists():
            temporary.unlink()
