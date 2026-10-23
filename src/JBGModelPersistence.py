"""Versioned serialization contract for JBG model artifacts.

Current model artifacts prefer the Python standard-library pickle serializer. Dill is
optional and is used only when a current artifact contains state that standard pickle
cannot serialize or deserialize. Pre-revision-176 artifact formats are unsupported.

Serialized model artifacts are trusted-input-only.  Loading pickle/dill data can
execute code and must never be used for untrusted files.
"""

from __future__ import annotations

import copy
import importlib
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from JBGPaths import relocate_saved_config_paths

MODEL_ARTIFACT_FORMAT = "JBGAutoClassification.model"
MODEL_ARTIFACT_VERSION = 2
MODEL_ARTIFACT_DEFAULT_SERIALIZER = "pickle"
MODEL_ARTIFACT_SUPPORTED_SERIALIZERS = frozenset(("pickle", "dill"))
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


def resolve_keras_model_sidecar_path(filename: str | Path, keras_name: str) -> Path:
    """Return the native Keras sidecar path used by current model artifacts."""
    return get_keras_model_sidecar_path(filename, keras_name)


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
    *,
    serializer: str = MODEL_ARTIFACT_DEFAULT_SERIALIZER,
    compatibility: Mapping | None = None,
) -> dict[str, Any]:
    """Build the current versioned model artifact envelope."""
    if serializer not in MODEL_ARTIFACT_SUPPORTED_SERIALIZERS:
        raise ModelArtifactError(f"Unsupported model artifact serializer {serializer!r}.")
    if not isinstance(compatibility, Mapping):
        raise ModelArtifactError("Current model artifacts require compatibility metadata.")
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "version": MODEL_ARTIFACT_VERSION,
        "serializer": serializer,
        "compatibility": dict(compatibility),
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
    """Validate a current versioned artifact and return its six-field payload."""
    if not isinstance(serialized, Mapping) or serialized.get("format") != MODEL_ARTIFACT_FORMAT:
        raise ModelArtifactError(
            "Unsupported pre-revision-176 model artifact. Retrain and save a new model."
        )

    version = serialized.get("version")
    if version != MODEL_ARTIFACT_VERSION:
        raise ModelArtifactError(
            f"Unsupported model artifact version {version!r}; "
            f"supported version is {MODEL_ARTIFACT_VERSION}. Retrain and save a new model."
        )

    serializer = serialized.get("serializer")
    if serializer not in MODEL_ARTIFACT_SUPPORTED_SERIALIZERS:
        raise ModelArtifactError(
            f"Unsupported model artifact serializer metadata {serializer!r}."
        )

    compatibility = serialized.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ModelArtifactError("Current model artifact is missing compatibility metadata.")

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

    values = tuple(payload[field] for field in MODEL_ARTIFACT_FIELDS)
    relocate_saved_config_paths(values[0])
    return values



def _import_optional_dill():
    """Return dill when installed, otherwise None without making it a hard dependency."""
    try:
        return importlib.import_module("dill")
    except ModuleNotFoundError as error:
        if error.name == "dill":
            return None
        raise


def _load_serialized(filename: str | Path):
    """Load trusted current-artifact data, preferring stdlib pickle before dill."""
    with open(filename, "rb") as infile:
        try:
            return pickle.load(infile)
        except Exception as pickle_error:
            infile.seek(0)
            dill_module = _import_optional_dill()
            if dill_module is None:
                raise ModelArtifactError(
                    "Could not load the model artifact with standard pickle. If the current "
                    "artifact requires dill, install the optional 'dill' package and retry. "
                    f"Original pickle error: {type(pickle_error).__name__}: {pickle_error}"
                ) from pickle_error
            try:
                return dill_module.load(infile)
            except Exception as dill_error:
                raise ModelArtifactError(
                    "Could not load the model artifact with standard pickle or dill. "
                    f"Pickle error: {type(pickle_error).__name__}: {pickle_error}; "
                    f"dill error: {type(dill_error).__name__}: {dill_error}"
                ) from dill_error


def unpack_model_compatibility(serialized: Any):
    """Return required compatibility metadata from a current model artifact."""
    unpack_model_artifact(serialized)
    compatibility = serialized.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ModelArtifactError("Current model artifact is missing compatibility metadata.")
    return dict(compatibility)


def load_model_artifact(filename: str | Path) -> tuple:
    """Load and validate a trusted current model artifact from disk."""
    return unpack_model_artifact(_load_serialized(filename))


def load_model_compatibility(filename: str | Path):
    """Load required compatibility metadata from a trusted current artifact."""
    return unpack_model_compatibility(_load_serialized(filename))


def load_model_artifact_with_compatibility(filename: str | Path) -> tuple[tuple, Mapping]:
    """Load the payload and required compatibility metadata in one file read."""
    serialized = _load_serialized(filename)
    return unpack_model_artifact(serialized), unpack_model_compatibility(serialized)


def load_model_config(filename: str | Path):
    """Load persisted configuration from a current model artifact."""
    return unpack_model_artifact(_load_serialized(filename))[0]



def save_model_artifact(
    filename: str | Path,
    config,
    text_converter,
    model_components,
    pipeline,
    keras_name,
    n_features_out,
    *,
    compatibility: Mapping | None = None,
) -> None:
    """Atomically persist a model, preferring stdlib pickle with optional dill fallback."""
    filename = Path(filename)
    temporary = filename.with_name(filename.name + ".tmp")
    artifact = build_model_artifact(
        config=config,
        text_converter=text_converter,
        model_components=model_components,
        pipeline=pipeline,
        keras_name=keras_name,
        n_features_out=n_features_out,
        serializer="pickle",
        compatibility=compatibility,
    )

    try:
        try:
            with open(temporary, "wb") as outfile:
                pickle.dump(artifact, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        except (OSError, MemoryError):
            raise
        except Exception as pickle_error:
            dill_module = _import_optional_dill()
            if dill_module is None:
                raise ModelArtifactError(
                    "This model contains state that standard pickle cannot serialize. "
                    "Install the optional 'dill' package to allow the serializer fallback, "
                    "or use a pickle-compatible model. "
                    f"Original pickle error: {type(pickle_error).__name__}: {pickle_error}"
                ) from pickle_error
            artifact = build_model_artifact(
                config=config,
                text_converter=text_converter,
                model_components=model_components,
                pipeline=pipeline,
                keras_name=keras_name,
                n_features_out=n_features_out,
                serializer="dill",
                compatibility=compatibility,
            )
            with open(temporary, "wb") as outfile:
                dill_module.dump(artifact, outfile)
        os.replace(temporary, filename)
    finally:
        if temporary.exists():
            temporary.unlink()
