import os
import subprocess
import sys
from pathlib import Path

import dill
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from JBGModelPersistence import (
    MODEL_ARTIFACT_FORMAT,
    MODEL_ARTIFACT_VERSION,
    ModelArtifactError,
    attach_keras_model_to_pipeline,
    detach_keras_model_from_pipeline,
    get_keras_model_sidecar_path,
    get_legacy_keras_model_sidecar_path,
    resolve_keras_model_sidecar_path,
    build_model_artifact,
    load_model_artifact,
    load_model_config,
    save_model_artifact,
)


class _FakeFittedKerasWrapper:
    def __init__(self, model):
        self.model_ = model
        self.classes_ = np.array(["B", "M"])
        self.n_features_in_ = 30
        self.target_encoder_ = {"fitted": True}
        self.feature_encoder_ = {"fitted": True}


def _payload(pipeline=None):
    return (
        {"safe_config": True},
        None,
        ("NOG", "NUG", "STA", "NOR", "LRN"),
        pipeline,
        None,
        2,
    )


def test_keras_sidecar_path_uses_native_keras_extension(tmp_path):
    artifact = tmp_path / "model.sav"

    assert get_keras_model_sidecar_path(artifact, "KERA") == tmp_path / "model.sav.KERA.keras"
    assert get_legacy_keras_model_sidecar_path(artifact, "KERA") == tmp_path / "model.sav.KERA"


def test_keras_sidecar_resolver_prefers_native_and_falls_back_to_legacy(tmp_path):
    artifact = tmp_path / "model.sav"
    preferred = get_keras_model_sidecar_path(artifact, "KERA")
    legacy = get_legacy_keras_model_sidecar_path(artifact, "KERA")

    assert resolve_keras_model_sidecar_path(artifact, "KERA") == preferred

    legacy.touch()
    assert resolve_keras_model_sidecar_path(artifact, "KERA") == legacy

    preferred.touch()
    assert resolve_keras_model_sidecar_path(artifact, "KERA") == preferred


def test_keras_pipeline_metadata_survives_model_detach_and_reattach():
    original_model = object()
    loaded_model = object()
    wrapper = _FakeFittedKerasWrapper(original_model)
    pipeline = Pipeline([("scale", StandardScaler()), ("KERA", wrapper)])

    persisted = detach_keras_model_from_pipeline(pipeline, "KERA")

    # Saving must not mutate the live fitted estimator.
    assert pipeline.named_steps["KERA"].model_ is original_model

    persisted_wrapper = persisted.named_steps["KERA"]
    assert persisted_wrapper.model_ is None
    assert persisted_wrapper.classes_.tolist() == ["B", "M"]
    assert persisted_wrapper.n_features_in_ == 30
    assert persisted_wrapper.target_encoder_ == {"fitted": True}
    assert persisted_wrapper.feature_encoder_ == {"fitted": True}

    assert attach_keras_model_to_pipeline(persisted, "KERA", loaded_model) is True
    assert persisted.named_steps["KERA"].model_ is loaded_model


def test_keras_pipeline_attach_rejects_legacy_pipeline_without_wrapper_metadata():
    pipeline = Pipeline([("scale", StandardScaler())])

    assert attach_keras_model_to_pipeline(pipeline, "KERA", object()) is False


def test_model_artifact_round_trip_uses_versioned_envelope(tmp_path):
    path = tmp_path / "model.sav"
    payload = _payload()

    save_model_artifact(path, *payload)

    with path.open("rb") as infile:
        raw = dill.load(infile)

    assert raw["format"] == MODEL_ARTIFACT_FORMAT
    assert raw["version"] == MODEL_ARTIFACT_VERSION
    assert raw["serializer"] == "dill"
    assert load_model_artifact(path) == payload
    assert not (tmp_path / "model.sav.tmp").exists()


def test_model_artifact_loader_accepts_legacy_six_item_payload(tmp_path):
    path = tmp_path / "legacy.sav"
    payload = _payload()

    with path.open("wb") as outfile:
        dill.dump(list(payload), outfile)

    assert load_model_artifact(path) == payload


def test_config_loader_keeps_legacy_config_only_fixture_compatibility(tmp_path):
    path = tmp_path / "legacy-config.sav"
    legacy = [{"legacy_config": True}, None, None, None, 4]

    with path.open("wb") as outfile:
        dill.dump(legacy, outfile)

    assert load_model_config(path) == {"legacy_config": True}


def test_model_artifact_rejects_unknown_version(tmp_path):
    path = tmp_path / "future.sav"
    artifact = build_model_artifact(*_payload())
    artifact["version"] = MODEL_ARTIFACT_VERSION + 1

    with path.open("wb") as outfile:
        dill.dump(artifact, outfile)

    with pytest.raises(ModelArtifactError, match="Unsupported model artifact version"):
        load_model_artifact(path)


def test_model_artifact_pipeline_reloads_predicts_and_retrains_in_fresh_process(tmp_path):
    path = tmp_path / "pipeline.sav"
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(random_state=1)),
    ])
    pipeline.fit(X, y)
    save_model_artifact(path, *_payload(pipeline=pipeline))

    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "src" / "JBGclassification"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(module_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child = r'''
import sys
import numpy as np
from JBGModelPersistence import load_model_artifact

_, _, _, pipeline, _, _ = load_model_artifact(sys.argv[1])
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0, 0, 1, 1])
before = pipeline.predict(X).tolist()
pipeline.fit(X, y)
after = pipeline.predict(X).tolist()
print(before)
print(after)
'''

    result = subprocess.run(
        [sys.executable, "-c", child, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["[0, 0, 1, 1]", "[0, 0, 1, 1]"]
