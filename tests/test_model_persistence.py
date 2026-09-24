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
    build_model_artifact,
    load_model_artifact,
    load_model_config,
    save_model_artifact,
)


def _payload(pipeline=None):
    return (
        {"safe_config": True},
        None,
        ("NOG", "NUG", "STA", "NOR", "LRN"),
        pipeline,
        None,
        2,
    )


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
