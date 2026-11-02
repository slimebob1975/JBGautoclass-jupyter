"""Contract checks for the legacy model artifact reconstruction."""

from dataclasses import dataclass
import pytest

from JBGModelArtifactPayload import (
    build_model_artifact_payload, restore_model_from_artifact_payload,
)


@dataclass
class DummyModel:
    text_converter: object = None
    oversampler: object = None
    undersampler: object = None
    preprocess: object = None
    reduction: object = None
    algorithm: object = None
    pipeline: object = None
    n_features_out: int = None


def test_roundtrip_preserves_field_identity_and_order():
    config = object()
    objects = [object() for _ in range(7)]
    original = DummyModel(*objects[:6], pipeline=objects[6], n_features_out=13)
    payload = build_model_artifact_payload(config, original, original.pipeline, None)
    restored = restore_model_from_artifact_payload(payload, DummyModel)
    assert payload[0] is config
    assert [getattr(restored, field) for field in (
        "text_converter", "oversampler", "undersampler", "preprocess",
        "reduction", "algorithm", "pipeline",
    )] == objects
    assert restored.n_features_out == 13


def test_keras_sidecar_attached_pipeline_is_used():
    original = DummyModel(pipeline=object(), n_features_out=7)
    payload = build_model_artifact_payload(object(), original, original.pipeline, "keras_sidecar")
    replacement_pipeline = object()
    payload[3] = replacement_pipeline
    restored = restore_model_from_artifact_payload(payload, DummyModel)
    assert restored.pipeline is replacement_pipeline
    assert restored.n_features_out == 7


def test_invalid_payload_shape_still_raises_unpack_error():
    with pytest.raises((ValueError, TypeError)):
        restore_model_from_artifact_payload([1, 2, 3], DummyModel)


def test_factory_receives_exact_existing_field_names():
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return kwargs
    payload = [object(), object(), (1, 2, 3, 4, 5), object(), None, 8]
    result = restore_model_from_artifact_payload(payload, factory)
    assert result == captured
    assert list(captured) == [
        "text_converter", "oversampler", "undersampler", "preprocess",
        "reduction", "algorithm", "pipeline", "n_features_out",
    ]
