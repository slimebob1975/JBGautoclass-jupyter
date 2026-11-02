"""Revision 280g: guard the saved artifact payload schema."""
from types import SimpleNamespace

from JBGModelArtifactPayload import build_model_artifact_payload


def _model():
    tokens = [object() for _ in range(8)]
    return SimpleNamespace(
        text_converter=tokens[0],
        oversampler=tokens[1],
        undersampler=tokens[2],
        preprocess=tokens[3],
        reduction=tokens[4],
        algorithm=tokens[5],
        pipeline=tokens[6],
        n_features_out=17,
    )


def test_non_keras_payload_preserves_legacy_shape_and_references():
    model = _model()
    config = object()
    payload = build_model_artifact_payload(config, model, model.pipeline, None)
    assert len(payload) == 6
    assert payload[0] is config
    assert payload[1] is model.text_converter
    assert payload[2] == (
        model.oversampler, model.undersampler, model.preprocess,
        model.reduction, model.algorithm,
    )
    assert all(a is b for a, b in zip(payload[2], (
        model.oversampler, model.undersampler, model.preprocess,
        model.reduction, model.algorithm,
    )))
    assert payload[3] is model.pipeline
    assert payload[4] is None
    assert payload[5] == 17


def test_keras_payload_keeps_detached_pipeline_and_sidecar_name():
    model = _model()
    detached = object()
    payload = build_model_artifact_payload(object(), model, detached, 'keras_step')
    assert payload[3] is detached
    assert payload[3] is not model.pipeline
    assert payload[4] == 'keras_step'
    assert payload[5] == 17


def test_payload_construction_does_not_mutate_model():
    model = _model()
    original_pipeline = model.pipeline
    build_model_artifact_payload(object(), model, object(), 'keras_step')
    assert model.pipeline is original_pipeline
    assert model.n_features_out == 17
