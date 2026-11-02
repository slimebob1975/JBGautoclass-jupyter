"""Revision 280i: preserve the ordered artifact loading contract."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from JBGModelArtifactService import load_checked_model_artifact


class ArtifactRejected(Exception):
    pass


def run_case(*, source_issues=(), semantic_issues=(), keras_name=None,
             attach_result=True, warnings=(), raise_at=None):
    calls = []
    saved_config = object()
    converter = object()
    pipeline = object()
    artifact = [saved_config, converter, (1, 2, 3, 4, 5), pipeline, keras_name, 17]
    logger = SimpleNamespace(print_warning=lambda text: calls.append(('warning', text)))

    def load_artifact(path):
        calls.append('load')
        return artifact, object()

    def review_source(current, saved, meta):
        calls.append('source')
        assert saved is saved_config
        return SimpleNamespace(issues=source_issues)

    def review_semantics(saved, text, meta):
        calls.append('semantics')
        assert text is converter
        return SimpleNamespace(issues=semantic_issues, warnings=warnings)

    def resolve(path, name):
        calls.append('resolve')
        return Path('keras-sidecar.keras')

    def load_keras(path):
        calls.append('keras')
        assert path == 'keras-sidecar.keras'
        return object()

    def attach(target, name, keras_model):
        calls.append('attach')
        assert target is pipeline
        return attach_result

    def restore(payload, factory):
        calls.append('restore')
        assert payload is artifact
        return factory()

    def factory():
        return object()

    def execute():
        return load_checked_model_artifact(
            Path('model'), object(), logger, factory,
            load_artifact=load_artifact, review_source=review_source,
            review_semantics=review_semantics, resolve_sidecar=resolve,
            load_keras=load_keras, attach_keras=attach,
            restore_model=restore, initialization_error=ArtifactRejected,
        )
    return execute, calls


def test_valid_non_keras_order_and_feature_count():
    execute, calls = run_case()
    model, n_features = execute()
    assert model is not None and n_features == 17
    assert calls == ['load', 'source', 'semantics', 'restore']


def test_source_rejection_prevents_semantics_and_reconstruction():
    execute, calls = run_case(source_issues=('source mismatch',), keras_name='sidecar')
    with pytest.raises(ArtifactRejected, match='source mismatch'):
        execute()
    assert calls == ['load', 'source']


def test_semantic_rejection_prevents_sidecar_loading():
    execute, calls = run_case(semantic_issues=('invalid metadata',), keras_name='sidecar')
    with pytest.raises(ArtifactRejected, match='Saved model compatibility metadata'):
        execute()
    assert calls == ['load', 'source', 'semantics']


def test_keras_sidecar_loaded_after_validation_before_reconstruction():
    execute, calls = run_case(keras_name='sidecar', warnings=('legacy warning',))
    _, count = execute()
    assert count == 17
    assert calls == ['load', 'source', 'semantics', ('warning', 'legacy warning'),
                     'resolve', 'keras', 'attach', 'restore']


def test_incomplete_keras_wrapper_rejected_before_reconstruction():
    execute, calls = run_case(keras_name='sidecar', attach_result=False)
    with pytest.raises(ArtifactRejected, match='retrain and save'):
        execute()
    assert calls == ['load', 'source', 'semantics', 'resolve', 'keras', 'attach']
