"""Revision 280j: preserve save ordering and payload identity."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from JBGModelArtifactService import save_checked_model_artifact


def scenario(keras=False, fail_at=None):
    calls = []
    lib = object()
    other = object()
    config, compatibility = object(), object()
    saved_model = SimpleNamespace(save=lambda path: calls.append(('sidecar-save', path)))
    last_step = SimpleNamespace(model_=saved_model)
    pipeline = SimpleNamespace(steps=[('preprocess', object()), ('final', last_step)])
    model = SimpleNamespace(algorithm=SimpleNamespace(lib=lib if keras else other), pipeline=pipeline)
    persisted_pipeline = object()
    payload = [object()] * 6

    def detach(source, name):
        calls.append('detach')
        assert source is pipeline and name == 'final'
        return persisted_pipeline

    def build(conf, mdl, pipe, name):
        calls.append('build')
        assert conf is config and mdl is model
        assert pipe is (persisted_pipeline if keras else pipeline)
        assert name == ('final' if keras else None)
        return payload

    def save(path, *items, compatibility=None):
        calls.append('artifact-save')
        assert path == Path('model.pkl')
        assert all(a is b for a, b in zip(items, payload))
        assert compatibility is expected_compatibility
        if fail_at == 'artifact':
            raise OSError('artifact failed')

    expected_compatibility = compatibility

    def sidecar(path, name):
        calls.append('sidecar-path')
        assert name == 'final'
        return Path('model.keras')

    def execute():
        return save_checked_model_artifact(
            Path('model.pkl'), config, model, compatibility,
            keras_library=lib, detach_keras=detach,
            build_payload=build, save_artifact=save, sidecar_path=sidecar,
        )
    return execute, calls


def test_non_keras_saves_payload_only():
    execute, calls = scenario()
    assert execute() is None
    assert calls == ['build', 'artifact-save']


def test_keras_saves_artifact_before_sidecar():
    execute, calls = scenario(keras=True)
    assert execute() is None
    assert calls == ['detach', 'build', 'artifact-save', 'sidecar-path',
                     ('sidecar-save', 'model.keras')]


@pytest.mark.parametrize('keras', [False, True])
def test_artifact_save_failure_propagates_for_handler_to_decide(keras):
    execute, calls = scenario(keras=keras, fail_at='artifact')
    with pytest.raises(OSError, match='artifact failed'):
        execute()
    assert calls[-1] == 'artifact-save'
    assert 'sidecar-path' not in calls
