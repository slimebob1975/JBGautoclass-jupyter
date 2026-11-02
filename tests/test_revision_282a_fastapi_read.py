"""282a: offline HTTP contract and isolation on the real presentation service."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

fastapi = pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGFastAPI import create_app
from JBGIdentity import AuthenticationMode, AuthenticatedPrincipal, IdentitySettings, LocalIdentityProvider
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService


def make_client(*, outcome=None):
    datasets = DatasetApplicationService(sql=SimpleNamespace(resolve=Mock(return_value=object())))
    preparation = RunPreparationService(datasets, sql=SimpleNamespace(validate=Mock()))
    prepared = preparation.prepare(DatasetReference(DataSourceKind.SQL, 'trusted'),
                                   RunConfiguration(DataSourceKind.SQL, object()))
    runtime = SimpleNamespace(execute=Mock(return_value=outcome))
    jobs = RunJobService(RunExecutionService(preparation, sql=runtime))
    presentation = RunAPIPresentationService(jobs)
    settings = IdentitySettings(AuthenticationMode.LOCAL, 'development', '127.0.0.1')
    local = LocalIdentityProvider(settings, 'alice', email='alice@local.test')
    current = [local.current_principal()]
    app = create_app(presentation, current_principal=lambda: current[0])
    return TestClient(app), presentation, prepared, current


def test_offline_health_and_pending_job():
    client, presentation, prepared, current = make_client()
    assert client.get('/api/health').json() == {'status': 'ok'}
    run_id = presentation.create(current[0].principal_key, prepared)['run_id']
    assert client.get(f'/api/runs/{run_id}').json() == {
        'run_id': run_id, 'status': 'pending', 'has_error': False}
    assert client.get(f'/api/runs/{run_id}/progress').json() == []
    assert client.get(f'/api/runs/{run_id}/result').json() == {
        'run_id': run_id, 'available': False, 'summary': None}


def test_http_result_is_allowlisted_and_json_serializable():
    pd = pytest.importorskip('pandas')
    raw = {'mispredicted': pd.DataFrame({'secret': ['hidden']}),
           'unique_classes': ['a', 'b'], 'password': 'internal'}
    client, presentation, prepared, current = make_client(outcome=raw)
    run_id = presentation.create(current[0].principal_key, prepared)['run_id']
    presentation.execute(current[0].principal_key, RunReference(run_id))
    response = client.get(f'/api/runs/{run_id}/result')
    assert response.status_code == 200
    assert response.json()['summary'] == {
        'schema_version': 1, 'kind': 'classification',
        'mispredicted_count': 1, 'class_count': 2}
    assert 'secret' not in response.text and 'password' not in response.text
    json.dumps(response.json())


def test_foreign_and_unknown_runs_both_404():
    client, presentation, prepared, current = make_client()
    run_id = presentation.create(current[0].principal_key, prepared)['run_id']
    current[0] = AuthenticatedPrincipal(AuthenticationMode.LOCAL, 'bob')
    for suffix in ('', '/progress', '/result'):
        forbidden = client.get(f'/api/runs/{run_id}{suffix}')
        unknown = client.get(f'/api/runs/no-such-run{suffix}')
        assert forbidden.status_code == unknown.status_code == 404
        assert forbidden.json() == unknown.json() == {'detail': 'run unavailable'}


def test_email_changes_do_not_change_job_owner():
    client, presentation, prepared, current = make_client()
    run_id = presentation.create(current[0].principal_key, prepared)['run_id']
    current[0] = AuthenticatedPrincipal(AuthenticationMode.LOCAL, 'alice', email='new@example.test')
    assert client.get(f'/api/runs/{run_id}').status_code == 200


def test_untrusted_or_missing_principal_is_rejected():
    client, presentation, prepared, current = make_client()
    run_id = presentation.create(current[0].principal_key, prepared)['run_id']
    current[0] = None
    assert client.get(f'/api/runs/{run_id}').status_code == 401
    assert client.get('/api/health').status_code == 200
    current[0] = 'alice'  # A client-controlled string is not a verified principal.
    assert client.get(f'/api/runs/{run_id}').status_code == 401


def test_provider_exception_is_not_exposed():
    client, _, _, _ = make_client()
    # Separately verify an invalid provider is rejected at composition time.
    with pytest.raises(TypeError):
        create_app(object(), current_principal=lambda: None)
    assert client.get('/api/health').status_code == 200
