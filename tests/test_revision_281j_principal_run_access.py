"""281j: verified identity and trusted scope at the job boundary."""
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGIdentity import AuthenticationMode, AuthenticatedPrincipal
from JBGPrincipalRunAccess import PrincipalRunAccessService
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from Storage.ArtifactContract import StorageScope


def principal(subject='developer', email=None, provider=AuthenticationMode.LOCAL):
    return AuthenticatedPrincipal(provider, subject,
                                  tenant_id='tenant' if provider is AuthenticationMode.ENTRA else None,
                                  email=email)


def setup(source=DataSourceKind.CSV):
    scope = StorageScope(str(uuid4()), str(uuid4()))
    other_scope = StorageScope(str(uuid4()), str(uuid4()))
    resolved = SimpleNamespace(scope=scope)
    resolver = SimpleNamespace(resolve=Mock(return_value=resolved))
    datasets = DatasetApplicationService(**{source.value: resolver})
    preparation = RunPreparationService(
        datasets, **{source.value: SimpleNamespace(validate=Mock())})
    prepared = preparation.prepare(DatasetReference(source, 'dataset'),
                                   RunConfiguration(source, object()))
    runtime = SimpleNamespace(execute=Mock(return_value={'model': object()}))
    execution = RunExecutionService(preparation, **{source.value: runtime})
    presentation = RunAPIPresentationService(RunJobService(execution))
    scopes = {principal().principal_key: scope,
              principal('someone-else').principal_key: other_scope,
              principal('developer', provider=AuthenticationMode.ENTRA).principal_key: other_scope}
    authorizer = Mock()
    access = PrincipalRunAccessService(
        presentation, scope_for=lambda p: scopes[p.principal_key],
        authorize_dataset=authorizer)
    return access, prepared, runtime, authorizer, scopes


def test_local_offline_identity_creates_and_executes_without_entra():
    access, prepared, runtime, _, _ = setup()
    me = principal()
    run = RunReference(access.create(me, prepared)['run_id'])
    assert access.execute(me, run)['status'] == 'succeeded'
    assert access.result(me, run)['available'] is True
    runtime.execute.assert_called_once()


def test_changed_email_preserves_owner_identity():
    access, prepared, _, _, _ = setup()
    run = RunReference(access.create(principal(email='old@example.com'), prepared)['run_id'])
    assert access.snapshot(principal(email='new@example.com'), run)['status'] == 'pending'


def test_different_local_and_entra_principals_are_isolated():
    access, prepared, runtime, _, _ = setup()
    run = RunReference(access.create(principal(), prepared)['run_id'])
    for intruder in (principal('someone-else'),
                     principal('developer', provider=AuthenticationMode.ENTRA)):
        for method in (access.snapshot, access.execute, access.progress, access.result):
            with pytest.raises(PermissionError, match='unavailable'):
                method(intruder, run)
    runtime.execute.assert_not_called()


def test_csv_scope_mismatch_rejected_before_job_creation():
    access, prepared, _, authorizer, scopes = setup()
    scopes[principal().principal_key] = StorageScope(str(uuid4()), str(uuid4()))
    with pytest.raises(PermissionError, match='dataset unavailable'):
        access.create(principal(), prepared)
    authorizer.assert_not_called()


def test_scope_revocation_rechecked_before_execution():
    access, prepared, runtime, _, scopes = setup()
    run = RunReference(access.create(principal(), prepared)['run_id'])
    scopes[principal().principal_key] = StorageScope(str(uuid4()), str(uuid4()))
    with pytest.raises(PermissionError, match='dataset unavailable'):
        access.execute(principal(), run)
    assert access.snapshot(principal(), run)['status'] == 'pending'
    runtime.execute.assert_not_called()


def test_sql_requires_explicit_authorization_not_just_scope():
    access, prepared, runtime, authorizer, _ = setup(DataSourceKind.SQL)
    authorizer.side_effect = PermissionError('SQL dataset unavailable')
    with pytest.raises(PermissionError, match='SQL dataset unavailable'):
        access.create(principal(), prepared)
    runtime.execute.assert_not_called()


def test_unverified_identity_objects_and_missing_scope_resolver_are_rejected():
    access, prepared, _, _, _ = setup()
    with pytest.raises(PermissionError, match='authenticated principal'):
        access.create('developer', prepared)
    with pytest.raises(TypeError, match='trusted scope'):
        PrincipalRunAccessService(access._presentation, scope_for=None)
