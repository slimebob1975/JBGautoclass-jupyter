"""281k: no custom Entra policy required for CSV; SQL remains fail-closed."""
import pytest

from JBGApplicationContracts import DataSourceKind, RunReference
from JBGPrincipalRunAccess import PrincipalRunAccessService
from test_revision_281j_principal_run_access import principal, setup


def test_csv_without_custom_policy_still_enforces_owner_and_scope():
    access, prepared, runtime, _, scopes = setup()
    access = PrincipalRunAccessService(access._presentation, scope_for=lambda p: scopes[p.principal_key])
    me = principal(email='before@example.com')
    run = RunReference(access.create(me, prepared)['run_id'])
    with pytest.raises(PermissionError, match='unavailable'):
        access.snapshot(principal('someone-else'), run)
    assert access.execute(principal(email='after@example.com'), run)['status'] == 'succeeded'
    runtime.execute.assert_called_once()


def test_csv_without_policy_rechecks_scope_before_execution():
    from uuid import uuid4
    from Storage.ArtifactContract import StorageScope
    access, prepared, runtime, _, scopes = setup()
    access = PrincipalRunAccessService(access._presentation, scope_for=lambda p: scopes[p.principal_key])
    me = principal()
    run = RunReference(access.create(me, prepared)['run_id'])
    scopes[me.principal_key] = StorageScope(str(uuid4()), str(uuid4()))
    with pytest.raises(PermissionError, match='dataset unavailable'):
        access.execute(me, run)
    runtime.execute.assert_not_called()


def test_sql_without_explicit_host_authorizer_is_denied():
    access, prepared, runtime, _, scopes = setup(DataSourceKind.SQL)
    access = PrincipalRunAccessService(access._presentation, scope_for=lambda p: scopes[p.principal_key])
    with pytest.raises(PermissionError, match='SQL dataset authorization required'):
        access.create(principal(), prepared)
    runtime.execute.assert_not_called()


def test_sql_with_explicit_authorizer_still_works():
    access, prepared, runtime, authorizer, _ = setup(DataSourceKind.SQL)
    run = RunReference(access.create(principal(), prepared)['run_id'])
    assert access.execute(principal(), run)['status'] == 'succeeded'
    assert authorizer.call_count == 2
    runtime.execute.assert_called_once()


def test_invalid_optional_authorizer_rejected():
    access, _, _, _, scopes = setup()
    with pytest.raises(TypeError, match='dataset authorizer must be callable'):
        PrincipalRunAccessService(access._presentation, scope_for=lambda p: scopes[p.principal_key],
                                  authorize_dataset='unsupported')
