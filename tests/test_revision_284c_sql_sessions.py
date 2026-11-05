"""284c: SQL runs through the user's own login (design gap 8, gap 1 for SQL).

A real Config and a real SQL DataLayer are used. Only the ODBC driver list,
the connection test and the final classifier run are replaced; no SQL Server
or network is needed.
"""
from contextlib import contextmanager
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
from JBGPrincipalRunAccess import PrincipalRunAccessService
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunComposition import compose_run, compose_sql_run
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration
from JBGRunProgress import RunProgressService
from JBGSQLSessions import SQLCredentials, SQLSessionRegistry
from Storage.ArtifactContract import StorageScope

DRIVER = 'SQL Server'  # a driver name Config knows how to format


def _local(subject='developer'):
    return AuthenticatedPrincipal(AuthenticationMode.LOCAL, subject)


def _entra(subject='developer'):
    return AuthenticatedPrincipal(AuthenticationMode.ENTRA, subject, tenant_id='tenant')


def _config():
    from Config import Config
    config = Config()
    config.connection.odbc_driver = DRIVER
    config.connection.host = 'sql.example.invalid'
    config.connection.data_catalog = 'ExampleDb'
    config.connection.data_table = 'dbo.Cases'
    config.connection.class_catalog = 'ExampleDb'
    config.connection.class_table = 'dbo.Results'
    return config


@contextmanager
def _sql_server(accept=True):
    """Pretend the driver exists; record which credentials each connection used."""
    from SQLDataLayer import DataLayer
    attempts = []

    def can_connect(layer, verbose=False):
        connection = layer.config.connection
        attempts.append((connection.trusted_connection, connection.sql_username,
                         connection.sql_password))
        return accept

    with patch('pyodbc.drivers', return_value=[DRIVER]), \
            patch.object(DataLayer, 'can_connect', autospec=True, side_effect=can_connect):
        yield attempts


def _access(sessions, default_execution):
    scope = StorageScope(str(uuid4()), str(uuid4()))
    return PrincipalRunAccessService(
        RunAPIPresentationService(RunJobService(default_execution)),
        scope_for=lambda p: scope, authorize_dataset=sessions.authorize)


def _prepare(composed):
    return composed.preparation.prepare(
        DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()),
        RunConfiguration(DataSourceKind.SQL, composed.configuration))


# --- Credentials ----------------------------------------------------------------

def test_credentials_validate_and_hide_the_password():
    login = SQLCredentials.sql_login('analyst', 'S3cret!')
    assert 'S3cret!' not in repr(login)
    assert SQLCredentials.windows_login().windows is True
    with pytest.raises(ValueError):
        SQLCredentials.sql_login('analyst', '')
    with pytest.raises(ValueError):
        SQLCredentials.sql_login('  ', 'pw')
    with pytest.raises(ValueError):
        SQLCredentials(windows=True, username='someone')


# --- Sign-in --------------------------------------------------------------------

def test_sql_login_sign_in_connects_with_a_copy_of_the_config():
    sessions, config = SQLSessionRegistry(), _config()
    with _sql_server() as attempts:
        layer = sessions.sign_in(_local(), SQLCredentials.sql_login('analyst', 'pw'),
                                 config, Mock())
    assert attempts == [(False, 'analyst', 'pw')]
    assert sessions.is_signed_in(_local())
    assert layer.config is not config
    assert config.connection.sql_username == '' and config.connection.sql_password == ''


def test_windows_sign_in_in_local_mode_uses_trusted_connection():
    sessions = SQLSessionRegistry()
    with _sql_server() as attempts:
        sessions.sign_in(_local(), SQLCredentials.windows_login(), _config(), Mock())
    assert attempts == [(True, '', '')]
    assert sessions.uses_windows_authentication(_local())


def test_windows_sign_in_is_refused_outside_local_mode_without_connecting():
    sessions = SQLSessionRegistry()
    with _sql_server() as attempts:
        with pytest.raises(PermissionError, match='only available in local mode'):
            sessions.sign_in(_entra(), SQLCredentials.windows_login(), _config(), Mock())
    assert attempts == []
    assert not sessions.is_signed_in(_entra())


def test_rejected_login_is_not_remembered():
    sessions = SQLSessionRegistry()
    with _sql_server(accept=False):
        with pytest.raises(ValueError, match='Connection to server failed'):
            sessions.sign_in(_local(), SQLCredentials.sql_login('analyst', 'wrong'),
                             _config(), Mock())
    assert not sessions.is_signed_in(_local())


def test_composition_requires_sql_sign_in():
    with _sql_server() as attempts:
        with pytest.raises(PermissionError, match='SQL sign-in required'):
            compose_sql_run(SQLSessionRegistry(), _local(), _config(), Mock())
    assert attempts == []


# --- Runs -------------------------------------------------------------------------

def test_sql_run_executes_with_the_users_own_session_layer():
    sessions, config, me = SQLSessionRegistry(), _config(), _local()
    progress = RunProgressService()
    with _sql_server() as attempts:
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), config, Mock())
        composed = compose_run(DataSourceKind.SQL, sessions=sessions, principal=me,
                               config=config, logger=Mock(), progress=progress)
        default = compose_sql_run(sessions, me, config, Mock())
    assert attempts[-2:] == [(False, 'analyst', 'pw')] * 2
    assert composed.sql_dataset_key() == '[ExampleDb].[dbo].[Cases]'
    assert config.connection.sql_password == ''

    prepared = _prepare(composed)
    assert prepared.resolved_dataset.config is composed.configuration
    access = _access(sessions, default.execution)
    run = RunReference(access.create(me, prepared, execution=composed.execution)['run_id'])
    with patch('AutomaticClassifier.AutomaticClassifier') as classifier:
        classifier.return_value.run.return_value = {'model': 'trained'}
        snapshot = access.execute(me, run)
    assert snapshot['status'] == 'succeeded'
    kwargs = classifier.call_args.kwargs
    assert kwargs['datalayer'] is prepared.resolved_dataset
    assert kwargs['config'] is composed.configuration


def test_another_user_cannot_run_on_my_sql_session():
    sessions, me, other = SQLSessionRegistry(), _local('me'), _local('other')
    with _sql_server():
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), _config(), Mock())
        sessions.sign_in(other, SQLCredentials.sql_login('colleague', 'pw2'), _config(), Mock())
        mine = compose_sql_run(sessions, me, _config(), Mock())
    prepared = _prepare(mine)
    access = _access(sessions, mine.execution)
    with pytest.raises(PermissionError, match='SQL dataset unavailable'):
        access.create(other, prepared, execution=mine.execution)


def test_sign_out_blocks_a_run_that_has_not_started():
    sessions, me = SQLSessionRegistry(), _local()
    with _sql_server():
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), _config(), Mock())
        composed = compose_sql_run(sessions, me, _config(), Mock())
    access = _access(sessions, composed.execution)
    run = RunReference(access.create(me, _prepare(composed),
                                     execution=composed.execution)['run_id'])
    sessions.sign_out(me)
    with patch('AutomaticClassifier.AutomaticClassifier') as classifier:
        with pytest.raises(PermissionError, match='SQL dataset unavailable'):
            access.execute(me, run)
    classifier.assert_not_called()
    assert not sessions.is_signed_in(me)


def test_a_layer_opened_outside_the_registry_is_not_authorized():
    from SQLDataLayer import DataLayer
    sessions, me = SQLSessionRegistry(), _local()
    with _sql_server():
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), _config(), Mock())
        composed = compose_sql_run(sessions, me, _config(), Mock())
        foreign_layer = DataLayer(config=composed.configuration, logger=Mock())
    prepared = _prepare(composed)
    forged = type(prepared)(dataset=prepared.dataset, resolved_dataset=foreign_layer,
                            configuration=prepared.configuration)
    with pytest.raises(PermissionError, match='SQL dataset unavailable'):
        sessions.authorize(me, forged)


def test_authorize_leaves_csv_runs_to_their_own_checks():
    prepared = Mock()
    prepared.dataset.source = DataSourceKind.CSV
    assert SQLSessionRegistry().authorize(_local(), prepared) is None


def test_sql_credentials_are_not_reused_for_another_target():
    sessions, config, me = SQLSessionRegistry(), _config(), _local()
    other = _config()
    other.connection.host = 'different.example.invalid'
    with _sql_server() as attempts:
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), config, Mock())
        with pytest.raises(PermissionError, match='SQL sign-in required'):
            sessions.data_layer_for(me, other, Mock())
    assert len(attempts) == 1
    assert sessions.credentials_for_owner(me.principal_key, other.connection) is None
    assert sessions.credentials_for_owner(me.principal_key, config.connection) is not None


def test_one_sign_in_serves_all_catalogs_on_the_same_server():
    # 288b (decision 2): bound to host and driver; SQL Server decides catalogs.
    sessions, config, me = SQLSessionRegistry(), _config(), _local()
    elsewhere = _config()
    elsewhere.connection.data_catalog = 'OtherDb'
    elsewhere.connection.class_catalog = 'ResultsDb'
    with _sql_server():
        sessions.sign_in(me, SQLCredentials.sql_login('analyst', 'pw'), config, Mock())
        sessions.data_layer_for(me, elsewhere, Mock())
    assert sessions.credentials_for_owner(me.principal_key, elsewhere.connection) is not None
