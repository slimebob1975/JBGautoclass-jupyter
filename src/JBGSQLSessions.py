"""Session-bound SQL access through the user's own login (revision 284c).

Design decision 5 / gap 8: SQL Server enforces each user's rights through that
user's own login. JBG adds no allowlist or role system. Its only check is that
a run's SQL DataLayer was opened by this registry with the requesting
principal's credentials in the current server session.

Credentials live in process memory only. They are never written to a file, the
job table or a log, and a server restart forgets them; affected runs must wait
until the user signs in again.

Windows authentication connects as the identity of the server process, so it is
accepted only for local principals (single-user local mode). Server deployments
use SQL login until Entra pass-through is added (design Phase 4).
"""
from __future__ import annotations

import copy
import threading
import weakref
from dataclasses import dataclass, field

from JBGApplicationContracts import DataSourceKind
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode


@dataclass(frozen=True)
class SQLCredentials:
    """Either Windows authentication or a SQL login; the password is never shown."""

    windows: bool = False
    username: str = ""
    password: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.windows, bool):
            raise TypeError("windows must be a bool")
        if not isinstance(self.username, str) or not isinstance(self.password, str):
            raise TypeError("SQL credentials must be strings")
        if self.windows and (self.username or self.password):
            raise ValueError("Windows authentication takes no username or password")
        if not self.windows and (not self.username.strip() or not self.password):
            raise ValueError("SQL login requires a username and a password")

    @classmethod
    def windows_login(cls) -> "SQLCredentials":
        return cls(windows=True)

    @classmethod
    def sql_login(cls, username: str, password: str) -> "SQLCredentials":
        return cls(windows=False, username=username, password=password)


def open_sql_data_layer(credentials: SQLCredentials, config, logger):
    """Open a DataLayer on a copy of ``config`` carrying ``credentials``.

    DataLayer construction checks the ODBC driver and connects, so a
    successful return is also the sign-in probe. The caller's ``config`` is
    never modified. Authorization is the caller's job: the session registry
    in the web process, or the dispatcher before it starts a worker (285b).
    """
    if not isinstance(credentials, SQLCredentials):
        raise TypeError("SQLCredentials required")
    if getattr(getattr(config, "data_source", None), "kind", None) != "sql":
        raise ValueError("SQL configuration required")
    from SQLDataLayer import DataLayer

    run_config = copy.deepcopy(config)
    connection = run_config.connection
    connection.trusted_connection = credentials.windows
    connection.sql_username = credentials.username
    connection.sql_password = credentials.password
    return DataLayer(config=run_config, logger=logger)


def _connection_target(config_or_connection) -> tuple[str, str]:
    """The SQL Server a login is bound to: host and driver (288b, decision 2).

    285f also bound the catalogs. A login is a server login, and SQL Server
    itself decides which catalogs it may use, so binding catalogs added no
    protection; one sign-in now serves all SQL projects on the same server.
    Credentials are still never handed out for a different server.
    """
    connection = getattr(config_or_connection, "connection", config_or_connection)
    fields = ("host", "odbc_driver")
    if isinstance(connection, dict):
        return tuple(str(connection.get(field, "")).strip().casefold() for field in fields)
    return tuple(str(getattr(connection, field, "")).strip().casefold() for field in fields)


def _principal_key(principal: AuthenticatedPrincipal) -> str:
    if not isinstance(principal, AuthenticatedPrincipal):
        raise PermissionError("authenticated principal required")
    return principal.principal_key


class SQLSessionRegistry:
    """In-memory SQL sessions per principal for one server process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._credentials: dict[str, SQLCredentials] = {}
        self._targets: dict[str, tuple[str, str]] = {}
        # DataLayer is an unhashable dataclass, so keep weak references and
        # compare by identity.
        self._layers: dict[str, list[weakref.ref]] = {}

    @staticmethod
    def _open(credentials: SQLCredentials, config, logger):
        return open_sql_data_layer(credentials, config, logger)

    def _remember(self, key: str, credentials: SQLCredentials, layer) -> None:
        with self._lock:
            if self._credentials.get(key) is not credentials:
                raise PermissionError("SQL sign-in required")
            alive = [ref for ref in self._layers.get(key, []) if ref() is not None]
            alive.append(weakref.ref(layer))
            self._layers[key] = alive

    def sign_in(self, principal: AuthenticatedPrincipal, credentials: SQLCredentials,
                config, logger):
        """Verify ``credentials`` by connecting; on success keep them for this session.

        Returns the verified DataLayer (usable for listing catalogs and tables).
        Raises PermissionError for Windows authentication outside local mode and
        ValueError when the server rejects the connection.
        """
        key = _principal_key(principal)
        if not isinstance(credentials, SQLCredentials):
            raise TypeError("SQLCredentials required")
        if credentials.windows and principal.provider is not AuthenticationMode.LOCAL:
            raise PermissionError("Windows authentication is only available in local mode")
        layer = self._open(credentials, config, logger)
        with self._lock:
            self._credentials[key] = credentials
            self._targets[key] = _connection_target(config)
            self._layers[key] = []
        self._remember(key, credentials, layer)
        return layer

    def is_signed_in(self, principal: AuthenticatedPrincipal) -> bool:
        key = _principal_key(principal)
        with self._lock:
            return key in self._credentials

    def uses_windows_authentication(self, principal: AuthenticatedPrincipal) -> bool:
        key = _principal_key(principal)
        with self._lock:
            credentials = self._credentials.get(key)
        return bool(credentials and credentials.windows)

    def credentials_for_owner(self, owner: str, connection=None) -> SQLCredentials | None:
        """Trusted, server-side lookup by principal key for the dispatcher (285c).

        Job records store only the owner's principal key. Never expose the
        returned credentials outside the dispatcher's worker pipe.
        """
        if not isinstance(owner, str) or not owner:
            return None
        with self._lock:
            if connection is not None and self._targets.get(owner) != _connection_target(connection):
                return None
            return self._credentials.get(owner)

    def sign_out(self, principal: AuthenticatedPrincipal) -> None:
        """Forget the credentials; runs not yet executed then fail authorization."""
        key = _principal_key(principal)
        with self._lock:
            self._credentials.pop(key, None)
            self._targets.pop(key, None)
            self._layers.pop(key, None)

    def data_layer_for(self, principal: AuthenticatedPrincipal, config, logger):
        """Open a run's DataLayer with the principal's own session credentials."""
        key = _principal_key(principal)
        with self._lock:
            credentials = self._credentials.get(key)
            authorized_target = self._targets.get(key)
        if credentials is None or authorized_target != _connection_target(config):
            raise PermissionError("SQL sign-in required")
        layer = self._open(credentials, config, logger)
        self._remember(key, credentials, layer)
        return layer

    def authorize(self, principal: AuthenticatedPrincipal, prepared) -> None:
        """Hook for PrincipalRunAccessService(authorize_dataset=...).

        SQL runs must use a DataLayer this registry opened for the same
        principal, still signed in. Other sources are left to their own checks.
        """
        if prepared.dataset.source is not DataSourceKind.SQL:
            return
        key = _principal_key(principal)
        with self._lock:
            refs = self._layers.get(key) if key in self._credentials else None
            allowed = bool(refs) and any(ref() is prepared.resolved_dataset for ref in refs)
        if not allowed:
            raise PermissionError("SQL dataset unavailable")
