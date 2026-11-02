"""Principal-bound job access with server-owned CSV scope (revision 281j).

The composition root must supply authenticated principals, a trusted scope
resolver, and an authorization policy for every data source. This is not a
JWT verifier, login endpoint, or authorization policy implementation.
"""
from __future__ import annotations

from typing import Callable

from JBGApplicationContracts import DataSourceKind, RunReference
from JBGIdentity import AuthenticatedPrincipal
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunPreparationService import PreparedRun
from Storage.ArtifactContract import StorageScope


class PrincipalRunAccessService:
    """Restrict job operations to an authenticated principal and authorized data."""

    def __init__(
        self,
        presentation: RunAPIPresentationService,
        *,
        scope_for: Callable[[AuthenticatedPrincipal], StorageScope],
        authorize_dataset: Callable[[AuthenticatedPrincipal, PreparedRun], None],
    ) -> None:
        if not isinstance(presentation, RunAPIPresentationService):
            raise TypeError("presentation service required")
        if not callable(scope_for) or not callable(authorize_dataset):
            raise TypeError("trusted scope resolver and dataset authorizer required")
        self._presentation = presentation
        self._scope_for = scope_for
        self._authorize_dataset = authorize_dataset
        self._prepared: dict[RunReference, PreparedRun] = {}

    @staticmethod
    def _owner(principal: AuthenticatedPrincipal) -> str:
        if not isinstance(principal, AuthenticatedPrincipal):
            raise PermissionError("authenticated principal required")
        return principal.principal_key

    def _authorize(self, principal: AuthenticatedPrincipal, prepared: PreparedRun) -> None:
        if not isinstance(prepared, PreparedRun):
            raise TypeError("prepared run required")
        scope = self._scope_for(principal)
        if not isinstance(scope, StorageScope):
            raise PermissionError("authorized storage scope unavailable")
        if prepared.dataset.source is DataSourceKind.CSV:
            # CSV resolvers return ArtifactReference, which carries a scope.
            # Missing or mismatched scope must not fall through to SQL policy.
            if getattr(prepared.resolved_dataset, "scope", None) != scope:
                raise PermissionError("dataset unavailable")
        # This mandatory trusted callback checks project membership and SQL
        # authorization, and may impose stricter CSV permissions as well.
        self._authorize_dataset(principal, prepared)

    def create(self, principal: AuthenticatedPrincipal, prepared: PreparedRun) -> dict:
        owner = self._owner(principal)
        self._authorize(principal, prepared)
        response = self._presentation.create(owner, prepared)
        run = RunReference(response["run_id"])
        self._prepared[run] = prepared
        return response

    def _owned(self, principal: AuthenticatedPrincipal, run: RunReference) -> str:
        owner = self._owner(principal)
        # Check ownership before even looking up the prepared run. Unknown and
        # foreign jobs must be indistinguishable to the caller.
        self._presentation.snapshot(owner, run)
        if run not in self._prepared:
            raise PermissionError("run unavailable")
        return owner

    def execute(self, principal: AuthenticatedPrincipal, run: RunReference) -> dict:
        owner = self._owned(principal, run)
        self._authorize(principal, self._prepared[run])
        return self._presentation.execute(owner, run)

    def snapshot(self, principal: AuthenticatedPrincipal, run: RunReference) -> dict:
        owner = self._owned(principal, run)
        return self._presentation.snapshot(owner, run)

    def progress(self, principal: AuthenticatedPrincipal, run: RunReference) -> list[dict]:
        owner = self._owned(principal, run)
        return self._presentation.progress(owner, run)

    def result(self, principal: AuthenticatedPrincipal, run: RunReference) -> dict:
        owner = self._owned(principal, run)
        return self._presentation.result(owner, run)
