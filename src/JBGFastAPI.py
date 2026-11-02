"""Read-only FastAPI adapter for the existing GUI-independent run presentation.

Authentication is performed by a trusted host-supplied principal provider. The
HTTP client cannot select a principal, tenant, owner or storage scope. Local
mode uses a server-configured LocalIdentityProvider; production must supply an
Entra-token-verifying provider before this adapter is exposed publicly.
"""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, FastAPI, HTTPException

from JBGApplicationContracts import RunReference
from JBGIdentity import AuthenticatedPrincipal
from JBGRunAPIPresentation import RunAPIPresentationService


def create_app(
    presentation: RunAPIPresentationService,
    *,
    current_principal: Callable[[], AuthenticatedPrincipal],
) -> FastAPI:
    """Create a read-only API with fail-closed, server-controlled identity."""
    if not isinstance(presentation, RunAPIPresentationService):
        raise TypeError("run presentation service required")
    if not callable(current_principal):
        raise TypeError("trusted principal provider required")

    app = FastAPI(title="JBG API", version="0.1.0")

    def principal_dependency() -> AuthenticatedPrincipal:
        try:
            principal = current_principal()
        except Exception:
            # Do not disclose token-verification or provider internals.
            raise HTTPException(status_code=401, detail="authentication required") from None
        if not isinstance(principal, AuthenticatedPrincipal):
            raise HTTPException(status_code=401, detail="authentication required")
        return principal

    def read_owned(reader: Callable, run_id: str, principal: AuthenticatedPrincipal):
        try:
            return reader(principal.principal_key, RunReference(run_id))
        except (PermissionError, ValueError):
            # Unknown run and another user's run must be indistinguishable.
            raise HTTPException(status_code=404, detail="run unavailable") from None

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/runs/{run_id}")
    def status(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(presentation.snapshot, run_id, principal)

    @app.get("/api/runs/{run_id}/progress")
    def progress(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(presentation.progress, run_id, principal)

    @app.get("/api/runs/{run_id}/result")
    def result(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(presentation.result_details, run_id, principal)

    return app
