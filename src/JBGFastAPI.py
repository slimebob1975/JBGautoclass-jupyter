"""Read-only FastAPI adapter for the existing GUI-independent run presentation.

Authentication is performed by a trusted host-supplied principal provider. The
HTTP client cannot select a principal, tenant, owner or storage scope. Local
mode uses a server-configured LocalIdentityProvider; production must supply an
Entra-token-verifying provider before this adapter is exposed publicly.
"""
from __future__ import annotations

from typing import Callable, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi_offline import FastAPIOffline
from pydantic import BaseModel

from JBGApplicationContracts import RunReference
from JBGIdentity import AuthenticatedPrincipal
from JBGRunAPIPresentation import RunAPIPresentationService


class RunRequest(BaseModel):
    """Start a run from a saved configuration (285d). Never a path or settings."""
    source: Literal["sql", "csv"]
    configuration: Optional[str] = None       # SQL: a name from /api/configurations
    project: Optional[str] = None              # CSV: project name
    configuration_id: Optional[str] = None     # CSV: saved configuration record


class SQLSignIn(BaseModel):
    method: Literal["windows", "sql"]
    username: str = ""
    password: str = ""
    configuration: str


def create_app(
    presentation,
    *,
    current_principal: Callable[[], AuthenticatedPrincipal],
    submission=None,
    sessions=None,
    lifespan=None,
) -> FastAPI:
    """Create the API with fail-closed, server-controlled identity.

    ``presentation`` is the in-memory RunAPIPresentationService (read-only) or,
    from 285d, the DurableRunService. With ``submission`` and ``sessions``, the
    app also starts and cancels runs and manages the user's SQL session.
    """
    from JBGDurableRuns import DurableRunService
    if not isinstance(presentation, (RunAPIPresentationService, DurableRunService)):
        raise TypeError("run presentation service required")
    if not callable(current_principal):
        raise TypeError("trusted principal provider required")
    if (submission is None) != (sessions is None):
        raise TypeError("submission and sessions are configured together")

    app = FastAPIOffline(title="JBG API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 body echoes the submitted input, which may
        # contain an SQL password. Report only that the request was invalid.
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs", status_code=307)

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

    if submission is not None:
        _add_run_and_session_endpoints(app, presentation, principal_dependency, read_owned,
                                       submission, sessions)
    return app


def _add_run_and_session_endpoints(app, runs, principal_dependency, read_owned,
                                   submission, sessions):
    """Endpoints that change state (285d)."""
    from JBGSavedConfigurations import UnknownConfiguration
    from JBGSQLSessions import SQLCredentials

    @app.get("/api/runs")
    def list_runs(principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return runs.list_runs(principal.principal_key)

    @app.post("/api/runs", status_code=201)
    def start_run(request: RunRequest = Body(...),
                  principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            if request.source == "sql":
                if not request.configuration:
                    raise HTTPException(status_code=422, detail="invalid request")
                return submission.submit_sql(principal, request.configuration)
            if not request.project or not request.configuration_id:
                raise HTTPException(status_code=422, detail="invalid request")
            return submission.submit_csv(principal, request.project, request.configuration_id)
        except UnknownConfiguration:
            raise HTTPException(status_code=404, detail="configuration unavailable") from None
        except PermissionError as error:
            if str(error) == "SQL sign-in required":
                raise HTTPException(status_code=409, detail="SQL sign-in required") from None
            raise HTTPException(status_code=403, detail="not permitted") from None
        except (ValueError, TypeError, OSError, KeyError):
            # Configuration, dataset or connection problem; details stay server-side.
            raise HTTPException(status_code=422, detail="configuration cannot be run") from None

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(runs.cancel, run_id, principal)

    @app.get("/api/configurations")
    def configurations(project: Optional[str] = None,
                       principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        catalog = submission.configurations
        listing = {"sql": catalog.sql_configurations(project), "csv": []}
        if project:
            try:
                listing["csv"] = catalog.csv_configurations(project)
            except (ValueError, OSError):
                listing["csv"] = []
        return listing

    @app.get("/api/sql/session")
    def sql_session(principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return {"signed_in": sessions.is_signed_in(principal),
                "windows": sessions.uses_windows_authentication(principal)}

    @app.post("/api/sql/session")
    def sql_sign_in(request: SQLSignIn = Body(...),
                    principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            credentials = (SQLCredentials.windows_login() if request.method == "windows"
                           else SQLCredentials.sql_login(request.username, request.password))
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="invalid request") from None
        try:
            config = submission.configurations.load_sql(request.configuration)
        except UnknownConfiguration:
            raise HTTPException(status_code=404, detail="configuration unavailable") from None
        try:
            sessions.sign_in(principal, credentials, config, _SilentLogger())
        except PermissionError:
            raise HTTPException(status_code=403, detail="not permitted") from None
        except Exception:
            # Never echo driver or server messages; they can contain host names.
            raise HTTPException(status_code=401, detail="SQL sign-in failed") from None
        return {"signed_in": True, "windows": credentials.windows}

    @app.delete("/api/sql/session")
    def sql_sign_out(principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        sessions.sign_out(principal)
        return {"signed_in": False, "windows": False}


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
