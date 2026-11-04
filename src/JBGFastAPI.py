"""Read-only FastAPI adapter for the existing GUI-independent run presentation.

Authentication is performed by a trusted host-supplied principal provider. The
HTTP client cannot select a principal, tenant, owner or storage scope. Local
mode uses a server-configured LocalIdentityProvider; production must supply an
Entra-token-verifying provider before this adapter is exposed publicly.
"""
from __future__ import annotations

from typing import Callable, Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
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


class ProjectCreate(BaseModel):
    """286a: a new project; its source type can never change afterwards."""
    name: str
    source: Literal["sql", "csv"]
    description: str = ""


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class SQLDatasetCreate(BaseModel):
    """286b: an SQL dataset names a catalog and table; never a host or login."""
    name: str
    role: Literal["training", "prediction", "mixed"]
    catalog: str
    table: str


class ConfigurationCreate(BaseModel):
    """286c: a named configuration; settings is a partial user view (see service)."""
    name: str
    settings: dict = {}


class ConfigurationSave(BaseModel):
    settings: dict
    expected_version: int


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
    projects=None,
    datasets=None,
    configurations=None,
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

    @app.get("/api/runs/{run_id}/overview")
    def overview(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(presentation.overview, run_id, principal)

    @app.get("/api/runs/{run_id}/result")
    def result(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return read_owned(presentation.result_details, run_id, principal)

    if projects is not None:
        _add_project_endpoints(app, principal_dependency, projects)
    if datasets is not None:
        _add_dataset_endpoints(app, principal_dependency, datasets)
    if configurations is not None:
        _add_configuration_endpoints(app, principal_dependency, configurations)
    if submission is not None:
        _add_run_and_session_endpoints(app, presentation, principal_dependency, read_owned,
                                       submission, sessions)
    return app


def _add_project_endpoints(app, principal_dependency, projects):
    """Projects (286a). No deletion until a cascade policy exists."""
    from JBGProjectService import ProjectConflict, ProjectUnavailable

    def call(action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except ProjectConflict:
            raise HTTPException(status_code=409, detail="project name already exists") from None
        except (ValueError, TypeError) as error:
            # Repository messages are fixed texts (e.g. length limits), never input.
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.get("/api/projects")
    def list_projects(principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(projects.list, principal.principal_key)

    @app.post("/api/projects", status_code=201)
    def create_project(request: ProjectCreate = Body(...),
                       principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(projects.create, principal.principal_key, name=request.name,
                    source=request.source, description=request.description)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str,
                    principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(projects.get, principal.principal_key, project_id)

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, request: ProjectUpdate = Body(...),
                       principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(projects.update, principal.principal_key, project_id,
                    name=request.name, description=request.description)


def _add_dataset_endpoints(app, principal_dependency, datasets):
    """Datasets of a project (286b). No deletion and no validation yet."""
    from JBGDatasetService import (DatasetConflict, DatasetUnavailable, UploadTooLarge,
                                   WrongSourceType)
    from JBGProjectService import ProjectUnavailable

    def call(action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except DatasetUnavailable:
            raise HTTPException(status_code=404, detail="dataset unavailable") from None
        except DatasetConflict:
            raise HTTPException(status_code=409, detail="dataset name already exists") from None
        except UploadTooLarge:
            raise HTTPException(status_code=413, detail="CSV file exceeds 25 MB") from None
        except WrongSourceType as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except (ValueError, TypeError) as error:
            # Fixed texts from the repository and the CSV structure check.
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.get("/api/projects/{project_id}/datasets")
    def list_datasets(project_id: str,
                      principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(datasets.list, principal.principal_key, project_id)

    @app.post("/api/projects/{project_id}/datasets", status_code=201)
    def create_sql_dataset(project_id: str, request: SQLDatasetCreate = Body(...),
                           principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(datasets.create_sql, principal.principal_key, project_id,
                    name=request.name, role=request.role, catalog=request.catalog,
                    table=request.table)

    @app.post("/api/projects/{project_id}/datasets/csv", status_code=201)
    def upload_csv_dataset(project_id: str, file: UploadFile = File(...),
                           name: str = Form(...),
                           role: Literal["training", "prediction", "mixed"] = Form(...),
                           delimiter: Optional[Literal[",", ";", "\t", "|"]] = Form(None),
                           encoding: Literal["utf-8-sig", "utf-8", "cp1252", "latin-1"]
                           = Form("utf-8-sig"),
                           principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            return call(datasets.upload_csv, principal.principal_key, project_id, name=name,
                        role=role, stream=file.file, filename=file.filename,
                        delimiter=delimiter, encoding=encoding)
        finally:
            file.file.close()

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}")
    def get_dataset(project_id: str, dataset_id: str,
                    principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(datasets.get, principal.principal_key, project_id, dataset_id)

    @app.post("/api/projects/{project_id}/datasets/{dataset_id}/activate")
    def activate_dataset(project_id: str, dataset_id: str,
                         principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(datasets.activate, principal.principal_key, project_id, dataset_id)


def _add_configuration_endpoints(app, principal_dependency, configurations):
    """Project configurations (286c). No deletion yet."""
    from JBGConfigurationService import (ConfigurationChanged, ConfigurationNameConflict,
                                         ConfigurationUnavailable, InvalidConfiguration)
    from JBGProjectService import ProjectUnavailable

    def call(action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except ConfigurationUnavailable:
            raise HTTPException(status_code=404, detail="configuration unavailable") from None
        except ConfigurationNameConflict:
            raise HTTPException(status_code=409,
                                detail="configuration name already exists") from None
        except ConfigurationChanged:
            raise HTTPException(status_code=409,
                                detail="configuration changed; reload before saving") from None
        except InvalidConfiguration as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (ValueError, TypeError) as error:
            # Fixed repository texts (name length, version, size limits).
            raise HTTPException(status_code=422, detail=str(error)) from None

    base = "/api/projects/{project_id}/configurations"

    @app.get(base)
    def list_configurations(project_id: str,
                            principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(configurations.list, principal.principal_key, project_id)

    @app.post(base, status_code=201)
    def create_configuration(project_id: str, request: ConfigurationCreate = Body(...),
                             principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(configurations.create, principal.principal_key, project_id,
                    name=request.name, settings=request.settings)

    @app.get(base + "/{configuration_id}")
    def get_configuration(project_id: str, configuration_id: str,
                          version: Optional[int] = None,
                          principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(configurations.get, principal.principal_key, project_id,
                    configuration_id, version)

    @app.put(base + "/{configuration_id}")
    def save_configuration(project_id: str, configuration_id: str,
                           request: ConfigurationSave = Body(...),
                           principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(configurations.save, principal.principal_key, project_id,
                    configuration_id, settings=request.settings,
                    expected_version=request.expected_version)


def _add_run_and_session_endpoints(app, runs, principal_dependency, read_owned,
                                   submission, sessions):
    """Endpoints that change state (285d)."""
    from JBGSavedConfigurations import UnknownConfiguration
    from JBGRunSubmission import InvalidSavedSQLConfiguration
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
        except InvalidSavedSQLConfiguration as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
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
