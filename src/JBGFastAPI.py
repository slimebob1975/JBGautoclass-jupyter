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


class ProjectRunRequest(BaseModel):
    """288a: a run from a project configuration (default: latest version) and a
    dataset (default: the project's active training dataset)."""
    configuration_id: str
    version: Optional[int] = None
    dataset_id: Optional[str] = None


class PredictionRequest(BaseModel):
    """295c: the saved model, and optionally the dataset to predict (default:
    the training dataset's rows without a class)."""
    model_id: str
    dataset_id: Optional[str] = None


class DatasetRolesRequest(BaseModel):
    """293a: one role per column (id, class, numeric, text, categorical, ignored);
    293a2: the categorical limit for text columns (2-1000, default 30)."""
    columns: dict
    categorical_limit: int = 30


class SQLSignIn(BaseModel):
    """Sign in to the server's SQL Server (288b), or, for the interim catalog,
    to the server of a saved Voilà configuration."""
    method: Literal["windows", "sql"]
    username: str = ""
    password: str = ""
    configuration: Optional[str] = None


def create_app(
    presentation,
    *,
    current_principal: Callable[[], AuthenticatedPrincipal],
    submission=None,
    sessions=None,
    projects=None,
    datasets=None,
    configurations=None,
    project_runs=None,
    models=None,
    dataset_inspection=None,
    run_results=None,
    run_logs=None,
    trust_app_service_headers: bool = False,
    sql_server=None,
    web_ui: bool = False,
    version_line: str = "",
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

    @app.middleware("http")
    async def refuse_cross_site_changes(request: Request, call_next):
        # 287a: with HTML forms in the interface, another web page in the same
        # browser could submit to this server (cross-site request forgery).
        # Browsers label such requests; refuse every state-changing one. Clients
        # without these headers (scripts, PowerShell, tests) are unaffected.
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            site = request.headers.get("sec-fetch-site")
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if (site is not None and site not in ("same-origin", "none")) or (
                    origin is not None and origin not in (f"http://{host}", f"https://{host}")):
                return JSONResponse(status_code=403,
                                    content={"detail": "cross-site request refused"})
        return await call_next(request)

    root_target = "/ui/" if web_ui else "/docs"

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url=root_target, status_code=307)

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
    if web_ui:
        if projects is None:
            raise TypeError("the web interface needs the project service")
        from JBGWebUI import add_web_ui
        add_web_ui(app, principal_dependency=principal_dependency, projects=projects,
                   version_line=version_line, datasets=datasets,
                   inspection=dataset_inspection, configurations=configurations,
                   sessions=sessions, sql_server=sql_server,
                   runs=presentation if hasattr(presentation, "list_project_runs") else None,
                   project_runs=project_runs, results=run_results, models=models,
                   logs=run_logs, trust_app_service_headers=trust_app_service_headers)
    if datasets is not None:
        _add_dataset_endpoints(app, principal_dependency, datasets)
    if configurations is not None:
        _add_configuration_endpoints(app, principal_dependency, configurations)
    if project_runs is not None:
        _add_project_run_endpoints(app, principal_dependency, project_runs)
    if models is not None:
        _add_model_endpoints(app, principal_dependency, models)
    if dataset_inspection is not None:
        _add_dataset_inspection_endpoints(app, principal_dependency, dataset_inspection)
    if run_results is not None:
        _add_run_results_endpoints(app, principal_dependency, run_results)
    if run_logs is not None:
        _add_run_log_endpoint(app, principal_dependency, run_logs)
    if submission is not None:
        _add_run_and_session_endpoints(app, presentation, principal_dependency, read_owned,
                                       submission, sessions, sql_server)
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
    from JBGDatasetRoles import IncompatibleDataset
    from JBGDatasetService import (DatasetConflict, DatasetUnavailable, UploadTooLarge,
                                   WrongSourceType)
    from JBGProjectRuns import RunRequestRefused
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
        except IncompatibleDataset as error:
            # The owner's own column names, to say what is missing.
            raise HTTPException(status_code=409, detail={
                "message": str(error), "missing_columns": list(error.missing),
                "different_kind": list(error.different)}) from None
        except PermissionError as error:
            if str(error) == "SQL sign-in required":
                raise HTTPException(status_code=409, detail="SQL sign-in required") from None
            raise HTTPException(status_code=403, detail="not permitted") from None
        except RunRequestRefused as error:
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
                    table=request.table, principal=principal)

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
                        delimiter=delimiter, encoding=encoding, principal=principal)
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

    @app.get("/api/config/schema")
    def configuration_schema(request: Request, source: Literal["sql", "csv"],
                             language: Optional[Literal["sv", "en"]] = None,
                             principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        """290a: user-owned settings with groups, kinds, defaults, options and
        labels in the user's language; the Configuration tab generates its forms
        from it. Technical option names are never translated."""
        from JBGConfigSchema import configuration_schema as build_schema
        from JBGConfigurationService import _server_template
        from JBGWebUI import request_language, translator
        chosen = language or request_language(request)
        schema = build_schema(source, translator(chosen), _server_template())
        schema["language"] = chosen
        return schema

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


def _add_project_run_endpoints(app, principal_dependency, project_runs):
    """Runs from a project's configuration and dataset (288a)."""
    from JBGConfigurationService import ConfigurationUnavailable
    from JBGDatasetService import DatasetUnavailable
    from JBGProjectRuns import RunDoesNotMatchDataset, RunRequestRefused
    from JBGProjectService import ProjectUnavailable

    @app.post("/api/projects/{project_id}/runs", status_code=201)
    def start_project_run(project_id: str, request: ProjectRunRequest = Body(...),
                          principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            return project_runs.submit(principal, project_id,
                                       configuration_id=request.configuration_id,
                                       version=request.version, dataset_id=request.dataset_id)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except ConfigurationUnavailable:
            raise HTTPException(status_code=404, detail="configuration unavailable") from None
        except DatasetUnavailable:
            raise HTTPException(status_code=404, detail="dataset unavailable") from None
        except RunRequestRefused as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except PermissionError as error:
            if str(error) == "SQL sign-in required":
                raise HTTPException(status_code=409, detail="SQL sign-in required") from None
            raise HTTPException(status_code=403, detail="not permitted") from None
        except RunDoesNotMatchDataset as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (ValueError, TypeError, KeyError, OSError):
            # Configuration, dataset or storage problem; details stay server-side.
            raise HTTPException(status_code=422, detail="the run cannot be started") from None

    @app.post("/api/projects/{project_id}/predictions", status_code=201)
    def start_prediction(project_id: str, request: PredictionRequest = Body(...),
                         principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        """295c: predict with a saved model (SQL projects for now)."""
        try:
            return project_runs.submit_prediction(principal, project_id,
                                                  model_id=request.model_id,
                                                  dataset_id=request.dataset_id)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except DatasetUnavailable:
            raise HTTPException(status_code=404, detail="dataset unavailable") from None
        except RunRequestRefused as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except PermissionError as error:
            if str(error) == "SQL sign-in required":
                raise HTTPException(status_code=409, detail="SQL sign-in required") from None
            raise HTTPException(status_code=403, detail="not permitted") from None
        except (ValueError, TypeError, KeyError, OSError):
            raise HTTPException(status_code=422, detail="the run cannot be started") from None


def _add_model_endpoints(app, principal_dependency, models):
    """Models registered by project runs (288c), for the Models tab."""
    from JBGModelService import ModelUnavailable
    from JBGProjectService import ProjectUnavailable

    def call(action, *args):
        try:
            return action(*args)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except ModelUnavailable:
            raise HTTPException(status_code=404, detail="model unavailable") from None

    @app.get("/api/projects/{project_id}/models")
    def list_models(project_id: str,
                    principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(models.list, principal.principal_key, project_id)

    @app.get("/api/projects/{project_id}/models/{model_id}")
    def get_model(project_id: str, model_id: str,
                  principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(models.get, principal.principal_key, project_id, model_id)


def _add_dataset_inspection_endpoints(app, principal_dependency, inspection):
    """Dataset validation and preview, the same for SQL and CSV (289a)."""
    from JBGConfigurationService import ConfigurationUnavailable
    from JBGDatasetInspection import ColumnUnavailable, DatasetTooLarge, DatasetUnreadable
    from JBGDatasetService import DatasetUnavailable
    from JBGProjectRuns import RunRequestRefused
    from JBGProjectService import ProjectUnavailable

    def call(action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except ProjectUnavailable:
            raise HTTPException(status_code=404, detail="project unavailable") from None
        except DatasetUnavailable:
            raise HTTPException(status_code=404, detail="dataset unavailable") from None
        except ConfigurationUnavailable:
            raise HTTPException(status_code=404, detail="configuration unavailable") from None
        except RunRequestRefused as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except PermissionError as error:
            if str(error) == "SQL sign-in required":
                raise HTTPException(status_code=409, detail="SQL sign-in required") from None
            raise HTTPException(status_code=403, detail="not permitted") from None
        except ColumnUnavailable:
            raise HTTPException(status_code=404, detail="column unavailable") from None
        except (DatasetUnreadable, DatasetTooLarge) as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (ValueError, TypeError) as error:
            # 293a: fixed texts from the role checks and plain-name rule.
            raise HTTPException(status_code=422, detail=str(error)) from None

    base = "/api/projects/{project_id}/datasets/{dataset_id}"

    @app.post(base + "/validate")
    def validate_dataset(project_id: str, dataset_id: str,
                         principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        # 293a: the dataset's own roles (or, for prediction datasets, the active
        # training dataset's) replace the 289a choice of a configuration.
        return call(inspection.validate, principal, project_id, dataset_id)

    def roles_response(dataset_id, roles):
        if roles is None:
            return {"dataset_id": dataset_id, "columns": None, "categorical_limit": None}
        return {"dataset_id": dataset_id, "columns": roles.as_mapping(),
                "categorical_limit": roles.categorical_limit}

    @app.get(base + "/roles")
    def get_roles(project_id: str, dataset_id: str,
                  principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return roles_response(dataset_id, call(inspection.get_roles, principal, project_id,
                                               dataset_id))

    @app.put(base + "/roles")
    def set_roles(project_id: str, dataset_id: str, request: DatasetRolesRequest = Body(...),
                  principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return roles_response(dataset_id, call(inspection.set_roles, principal, project_id,
                                               dataset_id, request.columns,
                                               request.categorical_limit))

    @app.get("/api/sql/catalogs")
    def sql_catalogs(principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(inspection.sql_catalogs, principal)

    @app.get("/api/sql/tables")
    def sql_tables(catalog: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(inspection.sql_tables, principal, catalog)

    @app.get(base + "/preview")
    def preview_dataset(project_id: str, dataset_id: str,
                        principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(inspection.preview, principal, project_id, dataset_id)

    @app.get(base + "/profile")
    def profile_dataset(project_id: str, dataset_id: str,
                        principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(inspection.profile, principal, project_id, dataset_id)

    @app.get(base + "/class-distribution")
    def class_distribution(project_id: str, dataset_id: str, column: str,
                           principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        return call(inspection.class_distribution, principal, project_id, dataset_id, column)


def _add_run_log_endpoint(app, principal_dependency, logs):
    """296a: the whole worker log of a run, as a download, for its owner only."""
    from fastapi.responses import FileResponse
    from JBGRunResults import RunResultsUnavailable

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            path = logs.path(principal.principal_key, run_id)
        except RunResultsUnavailable:
            raise HTTPException(status_code=404, detail="log unavailable") from None
        return FileResponse(path, media_type="text/plain; charset=utf-8",
                            filename=f"jbg_run_{run_id[:8]}.log")


def _add_run_results_endpoints(app, principal_dependency, results):
    """A project run's details for its owner (295a): class names and row values
    come from the run's own folder, never from the metadata database."""
    from JBGRunResults import RunResultsUnavailable

    @app.get("/api/runs/{run_id}/class-labels")
    def class_labels(run_id: str,
                     principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            return {"run_id": run_id,
                    "class_labels": results.class_labels(principal.principal_key, run_id)}
        except RunResultsUnavailable:
            raise HTTPException(status_code=404, detail="run unavailable") from None

    @app.get("/api/runs/{run_id}/predictions.csv")
    def predictions_download(run_id: str,
                             principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        """296b: a prediction run's predictions as a file, for its owner only."""
        from fastapi.responses import FileResponse
        try:
            path = results.predictions_file(principal.principal_key, run_id)
        except RunResultsUnavailable:
            raise HTTPException(status_code=404, detail="predictions unavailable") from None
        return FileResponse(path, media_type="text/csv; charset=utf-8",
                            filename=f"jbg_predictions_{run_id[:8]}.csv")

    @app.get("/api/runs/{run_id}/details/{kind}")
    def run_detail(run_id: str, kind: str, page: int = 1, page_size: int = 50,
                   principal: AuthenticatedPrincipal = Depends(principal_dependency)):
        try:
            return results.detail(principal.principal_key, run_id, kind, page=page,
                                  page_size=page_size)
        except RunResultsUnavailable as error:
            raise HTTPException(status_code=404, detail=str(error)) from None


def _add_run_and_session_endpoints(app, runs, principal_dependency, read_owned,
                                   submission, sessions, sql_server=None):
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
        if request.configuration is None:
            # 288b: the server's own SQL Server, for all SQL projects on it.
            if sql_server is None or not sql_server.configured:
                raise HTTPException(status_code=409,
                                    detail="no SQL Server is configured on this server") from None
            config = sql_server.probe_config()
        else:
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
