"""Server-rendered web interface: shell and Projects tab (revision 287a).

295d: the Overview and About tabs (JBGWebUIOverview, JBGWebUIAbout); the
title in the header links to the Overview.
295e: the JBG logo image in the header; /ui opens on the Overview.

Locked decisions: FastAPI + Jinja2 (server-rendered), Fluent 2 layout patterns
styled with the organization's stylesheet tokens, Swedish by default with
English available, technical terms untranslated. This first slice uses plain
HTML forms (post/redirect/get) and no JavaScript; htmx and Fluent web
components arrive with the first tab that needs them.

The interface uses the same services and identity as the JSON API; it adds no
business rules. Messages are chosen by fixed codes, never by echoing input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

WEB = Path(__file__).resolve().parent / "web"
# 295e: the header shows the logo the Voilà GUI uses (GUI/default_settings.json),
# served from where it is, so the two interfaces share one file.
LOGO = Path(__file__).resolve().parent / "GUI" / "assets" / "logo.png"
LANGUAGES = ("sv", "en")
DEFAULT_LANGUAGE = "sv"
LANGUAGE_COOKIE = "jbg_lang"
PROJECT_COOKIE = "jbg_project"
CONFIGURATION_COOKIE = "jbg_configuration"          # set on the Configuration tab
MODEL_COOKIE = "jbg_model"                          # set on the Models tab (295b)

# (key, English label, available in this revision)
# 296a: "Run" became "Training"; "Prediction" follows "Models" (the work order:
# train, look at the results, choose a model, predict).
# 295d: Overview is available; "About" is the last tab, right of the others.
TABS = (("overview", "Overview", True), ("projects", "Projects", True),
        ("dataset", "Dataset", False), ("configuration", "Configuration", False),
        ("run", "Training", False), ("results", "Results", False),
        ("models", "Models", False), ("prediction", "Prediction", False),
        ("history", "Run History", False), ("about", "About", True))

MESSAGES = {
    "created": ("ok", "Project created."), "saved": ("ok", "Changes saved."),
    "opened": ("ok", "Project opened."),
    "closed": ("ok", "Project closed."),
    "duplicate": ("error", "A project with that name already exists."),
    "invalid": ("error", "The input was not accepted. Check the name (1-200 characters)."),
    "unavailable": ("error", "The project is no longer available."),
}


def _load_catalog(language: str) -> dict[str, str]:
    if language == "en":
        return {}
    from babel.messages.pofile import read_po
    with (WEB / "i18n" / f"{language}.po").open("rb") as stream:
        catalog = read_po(stream, locale=language)
    return {message.id: message.string for message in catalog if message.id and message.string}


_CATALOGS = {language: _load_catalog(language) for language in LANGUAGES}
_CATALOG_TIMES: dict[str, float] = {}


def _current_catalog(language: str) -> dict[str, str]:
    """295c: reload a catalogue when its file has changed, as the templates are
    re-read, so new texts after an update need no server restart."""
    path = WEB / "i18n" / f"{language}.po"
    try:
        changed = path.stat().st_mtime
    except OSError:
        return _CATALOGS.get(language, {})
    if _CATALOG_TIMES.get(language) != changed:
        _CATALOGS[language] = _load_catalog(language)
        _CATALOG_TIMES[language] = changed
    return _CATALOGS[language]


def translator(language: str) -> Callable[[str], str]:
    catalog = _current_catalog(language)
    return lambda text: catalog.get(text, text)


def request_language(request: Request) -> str:
    """Saved choice (cookie), else the browser's first preference, else Swedish."""
    chosen = request.cookies.get(LANGUAGE_COOKIE)
    if chosen in LANGUAGES:
        return chosen
    header = request.headers.get("accept-language", "")
    first = header.split(",")[0].split(";")[0].strip().lower()
    return "en" if first.startswith("en") else DEFAULT_LANGUAGE


def _number_text(value, language: str = "sv") -> str:
    """295a: numbers in tables; decimal comma in Swedish, at most four
    significant digits for fractions; anything else as text."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e15):
        return str(int(value))
    if isinstance(value, float):
        text = f"{value:.4g}"
        return text.replace(".", ",") if language == "sv" else text
    return str(value)


def _local_time(iso: str) -> str:
    """294a: stored times are UTC; show them in the server's local time zone
    (in local mode the user's own)."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return ""


_STATIC_VERSIONS: dict = {}


def static_version(name: str) -> str:
    """298d: a short hash of a static file's content, added to its address
    (``?v=``) so that a browser never runs an old jbg.js or jbg.css after an
    update (found 2026-11-11: the files were served without a version and
    browsers kept using a cached copy). Recomputed when the file changes."""
    path = WEB / "static" / name
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _STATIC_VERSIONS.get(name)
    if cached is None or cached[0] != key:
        cached = (key, hashlib.sha256(path.read_bytes()).hexdigest()[:12])
        _STATIC_VERSIONS[name] = cached
    return cached[1]


def _redirect(path: str, **query) -> RedirectResponse:
    query = {key: value for key, value in query.items() if value}
    separator = "&" if "?" in path else "?"
    return RedirectResponse(path + (separator + urlencode(query) if query else ""), status_code=303)


def add_web_ui(app: FastAPI, *, principal_dependency, projects, version_line: str = "",
               datasets=None, inspection=None, configurations=None, sessions=None,
               sql_server=None, runs=None, project_runs=None, results=None,
               models=None, logs=None, trust_app_service_headers: bool = False) -> None:
    """Mount the interface under /ui on ``app``.

    The Dataset tab (291a) is mounted when its services are given.
    """
    from JBGProjectService import ProjectConflict, ProjectUnavailable

    templates = Jinja2Templates(directory=str(WEB / "templates"))
    # 293b2: shares with a decimal comma in Swedish, a point in English.
    templates.env.filters["localtime"] = _local_time            # 294a
    templates.env.filters["num"] = _number_text                  # 295a
    templates.env.filters["pct"] = lambda value, language="sv": (
        f"{float(value or 0):.1f}".replace(".", "," if language == "sv" else ".") + " %")
    templates.env.globals["static_url"] = (                       # 298d
        lambda name: f"/ui/static/{name}?v={static_version(name)}")
    app.mount("/ui/static", StaticFiles(directory=str(WEB / "static")), name="ui-static")

    def active_project(request: Request, owner: str):
        project_id = request.cookies.get(PROJECT_COOKIE)
        if not project_id:
            return None
        try:
            return projects.get(owner, project_id)
        except ProjectUnavailable:
            return None

    available = ({"overview", "projects", "about"} | ({"dataset"} if datasets is not None else set())
                 | ({"configuration"} if configurations is not None and datasets is not None
                    else set())
                 | ({"run"} if project_runs is not None and runs is not None else set())
                 | ({"results"} if results is not None and runs is not None else set())
                 | ({"models"} if models is not None and runs is not None else set())
                 | ({"prediction"} if project_runs is not None and runs is not None else set()))
    tabs = tuple((key, label, key in available) for key, label, _ in TABS)

    def active_dataset(owner: str, project):
        """The project's active training dataset, for the context bar."""
        if datasets is None or project is None:
            return None
        try:
            items = datasets.list(owner, project["project_id"])
        except Exception:
            return None
        return next((d for d in items if d["active_training"]), None)

    def active_configuration(request: Request, owner: str, project):
        """The configuration chosen on the Configuration tab (292a), for the context bar."""
        configuration_id = request.cookies.get("jbg_configuration")
        if configurations is None or project is None or not configuration_id:
            return None
        try:
            return configurations.get(owner, project["project_id"], configuration_id)
        except Exception:
            return None

    def display_name(request: Request):
        """296a: the name shown in the header, never the technical identity.
        X-MS-CLIENT-PRINCIPAL-NAME is read only behind Azure App Service
        authentication (JBG_TRUST_APP_SERVICE_HEADERS), which sets it and
        strips copies sent by clients; otherwise anyone could send one."""
        if not trust_app_service_headers:
            return None
        value = request.headers.get("x-ms-client-principal-name", "")
        value = "".join(ch for ch in value if ch.isprintable()).strip()[:200]
        return value or None

    def active_model(request: Request, owner: str, project):
        """The model chosen on the Models tab (295b), for the context bar."""
        model_id = request.cookies.get(MODEL_COOKIE)
        if models is None or project is None or not model_id:
            return None
        try:
            return models.get(owner, project["project_id"], model_id)
        except Exception:
            return None

    def render(request: Request, principal, template: str, tab: str, status_code: int = 200,
               **context):
        language = request_language(request)
        _ = translator(language)
        code = request.query_params.get("message", "")
        message = context.pop("tab_messages", {}).get(code) or MESSAGES.get(code)
        project = active_project(request, principal.principal_key)
        base = {"request": request, "_": _, "language": language, "tabs": tabs,
                "current_tab": tab, "user": display_name(request) or _("Unknown user"),
                "version_line": version_line,
                "has_logo": LOGO.is_file(),                      # 295e: else the text logo
                "active_project": project,
                "active_dataset": active_dataset(principal.principal_key, project),
                "active_configuration": active_configuration(request, principal.principal_key,
                                                             project),
                "active_model": active_model(request, principal.principal_key, project),
                "message": message}
        base.update(context)
        return templates.TemplateResponse(request, template, base, status_code=status_code)

    if datasets is not None:
        from JBGWebUIDatasets import add_dataset_tab
        add_dataset_tab(app, principal_dependency=principal_dependency, render=render,
                        redirect=_redirect, active_project=active_project, datasets=datasets,
                        inspection=inspection, configurations=configurations,
                        sessions=sessions, sql_server=sql_server)
    if models is not None and runs is not None:
        from JBGWebUIModels import add_models_tab
        add_models_tab(app, principal_dependency=principal_dependency, render=render,
                       redirect=_redirect, active_project=active_project, models=models,
                       runs=runs, datasets=datasets)
    if results is not None and runs is not None:
        from JBGWebUIResults import add_results_tab
        reclassification = None
        if sessions is not None and sql_server is not None:
            # 299a: corrections of mispredicted rows in SQL runs.
            from JBGReclassification import SQLReclassificationService
            reclassification = SQLReclassificationService(results._store, results, sessions,
                                                          sql_server)
        add_results_tab(app, principal_dependency=principal_dependency, render=render,
                        active_project=active_project, runs=runs, results=results,
                        reclassification=reclassification, redirect=_redirect)
    if project_runs is not None and runs is not None:
        from JBGWebUIRun import add_run_tab
        add_run_tab(app, principal_dependency=principal_dependency, render=render,
                    redirect=_redirect, active_project=active_project, runs=runs,
                    project_runs=project_runs, datasets=datasets,
                    configurations=configurations, sessions=sessions, inspection=inspection,
                    models=models, results=results, logs=logs)
        from JBGWebUIPrediction import add_prediction_tab
        add_prediction_tab(app, principal_dependency=principal_dependency, render=render,
                           redirect=_redirect, active_project=active_project, runs=runs,
                           project_runs=project_runs, datasets=datasets, sessions=sessions,
                           models=models, logs=logs, results=results)
    from JBGWebUIOverview import add_overview_tab                 # 295d
    add_overview_tab(app, principal_dependency=principal_dependency, render=render,
                     active_project=active_project, datasets=datasets,
                     configurations=configurations, runs=runs, models=models)
    from JBGWebUIAbout import add_about_tab                       # 295d
    add_about_tab(app, principal_dependency=principal_dependency, render=render)
    if configurations is not None and datasets is not None:
        from JBGWebUIConfigurations import add_configuration_tab
        add_configuration_tab(app, principal_dependency=principal_dependency, render=render,
                              redirect=_redirect, active_project=active_project,
                              configurations=configurations, datasets=datasets,
                              inspection=inspection)

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def ui_root():
        # 295e: the interface opens on the Overview (tab 1 of the design); without
        # an open project it points to the Projects tab.
        return _redirect("/ui/overview")

    @app.get("/ui/logo.png", include_in_schema=False)
    def logo():
        """295e: no sign-in needed; the file is the same public logo as in Voilà."""
        if not LOGO.is_file():
            return Response(status_code=404)
        return FileResponse(LOGO, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/ui/projects", response_class=HTMLResponse, include_in_schema=False)
    def projects_tab(request: Request, selected: str = "",
                     principal=Depends(principal_dependency)):
        owner = principal.principal_key
        items = projects.list(owner)
        chosen = next((item for item in items if item["project_id"] == selected), None)
        return render(request, principal, "projects.html", "projects", projects=items,
                      selected=chosen)

    @app.post("/ui/projects", include_in_schema=False)
    def create_project(name: str = Form(""), source: str = Form(""),
                       description: str = Form(""), principal=Depends(principal_dependency)):
        try:
            created = projects.create(principal.principal_key, name=name, source=source,
                                      description=description)
        except ProjectConflict:
            return _redirect("/ui/projects", message="duplicate")
        except (ValueError, TypeError):
            return _redirect("/ui/projects", message="invalid")
        return _redirect("/ui/projects", selected=created["project_id"], message="created")

    @app.post("/ui/projects/{project_id}", include_in_schema=False)
    def update_project(project_id: str, name: str = Form(""), description: str = Form(""),
                       principal=Depends(principal_dependency)):
        try:
            projects.update(principal.principal_key, project_id, name=name,
                            description=description)
        except ProjectUnavailable:
            return _redirect("/ui/projects", message="unavailable")
        except ProjectConflict:
            return _redirect("/ui/projects", selected=project_id, message="duplicate")
        except (ValueError, TypeError):
            return _redirect("/ui/projects", selected=project_id, message="invalid")
        return _redirect("/ui/projects", selected=project_id, message="saved")

    @app.post("/ui/projects/{project_id}/open", include_in_schema=False)
    def open_project(request: Request, project_id: str,
                     principal=Depends(principal_dependency)):
        try:
            projects.get(principal.principal_key, project_id)
        except ProjectUnavailable:
            return _redirect("/ui/projects", message="unavailable")
        response = _redirect("/ui/projects", selected=project_id, message="opened")
        if request.cookies.get(PROJECT_COOKIE) != project_id:
            # 294a2/295b: configuration and model belong to the previous project.
            response.delete_cookie(CONFIGURATION_COOKIE, path="/ui")
            response.delete_cookie(MODEL_COOKIE, path="/ui")
        response.set_cookie(PROJECT_COOKIE, project_id, httponly=True, samesite="lax",
                            path="/ui")
        return response

    @app.post("/ui/close-project", include_in_schema=False)
    def close_project(back: str = Form("/ui/projects"), principal=Depends(principal_dependency)):
        """294a2: forget the open project and its selected configuration."""
        target = back if back.startswith("/ui/") and "//" not in back else "/ui/projects"
        response = _redirect(target.split("?")[0], message="closed")
        response.delete_cookie(PROJECT_COOKIE, path="/ui")
        response.delete_cookie(CONFIGURATION_COOKIE, path="/ui")
        response.delete_cookie(MODEL_COOKIE, path="/ui")
        return response

    @app.post("/ui/language", include_in_schema=False)
    def choose_language(language: str = Form(DEFAULT_LANGUAGE), back: str = Form("/ui/projects"),
                        principal=Depends(principal_dependency)):
        # Only local interface paths: never redirect elsewhere.
        target = back if back.startswith("/ui/") and "//" not in back else "/ui/projects"
        response = RedirectResponse(target, status_code=303)
        if language in LANGUAGES:
            response.set_cookie(LANGUAGE_COOKIE, language, httponly=True, samesite="lax",
                                path="/ui", max_age=365 * 24 * 3600)
        return response
