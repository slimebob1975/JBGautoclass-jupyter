"""Server-rendered web interface: shell and Projects tab (revision 287a).

Locked decisions: FastAPI + Jinja2 (server-rendered), Fluent 2 layout patterns
styled with the organization's stylesheet tokens, Swedish by default with
English available, technical terms untranslated. This first slice uses plain
HTML forms (post/redirect/get) and no JavaScript; htmx and Fluent web
components arrive with the first tab that needs them.

The interface uses the same services and identity as the JSON API; it adds no
business rules. Messages are chosen by fixed codes, never by echoing input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

WEB = Path(__file__).resolve().parent / "web"
LANGUAGES = ("sv", "en")
DEFAULT_LANGUAGE = "sv"
LANGUAGE_COOKIE = "jbg_lang"
PROJECT_COOKIE = "jbg_project"

# (key, English label, available in this revision)
TABS = (("overview", "Overview", False), ("projects", "Projects", True),
        ("dataset", "Dataset", False), ("configuration", "Configuration", False),
        ("run", "Run", False), ("results", "Results", False),
        ("models", "Models", False), ("history", "Run History", False))

MESSAGES = {
    "created": ("ok", "Project created."), "saved": ("ok", "Changes saved."),
    "opened": ("ok", "Project opened."),
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


def translator(language: str) -> Callable[[str], str]:
    catalog = _CATALOGS.get(language, {})
    return lambda text: catalog.get(text, text)


def request_language(request: Request) -> str:
    """Saved choice (cookie), else the browser's first preference, else Swedish."""
    chosen = request.cookies.get(LANGUAGE_COOKIE)
    if chosen in LANGUAGES:
        return chosen
    header = request.headers.get("accept-language", "")
    first = header.split(",")[0].split(";")[0].strip().lower()
    return "en" if first.startswith("en") else DEFAULT_LANGUAGE


def _redirect(path: str, **query) -> RedirectResponse:
    query = {key: value for key, value in query.items() if value}
    separator = "&" if "?" in path else "?"
    return RedirectResponse(path + (separator + urlencode(query) if query else ""), status_code=303)


def add_web_ui(app: FastAPI, *, principal_dependency, projects, version_line: str = "",
               datasets=None, inspection=None, configurations=None, sessions=None,
               sql_server=None) -> None:
    """Mount the interface under /ui on ``app``.

    The Dataset tab (291a) is mounted when its services are given.
    """
    from JBGProjectService import ProjectConflict, ProjectUnavailable

    templates = Jinja2Templates(directory=str(WEB / "templates"))
    app.mount("/ui/static", StaticFiles(directory=str(WEB / "static")), name="ui-static")

    def active_project(request: Request, owner: str):
        project_id = request.cookies.get(PROJECT_COOKIE)
        if not project_id:
            return None
        try:
            return projects.get(owner, project_id)
        except ProjectUnavailable:
            return None

    available = ({"projects"} | ({"dataset"} if datasets is not None else set())
                 | ({"configuration"} if configurations is not None and datasets is not None
                    else set()))
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

    def render(request: Request, principal, template: str, tab: str, status_code: int = 200,
               **context):
        language = request_language(request)
        _ = translator(language)
        code = request.query_params.get("message", "")
        message = context.pop("tab_messages", {}).get(code) or MESSAGES.get(code)
        project = active_project(request, principal.principal_key)
        base = {"request": request, "_": _, "language": language, "tabs": tabs,
                "current_tab": tab, "user": principal.subject, "version_line": version_line,
                "active_project": project,
                "active_dataset": active_dataset(principal.principal_key, project),
                "active_configuration": active_configuration(request, principal.principal_key,
                                                             project),
                "message": message}
        base.update(context)
        return templates.TemplateResponse(request, template, base, status_code=status_code)

    if datasets is not None:
        from JBGWebUIDatasets import add_dataset_tab
        add_dataset_tab(app, principal_dependency=principal_dependency, render=render,
                        redirect=_redirect, active_project=active_project, datasets=datasets,
                        inspection=inspection, configurations=configurations,
                        sessions=sessions, sql_server=sql_server)
    if configurations is not None and datasets is not None:
        from JBGWebUIConfigurations import add_configuration_tab
        add_configuration_tab(app, principal_dependency=principal_dependency, render=render,
                              redirect=_redirect, active_project=active_project,
                              configurations=configurations, datasets=datasets,
                              inspection=inspection)

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def ui_root():
        return _redirect("/ui/projects")

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
        response.set_cookie(PROJECT_COOKIE, project_id, httponly=True, samesite="lax",
                            path="/ui")
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
