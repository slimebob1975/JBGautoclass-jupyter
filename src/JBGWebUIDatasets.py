"""Dataset tab of the web interface (revision 291a).

Plain forms and links, no JavaScript: adding (CSV upload or SQL table),
activating, SQL sign-in for the server, preview, profile, class distribution
and validation. Views are part of the address (``?dataset=...&view=profile``),
so they can be bookmarked. The tab uses the same services as the JSON API and
adds no rules of its own. Messages are fixed codes; input, passwords and
driver messages are never echoed. Preview values go to the page only.
"""
from __future__ import annotations

from fastapi import Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

VIEWS = ("preview", "profile", "classes")

MESSAGES = {
    "added": ("ok", "Dataset added."),
    "activated": ("ok", "Dataset activated for training."),
    "signed_in": ("ok", "Signed in to SQL Server."),
    "signed_out": ("ok", "Signed out of SQL Server."),
    "duplicate": ("error", "A dataset with that name already exists."),
    "too_large": ("error", "The file is larger than 25 MB."),
    "bad_csv": ("error", "The file was not accepted. Check that it is a CSV file with a header row."),
    "bad_sql": ("error", "The input was not accepted. Check the name, catalog and table (schema.table)."),
    "cannot_activate": ("error", "This dataset cannot be used for training."),
    "signin_failed": ("error", "SQL sign-in failed."),
    "no_sql_server": ("error", "No SQL Server is configured on this server."),
    "unavailable": ("error", "The dataset is no longer available."),
}
READ_ERRORS = {
    "signin_required": "Sign in to SQL Server to read this dataset.",
    "unreadable": "The dataset could not be read.",
    "too_large": "The table is too large for a quick profile.",
    "no_sql_server": "No SQL Server is configured on this server.",
}


class _QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def add_dataset_tab(app, *, principal_dependency, render, redirect, active_project,
                    datasets, inspection, configurations, sessions, sql_server) -> None:
    from JBGDatasetInspection import DatasetTooLarge, DatasetUnreadable
    from JBGDatasetService import (DatasetConflict, DatasetUnavailable, UploadTooLarge,
                                   WrongSourceType)
    from JBGProjectRuns import RunRequestRefused
    from JBGProjectService import ProjectUnavailable

    def read(action, *args, **kwargs):
        """Run a read; return (result, error code)."""
        try:
            return action(*args, **kwargs), None
        except PermissionError:
            return None, "signin_required"
        except RunRequestRefused:
            return None, "no_sql_server"
        except DatasetTooLarge:
            return None, "too_large"
        except (DatasetUnreadable, LookupError, ValueError):
            return None, "unreadable"

    def page(request, principal, *, dataset="", view="preview", column="", validation=None,
             status_code=200):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "datasets": [], "selected": None,
                   "view": view if view in VIEWS else "preview", "column": column,
                   "validation": validation, "configurations": [], "data": None,
                   "read_error": None, "read_errors": READ_ERRORS, "sql": None,
                   "tab_messages": MESSAGES}
        if project is not None:
            pid = project["project_id"]
            try:
                items = datasets.list(owner, pid)
            except ProjectUnavailable:
                items = []
            context["datasets"] = items
            context["configurations"] = configurations.list(owner, pid)
            selected = next((d for d in items if d["dataset_id"] == dataset), None)
            context["selected"] = selected
            if project["source"] == "sql":
                context["sql"] = {"configured": bool(sql_server and sql_server.configured),
                                  "signed_in": sessions.is_signed_in(principal),
                                  "windows": sessions.uses_windows_authentication(principal)}
            if selected is not None and validation is None:
                v = context["view"]
                if v == "preview":
                    context["data"], context["read_error"] = read(
                        inspection.preview, principal, pid, dataset)
                elif v == "profile":
                    context["data"], context["read_error"] = read(
                        inspection.profile, principal, pid, dataset)
                else:
                    columns, context["read_error"] = read(inspection.preview, principal, pid, dataset)
                    context["column_names"] = [c["name"] for c in columns["columns"]] if columns else []
                    if column and context["read_error"] is None:
                        context["data"], context["read_error"] = read(
                            inspection.class_distribution, principal, pid, dataset, column)
        return render(request, principal, "dataset.html", "dataset", status_code=status_code,
                      **context)

    def project_id(request, principal):
        project = active_project(request, principal.principal_key)
        return None if project is None else project["project_id"]

    @app.get("/ui/dataset", response_class=HTMLResponse, include_in_schema=False)
    def dataset_tab(request: Request, dataset: str = "", view: str = "preview", column: str = "",
                    principal=Depends(principal_dependency)):
        return page(request, principal, dataset=dataset, view=view, column=column)

    @app.post("/ui/dataset/csv", include_in_schema=False)
    def upload_csv(request: Request, file: UploadFile = File(...), name: str = Form(""),
                   role: str = Form("training"), principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        try:
            if pid is None:
                return redirect("/ui/dataset")
            created = datasets.upload_csv(principal.principal_key, pid, name=name, role=role,
                                          stream=file.file, filename=file.filename)
        except UploadTooLarge:
            return redirect("/ui/dataset", message="too_large")
        except DatasetConflict:
            return redirect("/ui/dataset", message="duplicate")
        except (WrongSourceType, ValueError, TypeError, ProjectUnavailable):
            return redirect("/ui/dataset", message="bad_csv")
        finally:
            file.file.close()
        return redirect("/ui/dataset", dataset=created["dataset_id"], message="added")

    @app.post("/ui/dataset/sql", include_in_schema=False)
    def register_sql(request: Request, name: str = Form(""), role: str = Form("training"),
                     catalog: str = Form(""), table: str = Form(""),
                     principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/dataset")
        try:
            created = datasets.create_sql(principal.principal_key, pid, name=name, role=role,
                                          catalog=catalog, table=table)
        except DatasetConflict:
            return redirect("/ui/dataset", message="duplicate")
        except (WrongSourceType, ValueError, TypeError, ProjectUnavailable):
            return redirect("/ui/dataset", message="bad_sql")
        return redirect("/ui/dataset", dataset=created["dataset_id"], message="added")

    @app.post("/ui/dataset/{dataset_id}/activate", include_in_schema=False)
    def activate(request: Request, dataset_id: str, principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/dataset")
        try:
            datasets.activate(principal.principal_key, pid, dataset_id)
        except (DatasetUnavailable, ProjectUnavailable):
            return redirect("/ui/dataset", message="unavailable")
        except (ValueError, TypeError):
            return redirect("/ui/dataset", dataset=dataset_id, message="cannot_activate")
        return redirect("/ui/dataset", dataset=dataset_id, message="activated")

    @app.post("/ui/dataset/{dataset_id}/validate", response_class=HTMLResponse,
              include_in_schema=False)
    def validate(request: Request, dataset_id: str, configuration_id: str = Form(""),
                 principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/dataset")
        result, error = read(inspection.validate, principal, pid, dataset_id,
                             configuration_id=configuration_id or None)
        if error is not None:
            if error == "unreadable":
                return redirect("/ui/dataset", message="unavailable")
            return page(request, principal, dataset=dataset_id,
                        validation={"error": error})
        # Shown directly (post and render): the check list is not stored.
        return page(request, principal, dataset=dataset_id, validation=result)

    @app.post("/ui/sql/session", include_in_schema=False)
    def sign_in(request: Request, method: str = Form("sql"), username: str = Form(""),
                password: str = Form(""), back: str = Form(""),
                principal=Depends(principal_dependency)):
        from JBGSQLSessions import SQLCredentials
        target = back if back.startswith("/ui/") and "//" not in back else "/ui/dataset"
        if sql_server is None or not sql_server.configured:
            return redirect(target, message="no_sql_server")
        try:
            credentials = (SQLCredentials.windows_login() if method == "windows"
                           else SQLCredentials.sql_login(username, password))
            sessions.sign_in(principal, credentials, sql_server.probe_config(), _QuietLogger())
        except Exception:
            return redirect(target, message="signin_failed")
        return redirect(target, message="signed_in")

    @app.post("/ui/sql/session/sign-out", include_in_schema=False)
    def sign_out(request: Request, principal=Depends(principal_dependency)):
        sessions.sign_out(principal)
        return redirect("/ui/dataset", message="signed_out")
