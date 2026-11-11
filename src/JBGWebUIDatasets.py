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

VIEWS = ("preview", "profile", "classes", "roles")
ROLE_CHOICES = (("id", "ID column"), ("class", "Class column"), ("numeric", "Numeric feature"),
                ("text", "Text feature"), ("categorical", "Categorical feature"),
                ("ignored", "Ignored"))
# Fixed texts from JBGDatasetRoles, shown on the roles view when saving fails.
ROLE_ERRORS = (
    "a column with a role does not exist in the dataset",
    "exactly one column must be the ID column",
    "exactly one column must be the class column",
    "at least one column must be a feature",
    "the categorical limit must be a whole number between 2 and 1000",
    "unknown column role",
)

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
    "incompatible": ("error", "The dataset is not compatible with the training dataset."),
    "signin_needed": ("error", "Sign in to SQL Server first."),
    "roles_saved": ("ok", "Column roles saved. Validate the dataset again."),
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
    from JBGDatasetRoles import IncompatibleDataset
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

    def optional(action, *args):
        """Lists are an aid: any failure shows the form without them (293b)."""
        try:
            return action(*args)
        except Exception:
            return None

    def page(request, principal, *, dataset="", view="preview", column="", validation=None,
             status_code=200, incompatible=None, roles_error=None, roles_draft=None,
             roles_from=""):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "datasets": [], "selected": None,
                   "view": view if view in VIEWS else "preview", "column": column,
                   "validation": validation, "configurations": [], "data": None,
                   "read_error": None, "read_errors": READ_ERRORS, "sql": None,
                   "tab_messages": MESSAGES, "incompatible": incompatible,
                   "role_choices": ROLE_CHOICES, "roles_error": roles_error,
                   "catalogs": None, "catalog": "", "add_role": "training", "table_choice": None}
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
                if context["sql"]["signed_in"]:
                    # 293b: catalogs and tables come from the user's login.
                    listed = optional(inspection.sql_catalogs, principal)
                    if listed:
                        context["catalogs"] = listed["catalogs"]
                        chosen = request.query_params.get("catalog") or listed["default"] or ""
                        context["catalog"] = chosen if chosen in listed["catalogs"] else ""
                    role = request.query_params.get("add_role", "training")
                    context["add_role"] = role if role in ("training", "prediction", "mixed") else "training"
                    if context["catalog"]:
                        context["table_choice"] = optional(inspection.table_choices, principal,
                                                           pid, context["catalog"], context["add_role"])
            # 293b2: the views stay available below a validation result.
            if selected is not None:
                v = context["view"]
                if v == "preview":
                    context["data"], context["read_error"] = read(
                        inspection.preview, principal, pid, dataset)
                elif v == "profile":
                    context["data"], context["read_error"] = read(
                        inspection.profile, principal, pid, dataset)
                elif v == "roles":
                    if selected["role"] != "prediction":
                        profile, context["read_error"] = read(inspection.profile, principal, pid, dataset)
                        stored, _ = read(inspection.get_roles, principal, pid, dataset)
                        context["data"] = profile
                        context["roles"] = roles_draft or (stored.as_mapping() if stored else {})
                        context["roles_stored"] = stored is not None
                        context["categorical_limit"] = (roles_draft or {}).get(
                            "__limit__", stored.categorical_limit if stored else 30)
                        if stored is None and not roles_draft and profile:
                            from JBGDatasetRoles import (roles_from_configuration,
                                                         suggest_roles, with_class_column)
                            names = [c["name"] for c in profile["columns"]]
                            # 298a: the column last shown on the Classes view
                            # (carried in the address) is the class column.
                            chosen = column if column in names else ""
                            suggested = suggest_roles(
                                profile["columns"], profile["basis"].get("rows_used") or 0,
                                context["categorical_limit"], chosen)
                            context["roles"] = suggested
                            context["roles_suggested"] = True
                            context["roles_class_chosen"] = chosen
                            # 293b2: a dataset registered before decision A has no
                            # roles; suggest those of the project's latest
                            # configuration that has any. Nothing is saved.
                            for item in sorted(context["configurations"],
                                               key=lambda c: c["updated_at"], reverse=True):
                                settings = optional(configurations.stored_settings, owner, pid,
                                                    item["configuration_id"])
                                mapping = roles_from_configuration(settings, names) if settings else {}
                                if mapping:
                                    if chosen:
                                        mapping = with_class_column(mapping, chosen, suggested)
                                    context["roles"] = mapping
                                    context["roles_suggested_from"] = item["name"]
                                    context["roles_suggested"] = False
                                    break
                        # 297d: the roles of another dataset of the project, to copy.
                        sources = []
                        for item in items:
                            if item["dataset_id"] == dataset or item["role"] == "prediction":
                                continue
                            roles = optional(inspection.get_roles, principal, pid,
                                             item["dataset_id"])
                            if roles is not None:
                                sources.append((item, roles))
                        context["roles_sources"] = [item for item, _ in sources]
                        source = next((pair for pair in sources
                                       if pair[0]["dataset_id"] == roles_from), None)
                        if source is not None and not roles_draft and profile:
                            from JBGDatasetRoles import copy_roles
                            kinds = {c["name"]: c["type"] for c in profile["columns"]}
                            mapping, missing, not_numeric = copy_roles(source[1], kinds)
                            context["roles"] = mapping
                            context["categorical_limit"] = source[1].categorical_limit
                            context["roles_copied_from"] = source[0]["name"]
                            context["roles_from"] = roles_from
                            context["roles_missing"] = missing
                            context["roles_not_numeric"] = not_numeric
                            for key in ("roles_suggested_from", "roles_suggested",
                                        "roles_class_chosen"):
                                context.pop(key, None)
                else:
                    columns, context["read_error"] = read(inspection.preview, principal, pid, dataset)
                    context["column_names"] = [c["name"] for c in columns["columns"]] if columns else []
                    # 298a: before roles are saved, the column shown here
                    # becomes the class column on the Column roles view.
                    context["roles_stored"] = (selected["role"] == "prediction" or optional(
                        inspection.get_roles, principal, pid, dataset) is not None)
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
                    roles_from: str = "", principal=Depends(principal_dependency)):
        return page(request, principal, dataset=dataset, view=view, column=column,
                    roles_from=roles_from)

    @app.post("/ui/dataset/csv", include_in_schema=False)
    def upload_csv(request: Request, file: UploadFile = File(...), name: str = Form(""),
                   role: str = Form("training"), principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        try:
            if pid is None:
                return redirect("/ui/dataset")
            created = datasets.upload_csv(principal.principal_key, pid, name=name, role=role,
                                          stream=file.file, filename=file.filename,
                                          principal=principal)
        except UploadTooLarge:
            return redirect("/ui/dataset", message="too_large")
        except DatasetConflict:
            return redirect("/ui/dataset", message="duplicate")
        except IncompatibleDataset as error:
            # Shown directly, naming the owner's own missing columns.
            return page(request, principal, incompatible=error, status_code=409)
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
                                          catalog=catalog, table=table, principal=principal)
        except DatasetConflict:
            return redirect("/ui/dataset", message="duplicate")
        except IncompatibleDataset as error:
            return page(request, principal, incompatible=error, status_code=409)
        except PermissionError:
            return redirect("/ui/dataset", message="signin_needed")
        except RunRequestRefused:
            return redirect("/ui/dataset", message="no_sql_server")
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
    def validate(request: Request, dataset_id: str, principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/dataset")
        # 293a: the dataset's own roles; prediction datasets against training.
        result, error = read(inspection.validate, principal, pid, dataset_id)
        if error is not None:
            if error == "unreadable":
                return redirect("/ui/dataset", message="unavailable")
            return page(request, principal, dataset=dataset_id,
                        validation={"error": error})
        # Shown directly (post and render): the check list is not stored.
        return page(request, principal, dataset=dataset_id, validation=result)

    @app.post("/ui/dataset/{dataset_id}/roles", response_class=HTMLResponse,
              include_in_schema=False)
    async def save_roles(request: Request, dataset_id: str,
                         principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/dataset")
        form = await request.form()
        mapping = {key[5:]: value for key, value in form.multi_items()
                   if key.startswith("role:") and value != "ignored"}
        try:
            limit = int(str(form.get("categorical_limit", "30")).strip())
        except ValueError:
            limit = 0                                       # refused below with a fixed text
        try:
            inspection.set_roles(principal, pid, dataset_id, mapping, limit)
        except (DatasetUnavailable, ProjectUnavailable):
            return redirect("/ui/dataset", message="unavailable")
        except PermissionError:
            return redirect("/ui/dataset", dataset=dataset_id, view="roles", message="signin_needed")
        except ValueError as error:
            draft = dict(mapping, __limit__=limit)
            message = str(error) if str(error) in ROLE_ERRORS else "unknown column role"
            return page(request, principal, dataset=dataset_id, view="roles",
                        roles_error=message, roles_draft=draft, status_code=422)
        return redirect("/ui/dataset", dataset=dataset_id, view="roles", message="roles_saved")

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
