"""Configuration tab of the web interface (revision 292a).

The form is generated from the configuration schema (290a): groups become
collapsible sections, advanced fields sit in a further collapsible part, and
every field kind has its control. Settings appear for the algorithms selected
in the saved version. No JavaScript: plain forms.

Form values are converted back by the schema's kinds (a missing checkbox is
"off"; empty numbers are "no value" only where allowed; decimal commas are
accepted). Conversion errors are shown per field; a configuration the Config
loader refuses is shown again with the submitted values kept. Saving uses the
286c version check.
"""
from __future__ import annotations

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse

CONFIGURATION_COOKIE = "jbg_configuration"

MESSAGES = {
    "created": ("ok", "Configuration created."),
    "saved": ("ok", "Configuration saved."),
    "selected": ("ok", "Configuration selected."),
    "duplicate": ("error", "A configuration with that name already exists."),
    "bad_name": ("error", "The name was not accepted (1-200 characters)."),
    "unavailable": ("error", "The configuration is no longer available."),
}
FIELD_ERRORS = {
    "integer": "Enter a whole number.",
    "number": "Enter a number.",
    "range": "The value is outside the allowed range.",
    "required": "A value is required.",
    "choice": "Choose one of the options.",
    "list": "Enter whole numbers separated by commas.",
}


def _number(text: str, kind: str):
    text = text.strip().replace(" ", "")
    if kind == "integer":
        return int(text)
    return float(text.replace(",", "."))          # decimal comma, as typed in Swedish


def convert_form(schema: dict, form) -> tuple[dict, dict]:
    """Turn submitted form values into the user view; return (settings, errors)."""
    settings, errors = {}, {}
    for field in schema["fields"]:
        key, kind = field["key"], field["kind"]
        if kind == "parameters":
            continue
        section, name = key.split(".", 1)
        form_key = "f:" + key
        raw = form.get(form_key)
        value = None
        try:
            if kind == "boolean":
                value = raw is not None
            elif kind == "multi_choice":
                values = form.getlist(form_key)
                allowed = {option["value"] for option in field["options"]}
                if field.get("allows_all") and ("ALL" in values or (values and set(values) == allowed)):
                    # 293c rules 1, 3 and 5: "All" (or every box) is sent as ALL only.
                    value = ["ALL"]
                elif any(v not in allowed for v in values):
                    raise LookupError("choice")
                else:
                    value = values or (None if field["nullable"] else [])
            elif kind == "columns":
                # Checkbox lists send one value each; the text fallback sends
                # one comma-separated value. Both become one list of names.
                value = [part.strip() for v in form.getlist(form_key)
                         for part in str(v).split(",") if part.strip()]
            elif kind == "choice":
                allowed = {option["value"] for option in field["options"]}
                if raw not in allowed:
                    raise LookupError("choice")
                value = raw
            elif kind in ("integer", "number"):
                text = (raw or "").strip()
                if not text:
                    if not field["nullable"]:
                        raise LookupError("required")
                else:
                    try:
                        value = _number(text, kind)
                    except ValueError:
                        raise LookupError(kind) from None
                    if ("minimum" in field and value < field["minimum"]) or \
                            ("maximum" in field and value > field["maximum"]):
                        raise LookupError("range")
            elif kind == "integer_list":
                text = (raw or "").strip()
                try:
                    value = [int(part) for part in text.replace(";", ",").split(",") if part.strip()]
                except ValueError:
                    raise LookupError("list") from None
                if not value and field["nullable"]:
                    value = None
            else:                                       # text, column
                value = (raw or "").strip()
        except LookupError as error:
            errors[key] = str(error.args[0])
            continue
        settings.setdefault(section, {})[name] = value

    parameters = {}
    for algorithm, specs in schema["model_parameters"].items():
        for spec in specs:
            form_key = f"p:{algorithm}:{spec['name']}"
            raw = (form.get(form_key) or "").strip()
            if not raw:
                continue                                 # the default applies
            try:
                if spec["kind"] == "choice":
                    match = [o for o in spec["options"] if str(o) == raw]
                    if not match:
                        raise LookupError("choice")
                    value = match[0]
                else:
                    try:
                        value = _number(raw, "integer" if spec["kind"] == "int" else "number")
                    except ValueError:
                        raise LookupError("integer" if spec["kind"] == "int" else "number") from None
                    if not (spec["minimum"] <= value <= spec["maximum"]):
                        raise LookupError("range")
            except LookupError as error:
                errors[form_key] = str(error.args[0])
                continue
            parameters.setdefault(algorithm, {})[spec["name"]] = value
    settings.setdefault("mode", {})["model_parameters"] = parameters
    return settings, errors


# 293c: shown first; the other groups sit under "More settings".
ESSENTIAL_GROUPS = ("run_type", "algorithms", "validation")


def combinations(schema: dict, values: dict) -> int:
    """Spot-check combinations: algorithms x scalers x reductions x
    oversamplers x undersamplers ("ALL" counts every option)."""
    total = 1
    for field in schema["fields"]:
        if not field.get("allows_all"):
            continue
        section, name = field["key"].split(".", 1)
        chosen = (values.get(section) or {}).get(name)
        if chosen and "ALL" in chosen:
            total *= len(field["options"])
        elif chosen:
            total *= len(chosen)
    return total


def add_configuration_tab(app, *, principal_dependency, render, redirect, active_project,
                          configurations, datasets, inspection) -> None:
    from JBGConfigSchema import configuration_schema
    from JBGConfigurationService import (ConfigurationChanged, ConfigurationNameConflict,
                                         ConfigurationUnavailable, InvalidConfiguration,
                                         _server_template)
    from JBGProjectService import ProjectUnavailable
    from JBGWebUI import request_language, translator

    def page(request, principal, *, configuration="", values=None, field_errors=None,
             form_message=None, status_code=200):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "items": [], "selected": None,
                   "schema": None, "values": values, "field_errors": field_errors or {},
                   "form_message": form_message, "combinations": None,
                   "essential_groups": ESSENTIAL_GROUPS,
                   "field_error_texts": FIELD_ERRORS, "tab_messages": MESSAGES}
        if project is not None:
            pid = project["project_id"]
            try:
                items = configurations.list(owner, pid)
            except ProjectUnavailable:
                items = []
            context["items"] = items
            selected = next((c for c in items if c["configuration_id"] == configuration), None)
            context["selected"] = selected
            if selected is not None:
                language = request_language(request)
                context["schema"] = configuration_schema(project["source"], translator(language),
                                                         _server_template())
                if values is None:
                    context["values"] = selected["settings"]
                context["combinations"] = combinations(context["schema"], context["values"])
        return render(request, principal, "configuration.html", "configuration",
                      status_code=status_code, **context)

    def project_id(request, principal):
        project = active_project(request, principal.principal_key)
        return None if project is None else project["project_id"]

    def remember(response, configuration_id):
        response.set_cookie(CONFIGURATION_COOKIE, configuration_id, httponly=True,
                            samesite="lax", path="/ui")
        return response

    @app.get("/ui/configuration", response_class=HTMLResponse, include_in_schema=False)
    def configuration_tab(request: Request, configuration: str = "",
                          principal=Depends(principal_dependency)):
        if not configuration:
            configuration = request.cookies.get(CONFIGURATION_COOKIE, "")
        return page(request, principal, configuration=configuration)

    @app.post("/ui/configuration", include_in_schema=False)
    def create(request: Request, name: str = Form(""), principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/configuration")
        try:
            created = configurations.create(principal.principal_key, pid, name=name, settings={})
        except ConfigurationNameConflict:
            return redirect("/ui/configuration", message="duplicate")
        except (ValueError, TypeError, ProjectUnavailable):
            return redirect("/ui/configuration", message="bad_name")
        return remember(redirect("/ui/configuration", configuration=created["configuration_id"],
                                 message="created"), created["configuration_id"])

    @app.post("/ui/configuration/{configuration_id}/select", include_in_schema=False)
    def select(request: Request, configuration_id: str, principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        try:
            if pid is None:
                raise ConfigurationUnavailable("configuration unavailable")
            configurations.get(principal.principal_key, pid, configuration_id)
        except (ConfigurationUnavailable, ProjectUnavailable):
            return redirect("/ui/configuration", message="unavailable")
        return remember(redirect("/ui/configuration", configuration=configuration_id,
                                 message="selected"), configuration_id)

    @app.post("/ui/configuration/{configuration_id}", response_class=HTMLResponse,
              include_in_schema=False)
    async def save(request: Request, configuration_id: str,
                   principal=Depends(principal_dependency)):
        pid = project_id(request, principal)
        if pid is None:
            return redirect("/ui/configuration")
        project = active_project(request, principal.principal_key)
        form = await request.form()
        language = request_language(request)
        schema = configuration_schema(project["source"], translator(language), _server_template())
        settings, errors = convert_form(schema, form)
        try:
            expected = int(form.get("expected_version", ""))
        except ValueError:
            expected = 0
        if errors:
            return page(request, principal, configuration=configuration_id,
                        values=_merged(settings, schema), field_errors=errors,
                        form_message="fields", status_code=422)
        try:
            configurations.save(principal.principal_key, pid, configuration_id,
                                settings=settings, expected_version=expected)
        except ConfigurationChanged:
            return page(request, principal, configuration=configuration_id,
                        values=_merged(settings, schema), form_message="changed",
                        status_code=409)
        except InvalidConfiguration:
            return page(request, principal, configuration=configuration_id,
                        values=_merged(settings, schema), form_message="invalid",
                        status_code=422)
        except (ConfigurationUnavailable, ProjectUnavailable):
            return redirect("/ui/configuration", message="unavailable")
        return remember(redirect("/ui/configuration", configuration=configuration_id,
                                 message="saved"), configuration_id)


def _merged(settings: dict, schema: dict) -> dict:
    """Submitted values for redisplay; fields that failed fall back to empty."""
    values = {}
    for field in schema["fields"]:
        section, name = field["key"].split(".", 1)
        values.setdefault(section, {})[name] = settings.get(section, {}).get(name)
    values["mode"]["model_parameters"] = settings.get("mode", {}).get("model_parameters", {})
    return values
