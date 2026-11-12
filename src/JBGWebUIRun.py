"""Run tab of the web interface (revision 294a).

Start a training run from the active training dataset and the selected
configuration (both in the context bar) after a preflight check list; follow
the run live; cancel it; see the overview's results when it has finished.
The page works without JavaScript (reload to update); ``jbg.js`` polls the
run's overview every few seconds and reloads the page once the run has ended.
"""
from __future__ import annotations

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse

TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")
STAGES = {
    "queued": "Queued", "awaiting_credentials": "Waiting for SQL sign-in",
    "preparing": "Preparing", "executing": "Running", "finalizing": "Finishing",
    "succeeded": "Finished", "failed": "Failed", "cancelled": "Cancelled",
    "interrupted": "Interrupted",
    # 295d: the stored statuses of runs that have not ended, as the run lists show them
    # (they showed the raw words "pending" and "running").
    "pending": "Queued", "running": "Running",
    "load_model": "Loading the model", "fetch_data": "Fetching data",
    "load_dataset": "Loading the dataset", "separate_dataset": "Splitting the data",
    "convert_to_numbers": "Converting text and categories",
    "train_model": "Training and comparing models", "evaluate_model": "Evaluating",
    "feature_importance": "Feature importance", "retrain_model": "Training the final model",
    "display_mispredicted": "Finding mispredictions", "make_predictions": "Predicting",
    "send_completetion_email": "Sending the notification",
}
# (code, passed text, failed text, blocking)
PREFLIGHT = (
    ("dataset", "An active training dataset is chosen.", "No active training dataset (Dataset tab).", True),
    ("roles", "The dataset has column roles.", "The dataset has no column roles (Dataset tab).", True),
    ("configuration", "A configuration is selected.", "No configuration is selected (Configuration tab).", True),
    ("trains", "The configuration trains a model.", "The configuration does not train a model.", True),
    ("train_only", "The configuration only trains (CSV).",
     "CSV runs cannot also predict: untick “Predict unlabelled rows” (Configuration tab).", True),
    # 298g: only shown when the configuration keeps groups apart.
    ("groups", "The dataset has a group column for keeping groups apart.",
     "The configuration keeps groups apart, but the dataset has no group column (Dataset tab, Column roles).",
     True),
    ("signed_in", "Signed in to SQL Server.", "Not signed in to SQL Server (Dataset tab).", True),
    ("validated", "The dataset is validated.", "The dataset has not been validated yet.", False),
)
# 298f: where a failed check is put right (the configuration's sections open
# from the address, 298e).
FIX_LINKS = {
    "dataset": "/ui/dataset", "roles": "/ui/dataset", "configuration": "/ui/configuration",
    "trains": "/ui/configuration#section-run_type",
    "train_only": "/ui/configuration#section-run_type", "signed_in": "/ui/dataset",
    "groups": "/ui/dataset",
    "validated": "/ui/dataset",
}
MESSAGES = {
    "started": ("ok", "The run has started."),
    "cancel_requested": ("ok", "Cancelling the run."),
    "refused": ("error", "The run could not be started. Check the list above."),
    "not_fitting": ("error", "The column roles do not fit the dataset."),
    "signin_needed": ("error", "Sign in to SQL Server first."),
    "unavailable": ("error", "The run is no longer available."),
    "id_not_integer": ("error", "The ID column of an SQL dataset must hold whole numbers "
                       "(int, bigint, smallint or tinyint). Choose another ID column on the "
                       "Dataset tab (Column roles)."),
}


def add_run_tab(app, *, principal_dependency, render, redirect, active_project, runs,
                project_runs, datasets, configurations, sessions, inspection,
                models=None, results=None, logs=None) -> None:
    from JBGApplicationContracts import RunReference
    from JBGProjectRuns import (IdColumnNotInteger, RunDoesNotMatchDataset, RunRequestRefused,
                                keeps_groups_apart)

    def preflight(request, principal, project):
        """Fixed check list; the start button needs every blocking check."""
        owner, pid = principal.principal_key, project["project_id"]
        try:
            items = datasets.list(owner, pid)
        except Exception:
            items = []
        active = next((d for d in items if d["active_training"]), None)
        roles = None
        if active is not None:
            try:
                roles = inspection.get_roles(principal, pid, active["dataset_id"])
            except Exception:
                roles = None
        configuration = None
        configuration_id = request.cookies.get("jbg_configuration")
        if configuration_id:
            try:
                configuration = configurations.get(owner, pid, configuration_id)
            except Exception:
                configuration = None
        outcome = {
            "dataset": active is not None,
            "roles": roles is not None,
            "configuration": configuration is not None,
            "trains": bool(configuration and configuration["settings"]["mode"].get("train")),
            "train_only": None if project["source"] != "csv" else bool(
                configuration and not configuration["settings"]["mode"].get("predict")),
            "groups": None if not (configuration and keeps_groups_apart(
                configuration["settings"])) else bool(roles and roles.group_column),
            "signed_in": None if project["source"] != "sql" else sessions.is_signed_in(principal),
            "validated": bool(active and active["validation_state"] == "valid"),
        }
        links = dict(FIX_LINKS)
        if active is not None:
            links["roles"] = links["validated"] = f"/ui/dataset?dataset={active['dataset_id']}"
            links["roles"] += "&view=roles"
            links["groups"] = links["roles"]
        if configuration is not None:
            links["trains"] = links["train_only"] = (
                f"/ui/configuration?configuration={configuration['configuration_id']}#section-run_type")
        checks = [{"code": code, "passed": outcome[code], "blocking": blocking,
                   "message": passed if outcome[code] else failed,
                   "link": None if outcome[code] else links.get(code)}
                  for code, passed, failed, blocking in PREFLIGHT if outcome[code] is not None]
        ready = all(c["passed"] for c in checks if c["blocking"])
        return checks, ready, configuration

    @app.get("/ui/run", response_class=HTMLResponse, include_in_schema=False)
    def run_tab(request: Request, run: str = "", principal=Depends(principal_dependency)):
        """296a: the Training tab (the path stays /ui/run for bookmarks)."""
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "checks": [], "ready": False, "recent": [],
                   "overview": None, "selected_run": None, "stages": STAGES,
                   "terminal": TERMINAL, "tab_messages": MESSAGES, "labels": [],
                   "has_log": False, "back": "/ui/run"}
        if project is not None:
            context["checks"], context["ready"], context["configuration"] = preflight(
                request, principal, project)
            context["recent"] = [r for r in runs.list_project_runs(owner, project["project_id"])
                                 if r.get("kind", "training") != "prediction"]
            chosen = run or (context["recent"][0]["run_id"] if context["recent"] else "")
            if chosen and any(r["run_id"] == chosen for r in context["recent"]):
                try:
                    context["overview"] = runs.overview(owner, RunReference(chosen))
                    context["selected_run"] = chosen
                except Exception:
                    context["overview"] = None
                context.update(run_extras(owner, chosen))
        return render(request, principal, "run.html", "run", **context)

    def run_extras(owner, run_id):
        """Class names (295a) and whether a log exists (296a), both owner-only."""
        extras = {"labels": [], "has_log": False}
        if results is not None:
            try:
                extras["labels"] = results.class_labels(owner, run_id)
            except Exception:
                extras["labels"] = []
        if logs is not None:
            try:
                logs.path(owner, run_id)
                extras["has_log"] = True
            except Exception:
                extras["has_log"] = False
        return extras

    @app.post("/ui/run", include_in_schema=False)
    def start(request: Request, principal=Depends(principal_dependency)):
        project = active_project(request, principal.principal_key)
        configuration_id = request.cookies.get("jbg_configuration", "")
        if project is None or not configuration_id:
            return redirect("/ui/run", message="refused")
        try:
            created = project_runs.submit(principal, project["project_id"],
                                          configuration_id=configuration_id)
        except RunDoesNotMatchDataset:
            return redirect("/ui/run", message="not_fitting")
        except IdColumnNotInteger:
            return redirect("/ui/run", message="id_not_integer")
        except PermissionError:
            return redirect("/ui/run", message="signin_needed")
        except (RunRequestRefused, LookupError, ValueError, TypeError, OSError):
            return redirect("/ui/run", message="refused")
        return redirect("/ui/run", run=created["run_id"], message="started")

    @app.post("/ui/run/{run_id}/cancel", include_in_schema=False)
    def cancel(request: Request, run_id: str, back: str = Form("/ui/run"),
               principal=Depends(principal_dependency)):
        target = back if back in ("/ui/run", "/ui/prediction") else "/ui/run"
        try:
            runs.cancel(principal.principal_key, RunReference(run_id))
        except Exception:
            return redirect(target, message="unavailable")
        return redirect(target, run=run_id, message="cancel_requested")

    @app.get("/ui/runs/{run_id}/log", response_class=HTMLResponse, include_in_schema=False)
    def run_log(request: Request, run_id: str, principal=Depends(principal_dependency)):
        """296a: the end of a run's worker log in the browser; the whole log
        downloads from /api/runs/{id}/log. Owner only."""
        text, truncated = None, False
        if logs is not None:
            try:
                text, truncated = logs.tail(principal.principal_key, run_id)
            except Exception:
                text = None
        return render(request, principal, "run_log.html", "run", run_id=run_id,
                      log_text=text, truncated=truncated,
                      status_code=200 if text is not None else 404)
