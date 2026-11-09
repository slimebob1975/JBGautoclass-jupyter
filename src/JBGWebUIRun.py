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
    ("signed_in", "Signed in to SQL Server.", "Not signed in to SQL Server (Dataset tab).", True),
    ("validated", "The dataset is validated.", "The dataset has not been validated yet.", False),
)
MESSAGES = {
    "started": ("ok", "The run has started."),
    "cancel_requested": ("ok", "Cancelling the run."),
    "refused": ("error", "The run could not be started. Check the list above."),
    "not_fitting": ("error", "The column roles do not fit the dataset."),
    "signin_needed": ("error", "Sign in to SQL Server first."),
    "unavailable": ("error", "The run is no longer available."),
    "prediction_started": ("ok", "The prediction has started."),
    "prediction_refused": ("error", "The prediction could not be started."),
    "model_missing": ("error", "The model is not available."),
}


def add_run_tab(app, *, principal_dependency, render, redirect, active_project, runs,
                project_runs, datasets, configurations, sessions, inspection,
                models=None) -> None:
    from JBGApplicationContracts import RunReference
    from JBGProjectRuns import RunDoesNotMatchDataset, RunRequestRefused

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
            "signed_in": None if project["source"] != "sql" else sessions.is_signed_in(principal),
            "validated": bool(active and active["validation_state"] == "valid"),
        }
        checks = [{"code": code, "passed": outcome[code], "blocking": blocking,
                   "message": passed if outcome[code] else failed}
                  for code, passed, failed, blocking in PREFLIGHT if outcome[code] is not None]
        ready = all(c["passed"] for c in checks if c["blocking"])
        return checks, ready, configuration

    @app.get("/ui/run", response_class=HTMLResponse, include_in_schema=False)
    def run_tab(request: Request, run: str = "", principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "checks": [], "ready": False, "recent": [],
                   "overview": None, "selected_run": None, "stages": STAGES,
                   "terminal": TERMINAL, "tab_messages": MESSAGES, "prediction": None,
                   "selected_kind": "training"}
        if project is not None:
            context["checks"], context["ready"], context["configuration"] = preflight(
                request, principal, project)
            context["recent"] = runs.list_project_runs(owner, project["project_id"])
            context["prediction"] = prediction_card(request, principal, project)
            chosen = run or (context["recent"][0]["run_id"] if context["recent"] else "")
            if chosen and any(r["run_id"] == chosen for r in context["recent"]):
                try:
                    context["overview"] = runs.overview(owner, RunReference(chosen))
                    context["selected_run"] = chosen
                    context["selected_kind"] = next(r for r in context["recent"]
                                                    if r["run_id"] == chosen).get("kind", "training")
                except Exception:
                    context["overview"] = None
        return render(request, principal, "run.html", "run", **context)

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
        except PermissionError:
            return redirect("/ui/run", message="signin_needed")
        except (RunRequestRefused, LookupError, ValueError, TypeError, OSError):
            return redirect("/ui/run", message="refused")
        return redirect("/ui/run", run=created["run_id"], message="started")

    def prediction_card(request, principal, project):
        """295c: predicting with the chosen model (SQL projects for now)."""
        owner, pid = principal.principal_key, project["project_id"]
        card = {"available": project["source"] == "sql", "model": None, "datasets": [],
                "signed_in": sessions.is_signed_in(principal) if project["source"] == "sql" else None}
        model_id = request.cookies.get("jbg_model")
        if models is not None and model_id:
            try:
                card["model"] = models.get(owner, pid, model_id)
            except Exception:
                card["model"] = None
        try:
            card["datasets"] = [d for d in datasets.list(owner, pid)
                                if d["role"] in ("prediction", "mixed")]
        except Exception:
            card["datasets"] = []
        card["ready"] = bool(card["available"] and card["model"]
                             and card["model"]["state"] == "available" and card["signed_in"])
        return card

    @app.post("/ui/run/predict", include_in_schema=False)
    def predict(request: Request, dataset_id: str = Form(""),
                principal=Depends(principal_dependency)):
        project = active_project(request, principal.principal_key)
        model_id = request.cookies.get("jbg_model", "")
        if project is None or not model_id:
            return redirect("/ui/run", message="model_missing")
        try:
            created = project_runs.submit_prediction(principal, project["project_id"],
                                                     model_id=model_id,
                                                     dataset_id=dataset_id or None)
        except PermissionError:
            return redirect("/ui/run", message="signin_needed")
        except (RunRequestRefused, LookupError, ValueError, TypeError, OSError):
            return redirect("/ui/run", message="prediction_refused")
        return redirect("/ui/run", run=created["run_id"], message="prediction_started")

    @app.post("/ui/run/{run_id}/cancel", include_in_schema=False)
    def cancel(request: Request, run_id: str, principal=Depends(principal_dependency)):
        try:
            runs.cancel(principal.principal_key, RunReference(run_id))
        except Exception:
            return redirect("/ui/run", message="unavailable")
        return redirect("/ui/run", run=run_id, message="cancel_requested")
