"""Prediction tab of the web interface (revision 296a).

Agreed 2026-11-09: the former Run tab is split into Training and Prediction;
Prediction follows Models (train, look at the results, choose a model,
predict). Predicting uses the model chosen on the Models tab (295b): SQL
projects into the prediction table (295c), CSV projects on the training
dataset's rows without a class (296b) or, since 297f, on another CSV dataset
of the project. Finished prediction runs show their
predictions, paged, and offer them as a file.
"""
from __future__ import annotations

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse

# (code, passed text, failed text)
PREFLIGHT = (
    ("model", "A model is chosen.", "No model is chosen (Models tab)."),
    ("available", "The model is available.", "The model is not available."),
    ("signed_in", "Signed in to SQL Server.", "Not signed in to SQL Server (Dataset tab)."),
)
MESSAGES = {
    "prediction_started": ("ok", "The prediction has started."),
    "prediction_refused": ("error", "The prediction could not be started."),
    "id_not_integer": ("error", "The ID column of an SQL dataset must hold whole numbers "
                       "(int, bigint, smallint or tinyint). Choose another ID column on the "
                       "Dataset tab (Column roles)."),
    "model_missing": ("error", "The model is not available."),
    "signin_needed": ("error", "Sign in to SQL Server first."),
    "cancel_requested": ("ok", "Cancelling the run."),
    "unavailable": ("error", "The run is no longer available."),
    # 297f: CSV prediction datasets.
    "prediction_incompatible": ("error", "The dataset does not fit the model: columns are missing, "
                                         "not numbers, or the ID column is not unique."),
    "nothing_to_predict": ("error", "Every row of the dataset has a class: there is nothing to "
                                    "predict."),
}


def add_prediction_tab(app, *, principal_dependency, render, redirect, active_project, runs,
                       project_runs, datasets, sessions, models, logs=None,
                       results=None) -> None:
    from JBGApplicationContracts import RunReference
    from JBGProjectRuns import (IdColumnNotInteger, NothingToPredict, RunDoesNotMatchDataset,
                                RunRequestRefused)
    from JBGWebUIRun import STAGES, TERMINAL

    def preflight(request, principal, project):
        owner, pid = principal.principal_key, project["project_id"]
        model = None
        model_id = request.cookies.get("jbg_model")
        if models is not None and model_id:
            try:
                model = models.get(owner, pid, model_id)
            except Exception:
                model = None
        outcome = {"model": model is not None,
                   "available": bool(model and model["state"] == "available"),
                   # 296b: CSV projects need no SQL sign-in.
                   "signed_in": None if project["source"] == "csv"
                   else sessions.is_signed_in(principal)}
        checks = [{"code": code, "passed": outcome[code], "blocking": True,
                   "message": passed if outcome[code] else failed}
                  for code, passed, failed in PREFLIGHT if outcome[code] is not None]
        return checks, all(c["passed"] for c in checks), model

    @app.get("/ui/prediction", response_class=HTMLResponse, include_in_schema=False)
    def prediction_tab(request: Request, run: str = "", page: int = 1, dataset: str = "",
                       principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "csv_project": False, "checks": [],
                   "ready": False, "model": None, "datasets": [], "recent": [],
                   "overview": None, "selected_run": None, "stages": STAGES,
                   "terminal": TERMINAL, "tab_messages": MESSAGES, "has_log": False,
                   "back": "/ui/prediction", "predictions": None, "chosen_dataset": ""}
        if project is not None:
            context["csv_project"] = project["source"] == "csv"
            context["checks"], context["ready"], context["model"] = preflight(
                request, principal, project)
            # 297f: CSV projects too (decision 4 changed 2026-11-10).
            try:
                context["datasets"] = [d for d in datasets.list(owner, project["project_id"])
                                       if d["role"] in ("prediction", "mixed")]
            except Exception:
                context["datasets"] = []
            context["recent"] = [r for r in runs.list_project_runs(owner, project["project_id"])
                                 if r.get("kind") == "prediction"]
            chosen = run or (context["recent"][0]["run_id"] if context["recent"] else "")
            if chosen and any(r["run_id"] == chosen for r in context["recent"]):
                try:
                    context["overview"] = runs.overview(owner, RunReference(chosen))
                    context["selected_run"] = chosen
                except Exception:
                    context["overview"] = None
                if logs is not None:
                    try:
                        logs.path(owner, chosen)
                        context["has_log"] = True
                    except Exception:
                        context["has_log"] = False
                if results is not None and context["overview"] and \
                        context["overview"]["status"] == "succeeded":
                    try:
                        context["predictions"] = results.detail(owner, chosen, "predictions",
                                                                page=page)
                    except Exception:
                        context["predictions"] = None
            # 297h: the dropdown shows the dataset of the run on show (it fell back
            # to its first option after "Start prediction"), or the one just
            # refused; the training dataset's rows are the empty choice.
            offered = {d["dataset_id"] for d in context["datasets"]}
            shown = next((r for r in context["recent"] if r["run_id"] == context["selected_run"]),
                         None)
            for candidate in (dataset, (shown or {}).get("dataset_id")):
                if candidate and candidate in offered:
                    context["chosen_dataset"] = candidate
                    break
        return render(request, principal, "prediction.html", "prediction", **context)

    @app.post("/ui/prediction", include_in_schema=False)
    def predict(request: Request, dataset_id: str = Form(""),
                principal=Depends(principal_dependency)):
        project = active_project(request, principal.principal_key)
        model_id = request.cookies.get("jbg_model", "")
        if project is None or not model_id:
            return redirect("/ui/prediction", message="model_missing")
        try:
            created = project_runs.submit_prediction(principal, project["project_id"],
                                                     model_id=model_id,
                                                     dataset_id=dataset_id or None)
        except PermissionError:
            return redirect("/ui/prediction", message="signin_needed", dataset=dataset_id)
        except RunDoesNotMatchDataset:
            return redirect("/ui/prediction", message="prediction_incompatible",
                            dataset=dataset_id)
        except IdColumnNotInteger:
            return redirect("/ui/prediction", message="id_not_integer", dataset=dataset_id)
        except NothingToPredict:
            return redirect("/ui/prediction", message="nothing_to_predict", dataset=dataset_id)
        except (RunRequestRefused, LookupError, ValueError, TypeError, OSError):
            return redirect("/ui/prediction", message="prediction_refused", dataset=dataset_id)
        return redirect("/ui/prediction", run=created["run_id"], message="prediction_started")
