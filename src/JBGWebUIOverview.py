"""Overview tab of the web interface (revision 295d).

A first, simple overview of the open project (design decision 7: scoped to
the active project): the project, its active training dataset, the
configuration chosen for runs, the latest training and prediction runs and
the models. Each part links to the tab where it is changed; a part that is
not there yet says so and links to the tab that fills it. Without an open
project the page is empty apart from a hint. The title in the header links
here.

Read-only: the page only shows what the other tabs keep, through the same
services, and leaves out what a service cannot give.
"""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse


def add_overview_tab(app, *, principal_dependency, render, active_project, datasets=None,
                     configurations=None, runs=None, models=None) -> None:
    from JBGApplicationContracts import RunReference
    from JBGWebUIRun import STAGES

    def listed(service, owner, project_id):
        if service is None:
            return []
        try:
            return list(service.list(owner, project_id))
        except Exception:
            return []

    def latest_runs(owner, project_id):
        """The newest training and prediction run, with the training run's key figures."""
        if runs is None:
            return None, None, None
        try:
            items = runs.list_project_runs(owner, project_id, limit=200)
        except Exception:
            return None, None, None
        training = next((r for r in items if r.get("kind", "training") != "prediction"), None)
        prediction = next((r for r in items if r.get("kind") == "prediction"), None)
        evaluation = None
        if training is not None and training["status"] == "succeeded":
            try:
                results = runs.overview(owner, RunReference(training["run_id"]))["results"]
                evaluation = (results or {}).get("evaluation")
            except Exception:
                evaluation = None
        return training, prediction, evaluation

    @app.get("/ui/overview", response_class=HTMLResponse, include_in_schema=False)
    def overview_tab(request: Request, principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "dataset_count": 0,
                   "configuration_count": 0, "latest_training": None,
                   "latest_prediction": None, "evaluation": None, "model_count": 0,
                   "available_count": 0, "newest_model": None, "stages": STAGES,
                   "has_runs": runs is not None, "has_models": models is not None,
                   "has_configurations": configurations is not None,
                   "has_datasets": datasets is not None}
        if project is not None:
            pid = project["project_id"]
            context["dataset_count"] = len(listed(datasets, owner, pid))
            context["configuration_count"] = len(listed(configurations, owner, pid))
            (context["latest_training"], context["latest_prediction"],
             context["evaluation"]) = latest_runs(owner, pid)
            items = listed(models, owner, pid)
            context["model_count"] = len(items)
            context["available_count"] = sum(1 for m in items if m.get("state") == "available")
            context["newest_model"] = items[0] if items else None
        return render(request, principal, "overview.html", "overview", **context)
