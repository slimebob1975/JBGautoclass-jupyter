"""Results tab of the web interface (revision 295a).

The project's finished runs; for the selected run an overview (key figures,
confusion matrix with class names, most important features) and its detail
tables: the spot-check leaderboard, the mispredictions (paged) and the
feature importance. Class names and row values come from the run's own
folder and are shown to its owner only. Plain links and forms.
"""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse

VIEWS = (("overview", "Overview"), ("leaderboard", "Model comparison"),
         ("mispredictions", "Mispredictions"), ("feature_importance", "Feature importance"))


def add_results_tab(app, *, principal_dependency, render, active_project, runs, results) -> None:
    from JBGApplicationContracts import RunReference
    from JBGRunResults import RunResultsUnavailable

    @app.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def results_tab(request: Request, run: str = "", view: str = "overview", page: int = 1,
                    principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        view = view if view in dict(VIEWS) else "overview"
        context = {"no_project": project is None, "finished": [], "selected_run": None,
                   "views": VIEWS, "view": view, "overview": None, "labels": [],
                   "detail": None, "detail_missing": False, "available": {},
                   "has_folder": False}
        if project is not None:
            context["finished"] = [r for r in runs.list_project_runs(owner, project["project_id"])
                                   if r["status"] == "succeeded"]
            chosen = run or (context["finished"][0]["run_id"] if context["finished"] else "")
            if any(r["run_id"] == chosen for r in context["finished"]):
                context["selected_run"] = chosen
                context["overview"] = runs.overview(owner, RunReference(chosen))
                try:
                    context["has_folder"] = results.has_folder(owner, chosen)
                    context["labels"] = results.class_labels(owner, chosen)
                    context["available"] = results.available(owner, chosen)
                    if view != "overview":
                        context["detail"] = results.detail(owner, chosen, view, page=page)
                except RunResultsUnavailable:
                    context["detail_missing"] = view != "overview"
        return render(request, principal, "results.html", "results", **context)
