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
         ("mispredictions", "Mispredictions"), ("feature_importance", "Feature importance"),
         ("dark_numbers", "Dark Numbers"))                                          # 298j


# 298j: the engine's Dark Number estimates (JBGDarkNumberReporting.INTERVAL_SCOPES),
# in their expected order from smallest to largest.
DARK_ESTIMATES = {"D_re_full": "Final model, all data", "D_cv_full": "Cross-trained model, all data",
                  "D_comb": "Combined", "D_cv_test": "Cross-trained model, test data"}
DARK_REASONS = {"unestimable": "cannot be estimated", "missing": "missing",
                "ambiguous": "ambiguous", "non-finite": "not a number",
                "unknown correction": "unknown correction",
                "inconsistent zero-FP result": "inconsistent"}
DARK_COLUMNS = {"Model type": "Estimate", "type": "Calculation", "target": "Target class",
                "alphas": "Alpha", "corr": "Correction (c)", "corr_model": "Correction from",
                "corr_source": "Basis of the correction", "dark_number": "Dark Number"}


# The engine's names for its estimate blocks ("D_cv_test - Cross-trained model").
_ESTIMATE_PREFIXES = (("D_re_full", "D_retrained_full"), ("D_cv_full", "D_cv_full"),
                      ("D_comb", "Combined"), ("D_cv_test", "D_cv_test"))
_CORRECTION_MODELS = {"Cross-trained model": "Cross-trained model", "Retrained model": "Final model"}


def readable_dark_table(detail: dict, translate) -> dict:
    """298j: the engine's Dark Number tables with readable estimate and model
    names and calculation types. Only these known columns are changed, never a
    class name."""
    columns = detail["columns"]
    for row in detail["rows"]:
        for index, column in enumerate(columns):
            value = row[index] if index < len(row) else ""
            if column == "Model type" and value:
                key = next((k for k, prefix in _ESTIMATE_PREFIXES if str(value).startswith(prefix)), None)
                row[index] = translate(DARK_ESTIMATES[key]) if key else value
            elif column == "corr_model" and value in _CORRECTION_MODELS:
                row[index] = translate(_CORRECTION_MODELS[value])
            elif column == "type" and isinstance(value, str):
                row[index] = value.replace("_", " ")
    return detail


# 299a: reclassification messages.
MESSAGES = {
    "corrected": ("ok", "The corrections are written to the source table."),
    "undone": ("ok", "The correction is undone: the old class is written back."),
    "correction_refused": ("error", "Nothing was corrected: choose a new class for at least one row."),
    "correction_failed": ("error", "The source table could not be updated, so nothing was written. "
                          "Check that your SQL sign-in may change the table."),
    "signin_needed": ("error", "Sign in to SQL Server first (Dataset tab)."),
}


def add_results_tab(app, *, principal_dependency, render, active_project, runs, results,
                    reclassification=None, redirect=None) -> None:
    from fastapi import Form
    from JBGApplicationContracts import RunReference
    from JBGRunResults import RunResultsUnavailable
    from JBGWebUI import request_language, translator

    @app.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def results_tab(request: Request, run: str = "", view: str = "overview", page: int = 1,
                    principal=Depends(principal_dependency)):
        return render(request, principal, "results.html", "results",
                      **context_for(request, principal, run, view, page))

    def context_for(request, principal, run, view, page):
        owner = principal.principal_key
        project = active_project(request, owner)
        view = view if view in dict(VIEWS) else "overview"
        context = {"no_project": project is None, "finished": [], "selected_run": None,
                   "views": VIEWS, "view": view, "overview": None, "labels": [],
                   "detail": None, "detail_missing": False, "available": {},
                   "has_folder": False, "dark_summary": [], "dark_matrix": None,
                   "correctable": False, "corrected": {}, "correction_log": [], "review": None,
                   "dark_estimates": DARK_ESTIMATES, "dark_reasons": DARK_REASONS,
                   "dark_columns": DARK_COLUMNS}
        if project is not None:
            # 296a: results belong to training runs; prediction runs are on the
            # Prediction tab.
            context["finished"] = [r for r in runs.list_project_runs(owner, project["project_id"])
                                   if r["status"] == "succeeded"
                                   and r.get("kind", "training") != "prediction"]
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
                    if view == "mispredictions" and reclassification is not None \
                            and reclassification.correctable(owner, chosen):
                        # 299a: SQL runs: pick the right class, review, confirm.
                        context["correctable"] = True
                        context["corrected"] = reclassification.current(owner, chosen)
                        context["correction_log"] = list(enumerate(reclassification.log(owner, chosen)))
                    if view == "dark_numbers":
                        # 298j: the estimates per target, then the calculation
                        # table (above) and the confusion matrices.
                        translate = translator(request_language(request))
                        readable_dark_table(context["detail"], translate)
                        context["dark_summary"] = results.dark_number_summary(owner, chosen)
                        try:
                            context["dark_matrix"] = readable_dark_table(results.detail(
                                owner, chosen, "dark_numbers_matrix", page_size=500), translate)
                        except RunResultsUnavailable:
                            context["dark_matrix"] = None
                except RunResultsUnavailable:
                    context["detail_missing"] = view != "overview"
        context["tab_messages"] = MESSAGES
        return context

    if reclassification is None or redirect is None:
        return
    from JBGReclassification import CorrectionFailed, CorrectionRefused

    def chosen_classes(form) -> dict:
        return {key[len("corr:"):]: value for key, value in form.multi_items()
                if key.startswith("corr:") and value}

    @app.post("/ui/results/{run_id}/corrections/review", response_class=HTMLResponse,
              include_in_schema=False)
    async def review_corrections(request: Request, run_id: str, page: int = 1,
                                 principal=Depends(principal_dependency)):
        """299a: the corrections to confirm; nothing is written yet."""
        form = await request.form()
        try:
            changes = reclassification.review(principal.principal_key, run_id, chosen_classes(form))
        except CorrectionRefused:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="correction_refused")
        context = context_for(request, principal, run_id, "mispredictions", page)
        context["review"] = changes
        return render(request, principal, "results.html", "results", **context)

    @app.post("/ui/results/{run_id}/corrections", include_in_schema=False)
    async def apply_corrections(request: Request, run_id: str,
                                principal=Depends(principal_dependency)):
        form = await request.form()
        try:
            reclassification.apply(principal, run_id, chosen_classes(form))
        except CorrectionRefused:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="correction_refused")
        except PermissionError:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="signin_needed")
        except CorrectionFailed:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="correction_failed")
        return redirect("/ui/results", run=run_id, view="mispredictions", message="corrected")

    @app.post("/ui/results/{run_id}/corrections/{index}/undo", include_in_schema=False)
    def undo_correction(run_id: str, index: int, principal=Depends(principal_dependency)):
        try:
            reclassification.undo(principal, run_id, index)
        except CorrectionRefused:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="correction_refused")
        except PermissionError:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="signin_needed")
        except CorrectionFailed:
            return redirect("/ui/results", run=run_id, view="mispredictions",
                            message="correction_failed")
        return redirect("/ui/results", run=run_id, view="mispredictions", message="undone")
