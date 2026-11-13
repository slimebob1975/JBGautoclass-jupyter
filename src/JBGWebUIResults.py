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


# 299a2 (asked 2026-11-12): the engine's English column names in the result
# tables, shown in the interface's language. Class names, P(<class>) and
# feature names are data and stay as they are.
HEADINGS = {
    "Actual": "Actual class", "Predicted": "Predicted class", "Model": "Model",
    "Rate": "Certainty",                                         # 299c: predictions
    "Oversampling": "Oversampling", "Undersampling": "Undersampling", "Preprocessor": "Scaling",
    "Feature Reduction": "Dimensionality reduction", "Algorithm (Library)": "Algorithm (library)",
    "Components": "Components", "Mean cv": "Mean CV score", "Stdev.": "Standard deviation",
    "CV target": "CV target class", "CV target precision": "CV precision",
    "CV target recall": "CV recall", "CV target F1": "CV F1",
    "CV predicted positives": "CV predicted positives", "CV target support": "CV target class rows",
    "CV validation rows": "CV validation rows", "CV diagnostic status": "CV diagnostic status",
    "Elapsed Time": "Time (s)", "Exception": "Error",
    "Rank": "Rank", "Feature": "Feature", "Mean score decrease": "Mean score decrease",
    "Std. across repeats": "Standard deviation across repeats",
}
# The mispredictions' Model column (JBGHandler.PredictionsHandler).
MODEL_VALUES = {"cross trained": "Cross-trained model", "retrained": "Final model",
                "no model": "No model"}


def headings(columns, translate, names=None) -> list:
    """The columns' headings in the interface's language (unknown names unchanged)."""
    import re
    names = HEADINGS if names is None else names
    result = []
    for column in columns:
        repeat = re.fullmatch(r"Repeat (\d+)", str(column))
        if repeat:
            result.append(f"{translate('Repeat')} {repeat.group(1)}")
        else:
            result.append(translate(names[column]) if column in names else column)
    return result


# The engine's names for its estimate blocks ("D_cv_test - Cross-trained model").
_ESTIMATE_PREFIXES = (("D_re_full", "D_retrained_full"), ("D_cv_full", "D_cv_full"),
                      ("D_comb", "Combined"), ("D_cv_test", "D_cv_test"))
_CORRECTION_MODELS = {"Cross-trained model": "Cross-trained model", "Retrained model": "Final model"}


def percent_text(value: float, language: str) -> str:
    """A share as a percentage with two decimals ("0,10 %" in Swedish)."""
    if value > 0 and value * 100 < 0.005:
        text = "< 0.01 %"
    else:
        text = f"{value * 100:.2f} %"
    return text.replace(".", ",") if language == "sv" else text


def with_percentages(summary: list, language: str) -> list:
    """299b (asked 2026-11-12): each estimate as a percentage, with its share of
    the target's largest estimate for the bar, and the range as text."""
    for interval in summary:
        values = [p["value"] for p in interval["points"] if p.get("value") is not None]
        top = max(values) if values else 0.0
        for point in interval["points"]:
            value = point.get("value")
            point["percent"] = None if value is None else percent_text(value, language)
            point["bar"] = 0.0 if value is None or top <= 0 else round(100 * value / top, 1)
        interval["range_text"] = (f"{percent_text(min(values), language)} – "
                                  f"{percent_text(max(values), language)}") if values else ""
    return summary


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


def estimate_first(detail: dict) -> dict:
    """299e (asked 2026-11-13): in the Dark Numbers confusion matrices the
    estimate ("Model type") comes first, left of the actual class (the
    engine writes the actual class first)."""
    columns = detail["columns"]
    if len(columns) >= 2 and columns[0] == "" and columns[1] == "Model type":
        detail["columns"] = [columns[1], columns[0]] + columns[2:]
        detail["rows"] = [([row[1], row[0]] + row[2:]) if len(row) >= 2 else row
                          for row in detail["rows"]]
    return detail


# 299a: reclassification messages.
MESSAGES = {
    "corrected": ("ok", "The corrections are written to the source table."),
    "corrected_csv": ("ok", "The corrections are saved in a new version of the dataset's file."),
    "correction_failed_csv": ("error", "The dataset's file could not be updated, so nothing was changed."),
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
                   "correction_source": "",
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
                        translate = translator(request_language(request))
                        detail = context["detail"]
                        detail["headings"] = headings(
                            detail["columns"], translate,
                            DARK_COLUMNS if view == "dark_numbers" else None)
                        if view == "mispredictions" and "Model" in detail["columns"]:
                            at = detail["columns"].index("Model")
                            for row in detail["rows"]:
                                if at < len(row) and row[at] in MODEL_VALUES:
                                    row[at] = translate(MODEL_VALUES[row[at]])
                    if view == "mispredictions" and reclassification is not None \
                            and reclassification.correctable(owner, chosen):
                        # 299a: SQL runs: pick the right class, review, confirm.
                        context["correctable"] = True
                        context["correction_source"] = reclassification.source(owner, chosen)
                        context["corrected"] = reclassification.current(owner, chosen)
                        context["correction_log"] = list(enumerate(reclassification.log(owner, chosen)))
                    if view == "dark_numbers":
                        # 298j: the estimates per target, then the calculation
                        # table (above) and the confusion matrices.
                        translate = translator(request_language(request))
                        readable_dark_table(context["detail"], translate)
                        context["dark_summary"] = with_percentages(
                            results.dark_number_summary(owner, chosen), request_language(request))
                        try:
                            context["dark_matrix"] = estimate_first(readable_dark_table(
                                results.detail(owner, chosen, "dark_numbers_matrix",
                                               page_size=500), translate))
                        except RunResultsUnavailable:
                            context["dark_matrix"] = None
                except RunResultsUnavailable:
                    context["detail_missing"] = view != "overview"
        context["tab_messages"] = MESSAGES
        return context

    if reclassification is None or redirect is None:
        return
    from JBGReclassification import CorrectionFailed, CorrectionRefused

    def is_csv(principal, run_id) -> bool:
        try:
            return reclassification.source(principal.principal_key, run_id) == "csv"
        except CorrectionRefused:
            return False

    def failed_message(principal, run_id) -> str:
        return "correction_failed_csv" if is_csv(principal, run_id) else "correction_failed"

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
        context["correction_source"] = reclassification.source(principal.principal_key, run_id)
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
                            message=failed_message(principal, run_id))
        return redirect("/ui/results", run=run_id, view="mispredictions",
                        message="corrected_csv" if is_csv(principal, run_id) else "corrected")

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
                            message=failed_message(principal, run_id))
        return redirect("/ui/results", run=run_id, view="mispredictions", message="undone")
