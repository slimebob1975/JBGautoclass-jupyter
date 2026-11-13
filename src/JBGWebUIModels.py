"""Models tab of the web interface (revision 295b).

The project's trained models (the 288c catalog) and a model card: where the
model came from (dataset, configuration version, run), its key figures and a
link to the run's results. "Choose model" puts an available model into the
context bar, ready for predicting with it (295c). No file paths are shown.
"""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse

MODEL_COOKIE = "jbg_model"
STATES = {"registered": "Registered", "available": "Available", "missing": "Missing",
          "retired": "Retired"}
MESSAGES = {
    "chosen": ("ok", "Model chosen."),
    "not_available": ("error", "Only available models can be chosen."),
    "unavailable": ("error", "The model is no longer available."),
}


def add_models_tab(app, *, principal_dependency, render, redirect, active_project, models,
                   runs, datasets) -> None:
    from JBGApplicationContracts import RunReference
    from JBGModelService import ModelUnavailable
    from JBGProjectService import ProjectUnavailable

    def card(principal, project, model):
        """Origin and key figures of one model; parts that are gone stay empty."""
        owner, pid = principal.principal_key, project["project_id"]
        info = {"dataset": None, "run": None, "evaluation": None, "predictions": []}
        if model.get("dataset_id"):
            try:
                info["dataset"] = datasets.get(owner, pid, model["dataset_id"])
            except Exception:
                info["dataset"] = None
        listed = runs.list_project_runs(owner, pid, limit=200)
        # 299e: the model's remaining prediction runs (deleted ones are not listed).
        info["predictions"] = [r for r in listed if r.get("kind") == "prediction"
                               and r.get("model_id") == model["model_id"]]
        if model.get("run_id"):
            info["run"] = next((r for r in listed if r["run_id"] == model["run_id"]), None)
            try:
                results = runs.overview(owner, RunReference(model["run_id"]))["results"]
                info["evaluation"] = (results or {}).get("evaluation")
            except Exception:
                info["evaluation"] = None
        return info

    @app.get("/ui/models", response_class=HTMLResponse, include_in_schema=False)
    def models_tab(request: Request, model: str = "", principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        context = {"no_project": project is None, "models": [], "selected": None,
                   "card": None, "states": STATES, "chosen_id": request.cookies.get(MODEL_COOKIE),
                   "tab_messages": MESSAGES}
        if project is not None:
            try:
                context["models"] = models.list(owner, project["project_id"])
            except ProjectUnavailable:
                context["models"] = []
            wanted = model or context["chosen_id"] or ""
            selected = next((m for m in context["models"] if m["model_id"] == wanted), None)
            if selected is None and context["models"]:
                selected = context["models"][0]
            if selected is not None:
                context["selected"] = selected
                context["card"] = card(principal, project, selected)
        return render(request, principal, "models.html", "models", **context)

    @app.post("/ui/models/{model_id}/choose", include_in_schema=False)
    def choose(request: Request, model_id: str, principal=Depends(principal_dependency)):
        owner = principal.principal_key
        project = active_project(request, owner)
        try:
            if project is None:
                raise ModelUnavailable("model unavailable")
            entry = models.get(owner, project["project_id"], model_id)
        except (ModelUnavailable, ProjectUnavailable):
            return redirect("/ui/models", message="unavailable")
        if entry["state"] != "available":
            return redirect("/ui/models", model=model_id, message="not_available")
        response = redirect("/ui/models", model=model_id, message="chosen")
        response.set_cookie(MODEL_COOKIE, model_id, httponly=True, samesite="lax", path="/ui")
        return response
