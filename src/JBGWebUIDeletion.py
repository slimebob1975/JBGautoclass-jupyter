"""Confirmation pages for deleting runs, models and configurations (revision 299e).

Each deletion opens a page listing exactly what goes and what stays, with an
orange (warning) button; nothing is deleted before it is confirmed. A
deletion that would take a queued or running run is refused, with the reason
and a red button. Owner only: other users' items are unknown. The cascade
itself is JBGDeletion's.
"""
from __future__ import annotations

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse

CONFIGURATION_COOKIE = "jbg_configuration"
MODEL_COOKIE = "jbg_model"
RUN_BACKS = {"/ui/run": "run", "/ui/prediction": "prediction", "/ui/results": "results"}
REFUSALS = {
    "running": "A run that would be deleted is queued or running. Wait for it to finish, "
               "or cancel it first.",
    "finishing": "A run that would be deleted is still being finished on the server. "
                 "Try again in a moment.",
}


def add_deletion_pages(app, *, principal_dependency, render, redirect, active_project,
                       deletion) -> None:
    from JBGDeletion import DeletionFailed, DeletionRefused, DeletionUnavailable

    def page(request, principal, plan, *, tab, action, back):
        return render(request, principal, "delete.html", tab, plan=plan, action=action,
                      back=back, refusals=REFUSALS,
                      source=(active_project(request, principal.principal_key) or {}).get("source"))

    def project_id(request, principal):
        project = active_project(request, principal.principal_key)
        return None if project is None else project["project_id"]

    def carry_out(principal, plan, back, done):
        try:
            deletion.delete(principal.principal_key, plan)
        except DeletionRefused:
            return None
        except DeletionUnavailable:
            return redirect(back, message="delete_unavailable")
        except DeletionFailed:
            return redirect(back, message="delete_failed")
        return redirect(back, message=done)

    def forget(request, response, plan):
        """Clear the context bar's configuration and model if they are deleted."""
        if plan.configuration is not None and request.cookies.get(CONFIGURATION_COOKIE) == \
                plan.configuration.configuration_id:
            response.delete_cookie(CONFIGURATION_COOKIE, path="/ui")
        if request.cookies.get(MODEL_COOKIE) in {m.model_id for m in plan.models}:
            response.delete_cookie(MODEL_COOKIE, path="/ui")
        return response

    # --- Runs -----------------------------------------------------------------------

    def run_plan(request, principal, run_id):
        pid = project_id(request, principal)
        if pid is None:
            raise DeletionUnavailable("no project")
        return deletion.plan_run(principal.principal_key, pid, run_id)

    @app.get("/ui/runs/{run_id}/delete", response_class=HTMLResponse, include_in_schema=False)
    def confirm_run(request: Request, run_id: str, back: str = "/ui/run",
                    principal=Depends(principal_dependency)):
        back = back if back in RUN_BACKS else "/ui/run"
        try:
            plan = run_plan(request, principal, run_id)
        except DeletionUnavailable:
            return redirect(back, message="delete_unavailable")
        return page(request, principal, plan, tab=RUN_BACKS[back],
                    action=f"/ui/runs/{run_id}/delete", back=back)

    @app.post("/ui/runs/{run_id}/delete", include_in_schema=False)
    def delete_run(request: Request, run_id: str, back: str = Form("/ui/run"),
                   principal=Depends(principal_dependency)):
        back = back if back in RUN_BACKS else "/ui/run"
        try:
            plan = run_plan(request, principal, run_id)
        except DeletionUnavailable:
            return redirect(back, message="delete_unavailable")
        done = carry_out(principal, plan, back, "run_deleted")
        return done or redirect(f"/ui/runs/{run_id}/delete", back=back)

    # --- Models ---------------------------------------------------------------------

    def model_plan(request, principal, model_id):
        pid = project_id(request, principal)
        if pid is None:
            raise DeletionUnavailable("no project")
        return deletion.plan_model(principal.principal_key, pid, model_id)

    @app.get("/ui/models/{model_id}/delete", response_class=HTMLResponse,
             include_in_schema=False)
    def confirm_model(request: Request, model_id: str, principal=Depends(principal_dependency)):
        try:
            plan = model_plan(request, principal, model_id)
        except DeletionUnavailable:
            return redirect("/ui/models", message="delete_unavailable")
        return page(request, principal, plan, tab="models",
                    action=f"/ui/models/{model_id}/delete",
                    back=f"/ui/models?model={model_id}")

    @app.post("/ui/models/{model_id}/delete", include_in_schema=False)
    def delete_model(request: Request, model_id: str, principal=Depends(principal_dependency)):
        try:
            plan = model_plan(request, principal, model_id)
        except DeletionUnavailable:
            return redirect("/ui/models", message="delete_unavailable")
        done = carry_out(principal, plan, "/ui/models", "model_deleted")
        if done is None:
            return redirect(f"/ui/models/{model_id}/delete")
        return forget(request, done, plan)

    # --- Configurations -------------------------------------------------------------

    def configuration_plan(request, principal, configuration_id):
        pid = project_id(request, principal)
        if pid is None:
            raise DeletionUnavailable("no project")
        return deletion.plan_configuration(principal.principal_key, pid, configuration_id)

    @app.get("/ui/configuration/{configuration_id}/delete", response_class=HTMLResponse,
             include_in_schema=False)
    def confirm_configuration(request: Request, configuration_id: str,
                              principal=Depends(principal_dependency)):
        try:
            plan = configuration_plan(request, principal, configuration_id)
        except DeletionUnavailable:
            return redirect("/ui/configuration", message="delete_unavailable")
        return page(request, principal, plan, tab="configuration",
                    action=f"/ui/configuration/{configuration_id}/delete",
                    back=f"/ui/configuration?configuration={configuration_id}")

    @app.post("/ui/configuration/{configuration_id}/delete", include_in_schema=False)
    def delete_configuration(request: Request, configuration_id: str,
                             principal=Depends(principal_dependency)):
        try:
            plan = configuration_plan(request, principal, configuration_id)
        except DeletionUnavailable:
            return redirect("/ui/configuration", message="delete_unavailable")
        done = carry_out(principal, plan, "/ui/configuration", "configuration_deleted")
        if done is None:
            return redirect(f"/ui/configuration/{configuration_id}/delete")
        return forget(request, done, plan)
