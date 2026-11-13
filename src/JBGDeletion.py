"""Deleting runs, models and configurations, for their owner (revision 299e).

Agreed 2026-11-13: what owns something takes it with it.

- A run: its log (worker and engine), its result folder (results,
  predictions, correction log) and its progress events. A training run whose
  model still exists is hidden instead of removed: only its record stays,
  with the settings the model needs to predict, out of every list.
- A model: its catalog entry and model files, the prediction runs made with
  it (as runs above), and its training run's record if that run was deleted.
- A configuration: every version, the training runs made with it, their
  models, and the prediction runs made with those models.

Nothing is deleted while a run that would go is queued, running or still
being finished by its worker. SQL tables are never touched: predictions
written to SQL Server and corrections already made stay. Files go first, then
the records, so a file that cannot be removed leaves everything listed and
the deletion can be tried again.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from JBGApplicationContracts import RunReference, RunStatus
from JBGRunJobStore import TERMINAL_STATES


class DeletionUnavailable(LookupError):
    """Unknown item, another user's item, or not part of the project."""


class DeletionRefused(RuntimeError):
    """Something that would be deleted is still in use (a fixed reason code)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DeletionFailed(OSError):
    """A file could not be removed; no record was changed."""


@dataclass
class DeletionPlan:
    """Exactly what a deletion removes. ``refused`` is a reason code or None."""
    kind: str                                   # "run", "model" or "configuration"
    name: str
    project_id: str
    runs: list = field(default_factory=list)            # JobRecords removed entirely
    hidden_runs: list = field(default_factory=list)     # JobRecords whose record stays
    models: list = field(default_factory=list)          # ModelEntries removed
    kept_models: list = field(default_factory=list)     # ModelEntries that stay (run deletion)
    configuration: object = None                        # Configuration removed
    versions: int = 0
    refused: str | None = None

    @property
    def training_runs(self) -> list:
        return [r for r in self.runs + self.hidden_runs if _kind(r) != "prediction"]

    @property
    def prediction_runs(self) -> list:
        return [r for r in self.runs + self.hidden_runs if _kind(r) == "prediction"]


def _meta(record) -> dict:
    meta = record.spec.get("project_run") if isinstance(record.spec, dict) else None
    return meta if isinstance(meta, dict) else {}


def _kind(record) -> str:
    return _meta(record).get("kind", "training")


class DeletionService:
    def __init__(self, store, *, catalog, configurations, results_root: Path,
                 log_dir: Path, model_root: Path,
                 active_runs: Callable[[], tuple] | None = None) -> None:
        self._store = store
        self._catalog = catalog
        self._configurations = configurations
        self._results_root = Path(results_root)
        self._log_dir = Path(log_dir)
        self._model_root = Path(model_root)
        self._active_runs = active_runs or (lambda: ())

    # --- What belongs to what ---------------------------------------------------------

    def _project_runs(self, owner: str, project_id: str) -> list:
        """Every run of the project, deleted (hidden) records included."""
        return [r for r in self._store.list_for_owner(owner, None, include_hidden=True)
                if _meta(r).get("project_id") == project_id]

    def _models(self, owner: str, project_id: str) -> list:
        try:
            return list(self._catalog.list_for_project(owner, project_id, limit=1000))
        except KeyError:
            raise DeletionUnavailable("project unavailable") from None

    def _refusal(self, records) -> str | None:
        busy = {r.key for r in self._active_runs()}
        for record in records:
            if record.status not in TERMINAL_STATES:
                return "running"
            if record.run.key in busy:
                return "finishing"
        return None

    def _predictions_with(self, runs: list, model_ids: set) -> list:
        return [r for r in runs if _kind(r) == "prediction"
                and _meta(r).get("model_id") in model_ids]

    # --- Plans ------------------------------------------------------------------------

    def plan_run(self, owner: str, project_id: str, run_id: str) -> DeletionPlan:
        try:
            record = self._store.get_owned(owner, RunReference(run_id))
        except (KeyError, ValueError):
            raise DeletionUnavailable("run unavailable") from None
        if _meta(record).get("project_id") != project_id:
            raise DeletionUnavailable("run unavailable")
        plan = DeletionPlan("run", _meta(record).get("configuration_name") or "", project_id)
        users = [m for m in self._models(owner, project_id) if m.run_id == record.run.key]
        if _kind(record) != "prediction" and record.status is RunStatus.SUCCEEDED and users:
            plan.hidden_runs.append(record)          # its model still predicts with it
            plan.kept_models = users
        else:
            plan.runs.append(record)
        plan.refused = self._refusal([record])
        return plan

    def plan_model(self, owner: str, project_id: str, model_id: str) -> DeletionPlan:
        try:
            model = self._catalog.get_owned(owner, project_id, model_id)
        except KeyError:
            raise DeletionUnavailable("model unavailable") from None
        plan = DeletionPlan("model", model.name, project_id, models=[model])
        runs = self._project_runs(owner, project_id)
        plan.runs = self._predictions_with(runs, {model.model_id})
        hidden = self._store.hidden_runs(owner)
        plan.runs += [r for r in runs if r.run.key == model.run_id and r.run.key in hidden]
        plan.refused = self._refusal(plan.runs)
        return plan

    def plan_configuration(self, owner: str, project_id: str,
                           configuration_id: str) -> DeletionPlan:
        try:
            configuration = self._configurations.get_owned(owner, project_id, configuration_id)
        except KeyError:
            raise DeletionUnavailable("configuration unavailable") from None
        plan = DeletionPlan("configuration", configuration.name, project_id,
                            configuration=configuration, versions=configuration.version)
        runs = self._project_runs(owner, project_id)
        training = [r for r in runs if _kind(r) != "prediction"
                    and _meta(r).get("configuration_id") == configuration_id]
        keys = {r.run.key for r in training}
        plan.models = [m for m in self._models(owner, project_id) if m.run_id in keys]
        plan.runs = training + self._predictions_with(runs, {m.model_id for m in plan.models})
        plan.refused = self._refusal(plan.runs)
        return plan

    # --- Deleting ---------------------------------------------------------------------

    def delete(self, owner: str, plan: DeletionPlan) -> DeletionPlan:
        """Carry out a plan made just now (it is made again, so nothing changed
        in between is missed)."""
        fresh = {"run": lambda: self.plan_run(owner, plan.project_id,
                                              (plan.runs + plan.hidden_runs)[0].run.key),
                 "model": lambda: self.plan_model(owner, plan.project_id,
                                                  plan.models[0].model_id),
                 "configuration": lambda: self.plan_configuration(
                     owner, plan.project_id, plan.configuration.configuration_id)}[plan.kind]()
        if fresh.refused:
            raise DeletionRefused(fresh.refused)
        self._remove_files(fresh)
        for model in fresh.models:
            try:
                self._catalog.delete(owner, fresh.project_id, model.model_id)
            except KeyError:
                pass                                       # already gone
        for record in fresh.runs:
            try:
                self._store.remove(record.run)
            except KeyError:
                pass
        for record in fresh.hidden_runs:
            self._store.hide(record.run)
        if fresh.configuration is not None:
            try:
                self._configurations.delete(owner, fresh.project_id,
                                            fresh.configuration.configuration_id)
            except KeyError:
                pass
        return fresh

    def _remove_files(self, plan: DeletionPlan) -> None:
        try:
            for record in plan.runs + plan.hidden_runs:
                key = record.run.key
                _remove_tree(self._results_root, self._results_root / plan.project_id / key)
                for name in (f"run_{key}.log", f"run_{key}_engine.log"):
                    _remove_file(self._log_dir, self._log_dir / name)
                if record in plan.runs and _kind(record) != "prediction":
                    # A removed training run takes its model file with it (its
                    # model, if any, goes in the same deletion).
                    self._remove_model_files(plan.project_id, _meta(record).get("model_id"))
            for model in plan.models:
                self._remove_model_files(plan.project_id, model.artifact_key)
        except OSError as error:
            raise DeletionFailed("a file could not be removed") from error

    def _remove_model_files(self, project_id: str, stem) -> None:
        if not isinstance(stem, str) or not stem or "/" in stem or "\\" in stem or ".." in stem:
            return
        folder = self._model_root / project_id
        if not folder.is_dir() or folder.is_symlink():
            return
        name = stem + ".sav"
        for path in folder.iterdir():
            if path.name == name or path.name.startswith(name + "."):
                _remove_file(self._model_root, path)


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _remove_file(root: Path, path: Path) -> None:
    if path.is_symlink():
        raise OSError("refusing to follow a link")
    if path.is_file():
        if not _inside(root, path):
            raise OSError("outside its folder")
        path.unlink()


def _remove_tree(root: Path, path: Path) -> None:
    if path.is_symlink():
        raise OSError("refusing to follow a link")
    if path.is_dir():
        if not _inside(root, path) or path.resolve() == root.resolve():
            raise OSError("outside its folder")
        shutil.rmtree(path)
