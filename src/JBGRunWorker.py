"""Run one stored job in its own process (revision 285b).

    python -m JBGRunWorker (--store-url URL | --store-from-env) --run RUN_ID --worker-id ID

With --store-from-env (285d) the worker reads the metadata database settings
from the repository's .env itself, so no password is passed on a command line.

The dispatcher (285c) starts one worker per job, after authorizing the run in
the web process. For SQL jobs it writes the user's credentials as one JSON line
to the worker's stdin; they are never stored. The worker claims the job, rebuilds
the run with JBGRunComposition, reports progress and heartbeats to the job store,
honours cancellation at the engine's own progress checkpoints, and stores only
the projected JSON result.

Exit codes: 0 the job was finished and its outcome recorded (succeeded, failed or
cancelled); 2 the job could not be claimed; 3 an SQL job arrived without
credentials (it stays pending); 1 unexpected worker error.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference, RunStatus
from JBGRunJobStore import RunJobStore
from JBGRunPreparationService import RunConfiguration
from JBGRunResultProjection import project_classifier_result
from Storage.ArtifactContract import StorageScope

SPEC_VERSION = 1
EXIT_FINISHED, EXIT_ERROR, EXIT_NOT_CLAIMED, EXIT_NO_CREDENTIALS = 0, 1, 2, 3


class RunCancelled(BaseException):
    """Raised at a progress checkpoint when cancellation was requested.

    A BaseException (like KeyboardInterrupt), so the engine's ``except
    Exception`` handlers cannot turn a cancellation into a skipped candidate.
    """


# --- Job specification ------------------------------------------------------------------

def build_job_spec(source: DataSourceKind, config, *, scope: StorageScope | None = None,
                   artifact_id: str | None = None,
                   configuration_id: str | None = None) -> dict:
    """Describe a run as plain JSON without secrets, for RunJobStore.create.

    The configuration is the existing schema-v2 JSON payload, with the SQL
    password (always) and username (here) blanked. CSV runs also name their trusted scope, dataset
    artifact and saved column-role configuration.
    """
    if not isinstance(source, DataSourceKind):
        raise TypeError("source must be a DataSourceKind")
    # The SQL login arrives through the worker's pipe; store neither name nor password.
    spec = {"spec_version": SPEC_VERSION, "configuration": config._config_to_json(username="")}
    if source is DataSourceKind.CSV:
        if not isinstance(scope, StorageScope):
            raise TypeError("trusted StorageScope required for CSV")
        for name, value in (("artifact_id", artifact_id), ("configuration_id", configuration_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} required for CSV")
        spec["csv"] = {"tenant_id": scope.tenant_id, "project_id": scope.project_id,
                       "artifact_id": artifact_id, "configuration_id": configuration_id}
    return spec


def _config_from_spec(spec: dict, script_path: Path | None):
    """Load the schema-v2 payload with the existing, file-based loader."""
    from Config import Config
    if spec.get("spec_version") != SPEC_VERSION:
        raise ValueError("unsupported job specification version")
    with tempfile.TemporaryDirectory(prefix="jbg_job_") as folder:
        path = Path(folder) / "configuration.json"
        path.write_text(json.dumps(spec["configuration"]), encoding="utf-8")
        config = Config.load_config_from_json(path)
    if script_path is not None:
        config.script_path = Path(script_path)
    return config


def read_credentials(stream) -> "SQLCredentials | None":
    """Read one optional JSON line {"windows": bool, "username": .., "password": ..}."""
    from JBGSQLSessions import SQLCredentials
    line = stream.readline() if stream is not None else ""
    if not line or not line.strip():
        return None
    data = json.loads(line)
    return SQLCredentials(windows=bool(data.get("windows", False)),
                          username=data.get("username", ""), password=data.get("password", ""))


# --- Progress, heartbeat and cancellation ----------------------------------------------------

class StoreProgressLogger:
    """Forward to the engine logger; record progress and check for cancellation.

    Every engine ``print_progress`` call is a checkpoint. Store errors are
    reported on stderr but never stop the training run.
    """

    def __init__(self, logger, store: RunJobStore, run: RunReference, heartbeat: "_Heartbeat"):
        self._logger = logger
        self._store = store
        self._run = run
        self._heartbeat = heartbeat
        self.evaluation = None
        self.feature_importance = None

    def record_task_stage(self, name):
        from JBGStructuredEvaluation import TASK_STAGES
        if not isinstance(name, str) or name not in TASK_STAGES:
            return
        try:
            self._store.record_progress(self._run, "jbg_stage:" + name)
        except Exception as error:
            print(f"[JBGRunWorker] stage not recorded: {type(error).__name__}", file=sys.stderr)
        self.checkpoint()

    def print_prediction_report(self, *, accuracy_score, confusion_matrix,
                                class_labels, classification_matrix, sample_rates):
        # The existing logger still owns presentation; this receives the exact
        # already-computed evaluation rather than re-parsing its text output.
        result = self._logger.print_prediction_report(
            accuracy_score=accuracy_score, confusion_matrix=confusion_matrix,
            class_labels=class_labels, classification_matrix=classification_matrix,
            sample_rates=sample_rates)
        from JBGStructuredEvaluation import evaluation_from_report
        self.evaluation = evaluation_from_report(
            accuracy_score=accuracy_score, confusion_matrix=confusion_matrix,
            class_labels=class_labels, classification_matrix=classification_matrix,
            sample_rates=sample_rates)
        return result

    def record_feature_importance(self, report):
        from JBGStructuredFeatureImportance import feature_importance_from_report
        self.feature_importance = feature_importance_from_report(report)
        self.checkpoint()

    def checkpoint(self) -> None:
        if self._heartbeat.lost:
            raise RunCancelled("worker no longer owns the job")
        try:
            cancelled = self._store.cancel_requested(self._run)
        except Exception as error:  # the database may be briefly unavailable
            print(f"[JBGRunWorker] cancel check failed: {type(error).__name__}", file=sys.stderr)
            return
        if cancelled:
            raise RunCancelled("cancellation requested")

    def print_progress(self, message=None, percent=None):
        result = self._logger.print_progress(message=message, percent=percent)
        try:
            self._store.record_progress(self._run, message, percent)
        except Exception as error:
            print(f"[JBGRunWorker] progress not recorded: {type(error).__name__}", file=sys.stderr)
        self.checkpoint()
        return result

    def __getattr__(self, name):
        return getattr(self._logger, name)


class _Heartbeat:
    """Background heartbeat, so long model fits without checkpoints look alive."""

    def __init__(self, store: RunJobStore, run: RunReference, worker_id: str, seconds: float):
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, args=(store, run, worker_id, seconds),
                                        name="jbg-worker-heartbeat", daemon=True)

    def _beat(self, store, run, worker_id, seconds):
        while not self._stop.wait(seconds):
            try:
                if not store.heartbeat(run, worker_id):
                    self.lost = True
                    return
            except Exception as error:
                print(f"[JBGRunWorker] heartbeat failed: {type(error).__name__}", file=sys.stderr)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)
        return False


# --- Running one job ------------------------------------------------------------------------

def _compose(record, config, logger, credentials, artifacts: Path | None):
    from JBGRunComposition import compose_csv_run, compose_sql_run_on_layer
    if record.source is DataSourceKind.CSV:
        from Storage.FileArtifactRepository import FileArtifactRepository
        if artifacts is None:
            raise ValueError("artifact repository path required for CSV jobs")
        csv = record.spec["csv"]
        scope = StorageScope(csv["tenant_id"], csv["project_id"])
        composed = compose_csv_run(FileArtifactRepository(artifacts), scope,
                                   csv["configuration_id"], logger)
        reference = DatasetReference(DataSourceKind.CSV, csv["artifact_id"])
        return composed, reference, config
    from JBGSQLSessions import open_sql_data_layer
    layer = open_sql_data_layer(credentials, config, logger)
    composed = compose_sql_run_on_layer(layer, logger)
    return composed, DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()), \
        composed.configuration


def run_job(store: RunJobStore, run: RunReference, worker_id: str, *, logger=None,
            credentials=None, artifacts: Path | None = None, script_path: Path | None = None,
            log_dir: Path | None = None, heartbeat_seconds: float = 30.0) -> int:
    """Claim and execute one job; return a worker exit code."""
    record = store.get(run)
    if record.source is DataSourceKind.SQL and credentials is None:
        return EXIT_NO_CREDENTIALS
    if not store.claim(run, worker_id):
        return EXIT_NOT_CLAIMED
    with _Heartbeat(store, run, worker_id, heartbeat_seconds) as heartbeat:
        try:
            config = _config_from_spec(record.spec, script_path)
            own_logger = logger is None
            if own_logger:
                from JBGLogger import JBGLogger
                logger = JBGLogger(quiet=not config.io.verbose, in_terminal=True, log_dir=log_dir)
            observed = StoreProgressLogger(logger, store, run, heartbeat)
            observed.checkpoint()                    # cancelled while it was queued
            composed, reference, run_config = _compose(record, config, observed,
                                                       credentials, artifacts)
            prepared = composed.preparation.prepare(
                reference, RunConfiguration(record.source, run_config))
            if own_logger:
                # As the command-line run does: persist stdout/stderr and warnings.
                with logger.capture_console_output():
                    result = composed.execution.execute(prepared)
            else:
                result = composed.execution.execute(prepared)
        except RunCancelled:
            store.finish(run, worker_id, RunStatus.CANCELLED)
            return EXIT_FINISHED
        except SystemExit:
            # Legacy logger.abort_cleanly() calls sys.exit(). This is an engine
            # failure, not a dead worker: record it now instead of leaving the
            # run RUNNING until its heartbeat expires. Never persist exit text,
            # which may include SQL or data-source details.
            traceback.print_exc(file=sys.stderr)
            store.finish(run, worker_id, RunStatus.FAILED, error_type="SystemExit")
            return EXIT_FINISHED
        except Exception as error:
            traceback.print_exc(file=sys.stderr)     # worker log only, never the store
            store.finish(run, worker_id, RunStatus.FAILED, error_type=type(error).__name__)
            return EXIT_FINISHED
        projected = project_classifier_result(result)
        if observed.evaluation is not None:
            projected["evaluation"] = observed.evaluation
        if observed.feature_importance is not None:
            projected["feature_importance"] = observed.feature_importance
        store.finish(run, worker_id, RunStatus.SUCCEEDED, result=projected)
        register_trained_model(store, record, run_config)
        return EXIT_FINISHED


def register_trained_model(store: RunJobStore, record, run_config) -> bool:
    """Record a project run's model in the catalog (288c); True if registered.

    Only project runs (``spec["project_run"]``) whose model file exists. The
    catalog accepts only succeeded runs, so this runs after ``finish``. A
    failure is logged but never turns the succeeded run into a failure: the
    model file is kept and can be registered later.
    """
    project_run = record.spec.get("project_run")
    if not isinstance(project_run, dict):
        return False
    try:
        model_file = Path(run_config.get_model_filename())
        if not model_file.is_file():
            print("[JBGRunWorker] no model file to register", file=sys.stderr)
            return False
        from JBGModelCatalog import ModelCatalog, ModelState
        catalog = ModelCatalog(store.engine, schema=store.schema)
        label = str(project_run.get("configuration_name") or "Model")[:150]
        name = f"{label} v{project_run.get('configuration_version', '?')} · {record.run.key[:8]}"
        entry = catalog.register(record.owner, project_run["project_id"], name,
                                 project_run["model_id"], dataset_id=project_run.get("dataset_id"),
                                 run_id=record.run.key)
        catalog.set_state(record.owner, project_run["project_id"], entry.model_id,
                          ModelState.AVAILABLE)
        return True
    except Exception as error:
        print(f"[JBGRunWorker] model not registered: {type(error).__name__}", file=sys.stderr)
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="JBGRunWorker", description=__doc__.splitlines()[0])
    store_source = parser.add_mutually_exclusive_group(required=True)
    store_source.add_argument("--store-url", help="SQLAlchemy URL without secrets (e.g. SQLite)")
    store_source.add_argument("--store-from-env", action="store_true",
                              help="Read JBG_METADATA_DB_* settings from .env (285d)")
    parser.add_argument("--store-schema", default=None)
    parser.add_argument("--run", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--artifacts", type=Path, default=None,
                        help="CSV artifact repository (FileArtifactRepository root)")
    parser.add_argument("--script-path", type=Path, default=None,
                        help="Base for models and output; defaults to src, as in Voila")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        if args.store_from_env:
            from JBGMetadataStore import open_metadata_store
            store = open_metadata_store()
        else:
            store = RunJobStore(args.store_url, schema=args.store_schema)
        credentials = read_credentials(sys.stdin)
        return run_job(store, RunReference(args.run), args.worker_id, credentials=credentials,
                       artifacts=args.artifacts, script_path=args.script_path,
                       log_dir=args.log_dir, heartbeat_seconds=args.heartbeat_seconds)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
