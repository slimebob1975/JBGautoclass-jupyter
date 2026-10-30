"""GUI-facing training-only service for original-column feature investigations."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
import socket
from contextlib import contextmanager
import pickle
from datetime import datetime
from pathlib import Path


from threadpoolctl import threadpool_limits

from JBGCVReporting import partition_signature
from JBGFeatureSearch import (SearchOptions, describe_feature_search_parallelism,
                              run_experiment)
from JBGHandler import JBGHandler
from JBGFeaturePreservation import validate_investigation_feature_preservation
from SQLDataLayer import DataLayer


def text_options_from_config(config) -> dict:
    """Build the fold-local converter settings used by the ordinary JBG representation."""
    return dict(
        text_columns=tuple(config.get_text_column_names()),
        category_columns=tuple(config.get_categorical_text_column_names()),
        limit_categorize=30,
        language="sv",
        stop_words=config.use_stop_words(),
        df=1.0,
        ngram_range=config.get_ngram_range().ngram_range,
        use_encryption=config.should_hex_encode(),
        use_categorization=config.use_categorization(),
    )



@contextmanager
def _exclusive_investigation(output):
    """Block duplicate active jobs, including after GUI reconnection."""
    lock = Path(output) / 'investigation.lock'
    token = {'pid': os.getpid(), 'hostname': socket.gethostname()}
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                previous = json.loads(lock.read_text(encoding='utf-8'))
                if previous.get('hostname') != token['hostname']:
                    raise RuntimeError('Investigation lock belongs to another machine; inspect before resuming')
                os.kill(int(previous['pid']), 0)
            except ProcessLookupError:
                # Kernel exited without cleanup; its lock is stale.
                lock.unlink(missing_ok=True)
                continue
            except (ValueError, OSError) as ex:
                raise RuntimeError('Cannot verify investigation lock owner; inspect the checkpoint before resuming') from ex
            raise RuntimeError('Investigation is already active in another JBG kernel/process')
        else:
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as writer:
                    json.dump(token, writer)
                    writer.flush()
                    os.fsync(writer.fileno())
                yield
            finally:
                lock.unlink(missing_ok=True)
            return
    raise RuntimeError('Unable to acquire investigation checkpoint lock')


def run_training_feature_search(config, logger, options: SearchOptions, on_event=None, output=None,
                                resume=False, pause_event=None, investigation_mode="selection_plus_diagnostics", subgroup_column=None,
                                csv_source=None):
    """Fetch and split data exactly once, then expose only original training rows to the search.

    The normal JBG holdout remains reserved. This service never runs the classifier task runner,
    saves a model, writes generated application state, changes Repeat Last, or writes predictions.
    """
    if not resume and not config.should_train():
        raise ValueError("Feature-selection investigation requires a training configuration")
    if not resume:
        options.validate()

    investigation_logger = _InvestigationLogger(logger)
    base_output = Path(config.script_path).parent / "output" if not resume else None
    if resume:
        if output is None:
            raise ValueError('Resume requires an investigation checkpoint directory')
        output = Path(output)
        snapshot = output / 'development_snapshot.pkl'
        checksum_path = output / 'development_snapshot.sha256'
        if not snapshot.is_file() or not checksum_path.is_file():
            raise ValueError('Missing immutable development snapshot; cannot safely resume')
        data = snapshot.read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum_path.read_text(encoding='ascii').strip():
            raise ValueError('Development snapshot checksum mismatch')
        # Pickle checkpoints are local, application-created trusted files only.
        record = pickle.loads(data)
        X, y, reserved_signature, context = (record[k] for k in
                                               ('X', 'y', 'holdout_signature', 'context'))
        source_sql_schema = record.get('source_sql_schema')
        # The immutable saved snapshot, not a newly initialized GUI or SQL sample,
        # is authoritative when recovering an existing investigation.
        text_options = record['text_options']
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        saved_options = summary['options']
        parameters = {field.name: saved_options[field.name] for field in fields(SearchOptions)}
        for key in ('sizes', 'non_linear_hidden_layers'):
            parameters[key] = tuple(parameters[key])
        options = SearchOptions(**parameters)
        options.validate()
        target = record['target']
        investigation_mode = summary.get('investigation_mode', 'selection_plus_diagnostics')
        subgroup_column = summary.get('subgroup_column')
    else:
        if csv_source is None:
            datalayer = DataLayer(config=config, logger=investigation_logger)
            source_sql_schema = datalayer.get_table_columns(
                config.connection.data_catalog, config.connection.data_table
            )
        else:
            from Storage.CSVTrainingAdapter import CSVTrainingAdapter
            from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
            from Storage.CSVModelConfiguration import load_csv_model_configuration
            repository, scope, configuration_id = csv_source
            saved = load_csv_model_configuration(repository, scope, configuration_id)
            source = config.data_source
            if (getattr(source, 'kind', None) != 'csv'
                    or source.dataset_id != saved.dataset_id
                    or source.artifact_id != saved.artifact_id):
                raise ValueError('Feature investigation CSV dataset identity mismatch')
            datalayer = CSVTrainingDataLayer(
                CSVTrainingAdapter(repository, scope, configuration_id, config))
            source_sql_schema = None
        rows = datalayer.get_dataset()
        if rows is None:
            raise ValueError("No data was fetched for feature-selection investigation")

        handler = JBGHandler(datalayer, config, investigation_logger)
        dataset = handler.add_handler("dataset")
        dataset.load_data(rows)
        # The shared DatasetHandler may coerce non-numeric values into NaN.
        # Validate the original raw records before its all-NaN column drop;
        # this protects SQL and CSV investigations without changing ordinary training.
        validate_investigation_feature_preservation(
            rows, config.get_column_names(), config, dataset.dataset)
        dataset.separate_dataset()
        dataset.split_dataset_for_training_and_validation()

        X = dataset.X_train.copy(deep=True)
        y = dataset.Y_train.copy(deep=True)
        if X.empty or y.empty:
            raise ValueError("Feature-selection investigation has no labelled training rows")
        if source_sql_schema is not None:
            missing_sql_schema = [column for column in X.columns if column not in source_sql_schema]
            if missing_sql_schema:
                raise ValueError(
                    f"Feature-selection SQL schema is missing {len(missing_sql_schema)} investigated predictor column(s)"
                )
            source_sql_schema = {column: source_sql_schema[column] for column in X.columns}
        classes = sorted({str(value) for value in y})
        if len(classes) != 2:
            raise ValueError(
                "Feature-selection investigation currently requires exactly two classes "
                f"(found {len(classes)}: {', '.join(classes)}). "
                "Ordinary JBG training supports multiclass data, but this investigative "
                "feature-search backend does not yet."
            )

        target = config.get_dark_number_target()
        if not target:
            counts = Counter(y)
            target = min(counts, key=lambda value: (counts[value], str(value)))

        text_options = text_options_from_config(config)
        output = Path(output or (base_output / ("feature_search_gui_" +
                                datetime.now().strftime("%Y%m%d_%H%M%S_%f"))))
        reserved_signature = partition_signature(dataset.X_validation.index, dataset.Y_validation)
    if not resume:
        context = dict(
            project=config.name,
            source_kind="csv" if csv_source is not None else "sql",
            source_tables=([f"dataset:{saved.dataset_id}:artifact:{saved.artifact_id}:correction:{datalayer.correction_revision or 'original'}"]
                           if csv_source is not None else
                           [f"{config.connection.data_catalog}.{config.connection.data_table}"]),
            class_column=config.get_class_column_name(),
            id_column=config.get_id_column_name(),
            excluded_columns=[],
            group_column="",
        )

    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        record = dict(X=X, y=y, holdout_signature=reserved_signature, context=context,
                      source_sql_schema=source_sql_schema, text_options=text_options, target=target)
        snapshot = output / 'development_snapshot.pkl'
        # Atomic creation avoids a partially written checkpoint after a crash.
        temp = snapshot.with_suffix('.tmp')
        data = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        temp.write_bytes(data)
        os.replace(temp, snapshot)
        (output / 'development_snapshot.sha256').write_text(hashlib.sha256(data).hexdigest(),
                                                              encoding='ascii')
    parallelism_message = describe_feature_search_parallelism(
        options.max_evaluations, options.parallel_workers
    )
    if logger is not None and getattr(logger, "log_file", None) is not None:
        logger.log_file.write_message(parallelism_message)

    with _exclusive_investigation(output):
        with threadpool_limits(limits=1):
            result = run_experiment(
                X,
                y,
                target,
                output,
                options=options,
                text_options=text_options,
                groups=None,
                reserved_holdout_signature=partition_signature(
                    dataset.X_validation.index, dataset.Y_validation
                ) if not resume else reserved_signature,
                progress=None,
                on_event=on_event,
                study_context=context,
                source_sql_schema=source_sql_schema,
                resume=resume,
                pause_event=pause_event,
                investigation_mode=investigation_mode, subgroup_column=subgroup_column,
            )
    result["output_directory"] = str(output)
    return result

class _InvestigationLogger:
    """Forward diagnostics without touching the ordinary Start progress widgets."""

    def __init__(self, logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def print_progress(self, message=None, percent=None):
        return None

    def update_progress(self, percent=None, message=None):
        return percent
