"""Opt-in CSV prediction I/O composition; does not execute an ML job.

A caller must authorize the storage scope before using this boundary.  The
existing classifier and prediction handler remain the sole ML implementation.
"""

from Storage.ArtifactContract import StorageKind, StorageScope
from Storage.CSVModelRuntime import prepare_csv_model_runtime
from Storage.CSVPredictionResults import CSVPredictionResultSink
from Storage.CSVRunContext import prepare_csv_run


def prepare_csv_prediction_io(model_filename, runtime_config, repository,
                              scope, configuration_id):
    """Build matching CSV row reader and prediction writer for a saved model.

    Never load rows or publish results if the saved roles or dataset identity
    conflict.  No SQL adapter or task runner is instantiated by this function.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid CSV prediction scope")
    config, reader = prepare_csv_model_runtime(
        model_filename, runtime_config, repository, scope, configuration_id)
    roles = reader.configuration
    source = getattr(config, "data_source", None)
    if (getattr(source, "kind", None) != "csv"
            or getattr(source, "dataset_id", None) != roles.dataset_id
            or getattr(source, "artifact_id", None) != roles.artifact_id):
        raise ValueError("CSV prediction source identity mismatch")
    context = prepare_csv_run(
        repository, scope, roles.artifact_id,
        id_column=roles.id_column, class_column=roles.class_column,
        feature_columns=tuple(roles.text_columns) + tuple(roles.data_columns))
    if (context.storage_kind is not StorageKind.FILES
            or context.dataset_id != roles.dataset_id
            or context.artifact_id != roles.artifact_id):
        raise ValueError("CSV prediction context mismatch")
    sink = CSVPredictionResultSink(repository, scope, context)
    return config, reader, sink
