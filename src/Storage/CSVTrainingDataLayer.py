"""Scoped, read-only CSV training data layer for the shared ML task runner.

This boundary intentionally cannot publish SQL predictions or mutate source CSV.
"""
from Storage.ArtifactContract import StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVDataLayer import CSVDataLayer


class CSVTrainingDataLayer(CSVDataLayer):
    execution_mode = 'train'
    def __init__(self, reader, diagnostic_sink=None):
        if not isinstance(reader, CSVTrainingAdapter):
            raise ValueError('Training requires a registered CSV reader')
        super().__init__(reader)
        self._diagnostic_sink = diagnostic_sink

    def get_dataset(self, num_rows=None):
        from collections import Counter
        from JBGExceptions import DatasetException

        rows = super().get_dataset(num_rows=num_rows)
        if rows is None or len(rows) == 0:
            raise DatasetException('CSV training has no labelled rows; model not saved')
        labels = [str(row[-1]).strip() for row in rows]
        counts = Counter(labels)
        if (any(not label for label in labels) or len(counts) < 2
                or min(counts.values()) < 2):
            raise DatasetException('CSV training needs at least two labelled classes with two rows each')
        return rows

    def save_mispredicted_data(self, frame):
        if self._diagnostic_sink is None:
            raise ValueError("CSV misprediction output is not enabled for this run")
        return self._diagnostic_sink.save_mispredicted_data(frame)



def prepare_csv_training_datalayer(runtime_config, repository, scope, configuration_id):
    """Validate saved roles and source, then pin the correction snapshot."""
    if not isinstance(scope, StorageScope):
        raise ValueError('CSV training requires an authorized scope')
    source = getattr(runtime_config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv'
            or not runtime_config.should_train() or runtime_config.should_predict()):
        raise ValueError('CSV training requires train-only CSV configuration')
    saved = load_csv_model_configuration(repository, scope, configuration_id)
    if (getattr(source, 'dataset_id', None) != saved.dataset_id
            or getattr(source, 'artifact_id', None) != saved.artifact_id):
        raise ValueError('Registered CSV source identity mismatch')
    reader = CSVTrainingAdapter(repository, scope, configuration_id, runtime_config)
    sink = None
    if runtime_config.should_display_mispredicted():
        from Storage.CSVPredictionResults import CSVPredictionResultSink
        from Storage.CSVRunContext import prepare_csv_run
        context = prepare_csv_run(
            repository, scope, saved.artifact_id,
            id_column=saved.id_column, class_column=saved.class_column,
            feature_columns=tuple(saved.text_columns) + tuple(saved.data_columns))
        sink = CSVPredictionResultSink(repository, scope, context)
    return CSVTrainingDataLayer(reader, diagnostic_sink=sink)
