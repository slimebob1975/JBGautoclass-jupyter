"""Prepared CSV read/write facade for a future combined train-and-predict run.

This module does not authorize, enable or dispatch combined execution. In
particular, the current train-only runtime guard remains in place until the
full task sequence is integration-tested against a pinned source snapshot.
"""
from Storage.ArtifactContract import StorageKind, StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.CSVPredictionResults import CSVPredictionResultSink
from Storage.CSVRunContext import prepare_csv_run
from Storage.CSVTrainingAdapter import CSVTrainingAdapter


class CSVCombinedDataLayer:
    """One immutable CSV row reader and a same-scope file prediction sink."""

    def __init__(self, reader, sink):
        if not isinstance(reader, CSVTrainingAdapter) or not isinstance(sink, CSVPredictionResultSink):
            raise ValueError('Combined CSV requires validated reader and result sink')
        if (reader.repository is not sink.repository or reader.scope != sink.scope
                or reader.configuration.dataset_id != sink.context.dataset_id
                or reader.configuration.artifact_id != sink.context.artifact_id):
            raise ValueError('Combined CSV reader/sink scope mismatch')
        self._reader = reader
        self._sink = sink

    @property
    def correction_revision(self):
        return self._reader.correction_revision

    def read_partitions(self, num_rows=None):
        """Read one pinned snapshot and separate labelled from unlabelled rows.

        Both partitions are derived from the same read, never from separately
        observed correction heads. This is a preflight API, not TaskRunner dispatch.
        """
        from collections import Counter

        rows = self._reader.get_dataset(num_rows=num_rows)
        if not isinstance(rows, (list, tuple)):
            raise ValueError('Combined CSV reader returned invalid rows')
        training, prediction = [], []
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) < 2:
                raise ValueError('Combined CSV reader returned invalid row')
            label = row[-1]
            if label is None or str(label).strip().casefold() in ('', 'null'):
                prediction.append(tuple(row))
            else:
                training.append(tuple(row))
        counts = Counter(str(row[-1]).strip() for row in training)
        if len(counts) < 2 or min(counts.values()) < 2:
            raise ValueError('Combined CSV needs two labelled classes with two rows each')
        if not prediction:
            raise ValueError('Combined CSV needs unlabelled prediction rows')
        return tuple(training), tuple(prediction)

    def get_dataset(self, num_rows=None):
        return self._reader.get_dataset(num_rows=num_rows)

    def save_mispredicted_data(self, frame):
        return self._sink.save_mispredicted_data(frame)

    def save_prediction_data(self, results, class_rate_type, model_name,
                             class_labels, create_tables=True, test_run=0):
        return self._sink.save_prediction_data(
            results, class_rate_type, model_name, class_labels,
            create_tables=create_tables, test_run=test_run)


def prepare_csv_combined_datalayer(config, repository, scope, configuration_id):
    """Validate identical registered identity and roles before creating I/O."""
    if not isinstance(scope, StorageScope):
        raise ValueError('Combined CSV requires an authorized scope')
    source = getattr(config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv' or not config.should_train()
            or not config.should_predict()):
        raise ValueError('Combined CSV requires train and predict mode')
    saved = load_csv_model_configuration(repository, scope, configuration_id)
    if (getattr(source, 'dataset_id', None) != saved.dataset_id
            or getattr(source, 'artifact_id', None) != saved.artifact_id
            or config.get_id_column_name() != saved.id_column
            or config.get_class_column_name() != saved.class_column
            or tuple(config.get_column_names()) !=
            tuple(saved.text_columns) + tuple(saved.data_columns) +
            (saved.id_column, saved.class_column)):
        raise ValueError('Combined CSV source or column roles mismatch')
    reader = CSVTrainingAdapter(repository, scope, configuration_id, config)
    context = prepare_csv_run(repository, scope, saved.artifact_id,
                              id_column=saved.id_column, class_column=saved.class_column,
                              feature_columns=tuple(saved.text_columns) + tuple(saved.data_columns))
    if (context.storage_kind is not StorageKind.FILES
            or context.dataset_id != saved.dataset_id
            or context.artifact_id != saved.artifact_id):
        raise ValueError('Combined CSV run context mismatch')
    return CSVCombinedDataLayer(reader, CSVPredictionResultSink(repository, scope, context))
