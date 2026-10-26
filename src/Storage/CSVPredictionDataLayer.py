"""Opt-in data-layer facade for CSV prediction, without task execution.

Uses the same two calls as the shared JBG dataset and prediction handlers.
The repository and scope must have been authorized by the caller.
"""

from Storage.CSVPredictionIO import prepare_csv_prediction_io


class CSVPredictionDataLayer:
    """Delegate shared data-layer calls to the prepared CSV reader and sink."""

    def __init__(self, reader, sink):
        if not callable(getattr(reader, "get_dataset", None)):
            raise ValueError("Invalid CSV prediction reader")
        if not callable(getattr(sink, "save_prediction_data", None)):
            raise ValueError("Invalid CSV prediction result sink")
        self._reader = reader
        self._sink = sink

    def get_dataset(self, num_rows=None):
        return self._reader.get_dataset(num_rows=num_rows)

    def save_prediction_data(self, results, class_rate_type, model_name,
                             class_labels, create_tables=True, test_run=0):
        # No SQL tables are created; the sink writes scoped file artifacts.
        return self._sink.save_prediction_data(
            results, class_rate_type, model_name, class_labels,
            create_tables=create_tables, test_run=test_run)


def prepare_csv_prediction_datalayer(model_filename, runtime_config,
                                     repository, scope, configuration_id):
    """Return the model config and a shared-interface CSV data layer.

    This function does not run the classifier or enable CSV in the GUI.
    """
    config, reader, sink = prepare_csv_prediction_io(
        model_filename, runtime_config, repository, scope, configuration_id)
    return config, CSVPredictionDataLayer(reader, sink)
