"""Opt-in data-layer facade for CSV prediction, without task execution.

Uses the same two calls as the shared JBG dataset and prediction handlers.
The repository and scope must have been authorized by the caller.
"""

from Storage.CSVPredictionIO import prepare_csv_prediction_io
from Storage.CSVDataLayer import CSVResultDataLayer


class CSVPredictionDataLayer(CSVResultDataLayer):
    execution_mode = 'predict'
    """Validated prediction I/O facade; cannot publish training diagnostics."""

    def __init__(self, reader, sink):
        if not callable(getattr(sink, "save_prediction_data", None)):
            raise ValueError("Invalid CSV prediction result sink")
        super().__init__(reader, sink)

    save_mispredicted_data = None


def prepare_csv_prediction_datalayer(model_filename, runtime_config,
                                     repository, scope, configuration_id):
    """Return the model config and a shared-interface CSV data layer.

    This function does not run the classifier or enable CSV in the GUI.
    """
    config, reader, sink = prepare_csv_prediction_io(
        model_filename, runtime_config, repository, scope, configuration_id)
    return config, CSVPredictionDataLayer(reader, sink)
