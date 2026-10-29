"""Shared CSV reader/result boundary for the existing ML handlers.

Mode-specific preparation remains at the caller boundary. This class never
creates SQL connections or chooses an artifact scope implicitly.
"""


class CSVDataLayer:
    execution_mode = None  # Only prepared, mode-bound facades may execute.
    def __init__(self, reader, sink=None):
        if not callable(getattr(reader, "get_dataset", None)):
            raise ValueError("Invalid CSV data reader")
        if sink is not None and not callable(getattr(sink, "save_prediction_data", None)):
            raise ValueError("Invalid CSV result sink")
        self._reader = reader
        self._sink = sink

    @property
    def correction_revision(self):
        return self._reader.correction_revision

    def get_dataset(self, num_rows=None):
        return self._reader.get_dataset(num_rows=num_rows)



class CSVResultDataLayer(CSVDataLayer):
    """Explicit file-output capability, never granted to train-only readers."""

    def save_prediction_data(self, results, class_rate_type, model_name,
                             class_labels, create_tables=True, test_run=0):
        if self._sink is None:
            raise ValueError("CSV result sink is not prepared")
        return self._sink.save_prediction_data(
            results, class_rate_type, model_name, class_labels,
            create_tables=create_tables, test_run=test_run)

    def get_mispredicted_export(self):
        if self._sink is None or not callable(getattr(self._sink, "get_mispredicted_export", None)):
            raise ValueError("CSV mispredicted export is unavailable")
        return self._sink.get_mispredicted_export()

    def save_mispredicted_data(self, frame):
        if self._sink is None or not callable(getattr(self._sink, "save_mispredicted_data", None)):
            raise ValueError("CSV mispredicted result sink is not prepared")
        return self._sink.save_mispredicted_data(frame)


def validate_csv_execution(datalayer, handler, config, tasks, regression_suite,
                           get_expected_tasks):
    """Authorize a prepared CSV I/O facade before shared ML task dispatch.

    This does not mutate the task list or the shared runner. The exact layer
    identity and mode restrictions are deliberate fail-closed boundaries.
    """
    mode = ('combined' if config.should_train() and config.should_predict()
            else 'train' if config.should_train() else 'predict')
    if (regression_suite or not isinstance(datalayer, CSVDataLayer)
            or getattr(datalayer, 'execution_mode', None) != mode
            or getattr(handler, 'datalayer', None) is not datalayer):
        if mode == 'combined':
            raise ValueError('CSV combined execution requires a prepared training and shared data layer')
        if mode == 'train':
            raise ValueError('CSV training requires a prepared training data layer')
        raise ValueError('CSV prediction data-layer mismatch: prepared prediction facade required')

    if mode in ('combined', 'train'):
        if not callable(get_expected_tasks):
            raise ValueError("CSV task sequence validator unavailable")
        expected = get_expected_tasks(config, regression_suite=False)
        if not isinstance(tasks, (list, tuple)) or list(tasks) != expected:
            raise ValueError('Unsupported CSV ' + mode + ' task sequence')
        return

    # Saved-model prediction has no training, diagnostic or regression tasks.
    allowed = (
        'load_model', 'fetch_data', 'load_dataset', 'separate_dataset',
        'convert_to_numbers', 'make_predictions', 'send_completetion_email',
    )
    if (not isinstance(tasks, (list, tuple)) or not tasks
            or any(not isinstance(task, str) for task in tasks)
            or len(tasks) != len(set(tasks))
            or any(task not in allowed for task in tasks)
            or tuple(tasks) != tuple(task for task in allowed if task in tasks)
            or not all(task in tasks for task in (
                'load_model', 'fetch_data', 'load_dataset',
                'separate_dataset', 'make_predictions'))):
        raise ValueError('Unsupported CSV prediction task sequence')
    if ('convert_to_numbers' in tasks) != bool(config.get_text_column_names()):
        raise ValueError('CSV prediction conversion task mismatch')
