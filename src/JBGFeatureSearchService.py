"""GUI-facing training-only service for original-column feature investigations."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from threadpoolctl import threadpool_limits

from JBGCVReporting import partition_signature
from JBGFeatureSearch import SearchOptions, run_experiment
from JBGHandler import JBGHandler
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


def run_training_feature_search(config, logger, options: SearchOptions, on_event=None, output=None):
    """Fetch and split data exactly once, then expose only original training rows to the search.

    The normal JBG holdout remains reserved. This service never runs the classifier task runner,
    saves a model, writes generated application state, changes Repeat Last, or writes predictions.
    """
    if not config.should_train():
        raise ValueError("Feature-selection investigation requires a training configuration")
    options.validate()

    investigation_logger = _InvestigationLogger(logger)
    datalayer = DataLayer(config=config, logger=investigation_logger)
    rows = datalayer.get_dataset()
    if rows is None:
        raise ValueError("No data was fetched for feature-selection investigation")

    handler = JBGHandler(datalayer, config, investigation_logger)
    dataset = handler.add_handler("dataset")
    dataset.load_data(rows)
    dataset.separate_dataset()
    dataset.split_dataset_for_training_and_validation()

    X = dataset.X_train.copy(deep=True)
    y = dataset.Y_train.copy(deep=True)
    if X.empty or y.empty:
        raise ValueError("Feature-selection investigation has no labelled training rows")
    classes = sorted(set(y), key=str)
    if len(classes) < 2:
        raise ValueError("Feature-selection investigation requires at least two labelled classes")

    target = config.get_dark_number_target()
    if target not in classes:
        counts = Counter(y)
        target = min(counts, key=lambda value: (counts[value], str(value)))

    output = Path(output or (
        Path(config.script_path).parent / "output" /
        ("feature_search_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    ))
    context = dict(
        project=config.name,
        source_kind="sql",
        source_tables=[f"{config.connection.data_catalog}.{config.connection.data_table}"],
        class_column=config.get_class_column_name(),
        id_column=config.get_id_column_name(),
        excluded_columns=[],
        group_column="",
    )

    with threadpool_limits(limits=1):
        result = run_experiment(
            X,
            y,
            target,
            output,
            options=options,
            text_options=text_options_from_config(config),
            groups=None,
            reserved_holdout_signature=partition_signature(
                dataset.X_validation.index, dataset.Y_validation
            ),
            progress=None,
            on_event=on_event,
            study_context=context,
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
