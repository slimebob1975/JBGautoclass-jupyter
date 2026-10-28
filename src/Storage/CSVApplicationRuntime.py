"""Explicit CSV prediction entry point through JBG's application classifier.

The caller must authorize the repository scope and supply a trusted local model
path. This module does not enable CSV in the legacy GUI or support training.
"""

from pathlib import Path

from Storage.ArtifactContract import StorageScope
from Storage.CSVPredictionDataLayer import (
    CSVPredictionDataLayer, prepare_csv_prediction_datalayer,
)


def prepare_csv_prediction_classifier(model_filename, runtime_config, repository,
                                      scope, configuration_id, logger):
    """Return an initialized AutomaticClassifier with verified CSV I/O.

    The prepared adapter, restored model configuration and handler registry are
    created through their regular production constructors. No SQL DataLayer is
    constructed. The caller may invoke classifier.run() when ready.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError('CSV prediction requires an authorized storage scope')
    if not isinstance(model_filename, (str, bytes)) or not model_filename:
        raise ValueError('CSV prediction requires a trusted model filename')
    source = getattr(runtime_config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv'
            or runtime_config.should_train()
            or not runtime_config.should_predict()):
        raise ValueError('CSV prediction requires prediction-only CSV configuration')

    restored_config, layer = prepare_csv_prediction_datalayer(
        model_filename, runtime_config, repository, scope, configuration_id)
    if not isinstance(layer, CSVPredictionDataLayer):
        raise ValueError('CSV prediction preparation returned a non-CSV data layer')
    # Reject an I/O facade that cannot prove the selected correction snapshot.
    # None is valid: it represents the immutable original dataset.
    reader = getattr(layer, '_reader', None)
    if reader is None or not hasattr(reader, 'correction_revision'):
        raise ValueError('CSV prediction requires a pinned correction snapshot')
    restored_source = getattr(restored_config, 'data_source', None)
    if (getattr(restored_source, 'kind', None) != 'csv'
            or restored_config.should_train()
            or not restored_config.should_predict()):
        raise ValueError('Restored model is not a CSV prediction configuration')

    # AutomaticClassifier reconstructs the checkpoint path from Config.io.
    # Bind it to the already scope-validated CSV filename, not the empty
    # training-time io.model_name in the saved configuration. An absolute
    # directory overrides script_path in Config.get_model_filename().
    checkpoint = Path(model_filename)
    restored_config.io.model_path = str(checkpoint.parent)
    restored_config.io.model_name = checkpoint.stem
    if restored_config.get_model_filename() != checkpoint:
        raise ValueError('CSV checkpoint path could not be bound')

    # Import only after repository and source validation; SQL remains untouched.
    from AutomaticClassifier import AutomaticClassifier
    classifier = AutomaticClassifier(config=restored_config, logger=logger,
                                     datalayer=layer, regression_suite=False)
    if (classifier.datalayer is not layer
            or getattr(getattr(classifier.config, 'data_source', None), 'kind', None) != 'csv'
            or getattr(classifier.handler, 'datalayer', None) is not layer):
        raise ValueError('CSV classifier lost its validated source binding')
    return classifier


def prepare_registered_csv_prediction_classifier(model_filename, runtime_config,
                                                 repository, scope, configuration_id,
                                                 logger):
    """Resolve the persisted CSV registration before entering application runtime.

    This opt-in entry point never infers identity from GUI widget values.  Model
    compatibility, the pinned correction snapshot, and application construction
    remain enforced by prepare_csv_prediction_classifier.  This does not run ML.
    """
    from Storage.CSVModelConfiguration import load_csv_model_configuration

    if not isinstance(scope, StorageScope):
        raise ValueError('CSV prediction requires an authorized storage scope')
    source = getattr(runtime_config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv'
            or runtime_config.should_train()
            or not runtime_config.should_predict()):
        raise ValueError('CSV prediction requires prediction-only CSV configuration')
    saved = load_csv_model_configuration(repository, scope, configuration_id)
    if (getattr(source, 'dataset_id', None) != saved.dataset_id
            or getattr(source, 'artifact_id', None) != saved.artifact_id):
        raise ValueError('Registered CSV source identity mismatch')
    if (runtime_config.get_id_column_name() != saved.id_column
            or runtime_config.get_class_column_name() != saved.class_column
            or tuple(runtime_config.get_column_names()) !=
                tuple(saved.text_columns) + tuple(saved.data_columns) +
                (saved.id_column, saved.class_column)):
        raise ValueError('Registered CSV model roles mismatch')
    return prepare_csv_prediction_classifier(
        model_filename, runtime_config, repository, scope, configuration_id, logger)
