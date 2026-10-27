"""Explicit CSV prediction entry point through JBG's application classifier.

The caller must authorize the repository scope and supply a trusted local model
path. This module does not enable CSV in the legacy GUI or support training.
"""

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
    restored_source = getattr(restored_config, 'data_source', None)
    if (getattr(restored_source, 'kind', None) != 'csv'
            or restored_config.should_train()
            or not restored_config.should_predict()):
        raise ValueError('Restored model is not a CSV prediction configuration')

    # Import only after repository and source validation; SQL remains untouched.
    from AutomaticClassifier import AutomaticClassifier
    classifier = AutomaticClassifier(config=restored_config, logger=logger,
                                     datalayer=layer, regression_suite=False)
    if (classifier.datalayer is not layer
            or getattr(getattr(classifier.config, 'data_source', None), 'kind', None) != 'csv'
            or getattr(classifier.handler, 'datalayer', None) is not layer):
        raise ValueError('CSV classifier lost its validated source binding')
    return classifier
