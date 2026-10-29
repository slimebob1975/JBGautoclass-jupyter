"""Registered CSV training assembly for explicit train-only application paths."""
from Storage.ArtifactContract import StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.CSVRegisteredSource import registered_csv_mismatch
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer, prepare_csv_training_datalayer


def prepare_registered_csv_training_classifier(runtime_config, repository, scope,
                                               configuration_id, logger):
    """Validate registered source and roles before constructing ML handlers."""
    if not isinstance(scope, StorageScope):
        raise ValueError('CSV training requires an authorized storage scope')
    source = getattr(runtime_config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv'
            or not runtime_config.should_train() or runtime_config.should_predict()):
        raise ValueError('CSV training requires a train-only CSV configuration')
    saved = load_csv_model_configuration(repository, scope, configuration_id)
    mismatch = registered_csv_mismatch(runtime_config, saved)
    if mismatch == 'identity':
        raise ValueError('Registered CSV training source mismatch')
    if mismatch == 'roles':
        raise ValueError('Registered CSV training column roles mismatch')
    layer = prepare_csv_training_datalayer(runtime_config, repository, scope, configuration_id)
    if not isinstance(layer, CSVTrainingDataLayer) or not hasattr(layer, 'correction_revision'):
        raise ValueError('CSV training requires a pinned correction snapshot')
    from AutomaticClassifier import AutomaticClassifier
    classifier = AutomaticClassifier(config=runtime_config, logger=logger,
                                     datalayer=layer, regression_suite=False)
    if (classifier.datalayer is not layer or classifier.config is not runtime_config
            or getattr(classifier.handler, 'datalayer', None) is not layer):
        raise ValueError('CSV training classifier lost validated source binding')
    return classifier
