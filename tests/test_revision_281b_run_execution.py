"""281b: application run execution must preserve source and runtime boundaries."""
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import sys
from uuid import uuid4

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunExecutionService import (RunExecutionService, SQLClassifierRuntime,
                                    RegisteredCSVClassifierRuntime)
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope


def setup_run(kind=DataSourceKind.SQL):
    resolved = object()
    resolver = SimpleNamespace(resolve=Mock(return_value=resolved))
    validator = SimpleNamespace(validate=Mock())
    prepare = RunPreparationService(
        DatasetApplicationService(**{kind.value: resolver}),
        **{kind.value: validator})
    ref = DatasetReference(kind, 'opaque')
    prepared = prepare.prepare(ref, RunConfiguration(kind, object()))
    runtime = SimpleNamespace(execute=Mock(return_value={'ok': True}))
    service = RunExecutionService(prepare, **{kind.value: runtime})
    return service, prepared, resolver, validator, runtime


def test_sql_runtime_revalidates_before_execution():
    service, prepared, resolver, validator, runtime = setup_run()
    assert service.execute(prepared) == {'ok': True}
    assert resolver.resolve.call_count == 2
    assert validator.validate.call_count == 2
    runtime.execute.assert_called_once()


def test_mutated_configuration_is_rejected_before_runtime():
    service, prepared, resolver, validator, runtime = setup_run()
    validator.validate.side_effect = ValueError('mutated configuration')
    with pytest.raises(ValueError, match='mutated'):
        service.execute(prepared)
    runtime.execute.assert_not_called()


def test_changed_dataset_binding_fails_closed():
    service, prepared, resolver, validator, runtime = setup_run()
    resolver.resolve.return_value = object()
    with pytest.raises(ValueError, match='binding changed'):
        service.execute(prepared)
    runtime.execute.assert_not_called()


def test_missing_runtime_does_not_resolve_again():
    service, prepared, resolver, validator, runtime = setup_run()
    missing = RunExecutionService(service._preparation)
    with pytest.raises(ValueError, match='not configured'):
        missing.execute(prepared)
    assert resolver.resolve.call_count == 1


def test_runtime_errors_propagate_without_fallback():
    service, prepared, _, _, runtime = setup_run()
    runtime.execute.side_effect = RuntimeError('engine failure')
    with pytest.raises(RuntimeError, match='engine failure'):
        service.execute(prepared)


def test_sql_runtime_uses_existing_datalayer_and_classifier():
    config = SimpleNamespace(data_source=SimpleNamespace(kind='sql'))
    layer = SimpleNamespace(config=config)
    prepared = SimpleNamespace(dataset=DatasetReference(DataSourceKind.SQL, 'table'),
                               configuration=RunConfiguration(DataSourceKind.SQL, config),
                               resolved_dataset=layer)
    logger = object()
    classifier = Mock()
    classifier.run.return_value = {'done': True}
    fake_module = ModuleType('AutomaticClassifier')
    constructor = Mock(return_value=classifier)
    fake_module.AutomaticClassifier = constructor
    with patch.dict(sys.modules, {'AutomaticClassifier': fake_module}):
        assert SQLClassifierRuntime(logger).execute(prepared) == {'done': True}
    constructor.assert_called_once_with(config=config, logger=logger,
                                       datalayer=layer, regression_suite=False)


def test_csv_runtime_delegates_to_registered_training_boundary():
    scope = StorageScope(str(uuid4()), str(uuid4()))
    dataset_id, artifact_id = str(uuid4()), str(uuid4())
    reference = ArtifactReference(scope, artifact_id, ArtifactKind.DATASET,
                                  'text/csv', dataset_id=dataset_id)
    source = SimpleNamespace(kind='csv', dataset_id=dataset_id, artifact_id=artifact_id)
    config = SimpleNamespace(data_source=source)
    prepared = SimpleNamespace(dataset=DatasetReference(DataSourceKind.CSV, artifact_id),
                               resolved_dataset=reference,
                               configuration=RunConfiguration(DataSourceKind.CSV, config))
    classifier = Mock()
    classifier.run.return_value = {'trained': True}
    repo, logger = object(), object()
    with patch('Storage.CSVTrainingApplication.prepare_registered_csv_training_classifier',
               return_value=classifier) as factory:
        assert RegisteredCSVClassifierRuntime(repo, scope, 'registered-config', logger).execute(prepared) == {'trained': True}
    factory.assert_called_once_with(config, repo, scope, 'registered-config', logger)
    classifier.run.assert_called_once_with()


def test_csv_rejects_foreign_scope_before_factory():
    scope = StorageScope(str(uuid4()), str(uuid4()))
    foreign = StorageScope(str(uuid4()), str(uuid4()))
    artifact_id, dataset_id = str(uuid4()), str(uuid4())
    reference = ArtifactReference(foreign, artifact_id, ArtifactKind.DATASET,
                                  'text/csv', dataset_id=dataset_id)
    config = SimpleNamespace(data_source=SimpleNamespace(kind='csv', artifact_id=artifact_id, dataset_id=dataset_id))
    prepared = SimpleNamespace(dataset=DatasetReference(DataSourceKind.CSV, artifact_id),
                               resolved_dataset=reference,
                               configuration=RunConfiguration(DataSourceKind.CSV, config))
    with patch('Storage.CSVTrainingApplication.prepare_registered_csv_training_classifier') as factory:
        with pytest.raises(ValueError, match='scope mismatch'):
            RegisteredCSVClassifierRuntime(object(), scope, 'registered-config', object()).execute(prepared)
        factory.assert_not_called()
