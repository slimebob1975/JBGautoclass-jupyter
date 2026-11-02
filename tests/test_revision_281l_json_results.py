"""281l: only known classifier summary fields cross the job API boundary."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunResultProjection import project_classifier_result


def build(result=None, error=None):
    datasets = DatasetApplicationService(sql=SimpleNamespace(resolve=Mock(return_value=object())))
    preparation = RunPreparationService(datasets, sql=SimpleNamespace(validate=Mock()))
    prepared = preparation.prepare(DatasetReference(DataSourceKind.SQL, 'trusted'),
                                   RunConfiguration(DataSourceKind.SQL, object()))
    runtime = SimpleNamespace(execute=Mock(return_value=result, side_effect=error))
    jobs = RunJobService(RunExecutionService(preparation, sql=runtime))
    return RunAPIPresentationService(jobs), prepared, runtime


def test_no_raw_classifier_data_is_exposed():
    pd = pytest.importorskip('pandas')
    raw = {'mispredicted': pd.DataFrame({'secret': ['private'], 'path': ['/hidden']}),
           'unique_classes': ['private', 'public'], 'password': 'secret'}
    api, prepared, _ = build(raw)
    run = RunReference(api.create('alice', prepared)['run_id'])
    assert api.result_details('alice', run)['available'] is False
    api.execute('alice', run)
    payload = api.result_details('alice', run)
    assert payload['summary'] == {'schema_version': 1, 'kind': 'classification',
                                  'mispredicted_count': 1, 'class_count': 2}
    serialized = json.dumps(payload)
    assert 'private' not in serialized and '/hidden' not in serialized and 'secret' not in serialized


def test_none_mispredicted_has_unknown_counts_not_zero():
    assert project_classifier_result({'mispredicted': None}) == {
        'schema_version': 1, 'kind': 'classification',
        'mispredicted_count': None, 'class_count': None}


@pytest.mark.parametrize('raw', [False, None, 13, {'model': object()}, {'mispredicted': object()}])
def test_unknown_objects_are_not_serialized(raw):
    value = project_classifier_result(raw)
    assert value == {'schema_version': 1, 'kind': 'unavailable'}
    json.dumps(value)


def test_owner_check_precedes_result_lookup():
    api, prepared, _ = build({'mispredicted': None})
    run = RunReference(api.create('alice', prepared)['run_id'])
    with pytest.raises(PermissionError, match='unavailable'):
        api.result_details('bob', run)
    with pytest.raises(PermissionError, match='unavailable'):
        api.result_details('alice', RunReference('unknown'))


def test_engine_failure_exposes_no_error_details():
    api, prepared, _ = build(error=RuntimeError('password=secret'))
    run = RunReference(api.create('alice', prepared)['run_id'])
    assert api.execute('alice', run)['status'] == 'failed'
    assert api.result_details('alice', run) == {
        'run_id': run.key, 'available': False, 'summary': None}
