"""Read scoped CSV prediction runs and export SQL-style joined CSV results.

Never log row content; run artifacts and source dataset must belong to one scope.
"""
import csv
from datetime import datetime
from io import StringIO, TextIOWrapper
import json
from uuid import UUID

from Storage.ArtifactContract import ArtifactKind, StorageScope

MAX_EXPORT_ROWS = 100000
MAX_EXPORT_BYTES = 5 * 1024 * 1024


def _read_json(repository, scope, artifact_id):
    with repository.open_artifact(scope, artifact_id) as stream:
        return json.load(stream)


def _run_manifest(repository, scope, dataset_artifact_id, manifest_id):
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid CSV scope')
    dataset = repository.get_reference(scope, dataset_artifact_id)
    ref = repository.get_reference(scope, manifest_id)
    if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv'
            or ref.kind != ArtifactKind.MANIFEST or ref.run_id is None
            or ref.dataset_id != dataset.dataset_id):
        raise ValueError('CSV prediction run does not belong to dataset')
    data = _read_json(repository, scope, manifest_id)
    if (not isinstance(data, dict) or set(data) != {
            'schema_version', 'run_id', 'dataset_id', 'dataset_artifact_id',
            'predictions_artifact_id', 'row_count', 'storage_kind', 'status', 'created_at'}
            or data['schema_version'] != 1 or data['run_id'] != ref.run_id
            or data['dataset_id'] != dataset.dataset_id
            or data['dataset_artifact_id'] != dataset_artifact_id
            or data['storage_kind'] != 'files' or data['status'] != 'predictions_saved'
            or type(data['row_count']) is not int or not 0 <= data['row_count'] <= MAX_EXPORT_ROWS):
        raise ValueError('Invalid CSV prediction manifest')
    try:
        if str(UUID(data['predictions_artifact_id'])) != data['predictions_artifact_id']:
            raise ValueError
        if datetime.fromisoformat(data['created_at']).tzinfo is None:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError('Invalid CSV prediction manifest') from None
    payload_ref = repository.get_reference(scope, data['predictions_artifact_id'])
    if (payload_ref.kind != ArtifactKind.PREDICTIONS or payload_ref.run_id != ref.run_id
            or payload_ref.dataset_id != dataset.dataset_id):
        raise ValueError('CSV prediction payload mismatch')
    return data


def list_csv_prediction_runs(repository, scope, dataset_artifact_id):
    """List validated (label, manifest_id) pairs, newest first, for one dataset."""
    dataset = repository.get_reference(scope, dataset_artifact_id)
    if dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv':
        raise ValueError('Invalid CSV dataset')
    entries = []
    for ref in repository.list_artifacts(scope):
        if (ref.kind != ArtifactKind.MANIFEST or ref.run_id is None
                or ref.dataset_id != dataset.dataset_id):
            continue
        try:
            data = _run_manifest(repository, scope, dataset_artifact_id, ref.artifact_id)
        except (ValueError, TypeError, OSError, KeyError, UnicodeError):
            continue
        entries.append((data['created_at'], ref.artifact_id, data['row_count']))
    return tuple((stamp.replace('T', ' ')[:19] + ' UTC | ' + str(count) + ' predictions | ' + ident[:8], ident)
                 for stamp, ident, count in sorted(entries, reverse=True))


def export_csv_prediction_run(repository, scope, dataset_artifact_id, manifest_id,
                              *, include_original=False):
    """Export a run as CSV bytes; join original rows on unique string IDs if requested."""
    manifest = _run_manifest(repository, scope, dataset_artifact_id, manifest_id)
    payload = _read_json(repository, scope, manifest['predictions_artifact_id'])
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'rows', 'rate_type'}
            or payload['schema_version'] != 1 or not isinstance(payload['rows'], list)
            or len(payload['rows']) != manifest['row_count']
            or not isinstance(payload['rate_type'], str)):
        raise ValueError('Invalid CSV prediction rows')
    rows, seen = [], set()
    for item in payload['rows']:
        if not isinstance(item, dict) or set(item) != {'key', 'prediction', 'rate', 'probabilities'}:
            raise ValueError('Invalid CSV prediction row')
        key = item['key']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Duplicate or invalid prediction ID')
        seen.add(key)
        rows.append(item)
    original = {}
    columns = []
    if include_original:
        # The immutable dataset is read again, not a potentially changed GUI upload.
        from Storage.CSVModelConfiguration import list_csv_model_configurations, load_csv_model_configuration
        candidates = []
        for _, config_id in list_csv_model_configurations(repository, scope, dataset_artifact_id):
            try:
                candidates.append(load_csv_model_configuration(repository, scope, config_id))
            except (ValueError, OSError):
                continue
        if not candidates:
            raise ValueError('No registered CSV column roles available')
        id_names = {c.id_column for c in candidates}
        dialects = {(c.csv_format or {}).get('delimiter', ',') for c in candidates}
        if len(id_names) != 1 or len(dialects) != 1:
            raise ValueError('Ambiguous CSV dataset column roles or delimiter')
        id_name = next(iter(id_names))
        fmt = candidates[0].csv_format or {}
        with repository.open_artifact(scope, dataset_artifact_id) as stream:
            with TextIOWrapper(stream, encoding=fmt.get('encoding', 'utf-8-sig'), newline='') as source:
                reader = csv.DictReader(source, delimiter=fmt.get('delimiter', ','),
                                        quotechar=fmt.get('quotechar', '"'))
                columns = reader.fieldnames or []
                if not columns or len(columns) != len(set(columns)) or id_name not in columns:
                    raise ValueError('Invalid source CSV header')
                for row in reader:
                    if None in row or row.get(id_name) is None:
                        raise ValueError('Invalid source CSV row')
                    key = row[id_name]
                    if key in original:
                        raise ValueError('Duplicate source CSV ID')
                    if key in seen:
                        original[key] = row
                if set(original) != seen:
                    raise ValueError('Prediction IDs do not match source CSV')
        if any(c in columns for c in ('predicted_class', 'prediction_rate', 'class_probabilities')):
            raise ValueError('Source CSV conflicts with result column names')
    output = StringIO(newline='')
    names = columns + ['predicted_class', 'prediction_rate', 'class_probabilities'] if include_original else [
        'id', 'predicted_class', 'prediction_rate', 'class_probabilities']
    writer = csv.writer(output, lineterminator='\r\n')
    writer.writerow(names)
    for item in rows:
        common = [item['key'], item['prediction'], item['rate'], item['probabilities']]
        if include_original:
            joined = [original[item['key']][name] for name in columns]
            writer.writerow(joined + common[1:])
        else:
            writer.writerow(common)
    result = output.getvalue().encode('utf-8-sig')
    if len(result) > MAX_EXPORT_BYTES:
        raise ValueError('CSV export exceeds browser download limit')
    return result
