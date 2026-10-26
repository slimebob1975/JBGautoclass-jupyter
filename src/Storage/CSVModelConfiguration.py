"""Scoped, immutable CSV model-role configurations; no SQL or ML execution.

Column names are private configuration data, never public catalog labels.
The caller must authorize StorageScope before invoking this local repository.
"""
from dataclasses import dataclass
from io import BytesIO
import json
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope


@dataclass(frozen=True)
class CSVModelConfiguration:
    configuration_id: str
    dataset_id: str
    artifact_id: str
    reference: ArtifactReference
    id_column: str
    class_column: str
    data_columns: tuple[str, ...]
    text_columns: tuple[str, ...]
    csv_format: dict | None = None
    shared_json: dict | None = None


def _roles(id_column, class_column, data_columns, text_columns):
    if (not isinstance(data_columns, (tuple, list)) or
            not isinstance(text_columns, (tuple, list)) or
            not data_columns and not text_columns):
        raise ValueError('Invalid CSV model column roles')
    roles = (id_column, class_column, *data_columns, *text_columns)
    if (any(not isinstance(name, str) or not name.strip() for name in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError('Invalid CSV model column roles')
    return tuple(data_columns), tuple(text_columns)


def _validate_csv_format(value):
    """Strict, portable CSV dialect metadata; no paths, credentials or free-form options."""
    if value is None:
        return None  # Backward compatible with configurations created before revision 197.
    if not isinstance(value, dict) or set(value) != {'delimiter', 'quotechar', 'encoding'}:
        raise ValueError('Invalid CSV format settings')
    delimiter, quotechar, encoding = (value[k] for k in ('delimiter', 'quotechar', 'encoding'))
    if (not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '\r\n\x00'
            or not isinstance(quotechar, str) or len(quotechar) != 1
            or quotechar in '\r\n\x00' or quotechar == delimiter
            or not isinstance(encoding, str) or encoding not in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252')):
        raise ValueError('Invalid CSV format settings')
    return dict(delimiter=delimiter, quotechar=quotechar, encoding=encoding)


def save_csv_model_configuration(repository, scope: StorageScope, dataset_artifact_id: str, *,
                                 id_column: str, class_column: str,
                                 data_columns, text_columns=(), csv_format=None, shared_json=None):
    """Persist a distinct configuration referencing one immutable dataset.

    This intentionally does not duplicate dataset bytes. Caller is responsible
    for checking actual CSV columns and semantic preprocessing before training.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid configuration scope')
    data_columns, text_columns = _roles(id_column, class_column, data_columns, text_columns)
    csv_format = _validate_csv_format(csv_format)
    dataset = repository.get_reference(scope, dataset_artifact_id)
    if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv'
            or dataset.dataset_id is None):
        raise ValueError('Invalid configuration dataset')
    if shared_json is not None:
        # Reuse JBG's strict JSON-v2 schema rather than inventing CSV-only
        # preprocessing/training settings. SQL credentials are forbidden here.
        from Config import Config
        Config._validate_json_payload(shared_json)
        source = shared_json['data_source']
        columns = shared_json['connection']
        if (source['kind'] != 'csv' or source['dataset_id'] != dataset.dataset_id
                or source['artifact_id'] != dataset.artifact_id
                or columns['id_column'] != id_column
                or columns['class_column'] != class_column
                or tuple(columns['data_numerical_columns']) != data_columns
                or tuple(columns['data_text_columns']) != text_columns
                or (csv_format is not None and
                    any(source[key] != value for key, value in csv_format.items()))):
            raise ValueError('Shared JSON configuration differs from selected CSV roles')
        if (columns['sql_password'] or columns['sql_username'] or columns['host']
                or columns['data_catalog'] or columns['data_table']
                or columns['class_catalog'] or columns['class_table']
                or shared_json['mail']['notification_email']):
            raise ValueError('CSV configuration must not retain SQL or email identity')
        shared_json = json.loads(json.dumps(shared_json, ensure_ascii=False))
    configuration_id = str(uuid4())
    reference = ArtifactReference(scope, configuration_id, ArtifactKind.MANIFEST,
                                  'application/json', dataset_id=dataset.dataset_id)
    payload = {
        'schema_version': 1, 'configuration_id': configuration_id,
        'dataset_id': dataset.dataset_id, 'dataset_artifact_id': dataset.artifact_id,
        'id_column': id_column, 'class_column': class_column,
        'data_columns': list(data_columns), 'text_columns': list(text_columns),
        'csv_format': csv_format,
        'shared_json': shared_json,
    }
    repository.save_artifact(scope, reference,
                             BytesIO(json.dumps(payload, ensure_ascii=False).encode('utf-8')))
    return CSVModelConfiguration(configuration_id, dataset.dataset_id, dataset.artifact_id,
                                 reference, id_column, class_column, data_columns, text_columns, csv_format, shared_json)


def load_csv_model_configuration(repository, scope: StorageScope, configuration_id: str):
    """Read the private scoped configuration and verify its dataset still exists."""
    reference = repository.get_reference(scope, configuration_id)
    if reference.kind != ArtifactKind.MANIFEST or reference.content_type != 'application/json':
        raise ValueError('Invalid CSV model configuration')
    try:
        with repository.open_artifact(scope, configuration_id) as stream:
            payload = json.load(stream)
        if (payload.get('schema_version') != 1 or
                payload.get('configuration_id') != configuration_id or
                payload.get('dataset_id') != reference.dataset_id):
            raise ValueError
        data, text = _roles(payload['id_column'], payload['class_column'],
                            payload['data_columns'], payload['text_columns'])
        csv_format = _validate_csv_format(payload.get('csv_format'))
        shared_json = payload.get('shared_json')
        if shared_json is not None:
            from Config import Config
            Config._validate_json_payload(shared_json)
            src = shared_json['data_source']
            conn = shared_json['connection']
            if (src['kind'] != 'csv' or src['dataset_id'] != reference.dataset_id
                    or src['artifact_id'] != payload['dataset_artifact_id']
                    or conn['id_column'] != payload['id_column']
                    or conn['class_column'] != payload['class_column']
                    or tuple(conn['data_numerical_columns']) != data
                    or tuple(conn['data_text_columns']) != text
                    or (csv_format is not None and
                        any(src[key] != value for key, value in csv_format.items()))):
                raise ValueError('Invalid shared JSON CSV configuration')
        dataset = repository.get_reference(scope, payload['dataset_artifact_id'])
        if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv'
                or dataset.dataset_id != reference.dataset_id):
            raise ValueError
        return CSVModelConfiguration(configuration_id, dataset.dataset_id, dataset.artifact_id,
                                     reference, payload['id_column'], payload['class_column'], data, text, csv_format, shared_json)
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid CSV model configuration') from None


def list_csv_model_configurations(repository, scope: StorageScope, dataset_artifact_id: str):
    """Return scoped GUI labels and opaque configuration IDs for one CSV dataset."""
    dataset = repository.get_reference(scope, dataset_artifact_id)
    if dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv':
        raise ValueError('Invalid CSV configuration dataset')
    result = []
    for ref in repository.list_artifacts(scope):
        if ref.kind != ArtifactKind.MANIFEST or ref.dataset_id != dataset.dataset_id or ref.run_id is not None:
            continue
        try:
            config = load_csv_model_configuration(repository, scope, ref.artifact_id)
        except (ValueError, TypeError, KeyError, OSError):
            continue
        if config.artifact_id == dataset_artifact_id:
            result.append(('Configuration ' + config.configuration_id[:8], config.configuration_id))
    return tuple(sorted(result))
