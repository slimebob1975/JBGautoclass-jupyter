"""Scoped CSV class corrections stored beside an immutable original dataset.

Each edit is a small immutable JSON event, not a rewritten CSV. The linked
revision chain permits deterministic replay and frozen historical views.
This is a single-writer local contract; multi-writer locking is not yet provided.
"""
import csv
import json
from io import BytesIO, StringIO
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration

SCHEMA = 'jbg.csv.class-correction.v1'
MAX_EVENTS = 10000
MAX_CSV_BYTES = 25 * 1024 * 1024


def _source(repository, scope, configuration_id):
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid CSV correction scope')
    config = load_csv_model_configuration(repository, scope, configuration_id)
    fmt = config.csv_format or {'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8-sig'}
    with repository.open_artifact(scope, config.artifact_id) as stream:
        raw = stream.read(MAX_CSV_BYTES + 1)
    if not 0 < len(raw) <= MAX_CSV_BYTES:
        raise ValueError('Invalid registered CSV size')
    return config, fmt, raw


def _events(repository, scope, config):
    """Reject branching or corrupted history rather than guessing which edit won."""
    events = {}
    for ref in repository.list_artifacts(scope):
        if ref.kind != ArtifactKind.MANIFEST or ref.dataset_id != config.dataset_id or ref.artifact_id == config.configuration_id:
            continue
        with repository.open_artifact(scope, ref.artifact_id) as stream:
            try:
                doc = json.load(stream)
            except (UnicodeError, ValueError):
                continue  # Other private manifests are unrelated.
        if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
            continue
        if (doc.get('dataset_id') != config.dataset_id or
                doc.get('source_artifact_id') != config.artifact_id or
                doc.get('artifact_id') != ref.artifact_id or
                not isinstance(doc.get('row_id'), str) or not doc['row_id'] or
                not isinstance(doc.get('new_class'), str) or not doc['new_class'].strip() or
                (doc.get('previous') is not None and not isinstance(doc['previous'], str))):
            raise ValueError('Invalid CSV correction history')
        events[ref.artifact_id] = doc
        if len(events) > MAX_EVENTS:
            raise ValueError('CSV correction history limit exceeded')
    ordered = []
    pending = dict(events)
    prior = None
    while pending:
        followers = [(key, doc) for key, doc in pending.items() if doc['previous'] == prior]
        if len(followers) != 1:
            raise ValueError('Ambiguous CSV correction history')
        key, doc = followers[0]
        ordered.append(doc)
        del pending[key]
        prior = key
    return ordered


def _rows(raw, fmt, config):
    try:
        reader = csv.DictReader(StringIO(raw.decode(fmt['encoding']), newline=''),
                                delimiter=fmt['delimiter'], quotechar=fmt['quotechar'], strict=True)
        names = reader.fieldnames
        if (not names or len(set(names)) != len(names) or config.id_column not in names
                or config.class_column not in names):
            raise ValueError
        rows = list(reader)
        ids = [r[config.id_column] for r in rows]
        if any(None in row for row in rows) or not ids or any(not v for v in ids) or len(ids) != len(set(ids)):
            raise ValueError
        return names, rows
    except (UnicodeError, csv.Error, ValueError, TypeError, KeyError):
        raise ValueError('Invalid registered CSV contents') from None


def read_corrected_csv(repository, scope, configuration_id, *, revision=None):
    """Build an in-memory CSV view at a chosen edit revision (None = latest).

    The original bytes are returned unchanged if there are no applied edits.
    """
    config, fmt, raw = _source(repository, scope, configuration_id)
    events = _events(repository, scope, config)
    if revision is not None and not any(e['artifact_id'] == revision for e in events):
        raise ValueError('Unknown CSV correction revision')
    if not events:
        return raw
    names, rows = _rows(raw, fmt, config)
    by_id = {row[config.id_column]: row for row in rows}
    for event in events:
        row = by_id.get(event['row_id'])
        if row is None:
            raise ValueError('CSV correction references missing row')
        row[config.class_column] = event['new_class']
        if revision == event['artifact_id']:
            break
    output = StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=names, delimiter=fmt['delimiter'],
                            quotechar=fmt['quotechar'], lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    rendered = output.getvalue().encode(fmt['encoding'])
    if len(rendered) > MAX_CSV_BYTES:
        raise ValueError('Corrected CSV size limit exceeded')
    return rendered


def list_csv_corrections(repository, scope, configuration_id):
    """Return opaque revision identifiers only; never expose row values in catalog."""
    config, _, _ = _source(repository, scope, configuration_id)
    return tuple(event['artifact_id'] for event in _events(repository, scope, config))


def correct_csv_class(repository, scope: StorageScope, configuration_id: str,
                      row_id: str, new_class: str, *, allowed_classes):
    """Append one class-change event; keep original dataset and IDs unchanged.

    Returns a correction-manifest reference, not a new dataset reference.
    """
    config, fmt, raw = _source(repository, scope, configuration_id)
    if (not isinstance(row_id, str) or not row_id or not isinstance(new_class, str)
            or not new_class.strip() or not isinstance(allowed_classes, (tuple, list, set))
            or not allowed_classes or new_class not in allowed_classes
            or any(not isinstance(v, str) or not v.strip() for v in allowed_classes)):
        raise ValueError('Invalid CSV correction request')
    _, rows = _rows(raw, fmt, config)
    matching = [row for row in rows if row[config.id_column] == row_id]
    if len(matching) != 1:
        raise ValueError('Unknown CSV row')
    events = _events(repository, scope, config)
    effective = matching[0][config.class_column]
    for event in events:
        if event['row_id'] == row_id:
            effective = event['new_class']
    if effective == new_class:
        raise ValueError('CSV class already has requested value')
    ref = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST, 'application/json',
                            dataset_id=config.dataset_id)
    event = {'schema': SCHEMA, 'artifact_id': ref.artifact_id,
             'dataset_id': config.dataset_id, 'source_artifact_id': config.artifact_id,
             'previous': events[-1]['artifact_id'] if events else None,
             'row_id': row_id, 'new_class': new_class}
    repository.save_artifact(scope, ref, BytesIO(json.dumps(event, ensure_ascii=False).encode('utf-8')))
    return ref
