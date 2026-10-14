"""Independent, paired validation of a fixed candidate against its original baseline.

This API requires development and externally collected labelled rows, with stable,
disjoint identifiers. It does not promote, approve, or apply a profile. The caller
is responsible for confirming that the external cohort was not used to choose the
candidate, preprocessing or validation policy.
"""
from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from JBGCVReporting import partition_signature
from JBGFeatureSearch import (SearchOptions, classification_counts,
                              metrics, multiclass_diagnostics, make_pipeline,
                              study_provenance)


def _ids(ids, expected_size, title):
    if ids is None or len(ids) != expected_size:
        raise ValueError(f"{title}: explicit row IDs are required for every row")
    series = pd.Series(list(ids), dtype='string')
    if series.isna().any() or (series.str.strip() == '').any() or series.duplicated().any():
        raise ValueError(f"{title}: row IDs must be nonempty and unique")
    return set(series.tolist())


def evaluate_external_candidate(record, X_development, y_development,
                                X_external, y_external, *, development_ids,
                                external_ids, cohort_description, output_path):
    """Fit two frozen models on development only; score once on external rows.

    Creates an immutable evidence file. Refuses profiles without the exact
    candidate configuration or with overlapping or malformed row identities.
    This is a *review* artifact only: even strong results confer no reuse right.
    """
    if (record.get('profile_schema') != 1 or record.get('validation_status') != 'candidate'
            or record.get('reuse_eligible') is not False):
        raise ValueError('Only unapproved schema-1 candidates can be evaluated')
    if not cohort_description or not str(cohort_description).strip():
        raise ValueError('Document the external cohort origin and collection process')
    if not all(isinstance(v, pd.DataFrame) for v in (X_development, X_external)):
        raise TypeError('Feature inputs must be pandas DataFrames')
    y_development, y_external = pd.Series(y_development), pd.Series(y_external)
    if (len(X_development) != len(y_development) or len(X_external) != len(y_external)
            or len(y_development) < 2 or len(y_external) < 2):
        raise ValueError('Feature and label lengths must match and cohorts must be nonempty')
    development = _ids(development_ids, len(X_development), 'Development')
    external = _ids(external_ids, len(X_external), 'External')
    if development & external:
        raise ValueError('Development and external datasets have overlapping row IDs')
    if y_development.isna().any() or y_external.isna().any():
        raise ValueError('All validation rows must have known labels')
    available = record.get('original_columns') or []
    selected = record.get('recommendation', {}).get('selected_columns') or []
    if (not selected or len(set(available)) != len(available)
            or not set(selected).issubset(set(available))
            or list(X_development.columns) != available
            or list(X_external.columns) != available):
        raise ValueError('Original input columns/order do not match the saved candidate')
    schema = record.get('provenance', {}).get('original_column_schema') or []
    for frame in (X_development, X_external):
        if [(x['name'], x['dtype']) for x in schema] != [(c, str(frame[c].dtype)) for c in available]:
            raise ValueError('Source data types differ from the recorded schema')
    # The exact original development features are needed, not simply a table with
    # matching column names. Old candidate profiles lacking this evidence fail closed.
    expected_hash = record.get('provenance', {}).get('development_feature_sha256')
    actual_hash = study_provenance(X_development)['development_feature_sha256']
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError('Development features differ from the original investigation')
    expected_partition = (record.get('signatures') or {}).get('development')
    if not expected_partition or partition_signature(X_development.index, y_development) != expected_partition:
        raise ValueError('Development row IDs/labels differ from the original investigation')
    saved_options = record.get('options')
    if not isinstance(saved_options, dict) or not isinstance(record.get('text_options'), dict):
        raise ValueError('Profile predates frozen options; run and save a new investigation')
    allowed = {field.name for field in fields(SearchOptions)}
    if set(saved_options) != allowed:
        raise ValueError('Incomplete or unknown frozen investigation settings')
    saved_options = dict(saved_options)
    saved_options['sizes'] = tuple(saved_options['sizes'])
    saved_options['non_linear_hidden_layers'] = tuple(saved_options['non_linear_hidden_layers'])
    options = SearchOptions(**saved_options)
    options.validate()
    if record.get('investigative_model', {}).get('model_type') != options.model_type:
        raise ValueError('Stored estimator identity is inconsistent')
    if set(y_external.unique()) - set(y_development.unique()):
        raise ValueError('External evaluation contains unseen classes')
    target = record.get('target')
    if target not in set(y_development):
        raise ValueError('Saved target label is not in development labels')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = {}
    predictions = {}
    for label, cols in (('selected', selected), ('full_baseline', available)):
        model = make_pipeline(cols, options, record['text_options'])
        model.fit(X_development, y_development)
        pred = model.predict(X_external)
        predictions[label] = pred
        counts = classification_counts(y_external, pred, target)
        row = dict(mcc=float(matthews_corrcoef(y_external, pred)), **metrics(counts))
        labels = sorted(set(y_development), key=str)
        row.update(multiclass_diagnostics(y_external, pred, labels))
        scores[label] = row
    report = dict(schema_version=1, evidence_id=str(uuid4()),
                  candidate_profile_id=record['profile_id'],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  validation_type='external_labelled_cohort',
                  cohort_description=str(cohort_description),
                  development_rows=len(X_development), external_rows=len(X_external),
                  selected_columns=selected, baseline_columns=available,
                  model_type=options.model_type,
                  selected=scores['selected'], full_baseline=scores['full_baseline'],
                  paired_mcc_difference=scores['selected']['mcc']-scores['full_baseline']['mcc'],
                  validation_status='evidence_recorded', reuse_eligible=False,
                  limitations=('Row-ID disjointness was checked. This tool cannot establish that '
                               'the external cohort was never used during model design, that '
                               'its labels are independent, or that schema meanings are stable. '
                               'Manual evidence review and approval remain required.'))
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    return report
