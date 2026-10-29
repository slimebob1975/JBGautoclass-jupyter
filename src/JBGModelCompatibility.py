"""Compatibility contracts for saved model artifacts and their input data.

The contract deliberately stores only a SHA-256 fingerprint for source provenance.
Column names are already part of the persisted model configuration, while database,
host and table names do not need to be duplicated in clear text just to notice that
a later run uses a different physical SQL source.  Physical source identity is
provenance, not a compatibility requirement: train and prediction tables may differ
when the fitted model's required columns and semantic type families remain compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

from pandas.api import types as pandas_types

MODEL_COMPATIBILITY_SCHEMA = 1

_TEXT_SQL_TYPES = frozenset((
    "char", "nchar", "varchar", "nvarchar", "text", "ntext", "xml", "enum", "set",
))
_NUMERIC_SQL_TYPES = frozenset((
    "bigint", "int", "smallint", "tinyint", "decimal", "numeric", "float", "real",
    "money", "smallmoney", "bit",
))
_TEMPORAL_SQL_TYPES = frozenset((
    "date", "datetime", "datetime2", "smalldatetime", "time", "datetimeoffset",
))
_BINARY_SQL_TYPES = frozenset(("binary", "varbinary", "image", "rowversion", "timestamp"))


class ModelCompatibilityError(ValueError):
    """Raised when a new model cannot be given a reliable compatibility contract."""


@dataclass(frozen=True)
class CompatibilityReview:
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


def _stable_digest(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_identity_fingerprint(*, host="", catalog="", table="") -> str:
    """Hash SQL source identity without persisting source names themselves."""
    return _stable_digest({
        "host": str(host or ""),
        "catalog": str(catalog or ""),
        "table": str(table or ""),
    })


def source_driver_fingerprint(driver="") -> str:
    """Track driver changes separately; a driver is execution context, not source identity."""
    return _stable_digest({"driver": str(driver or "")})


def source_identity_fingerprint_from_config(config) -> str:
    connection = config.connection
    return source_identity_fingerprint(
        host=getattr(connection, "host", ""),
        catalog=getattr(connection, "data_catalog", ""),
        table=getattr(connection, "data_table", ""),
    )


def sql_type_family(sql_type) -> str:
    value = str(sql_type or "").strip().lower()
    if value in _TEXT_SQL_TYPES:
        return "text"
    if value in _NUMERIC_SQL_TYPES:
        return "numeric"
    if value in _TEMPORAL_SQL_TYPES:
        return "temporal"
    if value in _BINARY_SQL_TYPES:
        return "binary"
    return "other:" + value


def pandas_dtype_family(dtype) -> str:
    """Reduce pandas dtypes to semantic families stable enough for model reuse."""
    if pandas_types.is_bool_dtype(dtype) or pandas_types.is_numeric_dtype(dtype):
        return "numeric"
    if pandas_types.is_datetime64_any_dtype(dtype) or pandas_types.is_timedelta64_dtype(dtype):
        return "temporal"
    if pandas_types.is_string_dtype(dtype) or pandas_types.is_object_dtype(dtype) or pandas_types.is_categorical_dtype(dtype):
        return "text"
    return "other:" + str(dtype).lower()


def _excluded_predictors(config) -> tuple[str, ...]:
    raw = str(getattr(config.mode, "excluded_predictor_columns", "") or "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _feature_roles(config) -> dict:
    return {
        "class_column": str(config.get_class_column_name() or ""),
        "id_column": str(config.get_id_column_name() or ""),
        "text_columns": tuple(config.get_text_column_names()),
        "numerical_columns": tuple(config.get_numerical_column_names()),
        "excluded_predictor_columns": _excluded_predictors(config),
    }


def _ngram_range(config) -> tuple[int, int]:
    value = config.get_ngram_range()
    return tuple(getattr(value, "ngram_range", value))


def _converter_config(config) -> dict:
    return {
        "text_columns": tuple(config.get_text_column_names()),
        "category_columns": tuple(config.get_categorical_text_column_names()),
        "limit_categorize": 30,
        "language": "sv",
        "stop_words": bool(config.use_stop_words()),
        "df": 1.0,
        "ngram_range": _ngram_range(config),
        "use_encryption": bool(config.should_hex_encode()),
        "use_categorization": bool(config.use_categorization()),
    }


def _required_columns(config, converter=None) -> tuple[str, ...]:
    """Columns whose source schema is required by the fitted model artifact.

    Configured predictors may be absent from the fitted representation when the
    fetched training dataset proves them globally all-NaN and the dataset layer
    drops them before converter fit.  The artifact contract must therefore be
    anchored to the converter's actual fitted input columns, plus ID/target.
    """
    if converter is not None:
        fitted = tuple(getattr(converter, "input_feature_columns_", ()) or ())
        values = list(fitted)
    else:
        values = list(_expected_predictor_columns(config))
    values.extend((config.get_id_column_name(), config.get_class_column_name()))
    return tuple(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def _expected_predictor_columns(config) -> tuple[str, ...]:
    excluded = set(_excluded_predictors(config))
    return tuple(column for column in config.get_data_column_names() if column not in excluded)


def _converter_setting_issues(config, converter) -> list[str]:
    if converter is None:
        return ["Fitted input converter is unavailable"]

    expected = _converter_config(config)
    attributes = {
        "text_columns": "configured_text_columns_",
        "category_columns": "configured_category_columns_",
        "limit_categorize": "limit_categorize_",
        "language": "language_",
        "stop_words": "stop_words_",
        "df": "df_",
        "ngram_range": "ngram_range_",
        "use_encryption": "use_encryption_",
        "use_categorization": "use_categorization_",
    }
    for key, attribute in attributes.items():
        if not hasattr(converter, attribute):
            return ["Fitted converter lacks revision-173 configuration metadata"]
        actual = getattr(converter, attribute)
        wanted = expected[key]
        if key in ("text_columns", "category_columns", "ngram_range"):
            actual = tuple(actual)
            wanted = tuple(wanted)
        if actual != wanted:
            return ["Saved converter settings differ from the saved model configuration"]

    actual_columns = tuple(getattr(converter, "input_feature_columns_", ()))
    expected_columns = tuple(_expected_predictor_columns(config))
    if not actual_columns or not set(actual_columns).issubset(set(expected_columns)):
        return ["Fitted converter input columns are inconsistent with the saved feature definition"]

    families = getattr(converter, "input_feature_type_families_", None)
    if not isinstance(families, Mapping) or set(families) != set(actual_columns):
        return ["Fitted converter lacks a complete training dtype-family schema"]
    return []


def build_model_compatibility_metadata(config, converter, source_schema: Mapping) -> dict:
    """Build the compatibility metadata required for newly saved model artifacts."""
    if not isinstance(source_schema, Mapping) or not source_schema:
        raise ModelCompatibilityError("Current SQL source schema is unavailable; model cannot be saved safely.")

    required = _required_columns(config, converter)
    normalized = {str(name): str(dtype or "").strip().lower() for name, dtype in source_schema.items()}
    missing = [column for column in required if column not in normalized]
    if missing:
        raise ModelCompatibilityError(
            f"Current SQL source schema is missing {len(missing)} configured model column(s)."
        )

    converter_issues = _converter_setting_issues(config, converter)
    if converter_issues:
        raise ModelCompatibilityError(converter_issues[0])

    input_columns = tuple(converter.input_feature_columns_)
    input_families = dict(converter.input_feature_type_families_)
    fitted_converter_roles = {
        "text_columns": tuple(getattr(converter, "text_columns_", ())),
        "category_columns": tuple(getattr(converter, "category_columns_", ())),
    }
    if set(fitted_converter_roles["text_columns"]) & set(fitted_converter_roles["category_columns"]):
        raise ModelCompatibilityError("Fitted converter has overlapping text and categorical roles.")
    if not set(fitted_converter_roles["text_columns"] + fitted_converter_roles["category_columns"]).issubset(set(input_columns)):
        raise ModelCompatibilityError("Fitted converter roles are inconsistent with its training input columns.")
    roles = _feature_roles(config)
    converter_config = _converter_config(config)
    return {
        "schema_version": MODEL_COMPATIBILITY_SCHEMA,
        "source_identity_sha256": source_identity_fingerprint_from_config(config),
        "source_driver_sha256": source_driver_fingerprint(getattr(config.connection, "odbc_driver", "")),
        "source_columns": tuple(
            {
                "name": column,
                "sql_type": normalized[column],
                "family": sql_type_family(normalized[column]),
            }
            for column in required
        ),
        "feature_roles": roles,
        "feature_semantics_sha256": _stable_digest(roles),
        "converter_config": converter_config,
        "converter_config_sha256": _stable_digest(converter_config),
        "converter_input_columns": input_columns,
        "converter_input_type_families": input_families,
        "ignored_configured_predictors": tuple(
            column for column in _expected_predictor_columns(config) if column not in set(input_columns)
        ),
        "fitted_converter_roles": fitted_converter_roles,
        "fitted_converter_roles_sha256": _stable_digest(fitted_converter_roles),
    }



# CSV uses a separate, explicit contract. Existing SQL v1 artifacts are unchanged.
CSV_MODEL_COMPATIBILITY_SCHEMA = 2
_CSV_FAMILIES = frozenset(("text", "numeric", "temporal", "binary"))


def _csv_column_families(config, converter, csv_configuration) -> dict:
    """Derive configured semantic roles; CSV raw text is not an SQL schema.

    A fitted converter is authoritative for numerical/text feature semantics.
    The ID and target are opaque strings regardless of their lexical contents.
    """
    if (config.get_id_column_name() != csv_configuration.id_column
            or config.get_class_column_name() != csv_configuration.class_column):
        raise ModelCompatibilityError("CSV identity roles differ from the model configuration.")
    if (set(csv_configuration.data_columns) & set(csv_configuration.text_columns)
            or not csv_configuration.data_columns and not csv_configuration.text_columns):
        raise ModelCompatibilityError("CSV feature roles are invalid.")
    predicted = tuple(_expected_predictor_columns(config))
    selected = set(csv_configuration.data_columns) | set(csv_configuration.text_columns)
    if set(predicted) != selected:
        raise ModelCompatibilityError("CSV feature roles differ from the model configuration.")
    result = {csv_configuration.id_column: "text", csv_configuration.class_column: "text"}
    result.update({name: "numeric" for name in csv_configuration.data_columns})
    result.update({name: "text" for name in csv_configuration.text_columns})
    fitted = getattr(converter, "input_feature_type_families_", None)
    if not isinstance(fitted, Mapping):
        raise ModelCompatibilityError("Fitted converter CSV type families are unavailable.")
    for column in getattr(converter, "input_feature_columns_", ()):
        if fitted.get(column) != result.get(column):
            raise ModelCompatibilityError("Fitted converter and CSV feature type families disagree.")
    return result


def build_csv_model_compatibility_metadata(config, converter, csv_configuration) -> dict:
    """Create CSV-native metadata without SQL fields or row/column values."""
    issues = _converter_setting_issues(config, converter)
    if issues:
        raise ModelCompatibilityError(issues[0])
    families = _csv_column_families(config, converter, csv_configuration)
    required = _required_columns(config, converter)
    if any(col not in families for col in required):
        raise ModelCompatibilityError("CSV model requires unavailable columns.")
    roles = _feature_roles(config)
    settings = _converter_config(config)
    fitted_roles = {
        "text_columns": tuple(getattr(converter, "text_columns_", ())),
        "category_columns": tuple(getattr(converter, "category_columns_", ())),
    }
    return {
        "schema_version": CSV_MODEL_COMPATIBILITY_SCHEMA,
        "source_kind": "csv",
        "source_identity_sha256": _stable_digest({
            "dataset_id": csv_configuration.dataset_id,
            "artifact_id": csv_configuration.artifact_id,
        }),
        "source_columns": tuple({"name": name, "family": families[name]} for name in required),
        "feature_roles": roles,
        "feature_semantics_sha256": _stable_digest(roles),
        "converter_config": settings,
        "converter_config_sha256": _stable_digest(settings),
        "converter_input_columns": tuple(converter.input_feature_columns_),
        "converter_input_type_families": dict(converter.input_feature_type_families_),
        "ignored_configured_predictors": tuple(
            name for name in _expected_predictor_columns(config)
            if name not in set(converter.input_feature_columns_)
        ),
        "fitted_converter_roles": fitted_roles,
        "fitted_converter_roles_sha256": _stable_digest(fitted_roles),
    }


def build_model_save_compatibility(config, converter, datalayer) -> dict:
    """Select the source-specific contract at the real model-save boundary.

    A CSV model must have an explicitly prepared, scoped CSV reader. Do not
    query SQL or infer a CSV schema from a SQL connection as a fallback.
    """
    kind = getattr(getattr(config, "data_source", None), "kind", "sql")
    if kind == "csv":
        from Storage.CSVTrainingAdapter import CSVTrainingAdapter
        from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
        from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
        from Storage.CSVCombinedDataLayer import CSVCombinedDataLayer

        reader = (datalayer._reader if isinstance(
            datalayer, (CSVPredictionDataLayer, CSVTrainingDataLayer, CSVCombinedDataLayer)) else datalayer)
        if not isinstance(reader, CSVTrainingAdapter):
            raise ModelCompatibilityError("CSV model saving requires a registered CSV reader.")
        if (getattr(reader, "config", None) is not config
                or getattr(reader, "configuration", None) is None):
            raise ModelCompatibilityError("CSV model source context is not validated.")
        return build_csv_model_compatibility_metadata(config, converter, reader.configuration)
    if kind == "sql":
        if datalayer is None or not callable(getattr(datalayer, "get_table_columns", None)):
            raise ModelCompatibilityError("SQL model saving requires an SQL schema provider.")
        connection = config.connection
        schema = datalayer.get_table_columns(connection.data_catalog, connection.data_table)
        return build_model_compatibility_metadata(config, converter, schema)
    raise ModelCompatibilityError("Unknown model data source type.")


def review_csv_source_compatibility(metadata, csv_configuration, *, column_families=None) -> CompatibilityReview:
    """Fail closed against an explicitly supplied CSV semantic schema.

    An untyped CSV file alone cannot prove feature families; callers must
    validate their source data and supply the semantic family per column.
    """
    if (not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != CSV_MODEL_COMPATIBILITY_SCHEMA
            or metadata.get("source_kind") != "csv"):
        return CompatibilityReview(issues=("Saved model does not have CSV compatibility metadata.",))
    if not isinstance(column_families, Mapping) or not column_families:
        return CompatibilityReview(issues=("Current CSV semantic column schema is unavailable.",))
    stored = metadata.get("source_columns")
    if not isinstance(stored, (tuple, list)) or not stored:
        return CompatibilityReview(issues=("Saved CSV model column schema is missing.",))
    issues = []
    seen = set()
    for entry in stored:
        if not isinstance(entry, Mapping):
            issues.append("Saved CSV model column schema is malformed.")
            continue
        name, family = entry.get("name"), entry.get("family")
        if not isinstance(name, str) or not name or name in seen or family not in _CSV_FAMILIES:
            issues.append("Saved CSV model column schema is malformed.")
            continue
        seen.add(name)
        if name not in column_families:
            issues.append("Current CSV is missing required model columns.")
        elif column_families[name] != family:
            issues.append("Current CSV has incompatible semantic column families.")
    if (csv_configuration.id_column not in seen
            or csv_configuration.class_column not in seen):
        issues.append("CSV identity or class column is missing from the model schema.")
    return CompatibilityReview(tuple(dict.fromkeys(issues)))


def _review_csv_artifact_semantics(config, converter, metadata):
    issues = []
    if metadata.get("source_kind") != "csv":
        issues.append("Saved CSV source type is invalid.")
    digest = metadata.get("source_identity_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        issues.append("Saved CSV source identity is incomplete.")
    roles, settings = _feature_roles(config), _converter_config(config)
    if metadata.get("feature_roles") != roles or metadata.get("feature_semantics_sha256") != _stable_digest(roles):
        issues.append("Saved CSV feature roles differ from model configuration.")
    if metadata.get("converter_config") != settings or metadata.get("converter_config_sha256") != _stable_digest(settings):
        issues.append("Saved CSV converter settings differ from model configuration.")
    issues.extend(_converter_setting_issues(config, converter))
    required = set(_required_columns(config, converter))
    source = metadata.get("source_columns")
    if (not isinstance(source, (tuple, list)) or not source
            or any(not isinstance(e, Mapping) or not isinstance(e.get("name"), str)
                   or e.get("family") not in _CSV_FAMILIES for e in source)
            or len({e["name"] for e in source}) != len(source)
            or {e["name"] for e in source} != required):
        issues.append("Saved CSV column schema is malformed or incomplete.")
    if converter is not None:
        columns = tuple(getattr(converter, "input_feature_columns_", ()))
        if tuple(metadata.get("converter_input_columns") or ()) != columns:
            issues.append("Saved CSV converter input columns differ from fitted converter.")
        if dict(metadata.get("converter_input_type_families") or {}) != dict(getattr(converter, "input_feature_type_families_", {}) or {}):
            issues.append("Saved CSV converter type families differ from fitted converter.")
        ignored = tuple(c for c in _expected_predictor_columns(config) if c not in set(columns))
        if tuple(metadata.get("ignored_configured_predictors") or ()) != ignored:
            issues.append("Saved CSV ignored predictors differ from fitted converter.")
        fitted_roles = {
            "text_columns": tuple(getattr(converter, "text_columns_", ())),
            "category_columns": tuple(getattr(converter, "category_columns_", ())),
        }
        if (metadata.get("fitted_converter_roles") != fitted_roles
                or metadata.get("fitted_converter_roles_sha256") != _stable_digest(fitted_roles)):
            issues.append("Saved CSV fitted roles differ from fitted converter.")
    return CompatibilityReview(tuple(dict.fromkeys(issues)))

def review_model_source_boundary(active_config, saved_config, metadata) -> CompatibilityReview:
    """Enforce SQL/CSV model isolation independently of feature compatibility.

    SQL v1 is an established legacy contract, not a source-neutral model.
    Missing or malformed source identity fails closed; no cross-source reuse.
    """
    active = getattr(getattr(active_config, "data_source", None), "kind", "sql")
    saved = getattr(getattr(saved_config, "data_source", None), "kind", "sql")
    if active not in ("sql", "csv") or saved not in ("sql", "csv"):
        return CompatibilityReview(issues=("Unknown model data source type.",))
    if not isinstance(metadata, Mapping):
        return CompatibilityReview(issues=("Saved model source compatibility metadata is missing.",))
    version = metadata.get("schema_version")
    if version == MODEL_COMPATIBILITY_SCHEMA and metadata.get("source_kind") in (None, "sql"):
        artifact_kind = "sql"
    elif version == CSV_MODEL_COMPATIBILITY_SCHEMA and metadata.get("source_kind") == "csv":
        artifact_kind = "csv"
    else:
        return CompatibilityReview(issues=("Saved model source compatibility metadata is unsupported.",))
    if active != saved or saved != artifact_kind:
        return CompatibilityReview(issues=("Saved model source type does not match the selected data source.",))
    return CompatibilityReview()


def review_artifact_semantics(config, converter, metadata) -> CompatibilityReview:
    """Verify the saved config, fitted converter and artifact metadata agree internally."""
    if metadata is None:
        return CompatibilityReview(issues=(
            "Current model artifacts require source/schema compatibility metadata. Retrain and save a new model.",
        ))
    if isinstance(metadata, Mapping) and metadata.get("schema_version") == CSV_MODEL_COMPATIBILITY_SCHEMA:
        return _review_csv_artifact_semantics(config, converter, metadata)
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != MODEL_COMPATIBILITY_SCHEMA:
        return CompatibilityReview(issues=("Saved model compatibility metadata is unsupported or malformed.",))

    issues = []
    identity_digest = metadata.get("source_identity_sha256")
    if not isinstance(identity_digest, str) or len(identity_digest) != 64:
        issues.append("Saved source-identity metadata is incomplete.")
    expected_driver = source_driver_fingerprint(getattr(config.connection, "odbc_driver", ""))
    if metadata.get("source_driver_sha256") != expected_driver:
        issues.append("Saved SQL-driver metadata differs from the saved model configuration.")

    stored_source = metadata.get("source_columns") or ()
    stored_names = []
    source_schema_valid = bool(stored_source)
    for entry in stored_source:
        if not isinstance(entry, Mapping) or not entry.get("name") or not entry.get("sql_type"):
            source_schema_valid = False
            continue
        stored_names.append(str(entry["name"]))
        if entry.get("family") != sql_type_family(entry.get("sql_type")):
            source_schema_valid = False
    if (not source_schema_valid or len(stored_names) != len(set(stored_names))
            or set(stored_names) != set(_required_columns(config, converter))):
        issues.append("Saved SQL source-schema metadata is incomplete or inconsistent.")

    roles = _feature_roles(config)
    converter_config = _converter_config(config)
    if metadata.get("feature_roles") != roles or metadata.get("feature_semantics_sha256") != _stable_digest(roles):
        issues.append("Saved feature-role metadata differs from the saved model configuration.")
    if metadata.get("converter_config") != converter_config or metadata.get("converter_config_sha256") != _stable_digest(converter_config):
        issues.append("Saved converter metadata differs from the saved model configuration.")
    issues.extend(_converter_setting_issues(config, converter))

    if converter is not None:
        converter_input_columns = tuple(getattr(converter, "input_feature_columns_", ()))
        if tuple(metadata.get("converter_input_columns") or ()) != converter_input_columns:
            issues.append("Saved converter input-column metadata differs from the fitted converter.")
        expected_ignored = tuple(
            column for column in _expected_predictor_columns(config) if column not in set(converter_input_columns)
        )
        if tuple(metadata.get("ignored_configured_predictors") or ()) != expected_ignored:
            issues.append("Saved ignored-predictor metadata differs from the fitted converter.")
        if dict(metadata.get("converter_input_type_families") or {}) != dict(getattr(converter, "input_feature_type_families_", {}) or {}):
            issues.append("Saved converter dtype-family metadata differs from the fitted converter.")
        fitted_roles = {
            "text_columns": tuple(getattr(converter, "text_columns_", ())),
            "category_columns": tuple(getattr(converter, "category_columns_", ())),
        }
        if (metadata.get("fitted_converter_roles") != fitted_roles
                or metadata.get("fitted_converter_roles_sha256") != _stable_digest(fitted_roles)):
            issues.append("Saved fitted converter roles differ from the persisted converter.")
    return CompatibilityReview(issues=tuple(dict.fromkeys(issues)))


def review_current_source_compatibility(
    metadata,
    *,
    driver="",
    host="",
    catalog="",
    table="",
    source_schema=None,
) -> CompatibilityReview:
    """Verify a currently selected SQL source against a saved model contract."""
    if metadata is None:
        return CompatibilityReview(issues=(
            "Current model artifacts require source/schema compatibility metadata. Retrain and save a new model.",
        ))
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != MODEL_COMPATIBILITY_SCHEMA:
        return CompatibilityReview(issues=("Saved model compatibility metadata is unsupported or malformed.",))

    issues = []
    warnings = []
    current_identity = source_identity_fingerprint(
        host=host, catalog=catalog, table=table,
    )
    if current_identity != metadata.get("source_identity_sha256"):
        warnings.append(
            "Selected SQL source differs from the model's training provenance; "
            "compatibility is being determined from the fitted model's required columns and SQL type families."
        )
    saved_driver = metadata.get("source_driver_sha256")
    if saved_driver and saved_driver != source_driver_fingerprint(driver):
        warnings.append(
            "SQL driver changed since training; source parsing equivalence is not independently proven."
        )

    if not isinstance(source_schema, Mapping) or not source_schema:
        issues.append("Current SQL source schema could not be verified.")
        return CompatibilityReview(tuple(issues), tuple(warnings))

    current = {str(name): str(dtype or "").strip().lower() for name, dtype in source_schema.items()}
    stored_columns = metadata.get("source_columns") or ()
    if not stored_columns:
        issues.append("Saved SQL source-schema metadata is missing.")
        return CompatibilityReview(tuple(dict.fromkeys(issues)), tuple(dict.fromkeys(warnings)))
    missing = 0
    incompatible = 0
    exact_type_changes = 0
    for entry in stored_columns:
        if not isinstance(entry, Mapping) or not entry.get("name"):
            issues.append("Saved SQL schema metadata is malformed.")
            continue
        name = str(entry["name"])
        if name not in current:
            missing += 1
            continue
        current_type = current[name]
        current_family = sql_type_family(current_type)
        if current_family != entry.get("family"):
            incompatible += 1
        elif current_type != str(entry.get("sql_type") or "").lower():
            exact_type_changes += 1
    if missing:
        issues.append(f"Current source is missing {missing} required model column(s).")
    if incompatible:
        issues.append(f"Current source has {incompatible} required column(s) in an incompatible SQL type family.")
    if exact_type_changes:
        warnings.append(
            f"{exact_type_changes} required source column(s) changed SQL type within a compatible type family; semantic equivalence is not independently proven."
        )
    return CompatibilityReview(tuple(dict.fromkeys(issues)), tuple(dict.fromkeys(warnings)))
