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


def review_artifact_semantics(config, converter, metadata) -> CompatibilityReview:
    """Verify the saved config, fitted converter and artifact metadata agree internally."""
    if metadata is None:
        return CompatibilityReview(warnings=(
            "Legacy model has no revision-173 compatibility metadata; source identity, SQL type compatibility and converter-setting equivalence cannot be fully verified.",
        ))
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
        return CompatibilityReview(warnings=(
            "Legacy model has no source/schema compatibility metadata; current source identity and SQL types are unverified.",
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
