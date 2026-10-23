"""Revision 173: saved-model/data compatibility contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from JBGModelCompatibility import (
    build_model_compatibility_metadata,
    pandas_dtype_family,
    review_artifact_semantics,
    review_current_source_compatibility,
)


class FakeConfig:
    def __init__(self):
        self.connection = SimpleNamespace(
            odbc_driver="driver_x", host="host_x", data_catalog="catalog_x", data_table="schema_x.table_x"
        )
        self.mode = SimpleNamespace(excluded_predictor_columns="")

    def get_class_column_name(self): return "target"
    def get_id_column_name(self): return "row_id"
    def get_text_column_names(self): return ["text_feature"]
    def get_numerical_column_names(self): return ["numeric_feature"]
    def get_data_column_names(self): return self.get_text_column_names() + self.get_numerical_column_names()
    def get_categorical_text_column_names(self): return ["text_feature"]
    def use_stop_words(self): return True
    def get_ngram_range(self): return SimpleNamespace(ngram_range=(1, 2))
    def should_hex_encode(self): return False
    def use_categorization(self): return True


class FakeConverter:
    text_columns_ = ()
    category_columns_ = ("text_feature",)
    configured_text_columns_ = ("text_feature",)
    configured_category_columns_ = ("text_feature",)
    limit_categorize_ = 30
    language_ = "sv"
    stop_words_ = True
    df_ = 1.0
    ngram_range_ = (1, 2)
    use_encryption_ = False
    use_categorization_ = True
    input_feature_columns_ = ("text_feature", "numeric_feature")
    input_feature_type_families_ = {"text_feature": "text", "numeric_feature": "numeric"}


def source_schema(numeric_type="int"):
    return {
        "text_feature": "nvarchar",
        "numeric_feature": numeric_type,
        "row_id": "bigint",
        "target": "varchar",
    }


def test_new_contract_hashes_source_identity_and_records_required_schema():
    config = FakeConfig()
    metadata = build_model_compatibility_metadata(config, FakeConverter(), source_schema())
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["schema_version"] == 1
    assert "catalog_x" not in serialized
    assert "schema_x.table_x" not in serialized
    assert "host_x" not in serialized
    assert "driver_x" not in serialized
    assert {entry["name"] for entry in metadata["source_columns"]} == {
        "text_feature", "numeric_feature", "row_id", "target"
    }
    assert metadata["fitted_converter_roles"] == {
        "text_columns": (), "category_columns": ("text_feature",)
    }



def test_globally_all_nan_configured_predictor_can_be_absent_from_fitted_converter():
    class ConfigWithDroppedPredictor(FakeConfig):
        def get_numerical_column_names(self):
            return ["numeric_feature", "all_nan_feature"]

    class ConverterAfterGlobalAllNanDrop(FakeConverter):
        input_feature_columns_ = ("text_feature", "numeric_feature")
        input_feature_type_families_ = {"text_feature": "text", "numeric_feature": "numeric"}

    config = ConfigWithDroppedPredictor()
    converter = ConverterAfterGlobalAllNanDrop()
    schema = dict(source_schema(), all_nan_feature="float")

    metadata = build_model_compatibility_metadata(config, converter, schema)

    assert metadata["ignored_configured_predictors"] == ("all_nan_feature",)
    assert {entry["name"] for entry in metadata["source_columns"]} == {
        "text_feature", "numeric_feature", "row_id", "target"
    }
    assert review_artifact_semantics(config, converter, metadata).ok

    current = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=schema,
    )
    assert current.ok


def test_converter_still_rejects_unconfigured_fitted_predictor():
    class ConverterWithUnexpectedPredictor(FakeConverter):
        input_feature_columns_ = ("text_feature", "numeric_feature", "unexpected_feature")
        input_feature_type_families_ = {
            "text_feature": "text", "numeric_feature": "numeric", "unexpected_feature": "numeric"
        }

    schema = dict(source_schema(), unexpected_feature="int")
    try:
        build_model_compatibility_metadata(FakeConfig(), ConverterWithUnexpectedPredictor(), schema)
    except Exception as exc:
        assert "inconsistent with the saved feature definition" in str(exc)
    else:
        raise AssertionError("Unexpected fitted predictors must remain fail-closed")

def test_current_source_allows_different_physical_source_when_required_schema_is_compatible():
    config = FakeConfig()
    metadata = build_model_compatibility_metadata(config, FakeConverter(), source_schema())
    same = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema(),
    )
    assert same.ok
    assert same.warnings == ()

    changed_driver = review_current_source_compatibility(
        metadata,
        driver="driver_y", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema(),
    )
    assert changed_driver.ok
    assert any("driver changed" in item for item in changed_driver.warnings)

    different_source = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.prediction_table",
        source_schema=dict(source_schema(), unused_metadata_column="bigint"),
    )
    assert different_source.ok
    assert not different_source.issues
    assert any("training provenance" in item for item in different_source.warnings)

    different_source_missing_required = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.prediction_table",
        source_schema={
            key: value for key, value in source_schema().items() if key != "numeric_feature"
        },
    )
    assert not different_source_missing_required.ok
    assert any("missing 1 required model column" in item for item in different_source_missing_required.issues)

    wrong_type = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema(numeric_type="varchar"),
    )
    assert not wrong_type.ok
    assert any("incompatible SQL type family" in item for item in wrong_type.issues)


    incomplete_metadata = dict(metadata, source_columns=())
    incomplete = review_current_source_compatibility(
        incomplete_metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema(),
    )
    assert not incomplete.ok
    assert any("source-schema metadata is missing" in item for item in incomplete.issues)


def test_compatible_sql_width_change_is_explicit_warning_not_silent_assumption():
    config = FakeConfig()
    metadata = build_model_compatibility_metadata(config, FakeConverter(), source_schema("int"))
    review = review_current_source_compatibility(
        metadata,
        driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema("bigint"),
    )
    assert review.ok
    assert any("semantic equivalence is not independently proven" in item for item in review.warnings)


def test_artifact_semantics_detect_converter_setting_drift():
    config = FakeConfig()
    metadata = build_model_compatibility_metadata(config, FakeConverter(), source_schema())
    assert review_artifact_semantics(config, FakeConverter(), metadata).ok

    converter = FakeConverter()
    converter.stop_words_ = False
    review = review_artifact_semantics(config, converter, metadata)
    assert not review.ok
    assert any("converter settings differ" in item for item in review.issues)

    converter = FakeConverter()
    converter.category_columns_ = ()
    converter.text_columns_ = ("text_feature",)
    review = review_artifact_semantics(config, converter, metadata)
    assert not review.ok
    assert any("fitted converter roles differ" in item for item in review.issues)

    broken_metadata = dict(metadata, source_columns=())
    review = review_artifact_semantics(config, FakeConverter(), broken_metadata)
    assert not review.ok
    assert any("source-schema metadata" in item for item in review.issues)


def test_current_artifact_requires_compatibility_metadata():
    review = review_artifact_semantics(FakeConfig(), FakeConverter(), None)
    assert not review.ok
    assert review.issues
    source = review_current_source_compatibility(
        None, driver="driver_x", host="host_x", catalog="catalog_x", table="schema_x.table_x",
        source_schema=source_schema(),
    )
    assert not source.ok
    assert source.issues


def test_pandas_dtype_families_are_stable_for_prediction_guard():
    assert pandas_dtype_family(pd.Series([1, 2]).dtype) == "numeric"
    assert pandas_dtype_family(pd.Series(["a", "b"]).dtype) == "text"
    assert pandas_dtype_family(pd.to_datetime(pd.Series(["2026-01-01"])).dtype) == "temporal"


def test_save_path_fails_closed_when_compatibility_contract_cannot_be_built():
    source = (Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py").read_text(encoding="utf-8")
    block = source.split("def save_model_to_file", 1)[1].split("@dataclass", 1)[0]
    assert "build_model_compatibility_metadata" in block
    assert "Model was not saved because source/schema compatibility metadata could not be established" in block
    assert "return" not in block.split("except Exception as compatibility_error:", 1)[1].split("# Handle KERAS", 1)[0]


def test_runtime_checks_source_before_saved_config_replaces_live_source():
    source = (Path(__file__).resolve().parents[1] / "src" / "AutomaticClassifier.py").read_text(encoding="utf-8")
    assert source.index("load_model_compatibility(model_filename)") < source.index("load_config_from_model_file(model_filename")
    assert "self.datalayer.get_table_columns(" in source
    assert "Saved model is not compatible with the selected data source" in source
