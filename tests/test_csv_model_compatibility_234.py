"""Revision 234 regression updated for source-specific CSV v2 contract.

Exercises production semantic/source review with synthetic CSV configuration.
Application bootstrap remains isolated; this is not a live deployment test.
"""
import json
import unittest
from types import SimpleNamespace

from JBGModelCompatibility import (build_csv_model_compatibility_metadata,
                                   review_artifact_semantics, review_model_source_boundary)
from JBGModelPersistence import save_model_artifact
from Storage.ArtifactContract import ArtifactKind
import test_csv_model_reload_233 as predecessor


class SyntheticCompatibilityConfig(predecessor.bridge.bridge.handlers.fixture216.SavedConfig):
    """Synthetic model config with real compatibility API; no source names."""
    def __init__(self, ref):
        super().__init__(ref)
        self.connection = SimpleNamespace(host='', data_catalog='', data_table='', odbc_driver='')
        self.mode = SimpleNamespace(excluded_predictor_columns='')

    def get_text_column_names(self): return []
    def get_numerical_column_names(self): return ['value']
    def get_categorical_text_column_names(self): return []
    def get_data_column_names(self): return ['value']
    def get_ngram_range(self): return (1, 1)
    def use_stop_words(self): return False
    def should_hex_encode(self): return False
    def use_categorization(self): return False


def fitted_converter():
    return SimpleNamespace(
        input_feature_columns_=('value',),
        input_feature_type_families_={'value': 'numeric'},
        configured_text_columns_=(), configured_category_columns_=(),
        limit_categorize_=30, language_='sv', stop_words_=False, df_=1.0,
        ngram_range_=(1, 1), use_encryption_=False, use_categorization_=False,
        text_columns_=(), category_columns_=(),
    )


class CSVModelCompatibility234(predecessor.CSVModelReload233):
    test_versioned_jbg_model_reloads_via_production_method_and_runner = None
    test_corrupt_model_format_refuses_publication = None
    test_versioned_artifact_contains_six_payload_fields = None

    def setUp(self):
        super().setUp()
        self.saved_config = SyntheticCompatibilityConfig(self.ref)
        self.converter = fitted_converter()
        self.csv_source = SimpleNamespace(
            dataset_id=self.ref.dataset_id, artifact_id=self.ref.artifact_id,
            id_column='uid', class_column='label', data_columns=('value',),
            text_columns=(),
        )
        self.compatibility = build_csv_model_compatibility_metadata(
            self.saved_config, self.converter, self.csv_source)
        self._save_current(self.compatibility)
        loader = predecessor.production_model_loader()
        # Replace the 233 stub with the real model semantic checker.
        for method_name in ('load_model_from_file', 'load_model'):
            method = loader[method_name]
            method.__globals__['review_artifact_semantics'] = review_artifact_semantics
            method.__globals__['review_model_source_boundary'] = review_model_source_boundary
            setattr(self.model, method_name, method.__get__(self.model))

    def _save_current(self, metadata):
        save_model_artifact(self.jbg_model_path, self.saved_config, self.converter,
                            (None, None, None, None, None), self.estimator,
                            None, 1, compatibility=metadata)

    def _published(self):
        return [ref for ref in self.repo.list_artifacts(self.scope)
                if ref.kind == ArtifactKind.PREDICTIONS]

    def test_source_boundary_accepts_csv_v2_but_rejects_legacy_sql(self):
        self.assertTrue(review_model_source_boundary(
            self.config, self.saved_config, self.compatibility).ok)
        self.assertFalse(review_model_source_boundary(
            self.config, self.saved_config,
            {'schema_version': 1}).ok)

    def test_real_semantic_metadata_passes_model_reload_and_csv_runner(self):
        self.assertTrue(review_artifact_semantics(self.saved_config, self.converter,
                                                  self.compatibility).ok)
        self.assertFalse(self.runner.run(predecessor.bridge.TASKS))
        self.assertEqual(len(self._published()), 1)

    def test_modified_feature_roles_rejected_before_result_publication(self):
        incompatible = dict(self.compatibility)
        incompatible['feature_semantics_sha256'] = '0' * 64
        self._save_current(incompatible)
        self.assertFalse(review_artifact_semantics(self.saved_config, self.converter,
                                                   incompatible).ok)
        with self.assertRaises(Exception):
            self.runner.run(predecessor.bridge.TASKS)
        self.assertFalse(self._published())

    def test_missing_converter_schema_rejected_before_publication(self):
        incompatible = dict(self.compatibility)
        incompatible.pop('converter_input_type_families')
        self._save_current(incompatible)
        self.assertFalse(review_artifact_semantics(self.saved_config, self.converter,
                                                   incompatible).ok)
        with self.assertRaises(Exception):
            self.runner.run(predecessor.bridge.TASKS)
        self.assertFalse(self._published())

    def test_unsupported_schema_version_rejected_before_publication(self):
        incompatible = dict(self.compatibility, schema_version='unsupported')
        self._save_current(incompatible)
        with self.assertRaises(Exception):
            self.runner.run(predecessor.bridge.TASKS)
        self.assertFalse(self._published())


if __name__ == '__main__':
    unittest.main()
