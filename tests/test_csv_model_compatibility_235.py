"""Synthetic CSV compatibility contract; never accesses real datasets."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

from JBGModelCompatibility import (
    ModelCompatibilityError, build_csv_model_compatibility_metadata,
    review_artifact_semantics, review_csv_source_compatibility,
    build_model_compatibility_metadata,
)
from test_csv_model_compatibility_234 import SyntheticCompatibilityConfig, fitted_converter


class CSVModelCompatibility235(unittest.TestCase):
    def setUp(self):
        self.config = SyntheticCompatibilityConfig(SimpleNamespace(dataset_id='ds-synthetic', artifact_id='file-synthetic'))
        self.converter = fitted_converter()
        self.csv = SimpleNamespace(
            configuration_id='cfg-synthetic', dataset_id='ds-synthetic',
            artifact_id='file-synthetic', id_column='uid', class_column='label',
            data_columns=('value',), text_columns=(),
        )
        self.metadata = build_csv_model_compatibility_metadata(self.config, self.converter, self.csv)
        self.families = {'uid': 'text', 'label': 'text', 'value': 'numeric'}

    def test_csv_metadata_uses_explicit_families_without_sql_identity(self):
        self.assertEqual(self.metadata['source_kind'], 'csv')
        self.assertEqual(self.metadata['schema_version'], 2)
        self.assertFalse('source_driver_sha256' in self.metadata)
        self.assertNotIn('sql_type', self.metadata['source_columns'][0])
        self.assertTrue(review_artifact_semantics(self.config, self.converter, self.metadata).ok)
        self.assertTrue(review_csv_source_compatibility(self.metadata, self.csv,
                            column_families=self.families).ok)

    def test_missing_or_mismatched_current_source_fails_closed(self):
        for families in (None, {'uid': 'text', 'label': 'text'},
                         {'uid': 'text', 'label': 'text', 'value': 'text'}):
            with self.subTest(families=families):
                self.assertFalse(review_csv_source_compatibility(self.metadata, self.csv,
                                 column_families=families).ok)

    def test_converter_and_saved_metadata_tampering_rejected(self):
        for changed in (dict(self.metadata, feature_semantics_sha256='0' * 64),
                        dict(self.metadata, source_columns=({'name': 'value', 'family': 'numeric'},)),
                        dict(self.metadata, converter_input_type_families={'value': 'text'})):
            with self.subTest(changed=changed.keys()):
                self.assertFalse(review_artifact_semantics(self.config, self.converter, changed).ok)
        wrong = SimpleNamespace(**vars(self.csv))
        wrong.data_columns = ('missing',)
        with self.assertRaises(ModelCompatibilityError):
            build_csv_model_compatibility_metadata(self.config, self.converter, wrong)

    def test_sql_v1_contract_stays_compatible(self):
        sql = build_model_compatibility_metadata(self.config, self.converter,
                     {'value': 'float', 'uid': 'varchar', 'label': 'varchar'})
        self.assertEqual(sql['schema_version'], 1)
        self.assertTrue(review_artifact_semantics(self.config, self.converter, sql).ok)
        self.assertFalse(review_csv_source_compatibility(sql, self.csv,
                         column_families=self.families).ok)


if __name__ == '__main__':
    unittest.main()
