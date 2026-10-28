"""Synthetic model-save source selection without access to production data."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from JBGModelCompatibility import ModelCompatibilityError, build_model_save_compatibility
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
from test_csv_model_compatibility_235 import CSVModelCompatibility235


class ModelSaveSource237(unittest.TestCase):
    def setUp(self):
        fixture = CSVModelCompatibility235()
        fixture.setUp()
        self.config = fixture.config
        self.converter = fixture.converter
        self.csv = fixture.csv

    def test_csv_requires_prepared_adapter_and_never_calls_sql(self):
        sql = Mock()
        with self.assertRaises(ModelCompatibilityError):
            build_model_save_compatibility(self.config, self.converter, sql)
        sql.get_table_columns.assert_not_called()

    def test_csv_prepared_adapter_produces_v2(self):
        adapter = object.__new__(CSVTrainingAdapter)
        adapter.configuration = self.csv
        adapter.config = self.config
        result = build_model_save_compatibility(self.config, self.converter, adapter)
        self.assertEqual(result['schema_version'], 2)
        self.assertEqual(result['source_kind'], 'csv')
        self.assertNotIn('source_driver_sha256', result)

    def test_csv_registered_training_facade_produces_v2(self):
        adapter = object.__new__(CSVTrainingAdapter)
        adapter.configuration = self.csv
        adapter.config = self.config
        facade = CSVTrainingDataLayer(adapter)
        result = build_model_save_compatibility(self.config, self.converter, facade)
        self.assertEqual(result['schema_version'], 2)
        self.assertEqual(result['source_kind'], 'csv')
        self.assertNotIn('source_driver_sha256', result)

    def test_csv_training_facade_rejects_foreign_config(self):
        adapter = object.__new__(CSVTrainingAdapter)
        adapter.configuration = self.csv
        adapter.config = object()
        facade = CSVTrainingDataLayer(adapter)
        with self.assertRaises(ModelCompatibilityError):
            build_model_save_compatibility(self.config, self.converter, facade)

    def test_csv_rejects_foreign_runtime_configuration(self):
        adapter = object.__new__(CSVTrainingAdapter)
        adapter.configuration = self.csv
        adapter.config = object()
        with self.assertRaises(ModelCompatibilityError):
            build_model_save_compatibility(self.config, self.converter, adapter)

    def test_sql_preserves_v1_and_queries_sql_schema(self):
        self.config.data_source.kind = 'sql'
        provider = Mock()
        provider.get_table_columns.return_value = {
            'uid': 'varchar', 'label': 'varchar', 'value': 'float'
        }
        metadata = build_model_save_compatibility(self.config, self.converter, provider)
        self.assertEqual(metadata['schema_version'], 1)
        self.assertNotIn('source_kind', metadata)
        provider.get_table_columns.assert_called_once()

    def test_unknown_source_rejected_without_sql_query(self):
        self.config.data_source.kind = 'invalid'
        provider = Mock()
        with self.assertRaises(ModelCompatibilityError):
            build_model_save_compatibility(self.config, self.converter, provider)
        provider.get_table_columns.assert_not_called()


if __name__ == '__main__':
    unittest.main()
