"""Source-neutral engine contract: all registered CSV modes share identity validation."""
import unittest
from types import SimpleNamespace
from Storage.CSVRegisteredSource import registered_csv_mismatch


class CSVRegisteredSource276(unittest.TestCase):
    def setUp(self):
        self.saved = SimpleNamespace(dataset_id='dataset', artifact_id='artifact',
            id_column='id', class_column='label', text_columns=('text',),
            data_columns=('number',))
        self.source = SimpleNamespace(kind='csv', dataset_id='dataset', artifact_id='artifact')
        self.config = SimpleNamespace(data_source=self.source,
            get_id_column_name=lambda: 'id', get_class_column_name=lambda: 'label',
            get_column_names=lambda: ['text', 'number', 'id', 'label'])

    def test_matching_roles_and_opaque_identity(self):
        self.assertIsNone(registered_csv_mismatch(self.config, self.saved))

    def test_wrong_dataset_or_artifact_rejected(self):
        self.source.dataset_id = 'other'
        self.assertEqual(registered_csv_mismatch(self.config, self.saved), 'identity')
        self.source.dataset_id = 'dataset'
        self.source.artifact_id = 'other'
        self.assertEqual(registered_csv_mismatch(self.config, self.saved), 'identity')

    def test_sql_source_rejected(self):
        self.source.kind = 'sql'
        self.assertEqual(registered_csv_mismatch(self.config, self.saved), 'identity')

    def test_reordered_features_and_changed_roles_rejected(self):
        self.config.get_column_names = lambda: ['number', 'text', 'id', 'label']
        self.assertEqual(registered_csv_mismatch(self.config, self.saved), 'roles')
        self.config.get_column_names = lambda: ['text', 'number', 'id', 'label']
        self.config.get_id_column_name = lambda: 'wrong'
        self.assertEqual(registered_csv_mismatch(self.config, self.saved), 'roles')


if __name__ == '__main__':
    unittest.main()
