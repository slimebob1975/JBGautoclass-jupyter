"""Synthetic SQL/CSV model source isolation tests."""
import unittest
from types import SimpleNamespace
from JBGModelCompatibility import review_model_source_boundary


def source(kind):
    return SimpleNamespace(data_source=SimpleNamespace(kind=kind))


class ModelSourceBoundary236(unittest.TestCase):
    def test_matching_csv_v2_is_accepted(self):
        self.assertTrue(review_model_source_boundary(source('csv'), source('csv'),
                        {'schema_version': 2, 'source_kind': 'csv'}).ok)

    def test_matching_legacy_sql_v1_is_accepted(self):
        self.assertTrue(review_model_source_boundary(source('sql'), source('sql'),
                        {'schema_version': 1}).ok)

    def test_both_cross_source_directions_are_blocked(self):
        fixtures = ((source('csv'), source('sql'), {'schema_version': 1}),
                    (source('sql'), source('csv'), {'schema_version': 2, 'source_kind': 'csv'}),
                    (source('csv'), source('csv'), {'schema_version': 1}),
                    (source('sql'), source('sql'), {'schema_version': 2, 'source_kind': 'csv'}))
        for active, saved, metadata in fixtures:
            with self.subTest(active=active.data_source.kind, saved=saved.data_source.kind,
                              version=metadata['schema_version']):
                self.assertFalse(review_model_source_boundary(active, saved, metadata).ok)

    def test_unknown_or_inconsistent_source_metadata_fails_closed(self):
        for metadata in (None, {}, {'schema_version': 8},
                         {'schema_version': 1, 'source_kind': 'csv'},
                         {'schema_version': 2, 'source_kind': 'sql'}):
            with self.subTest(metadata=metadata):
                self.assertFalse(review_model_source_boundary(source('sql'), source('sql'), metadata).ok)
        self.assertFalse(review_model_source_boundary(source('other'), source('sql'),
                         {'schema_version': 1}).ok)

if __name__ == '__main__':
    unittest.main()
