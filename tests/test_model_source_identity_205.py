import unittest
from types import SimpleNamespace

from JBGModelSourceIdentity import check_model_source_identity


def config(kind="sql", dataset_id="", artifact_id="", display_name=""):
    return SimpleNamespace(data_source=SimpleNamespace(kind=kind, dataset_id=dataset_id,
                                                       artifact_id=artifact_id, display_name=display_name))


class ModelSourceIdentityTests(unittest.TestCase):
    def test_legacy_sql_without_source_metadata_is_compatible(self):
        check_model_source_identity(SimpleNamespace(), config())
        check_model_source_identity(config(), config())

    def test_matching_csv_identity_allows_different_display_filename(self):
        check_model_source_identity(config("csv", "dataset-a", "blob-a", "old.csv"),
                                    config("csv", "dataset-a", "blob-a", "renamed.csv"))

    def test_same_filename_other_upload_must_fail(self):
        with self.assertRaisesRegex(ValueError, "different dataset"):
            check_model_source_identity(config("csv", "dataset-a", "blob-a", "same.csv"),
                                        config("csv", "dataset-b", "blob-b", "same.csv"))

    def test_cross_backend_must_fail(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            check_model_source_identity(config(), config("csv", "a", "b"))
        with self.assertRaisesRegex(ValueError, "differs"):
            check_model_source_identity(config("csv", "a", "b"), config())

    def test_missing_csv_identity_and_invalid_kind_must_fail(self):
        with self.assertRaises(ValueError):
            check_model_source_identity(config("csv", "a", ""), config("csv", "a", ""))
        with self.assertRaises(ValueError):
            check_model_source_identity(config("other"), config())


if __name__ == "__main__":
    unittest.main()
