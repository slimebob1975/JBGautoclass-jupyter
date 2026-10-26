"""Dependency-free checks for logical, web-facing storage identifiers."""
import unittest
from uuid import uuid4

from Storage.ArtifactContract import (
    ArtifactKind, ArtifactReference, StorageKind, StorageScope,
)


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = StorageScope(str(uuid4()), str(uuid4()))

    def test_valid_sql_and_file_references_are_backend_neutral(self):
        self.assertEqual(StorageKind.SQL_SERVER.value, "sql_server")
        self.assertEqual(StorageKind.FILES.value, "files")
        ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.PREDICTIONS,
                                "text/csv", run_id=str(uuid4()), dataset_id=str(uuid4()))
        self.assertEqual(ref.scope, self.scope)
        self.assertEqual(ref.kind, ArtifactKind.PREDICTIONS)
        self.assertFalse(hasattr(ref, "path"))

    def test_scope_rejects_paths_and_nonopaque_ids(self):
        for invalid in ("C:\\sensitive\\data.csv", "../private", "", "project_a", str(uuid4()).upper()):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "Invalid tenant identifier"):
                    StorageScope(invalid, str(uuid4()))

    def test_reference_rejects_invalid_ids_and_mime(self):
        for bad in ("../data.csv", "1", ""):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    ArtifactReference(self.scope, bad, ArtifactKind.DATASET, "text/csv")
        with self.assertRaisesRegex(ValueError, "Unsupported artifact content type"):
            ArtifactReference(self.scope, str(uuid4()), ArtifactKind.MODEL, "text/html")
        with self.assertRaisesRegex(ValueError, "Invalid run identifier"):
            ArtifactReference(self.scope, str(uuid4()), ArtifactKind.METRICS,
                              "application/json", run_id="/tmp/private")

    def test_no_io_or_credential_fields_in_contract(self):
        ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.MANIFEST, "application/json")
        self.assertEqual(set(vars(ref)), {"scope", "artifact_id", "kind", "content_type", "run_id", "dataset_id"})


if __name__ == "__main__":
    unittest.main()
