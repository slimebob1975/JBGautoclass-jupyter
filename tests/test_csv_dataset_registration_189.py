"""Synthetic tests for scoped, persistent CSV dataset registration."""
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVDatasetRegistrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "input.csv"
        self.source.write_bytes(b'id;class;feature\n001;A;alpha\n002;B;beta\n')
        self.repo = FileArtifactRepository(self.root / "artifacts")
        self.scope = StorageScope(str(uuid4()), str(uuid4()))

    def register(self, **changes):
        kwargs = dict(id_column="id", class_column="class", feature_columns=("feature",), delimiter=";")
        kwargs.update(changes)
        return register_csv_dataset(self.repo, self.scope, self.source, **kwargs)

    def test_roundtrip_and_opaque_identifiers(self):
        item = self.register()
        self.assertEqual((item.rows, item.columns), (2, 3))
        self.assertEqual(item.reference.kind, ArtifactKind.DATASET)
        self.assertEqual(item.reference.dataset_id, item.dataset_id)
        with self.repo.open_artifact(self.scope, item.reference.artifact_id) as stream:
            self.assertEqual(stream.read(), self.source.read_bytes())
        self.assertNotIn(str(self.source), repr(item))

    def test_bad_roles_do_not_publish(self):
        with self.assertRaises(ValueError):
            self.register(id_column="class")
        self.assertEqual(self.repo.list_artifacts(self.scope), ())

    def test_size_limit_does_not_publish(self):
        with self.assertRaises(ValueError):
            self.register(max_bytes=4)
        self.assertEqual(self.repo.list_artifacts(self.scope), ())

    def test_repeated_uploads_are_distinct(self):
        a, b = self.register(), self.register()
        self.assertNotEqual(a.dataset_id, b.dataset_id)
        self.assertNotEqual(a.reference.artifact_id, b.reference.artifact_id)

    def test_scope_isolation(self):
        a = self.register()
        other = StorageScope(str(uuid4()), str(uuid4()))
        self.assertEqual(self.repo.list_artifacts(other), ())
        with self.assertRaises(FileNotFoundError):
            self.repo.open_artifact(other, a.reference.artifact_id)

    def test_duplicate_identifiers_rejected(self):
        self.source.write_bytes(b'id;class;feature\n001;A;alpha\n001;B;beta\n')
        with self.assertRaises(ValueError):
            self.register()
        self.assertEqual(self.repo.list_artifacts(self.scope), ())


if __name__ == "__main__":
    unittest.main()
