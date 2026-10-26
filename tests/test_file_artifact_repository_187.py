"""File artifact storage regression tests with synthetic, non-sensitive data."""
import io
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository


def uid():
    return str(uuid4())


class FileArtifactRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = FileArtifactRepository(self.temp.name)
        self.scope = StorageScope(uid(), uid())
        self.run = uid()
        self.ref = ArtifactReference(self.scope, uid(), ArtifactKind.PREDICTIONS,
                                     "text/csv", run_id=self.run)

    def test_binary_roundtrip_and_scoped_catalog(self):
        data = b"id,prediction\n001,A\n"
        self.repo.save_artifact(self.scope, self.ref, io.BytesIO(data))
        self.assertEqual(self.repo.get_reference(self.scope, self.ref.artifact_id), self.ref)
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as reader:
            self.assertEqual(reader.read(), data)
        self.assertEqual(self.repo.list_artifacts(self.scope, run_id=self.run), (self.ref,))
        self.assertEqual(self.repo.list_artifacts(self.scope, run_id=uid()), ())
        self.assertFalse(hasattr(self.ref, "path"))

    def test_immutable_no_overwrite(self):
        self.repo.save_artifact(self.scope, self.ref, io.BytesIO(b"original"))
        with self.assertRaises(FileExistsError):
            self.repo.save_artifact(self.scope, self.ref, io.BytesIO(b"replacement"))
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as reader:
            self.assertEqual(reader.read(), b"original")

    def test_cross_scope_is_invisible_and_mismatch_cannot_write(self):
        other = StorageScope(uid(), uid())
        self.repo.save_artifact(self.scope, self.ref, io.BytesIO(b"private"))
        self.assertEqual(self.repo.list_artifacts(other), ())
        with self.assertRaises(FileNotFoundError):
            self.repo.get_reference(other, self.ref.artifact_id)
        with self.assertRaises(FileNotFoundError):
            self.repo.open_artifact(other, self.ref.artifact_id)
        with self.assertRaises(ValueError):
            self.repo.save_artifact(other, self.ref, io.BytesIO(b"bad"))

    def test_opaque_ids_and_missing_artifacts(self):
        with self.assertRaises(ValueError):
            self.repo.get_reference(self.scope, "../../secret")
        with self.assertRaises(ValueError):
            self.repo.list_artifacts(self.scope, run_id="../other")
        with self.assertRaises(FileNotFoundError):
            self.repo.get_reference(self.scope, uid())

    def test_failed_writer_does_not_publish_result(self):
        class BrokenStream:
            def read(self, size):
                raise OSError("private stream detail")
        with self.assertRaises(OSError):
            self.repo.save_artifact(self.scope, self.ref, BrokenStream())
        self.assertEqual(self.repo.list_artifacts(self.scope), ())

    def test_rejects_symlinked_metadata(self):
        self.repo.save_artifact(self.scope, self.ref, io.BytesIO(b"hello"))
        directory = Path(self.temp.name) / self.scope.tenant_id / self.scope.project_id
        meta = directory / (self.ref.artifact_id + ".json")
        meta.unlink()
        try:
            meta.symlink_to(Path(self.temp.name) / "outside")
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not available")
        with self.assertRaises(FileNotFoundError):
            self.repo.get_reference(self.scope, self.ref.artifact_id)
        self.assertEqual(self.repo.list_artifacts(self.scope), ())


if __name__ == "__main__":
    unittest.main()
