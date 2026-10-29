"""273j: private and scoped CSV misprediction publication (no GUI activation)."""
from io import BytesIO
from pathlib import Path
import json
import sys
import unittest
from uuid import uuid4
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageKind, StorageScope
from Storage.CSVRunContext import CSVRunContext
from Storage.CSVPredictionResults import CSVPredictionResultSink

class DiagnosticStorage273j(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset_id, self.artifact_id = str(uuid4()), str(uuid4())
        self.context = CSVRunContext(self.scope, self.dataset_id, self.artifact_id,
                                     "row_id", "target", ("score",), StorageKind.FILES)
        self.repository = Mock()
        self.repository.get_reference.return_value = ArtifactReference(
            self.scope, self.artifact_id, ArtifactKind.DATASET, "text/csv", dataset_id=self.dataset_id)
        self.saved = []
        self.repository.save_artifact.side_effect = lambda scope, ref, content: self.saved.append(
            (scope, ref, content.read()))
        self.sink = CSVPredictionResultSink(self.repository, self.scope, self.context)
        self.frame = pd.DataFrame({"score": [0.1, 0.4]}, index=["0001", "0002"])

    def test_publishes_private_payload_then_manifest(self):
        result = self.sink.save_mispredicted_data(self.frame)
        self.assertEqual(len(self.saved), 2)
        self.assertEqual([x[1].kind for x in self.saved],
                         [ArtifactKind.SPOT_CHECK, ArtifactKind.MANIFEST])
        self.assertEqual([x[0] for x in self.saved], [self.scope, self.scope])
        self.assertEqual(result["row_count"], 2)
        self.assertIn(b"0001", self.saved[0][2])
        self.assertNotIn(b"0001", self.saved[1][2])
        manifest = json.loads(self.saved[1][2])
        self.assertEqual(manifest["dataset_id"], self.dataset_id)
        self.assertEqual(manifest["diagnostic_artifact_id"], self.saved[0][1].artifact_id)

    def test_rejects_dataset_mismatch_before_write(self):
        self.repository.get_reference.return_value = ArtifactReference(
            self.scope, self.artifact_id, ArtifactKind.DATASET, "text/csv", dataset_id=str(uuid4()))
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            self.sink.save_mispredicted_data(self.frame)
        self.assertEqual(self.saved, [])

    def test_rejects_duplicate_row_ids(self):
        self.frame.index = ["0001", "0001"]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.sink.save_mispredicted_data(self.frame)
        self.assertEqual(self.saved, [])

    def test_does_not_publish_manifest_when_payload_fails(self):
        self.repository.save_artifact.side_effect = RuntimeError("synthetic write error")
        with self.assertRaisesRegex(RuntimeError, "synthetic write error"):
            self.sink.save_mispredicted_data(self.frame)
        self.repository.save_artifact.assert_called_once()

if __name__ == "__main__":
    unittest.main()
