"""273l2: positive-misprediction acceptance using the real combined CSV runner.

This fixture creates contradictory targets for identical features; a deterministic
classifier cannot fit both labels correctly. No ML model or task is mocked.
"""
import csv
from io import BytesIO, StringIO
from uuid import uuid4

from test_csv_combined_mispredicted_273l import CSVCombinedMispredicted273l
from Storage.ArtifactContract import ArtifactKind, ArtifactReference
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVPositiveMispredictions273l2(CSVCombinedMispredicted273l):
    # Run the inherited full-stack assertion once with our noisy fixture.
    test_real_combined_fit_predict_and_mispredicted_publication = None

    def setUp(self):
        super().setUp()
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            source_csv = stream.read().decode("utf-8")
        data = list(csv.reader(StringIO(source_csv)))
        header, rows = data[0], data[1:]
        self.assertEqual(header, ["f1", "f2", "f3", "f4", "row_id", "target"])
        # Nine pairs of contradictory labelled rows. Identical predictor vectors
        # require identical predictions, but the true classes differ.
        for index in range(9):
            left, right = rows[index], rows[50 + index]
            self.assertEqual((left[-1], right[-1]), ("a", "b"))
            right[:4] = left[:4]
        self.assertEqual(sum(bool(row[-1]) for row in rows), 135)
        self.repo = FileArtifactRepository(self.root / "overlap_artifacts")
        self.ref = ArtifactReference(
            self.scope, str(uuid4()), ArtifactKind.DATASET, "text/csv",
            dataset_id=str(uuid4()))
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        self.repo.save_artifact(
            self.scope, self.ref, BytesIO(buffer.getvalue().encode("utf-8")))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id, id_column="row_id",
            class_column="target", data_columns=("f1", "f2", "f3", "f4"),
            csv_format={"delimiter": ",", "quotechar": '"', "encoding": "utf-8"})
        self.config.data_source.dataset_id = self.ref.dataset_id
        self.config.data_source.artifact_id = self.ref.artifact_id

    def test_real_mispredictions_are_published_for_overlapping_classes(self):
        super().test_real_combined_fit_predict_and_mispredicted_publication()
        # The base test verifies the real fitted model, prediction publication,
        # diagnostic artifact contents and row IDs when errors exist. This
        # positive-case test additionally makes that path mandatory.
        from Storage.ArtifactContract import ArtifactKind
        import json
        self.assertGreater(self._positive_misprediction_count(), 0)
        diagnostics = []
        for ref in self.repo.list_artifacts(self.scope):
            if ref.kind != ArtifactKind.MANIFEST or ref.run_id is None:
                continue
            with self.repo.open_artifact(self.scope, ref.artifact_id) as stream:
                item = json.load(stream)
            if item.get("status") == "mispredicted_saved":
                diagnostics.append(item)
        self.assertEqual(len(diagnostics), 1)
        self.assertGreater(diagnostics[0]["row_count"], 0)

    def _positive_misprediction_count(self):
        # The base integration assertion already compared the published count
        # with classifier.ph.num_mispredicted. Only the manifest is needed here.
        import json
        from Storage.ArtifactContract import ArtifactKind
        for ref in self.repo.list_artifacts(self.scope):
            if ref.kind == ArtifactKind.MANIFEST and ref.run_id is not None:
                with self.repo.open_artifact(self.scope, ref.artifact_id) as stream:
                    item = json.load(stream)
                if item.get("status") == "mispredicted_saved":
                    return item["row_count"]
        return 0


if __name__ == "__main__":
    import unittest
    unittest.main()
