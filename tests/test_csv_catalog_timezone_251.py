"""Synthetic, deterministic local-time CSV catalog tests."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from Storage.ArtifactContract import StorageScope
from Storage.CSVDatasetCatalog import format_csv_catalog_time, list_csv_dataset_labels
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.FileArtifactRepository import FileArtifactRepository


class CatalogTimezone251Tests(unittest.TestCase):
    def test_stockholm_winter_and_summer(self):
        stockholm = ZoneInfo("Europe/Stockholm")
        self.assertEqual(format_csv_catalog_time(datetime(2026, 1, 15, 12, tzinfo=timezone.utc), stockholm),
                         "2026-01-15 13:00 CET+0100")
        self.assertEqual(format_csv_catalog_time(datetime(2026, 7, 15, 12, tzinfo=timezone.utc), stockholm),
                         "2026-07-15 14:00 CEST+0200")

    def test_dst_fallback_keeps_two_distinct_instants(self):
        stockholm = ZoneInfo("Europe/Stockholm")
        a = format_csv_catalog_time(datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc), stockholm)
        b = format_csv_catalog_time(datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc), stockholm)
        self.assertEqual(a, "2026-10-25 02:30 CEST+0200")
        self.assertEqual(b, "2026-10-25 02:30 CET+0100")

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            format_csv_catalog_time(datetime(2026, 1, 15))

    def test_repository_label_preserves_opaque_artifact_id(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "synthetic.csv"
            source.write_text("uid;label;x\nA1;red;2\nB2;blue;3\n", encoding="utf-8")
            repository = FileArtifactRepository(base / "repo")
            scope = StorageScope(str(uuid4()), str(uuid4()))
            registered = register_csv_dataset(repository, scope, source,
                id_column="uid", class_column="label", feature_columns=("x",),
                delimiter=";", original_name="synthetic.csv")
            labels = list_csv_dataset_labels(repository, scope, display_timezone=ZoneInfo("Europe/Stockholm"))
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0][1], registered.reference.artifact_id)
            self.assertTrue("CET+0100" in labels[0][0] or "CEST+0200" in labels[0][0])
            self.assertNotIn(str(base), str(labels))


if __name__ == "__main__":
    unittest.main()
