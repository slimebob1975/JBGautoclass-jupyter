"""Dependency-light contract tests for the opt-in CSV reader."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from CSVDataSource import CSVDataSource, CSVSourceSettings


class CSVDataSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"

    def source(self, **kwargs):
        return CSVDataSource(CSVSourceSettings(path=self.path, id_column="row_id", class_column="target", **kwargs))

    def test_bom_semicolon_preserves_identifiers_and_empty_labels(self):
        self.path.write_text("row_id;feature;target\n001;0007;yes\n002;;\n", encoding="utf-8-sig")
        data = self.source(delimiter=";").read_frame()
        self.assertEqual(data["row_id"].tolist(), ["001", "002"])
        self.assertEqual(data["feature"].tolist(), ["0007", ""])
        self.assertEqual(data["target"].tolist(), ["yes", ""])

    def test_reader_uses_portable_dtype_and_retains_text(self):
        self.path.write_text("row_id,target,feature\n001,yes,0007\n", encoding="utf-8")
        from CSVDataSource import pd
        original_read_csv = pd.read_csv
        seen = []

        def checked_read_csv(*args, **kwargs):
            seen.append(kwargs.get("dtype"))
            return original_read_csv(*args, **kwargs)

        with patch.object(pd, "read_csv", side_effect=checked_read_csv):
            frame = self.source().read_frame()
        self.assertEqual(seen, [object])
        self.assertEqual(frame.loc[0, "row_id"], "001")
        self.assertEqual(frame.loc[0, "feature"], "0007")

    def test_rejects_duplicate_ids(self):
        self.path.write_text("row_id,target,x\n1,a,foo\n1,b,bar\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.source().read_frame()

    def test_rejects_missing_role_column_without_leaking_its_name(self):
        self.path.write_text("row_id,feature\n1,private-value\n", encoding="utf-8")
        with self.assertRaises(ValueError) as context:
            self.source().read_frame()
        self.assertNotIn("private-value", str(context.exception))
        self.assertNotIn(str(self.path), str(context.exception))

    def test_rejects_missing_file_and_ambiguous_roles(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.source().read_frame()
        with self.assertRaisesRegex(ValueError, "distinct"):
            CSVSourceSettings(path=self.path, id_column="same", class_column="same")

    def test_rejects_duplicate_headers(self):
        self.path.write_text("row_id,target,target\n1,a,b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.source().read_frame()


if __name__ == "__main__":
    unittest.main()
