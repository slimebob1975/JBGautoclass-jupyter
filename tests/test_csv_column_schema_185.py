"""Private, SQL-independent CSV column role contract tests."""
import tempfile
import unittest
from pathlib import Path
from CSVColumnSchema import inspect_csv_columns, validate_csv_column_roles


class CSVColumnSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "confidential.csv"

    def test_text_id_and_role_candidates(self):
        self.path.write_text("key;target;feature\n001;yes;red\n002;no;blue\n003;yes;green\n", encoding="utf-8-sig")
        schema = inspect_csv_columns(self.path, delimiter=";")
        self.assertEqual(schema.rows, 3)
        self.assertIn("key", schema.unique_id_candidates)
        self.assertIn("target", schema.class_candidates)
        self.assertEqual(validate_csv_column_roles(schema, id_column="key", class_column="target", feature_columns=["feature"]), ("feature",))

    def test_duplicate_and_blank_ids_not_eligible(self):
        self.path.write_text("key,target,x\n1,a,z\n1,b,y\n,a,x\n")
        schema = inspect_csv_columns(self.path)
        self.assertNotIn("key", schema.unique_id_candidates)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_csv_column_roles(schema, id_column="key", class_column="target", feature_columns=["x"])

    def test_roles_reject_overlap_and_empty_features(self):
        self.path.write_text("key,target,x\n1,a,z\n2,b,y\n")
        schema = inspect_csv_columns(self.path)
        for values in [[], ["key"], ["target"], ["x", "x"]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_csv_column_roles(schema, id_column="key", class_column="target", feature_columns=values)

    def test_invalid_file_shape_is_private(self):
        self.path.write_text("private_key,target\n1,secret,extra\n")
        with self.assertRaises(ValueError) as error:
            inspect_csv_columns(self.path)
        self.assertNotIn("private_key", str(error.exception))
        self.assertNotIn("secret", str(error.exception))
        self.assertNotIn(str(self.path), str(error.exception))

    def test_no_sql_import_needed(self):
        import sys
        self.assertNotIn("SQLDataLayer", sys.modules)


if __name__ == "__main__":
    unittest.main()
