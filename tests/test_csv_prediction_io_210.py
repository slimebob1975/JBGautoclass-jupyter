"""Revision 210: CSV read/write composition without SQL or ML execution."""
import unittest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

from Storage.ArtifactContract import StorageKind, StorageScope
from Storage.CSVPredictionIO import prepare_csv_prediction_io
from Storage.CSVRunContext import CSVRunContext


class CSVPredictionIOTests(unittest.TestCase):
    def setUp(self):
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset_id = str(uuid4())
        self.artifact_id = str(uuid4())
        self.roles = SimpleNamespace(dataset_id=self.dataset_id, artifact_id=self.artifact_id,
                                     id_column="row_id", class_column="target",
                                     text_columns=("text",), data_columns=("number",))
        self.reader = SimpleNamespace(configuration=self.roles)
        self.config = SimpleNamespace(data_source=SimpleNamespace(
            kind="csv", dataset_id=self.dataset_id, artifact_id=self.artifact_id))
        self.context = CSVRunContext(self.scope, self.dataset_id, self.artifact_id, "row_id",
                                     "target", ("text", "number"))

    def _prepare(self, config=None, context=None):
        with patch("Storage.CSVPredictionIO.prepare_csv_model_runtime",
                   return_value=(config or self.config, self.reader)) as load, patch(
                   "Storage.CSVPredictionIO.prepare_csv_run",
                   return_value=context or self.context) as make:
            result = prepare_csv_prediction_io("model", object(), object(), self.scope, "cfg")
        return result, load, make

    def test_connects_reader_and_sink_to_same_scope(self):
        (config, reader, sink), load, make = self._prepare()
        self.assertIs(config, self.config)
        self.assertIs(reader, self.reader)
        self.assertIs(sink.context, self.context)
        self.assertEqual(sink.scope, self.scope)
        self.assertEqual(make.call_args.kwargs["feature_columns"], ("text", "number"))
        load.assert_called_once()

    def test_rejects_invalid_scope_before_model_loading(self):
        with patch("Storage.CSVPredictionIO.prepare_csv_model_runtime") as load:
            with self.assertRaises(ValueError):
                prepare_csv_prediction_io("model", object(), object(), None, "cfg")
            load.assert_not_called()

    def test_rejects_mismatched_saved_dataset(self):
        mismatch = SimpleNamespace(data_source=SimpleNamespace(
            kind="csv", dataset_id=str(uuid4()), artifact_id=self.artifact_id))
        with self.assertRaisesRegex(ValueError, "identity"):
            self._prepare(config=mismatch)

    def test_rejects_sql_saved_configuration(self):
        mismatch = SimpleNamespace(data_source=SimpleNamespace(kind="sql"))
        with self.assertRaisesRegex(ValueError, "identity"):
            self._prepare(config=mismatch)

    def test_rejects_context_mismatch(self):
        wrong = CSVRunContext(self.scope, str(uuid4()), self.artifact_id, "row_id", "target",
                              ("text", "number"), StorageKind.FILES)
        with self.assertRaisesRegex(ValueError, "context"):
            self._prepare(context=wrong)


if __name__ == "__main__":
    unittest.main()
