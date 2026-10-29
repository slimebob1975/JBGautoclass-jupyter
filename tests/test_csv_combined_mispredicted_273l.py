"""273l: combined CSV training, prediction and shared misprediction acceptance.

This is intentionally a full-stack regression gate, not a mocked TaskRunner
unit test. GUI activation requires this test to PASS on the Windows ML host.
"""
import json
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest import skipIf
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_csv_native_handler_bridge_258 import (
    AutomaticClassifier, CSVNativeHandlerBridge258,
)


@skipIf(AutomaticClassifier is None, "Full JBG ML dependencies unavailable")
class CSVCombinedMispredicted273l(CSVNativeHandlerBridge258):
    test_native_dataset_handler_and_real_estimator_roundtrip = None
    test_registered_train_only_reader_does_not_include_unlabelled_rows = None

    def test_real_combined_fit_predict_and_mispredicted_publication(self):
        from JBGMeta import (
            Algorithm, AlgorithmTuple, Oversampling, Undersampling,
            Preprocess, PreprocessTuple, Reduction, ReductionTuple,
        )
        from JBGModelPersistence import load_model_artifact_with_compatibility
        from Storage.ArtifactContract import ArtifactKind
        from Storage.CSVCombinedDataLayer import prepare_csv_combined_datalayer

        config = self.config
        config.mode.predict = True
        config.mode.mispredicted = True
        config.script_path = self.root
        config.io.model_path = "models"
        config.mode.algorithm = AlgorithmTuple([Algorithm.LRN])
        config.mode.preprocessor = PreprocessTuple([Preprocess.NOS])
        config.mode.feature_selection = ReductionTuple([Reduction.NOR])
        config.mode.oversampler = Oversampling.NOG
        config.mode.undersampler = Undersampling.NUG
        config.mode.oversamplers = ["NOG"]
        config.mode.undersamplers = ["NUG"]
        config.mode.max_iterations = 300
        (self.root / "models").mkdir()
        (self.root / "output" / "csvs").mkdir(parents=True)

        from Helpers import build_model_performance_matrix, save_matrix_as_csv
        self.logger.print_test_performance.side_effect = (
            lambda results, filepath: save_matrix_as_csv(
                build_model_performance_matrix(results), filepath))
        self.logger.get_log_output.return_value = ""
        self.logger.get_log_output_html.return_value = ""
        model_path = Path(config.get_model_filename())
        self.assertFalse(model_path.exists())
        layer = prepare_csv_combined_datalayer(
            config, self.repo, self.scope, self.roles.configuration_id)
        training, unknown = layer.read_partitions()
        self.assertEqual((len(training), len(unknown)), (135, 15))
        self.assertEqual(len({row[-2] for row in training + unknown}), 150)
        self.assertTrue(all(row[-1] == "" for row in unknown))

        with ExitStack() as guards:
            guards.enter_context(patch("socket.create_connection",
                side_effect=AssertionError("Network access forbidden")))
            import pyodbc
            guards.enter_context(patch.object(pyodbc, "connect",
                side_effect=AssertionError("SQL connection forbidden")))
            from SQLDataLayer import DataLayer as SQLDataLayer
            guards.enter_context(patch.object(SQLDataLayer, "__init__",
                side_effect=AssertionError("SQL layer forbidden")))
            mock_mail = guards.enter_context(patch(
                "JBGTaskRunner.send_email", return_value=False))
            classifier = AutomaticClassifier(
                config=config, logger=self.logger, datalayer=layer,
                regression_suite=False)
            self.assertIs(classifier.handler.datalayer, layer)
            result = classifier.run()
            self.assertIsInstance(result, dict)
            mock_mail.assert_called_once()

        self.assertTrue(model_path.is_file(), "Combined run did not save model")
        payload, metadata = load_model_artifact_with_compatibility(model_path)
        self.assertEqual(metadata["source_kind"], "csv")
        self.assertIsNotNone(payload[3], "Saved model is not fitted")
        manifests = [ref for ref in self.repo.list_artifacts(self.scope)
                     if ref.kind == ArtifactKind.MANIFEST and ref.run_id is not None]
        self.logger.print_task_header.assert_any_call(title="Calculating mispredictions")
        # A fully correct model produces no mispredicted rows, so the shared
        # SQL/CSV analysis intentionally does not publish an empty report.
        published = []
        for ref in manifests:
            with self.repo.open_artifact(self.scope, ref.artifact_id) as stream:
                published.append(json.load(stream))
        diagnostics = [item for item in published
                       if item.get("status") == "mispredicted_saved"]
        predictions = [item for item in published
                       if "predictions_artifact_id" in item]
        self.assertEqual(len(predictions), 1)
        mispredicted_count = classifier.ph.num_mispredicted
        self.assertGreaterEqual(mispredicted_count, 0)
        self.assertEqual(len(diagnostics), 1 if mispredicted_count else 0)
        self.assertEqual(len(manifests), 2 if mispredicted_count else 1)
        if mispredicted_count:
            diagnostic = diagnostics[0]
            self.assertEqual(diagnostic["dataset_id"], self.ref.dataset_id)
            self.assertEqual(diagnostic["dataset_artifact_id"], self.ref.artifact_id)
            self.assertEqual(diagnostic["row_count"], mispredicted_count)
            with self.repo.open_artifact(self.scope, diagnostic["diagnostic_artifact_id"]) as stream:
                import pandas as pd
                from io import BytesIO
                table = pd.read_csv(BytesIO(stream.read()), dtype={"row_id": str})
            self.assertEqual(len(table), diagnostic["row_count"])
            self.assertIn("Actual", table.columns)
            self.assertIn("Predicted", table.columns)
            self.assertIn("Model", table.columns)
            self.assertTrue(set(table["row_id"]).issubset({row[-2] for row in training}))
        manifest = predictions[0]
        self.assertEqual(manifest["dataset_id"], self.ref.dataset_id)
        self.assertEqual(manifest["dataset_artifact_id"], self.ref.artifact_id)
        self.assertEqual(manifest["row_count"], 15)
        with self.repo.open_artifact(
                self.scope, manifest["predictions_artifact_id"]) as stream:
            rows = json.load(stream)["rows"]
        self.assertEqual(len(rows), 15)
        self.assertEqual({row["key"] for row in rows},
                         {row[-2] for row in unknown})
        self.assertTrue(all(isinstance(row["key"], str) for row in rows))
        self.assertTrue(all(row["prediction"] in {"a", "b", "c"}
                            for row in rows))


if __name__ == "__main__":
    import unittest
    unittest.main()
