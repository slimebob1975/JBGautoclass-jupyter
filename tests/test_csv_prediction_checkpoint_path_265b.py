"""CSV checkpoint location must survive entry into the shared classifier."""
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys


class CSVCheckpointPathTests(unittest.TestCase):
    def test_bound_filename_is_exact_scoped_checkpoint(self):
        from Storage.CSVApplicationRuntime import prepare_csv_prediction_classifier
        from Storage.ArtifactContract import StorageScope

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'csv_model_example.sav'
            filename.touch()
            runtime = SimpleNamespace(data_source=SimpleNamespace(kind='csv'), should_train=lambda: False, should_predict=lambda: True)
            class Restored:
                data_source = SimpleNamespace(kind='csv')
                io = SimpleNamespace(model_name='', model_path='')
                def should_train(self): return False
                def should_predict(self): return True
                def get_model_filename(self): return Path(self.io.model_path) / (self.io.model_name + '.sav')
            restored = Restored()
            reader = SimpleNamespace(correction_revision=None)
            from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
            reader.get_dataset = lambda num_rows=None: None
            sink = SimpleNamespace(save_prediction_data=lambda *args, **kwargs: None)
            layer = CSVPredictionDataLayer(reader, sink)
            received = []

            class FakeClassifier:
                def __init__(self, *, config, logger, datalayer, regression_suite):
                    received.append(config.get_model_filename())
                    self.datalayer = datalayer
                    self.config = config
                    self.handler = SimpleNamespace(datalayer=datalayer)

            module = ModuleType('AutomaticClassifier')
            module.AutomaticClassifier = FakeClassifier
            scope = object.__new__(StorageScope)
            with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer', return_value=(restored, layer)), \
                    patch.dict(sys.modules, {'AutomaticClassifier': module}):
                classifier = prepare_csv_prediction_classifier(str(filename), runtime, None, scope, 'id', None)
            self.assertEqual(received, [filename])
            self.assertIs(classifier.datalayer, layer)

    def test_sql_constructor_untouched(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1] / 'src/AutomaticClassifier.py').read_text()
        self.assertIn('elif source_kind == "sql":', text)


if __name__ == '__main__':
    unittest.main()
