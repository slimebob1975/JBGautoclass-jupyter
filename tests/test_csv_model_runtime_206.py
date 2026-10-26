import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import sys

from Storage.CSVModelRuntime import prepare_csv_model_runtime


def config(kind='csv', train=False):
    return SimpleNamespace(data_source=SimpleNamespace(kind=kind), should_train=lambda: train)


class CSVModelRuntimeTests(unittest.TestCase):
    def test_reuses_model_config_and_scoped_adapter(self):
        runtime = config()
        saved = config()
        repository, scope = object(), object()
        load = MagicMock(return_value=saved)
        adapter = MagicMock()
        with patch.dict(sys.modules, {"Config": SimpleNamespace(Config=SimpleNamespace(load_config_from_model_file=load)),
                                      "Storage.CSVTrainingAdapter": SimpleNamespace(CSVTrainingAdapter=adapter)}):
            cfg, reader = prepare_csv_model_runtime('existing.sav', runtime, repository, scope, 'configuration-id')
        self.assertIs(cfg, saved)
        self.assertIs(reader, adapter.return_value)
        load.assert_called_once_with('existing.sav', config=runtime)
        adapter.assert_called_once_with(repository, scope, 'configuration-id', saved)

    def test_sql_mode_cannot_enter_csv_preparation(self):
        with self.assertRaisesRegex(ValueError, 'CSV'):
            prepare_csv_model_runtime('unused', config('sql'), None, None, 'id')

    def test_training_mode_cannot_reuse_model(self):
        with self.assertRaisesRegex(ValueError, 'prediction-only'):
            prepare_csv_model_runtime('unused', config(train=True), None, None, 'id')

    def test_invalid_saved_model_fails_before_data_access(self):
        adapter = MagicMock()
        with patch.dict(sys.modules, {"Config": SimpleNamespace(Config=SimpleNamespace(load_config_from_model_file=MagicMock(side_effect=ValueError("bad model")))),
                                      "Storage.CSVTrainingAdapter": SimpleNamespace(CSVTrainingAdapter=adapter)}):
            with self.assertRaisesRegex(ValueError, 'bad model'):
                prepare_csv_model_runtime('bad.sav', config(), None, None, 'id')
            adapter.assert_not_called()

    def test_scope_mismatch_from_adapter_is_not_suppressed(self):
        with patch.dict(sys.modules, {"Config": SimpleNamespace(Config=SimpleNamespace(load_config_from_model_file=MagicMock(return_value=config()))),
                                      "Storage.CSVTrainingAdapter": SimpleNamespace(CSVTrainingAdapter=MagicMock(side_effect=ValueError("dataset identity mismatch")))}):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                prepare_csv_model_runtime('existing.sav', config(), None, None, 'id')


if __name__ == '__main__':
    unittest.main()
