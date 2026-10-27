"""Revision 256: fail-closed CSV model checkpoint at the real retraining task.

Synthetic objects only. The live Iris CSV GUI remains disabled.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'


def portable_retrain():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRunner')
    method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'retrain_model__task')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    context = {'ModelException': RuntimeError}
    exec(compile(module, str(SOURCE), 'exec'), context)
    return context['retrain_model__task']


class CheckpointGate256(TestCase):
    def setup_runner(self, kind='csv', restored=None):
        config = Mock()
        config.data_source.kind = kind
        config.get_model_filename.return_value = '/synthetic/models/checkpoint.sav'
        logger = Mock()
        model = SimpleNamespace(pipeline=object())
        mh = Mock(model=model)
        mh.load_pipeline_from_file.return_value = restored
        mh.retrain_picked_model.return_value = object()
        dh = SimpleNamespace(X=[[1], [2]], Y=['a', 'b'])
        handler = SimpleNamespace(logger=logger)
        runner = SimpleNamespace(config=config, logger=logger, mh=mh, dh=dh, handler=handler)
        return runner

    def test_csv_unreadable_checkpoint_is_fatal_before_refit_or_save(self):
        runner = self.setup_runner()
        with self.assertRaisesRegex(RuntimeError, 'checkpoint could not be reloaded'):
            portable_retrain()(runner)
        runner.mh.retrain_picked_model.assert_not_called()
        runner.mh.save_model_to_file.assert_not_called()
        runner.handler.logger.print_warning.assert_not_called()

    def test_csv_successful_checkpoint_refits_and_saves_full_data(self):
        restored = object()
        runner = self.setup_runner(restored=restored)
        original = runner.mh.model.pipeline
        result = portable_retrain()(runner)
        self.assertIs(result['cross_trained_model'], restored)
        self.assertIs(result['trained_model'], runner.mh.model.pipeline)
        runner.mh.retrain_picked_model.assert_called_once_with(original, runner.dh.X, runner.dh.Y)
        runner.mh.save_model_to_file.assert_called_once_with('/synthetic/models/checkpoint.sav')

    def test_csv_save_failure_propagates(self):
        runner = self.setup_runner(restored=object())
        runner.mh.save_model_to_file.side_effect = OSError('synthetic disk failure')
        with self.assertRaisesRegex(OSError, 'synthetic disk failure'):
            portable_retrain()(runner)

    def test_sql_legacy_fallback_preserved(self):
        runner = self.setup_runner(kind='sql')
        portable_retrain()(runner)
        runner.mh.retrain_picked_model.assert_called_once()
        runner.handler.logger.print_warning.assert_called_once()


if __name__ == '__main__':
    main()
