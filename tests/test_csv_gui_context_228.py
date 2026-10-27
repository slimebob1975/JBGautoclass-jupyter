"""Synthetic CSV GUI boundary: explicit scoped identity and no SQL fallback."""
import ast
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4
import unittest

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import list_csv_corrections
from Storage.CSVReclassificationWorkflow import (
    CSVReclassificationContext, prepare_csv_reclassification_context)
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVGUIContext228(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                    'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, dataset,
                                BytesIO(b'uid;class;feature\n005;no;1\nA-17;yes;2\n'))
        config = save_csv_model_configuration(self.repo, self.scope, dataset.artifact_id,
                    id_column='uid', class_column='class', data_columns=['feature'],
                    csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        self.config = config
        self.args = dict(repository=self.repo, scope=self.scope,
                         configuration_id=config.configuration_id,
                         dataset_id=config.dataset_id, dataset_artifact_id=config.artifact_id,
                         id_column='uid', class_column='class',
                         allowed_classes=('no', 'yes'))

    def test_context_matches_exact_dataset_roles_and_scope(self):
        ctx = prepare_csv_reclassification_context(**self.args)
        self.assertIsInstance(ctx, CSVReclassificationContext)
        for field in ('dataset_id', 'dataset_artifact_id', 'id_column', 'class_column'):
            args = dict(self.args, **{field: 'not-the-expected-value'})
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_csv_reclassification_context(**args)
        with self.assertRaises((ValueError, FileNotFoundError)):
            prepare_csv_reclassification_context(**dict(self.args, scope=StorageScope(str(uuid4()), str(uuid4()))))

    def test_confirmation_and_original_file_are_preserved(self):
        ctx = prepare_csv_reclassification_context(**self.args)
        with self.assertRaises(ValueError):
            ctx.correct('005', 'yes')
        self.assertEqual(list_csv_corrections(self.repo, self.scope, self.config.configuration_id), ())
        ctx.correct('005', 'yes', confirmed=True)
        self.assertEqual(len(list_csv_corrections(self.repo, self.scope, self.config.configuration_id)), 1)
        with self.repo.open_artifact(self.scope, self.config.artifact_id) as stream:
            self.assertIn(b'005;no;1', stream.read())

    def test_rejects_ambiguous_classes_and_wrong_row(self):
        for labels in ((), ('yes', 'yes'), ('', 'no'), ('no', 1)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                prepare_csv_reclassification_context(**dict(self.args, allowed_classes=labels))
        ctx = prepare_csv_reclassification_context(**self.args)
        with self.assertRaises(ValueError):
            ctx.correct('missing', 'yes', confirmed=True)

    @staticmethod
    def _gui_methods():
        # Execute unchanged production method bodies without importing optional GUI/ML bootstraps.
        source = Path(__file__).resolve().parents[1] / 'src' / 'GUIHandler.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GUIHandler')
        selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
                    n.name in {'correct_mispredicted_data', 'correct_csv_mispredicted_data'}]
        isolated = ast.Module(body=[ast.ClassDef(name='TestGUI', bases=[], keywords=[], body=selected,
                                                  decorator_list=[])], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {}
        exec(compile(isolated, str(source), 'exec'), namespace)
        return namespace['TestGUI']

    def test_sql_path_preserved_and_csv_dropdown_never_writes_sql(self):
        gui = self._gui_methods()()
        sent = []
        gui.classifier_datalayer = SimpleNamespace(correct_mispredicted_data=lambda *v: sent.append(v))
        gui.widgets = SimpleNamespace(_csv_view_active=lambda: False)
        gui.correct_mispredicted_data('yes', 5)
        self.assertEqual(sent, [('yes', 5)])
        gui.widgets = SimpleNamespace(_csv_view_active=lambda: True)
        with self.assertRaises(ValueError):
            gui.correct_mispredicted_data('yes', '005')
        self.assertEqual(sent, [('yes', 5)])

    def test_explicit_gui_service_checks_source_and_confirmation(self):
        gui = self._gui_methods()()
        gui.widgets = SimpleNamespace(_csv_view_active=lambda: True)
        ctx = prepare_csv_reclassification_context(**self.args)
        with self.assertRaises(ValueError):
            gui.correct_csv_mispredicted_data(ctx, 'yes', '005')
        with self.assertRaises(ValueError):
            gui.correct_csv_mispredicted_data(object(), 'yes', '005', confirmed=True)
        gui.widgets = SimpleNamespace(_csv_view_active=lambda: False)
        with self.assertRaises(ValueError):
            gui.correct_csv_mispredicted_data(ctx, 'yes', '005', confirmed=True)
        gui.widgets = SimpleNamespace(_csv_view_active=lambda: True)
        gui.correct_csv_mispredicted_data(ctx, 'yes', '005', confirmed=True)
        self.assertEqual(len(list_csv_corrections(self.repo, self.scope, self.config.configuration_id)), 1)


if __name__ == '__main__':
    unittest.main()
