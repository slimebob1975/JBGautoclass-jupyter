"""Synthetic, dependency-light tests for explicit CSV GUI context lifecycle."""
import ast
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import list_csv_corrections
from Storage.CSVReclassificationWorkflow import prepare_csv_reclassification_context
from Storage.FileArtifactRepository import FileArtifactRepository

ROOT = Path(__file__).resolve().parents[1]


def isolated_methods(source_file, class_name, names):
    tree = ast.parse(source_file.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    module = ast.Module(body=[ast.ClassDef(name='Isolated', bases=[], keywords=[], body=methods,
                                           decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(source_file), 'exec'), ns)
    return ns['Isolated']


class ContextBinding229(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                    'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, dataset, BytesIO(b'uid;class;feature\n005;no;1\n'))
        roles = save_csv_model_configuration(self.repo, self.scope, dataset.artifact_id,
                  id_column='uid', class_column='class', data_columns=['feature'],
                  csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        self.context = prepare_csv_reclassification_context(
            self.repo, self.scope, roles.configuration_id, dataset_id=roles.dataset_id,
            dataset_artifact_id=roles.artifact_id, id_column='uid', class_column='class',
            allowed_classes=('no', 'yes'))
        cls = isolated_methods(ROOT/'src'/'GUIHandler.py', 'GUIHandler', {
            'bind_csv_reclassification_context', 'clear_csv_reclassification_context',
            'correct_bound_csv_mispredicted_data', 'correct_csv_mispredicted_data'})
        self.gui = cls()
        self.gui.widgets = SimpleNamespace(_csv_view_active=lambda: True,
            _csv_catalog_storage=lambda: (self.repo, self.scope),
            _csv_current_dataset_artifact=lambda: dataset.artifact_id)

    def test_valid_binding_and_confirmed_edit(self):
        self.gui.bind_csv_reclassification_context(self.context)
        with self.assertRaises(ValueError):
            self.gui.correct_bound_csv_mispredicted_data('yes', '005')
        self.gui.correct_bound_csv_mispredicted_data('yes', '005', confirmed=True)
        self.assertEqual(len(list_csv_corrections(self.repo, self.scope,
                                                   self.context.configuration_id)), 1)

    def test_wrong_scope_or_dataset_fails_closed(self):
        self.gui.widgets._csv_current_dataset_artifact = lambda: str(uuid4())
        with self.assertRaises(ValueError):
            self.gui.bind_csv_reclassification_context(self.context)
        self.assertIsNone(self.gui._active_csv_reclassification_context)
        self.gui.widgets._csv_current_dataset_artifact = lambda: self.context.dataset_artifact_id
        self.gui.widgets._csv_catalog_storage = lambda: (self.repo, StorageScope(str(uuid4()), str(uuid4())))
        with self.assertRaises(ValueError):
            self.gui.bind_csv_reclassification_context(self.context)

    def test_source_change_after_binding_rejects_and_clears(self):
        self.gui.bind_csv_reclassification_context(self.context)
        self.gui.widgets._csv_view_active = lambda: False
        with self.assertRaises(ValueError):
            self.gui.correct_bound_csv_mispredicted_data('yes', '005', confirmed=True)
        self.assertIsNone(self.gui._active_csv_reclassification_context)

    def test_context_is_not_inferred_or_auto_bound(self):
        with self.assertRaises(ValueError):
            self.gui.correct_bound_csv_mispredicted_data('yes', '005', confirmed=True)
        with self.assertRaises(ValueError):
            self.gui.bind_csv_reclassification_context(object())

    def test_csv_widget_lifecycle_invalidates_binding(self):
        tree = ast.parse((ROOT/'src'/'GUI'/'Widgets.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Widgets')
        for name in ('_csv_source_selected', '_csv_file_changed', '_csv_register_dataset',
                     '_csv_load_saved', '_csv_erase_saved'):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            first = method.body[1]  # after docstring
            self.assertIsInstance(first, ast.Expr)
            self.assertEqual(first.value.func.attr, 'clear_csv_reclassification_context')
