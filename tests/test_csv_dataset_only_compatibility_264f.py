"""Revision 264f: dataset roles, not historical training choices, gate reuse."""
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVSavedModelInspection import inspect_saved_csv_model

try:
    from Config import Config
except ImportError:
    Config = None


@unittest.skipIf(Config is None, 'Full JBG dependencies unavailable')
class DatasetOnlyCompatibility264f(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.repo = FileArtifactRepository(root / 'objects')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref, BytesIO(b'id,feature,class\n1,1,a\n2,2,b\n'))
        from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
        self.runtime, self.saved = prepare_csv_gui_training_config(
            Config(), self.repo, self.scope, self.ref.artifact_id,
            id_column='id', class_column='class', data_columns=('feature',),
            csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8'},
            model_root=root / 'models')
        self.root = root / 'models'
        self.filename = Path(self.runtime.get_model_filename())
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.filename.write_bytes(b'synthetic-not-an-actual-pickle')

    def inspect(self, saved, source_kind="csv"):
        with patch('JBGModelPersistence.load_model_artifact_with_compatibility',
                   return_value=((saved, None), {'source_kind': source_kind})), \
             patch('JBGModelCompatibility.review_model_source_boundary', return_value=SimpleNamespace(ok=True)), \
             patch('JBGModelCompatibility.review_artifact_semantics', return_value=SimpleNamespace(ok=True)):
            return inspect_saved_csv_model(self.repo, self.scope, self.ref.artifact_id,
                                           self.saved.configuration_id, model_root=self.root)

    def test_saved_training_recipe_does_not_need_to_match_registration(self):
        self.assertEqual(self.inspect(self.runtime), self.filename)
        # Exact recipe differences are verified structurally: inspection has
        # no dependency on the immutable registration's mode settings.
        import inspect as pyinspect
        source = pyinspect.getsource(inspect_saved_csv_model)
        self.assertNotIn('roles.shared_json.get(\'mode\'', source)

    def test_wrong_feature_roles_are_rejected(self):
        with patch.object(self.runtime, 'get_numerical_column_names', return_value=['wrong']):
            with self.assertRaisesRegex(ValueError, 'roles differ'):
                self.inspect(self.runtime)

    def test_sql_source_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CSV source compatibility'):
            self.inspect(self.runtime, source_kind='sql')

if __name__ == '__main__':
    unittest.main()
