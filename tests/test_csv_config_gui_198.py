"""Scoped CSV configuration catalog and GUI wiring regression tests."""
import ast
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration, list_csv_model_configurations
from Storage.FileArtifactRepository import FileArtifactRepository

class CSVConfigurationGUI198Tests(unittest.TestCase):
    def test_catalog_only_returns_current_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = FileArtifactRepository(Path(tmp))
            scope = StorageScope(str(uuid4()), str(uuid4()))
            datasets = [ArtifactReference(scope, str(uuid4()), ArtifactKind.DATASET, 'text/csv',dataset_id=str(uuid4())) for _ in range(2)]
            for d in datasets:
                repo.save_artifact(scope, d, BytesIO(b'uid,label,x\n001,A,1\n'))
            a,b = [save_csv_model_configuration(repo,scope,d.artifact_id,id_column='uid',class_column='label',data_columns=('x',),csv_format={'delimiter':',','quotechar':'"','encoding':'utf-8'}) for d in datasets]
            self.assertEqual(list_csv_model_configurations(repo,scope,datasets[0].artifact_id),(('Configuration '+a.configuration_id[:8],a.configuration_id),))
            self.assertEqual(list_csv_model_configurations(repo,scope,datasets[1].artifact_id),(('Configuration '+b.configuration_id[:8],b.configuration_id),))
            other = StorageScope(str(uuid4()),str(uuid4()))
            with self.assertRaises(FileNotFoundError):
                list_csv_model_configurations(repo,other,datasets[0].artifact_id)

    def test_gui_handlers_are_registered_without_sql_execution(self):
        root = Path(__file__).resolve().parents[1] / 'src/GUI'
        tree=ast.parse((root/'Widgets.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Widgets')
        methods={n.name: ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertIn('save_configuration_button.on_click(self._csv_save_configuration)',methods['connection_form'])
        self.assertIn('load_configuration_button.on_click(self._csv_load_configuration)',methods['connection_form'])
        for handler in ('_csv_save_configuration','_csv_load_configuration'):
            self.assertNotIn('run_classifier',methods[handler])
            self.assertNotIn('start_button_actions',methods[handler])
        self.assertIn('self._restore_source_roles',methods['_csv_load_configuration'])

if __name__=='__main__': unittest.main()
