"""Synthetic regression coverage for complete, scoped CSV dataset erasure."""
from io import BytesIO
from pathlib import Path
import json
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVSavedModelInspection import _model_filename
from Storage.CSVDatasetCascadeErasure import erase_csv_dataset_cascade


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.repo = FileArtifactRepository(root / 'objects')
        self.models = (root / 'models').resolve()
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = self.new_dataset(self.scope)
        self.config = save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
            id_column='id', class_column='class', data_columns=('x',))
        self.checkpoint = _model_filename(self.models, self.scope, self.config.dataset_id, self.config.artifact_id)
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'checkpoint')
        self.manifest, self.payload = self.new_prediction(self.scope, self.dataset)

    def new_dataset(self, scope):
        ref = ArtifactReference(scope, str(uuid4()), ArtifactKind.DATASET, 'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(scope, ref, BytesIO(b'id,class,x\n001,,1\n'))
        return ref

    def new_prediction(self, scope, dataset):
        run_id = str(uuid4())
        payload = ArtifactReference(scope, str(uuid4()), ArtifactKind.PREDICTIONS,
                                    'application/json', run_id=run_id, dataset_id=dataset.dataset_id)
        manifest = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST,
                                     'application/json', run_id=run_id, dataset_id=dataset.dataset_id)
        self.repo.save_artifact(scope, payload, BytesIO(json.dumps({'schema_version':1,'rows': [],'rate_type':'probability'}).encode()))
        obj = dict(schema_version=1,run_id=run_id,dataset_id=dataset.dataset_id,
            dataset_artifact_id=dataset.artifact_id,predictions_artifact_id=payload.artifact_id,
            row_count=0,storage_kind='files',status='predictions_saved',created_at='2026-10-28T12:00:00+00:00')
        self.repo.save_artifact(scope, manifest, BytesIO(json.dumps(obj).encode()))
        return manifest, payload

    def test_cascade_erases_only_selected_dataset(self):
        other = self.new_dataset(self.scope)
        other_manifest, other_payload = self.new_prediction(self.scope, other)
        self.assertTrue(erase_csv_dataset_cascade(self.repo, self.scope, self.dataset.artifact_id, model_root=self.models))
        self.assertFalse(self.checkpoint.exists())
        for ref in (self.dataset, self.config.reference, self.manifest, self.payload):
            with self.assertRaises(FileNotFoundError):
                self.repo.get_reference(self.scope, ref.artifact_id)
        for ref in (other, other_manifest, other_payload):
            self.assertEqual(self.repo.get_reference(self.scope, ref.artifact_id), ref)

    def test_wrong_scope_and_other_dataset_fail_closed(self):
        other = self.new_dataset(self.scope)
        foreign = StorageScope(str(uuid4()),str(uuid4()))
        for scope, identifier in ((foreign,self.dataset.artifact_id), (self.scope,str(uuid4()))):
            with self.assertRaises((ValueError, FileNotFoundError)):
                erase_csv_dataset_cascade(self.repo, scope, identifier, model_root=self.models)
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual(self.repo.get_reference(self.scope, other.artifact_id), other)

    def test_orphan_prediction_blocks_without_deletion(self):
        orphan = ArtifactReference(self.scope,str(uuid4()),ArtifactKind.PREDICTIONS,
            'application/json',run_id=str(uuid4()),dataset_id=self.dataset.dataset_id)
        self.repo.save_artifact(self.scope, orphan, BytesIO(b'{}'))
        with self.assertRaises(ValueError):
            erase_csv_dataset_cascade(self.repo,self.scope,self.dataset.artifact_id,model_root=self.models)
        self.assertTrue(self.checkpoint.exists())
        self.repo.get_reference(self.scope,self.dataset.artifact_id)

    def test_symlink_checkpoint_blocks_without_deletion(self):
        self.checkpoint.unlink()
        outside = self.models / 'outside.sav'
        outside.write_bytes(b'untouched')
        try:
            self.checkpoint.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest('No symlink support')
        with self.assertRaises(ValueError):
            erase_csv_dataset_cascade(self.repo,self.scope,self.dataset.artifact_id,model_root=self.models)
        self.assertEqual(outside.read_bytes(),b'untouched')
        self.repo.get_reference(self.scope,self.dataset.artifact_id)

    def test_ambiguous_model_manifest_blocks_before_deletion(self):
        unknown = ArtifactReference(self.scope,str(uuid4()),ArtifactKind.MANIFEST,
            'application/json',dataset_id=self.dataset.dataset_id)
        self.repo.save_artifact(self.scope,unknown,BytesIO(b'{}'))
        with self.assertRaises(ValueError):
            erase_csv_dataset_cascade(self.repo,self.scope,self.dataset.artifact_id,model_root=self.models)
        self.assertTrue(self.checkpoint.exists())
        self.repo.get_reference(self.scope,self.dataset.artifact_id)


if __name__ == '__main__':
    unittest.main()
