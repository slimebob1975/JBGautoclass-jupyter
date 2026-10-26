"""No-ML, no-SQL synthetic tests for CSV run artifact publication."""
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.CSVRunContext import prepare_csv_run
from Storage.CSVRunArtifacts import publish_csv_run
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVRunArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.repo = FileArtifactRepository(root / 'artifacts')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        source = root / 'synthetic.csv'
        source.write_text('uid;class;x\n001;A;one\n002;B;two\n', encoding='utf-8')
        registered = register_csv_dataset(self.repo, self.scope, source,
            id_column='uid', class_column='class', feature_columns=('x',), delimiter=';')
        self.ctx = prepare_csv_run(self.repo, self.scope, registered.reference.artifact_id,
            id_column='uid', class_column='class', feature_columns=('x',))

    def test_manifest_and_aggregate_metrics_are_reopenable_and_scoped(self):
        result = publish_csv_run(self.repo, self.scope, self.ctx, metrics={'mcc': 0.6, 'folds': 3})
        reopened = FileArtifactRepository(self.repo._root)
        with reopened.open_artifact(self.scope, result.manifest.artifact_id) as reader:
            manifest = json.load(reader)
        self.assertEqual(manifest['run_id'], result.run_id)
        self.assertEqual(manifest['dataset_id'], self.ctx.dataset_id)
        self.assertEqual(manifest['status'], 'prepared')
        self.assertNotIn('uid', str(manifest))
        self.assertNotIn('synthetic.csv', str(manifest))
        refs = reopened.list_artifacts(self.scope, run_id=result.run_id)
        self.assertEqual({r.kind for r in refs}, {ArtifactKind.MANIFEST, ArtifactKind.METRICS})
        with reopened.open_artifact(self.scope, result.metrics.artifact_id) as reader:
            self.assertEqual(json.load(reader), {'folds': 3, 'mcc': 0.6})
        with self.assertRaises(FileNotFoundError):
            reopened.get_reference(StorageScope(str(uuid4()), str(uuid4())), result.manifest.artifact_id)

    def test_each_publication_gets_distinct_ids(self):
        first = publish_csv_run(self.repo, self.scope, self.ctx)
        second = publish_csv_run(self.repo, self.scope, self.ctx)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.manifest.artifact_id, second.manifest.artifact_id)
        self.assertIsNone(first.metrics)

    def test_invalid_metrics_rejected_before_any_publication(self):
        original = len(self.repo.list_artifacts(self.scope))
        for bad in ({'username': 'alice'}, {'mcc': float('nan')}, {'path/hack': 1}, {'flag': True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                publish_csv_run(self.repo, self.scope, self.ctx, metrics=bad)
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), original)

    def test_cross_scope_and_tampered_context_rejected(self):
        from dataclasses import replace
        other = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises(ValueError):
            publish_csv_run(self.repo, other, self.ctx)
        with self.assertRaises(FileNotFoundError):
            publish_csv_run(self.repo, self.scope, replace(self.ctx, artifact_id=str(uuid4())))
        self.assertFalse(any(r.run_id for r in self.repo.list_artifacts(self.scope)))


if __name__ == '__main__':
    unittest.main()
