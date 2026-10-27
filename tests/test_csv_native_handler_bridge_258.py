"""258: real CSV repository -> native JBG DatasetHandler -> fitted model probe.

Synthetic records only. GUI is deliberately not activated. A full ordinary
ModelHandler.train_model / AutomaticClassifier.run fit remains a separate gate.
"""
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, skipIf
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVTrainingDataLayer import prepare_csv_training_datalayer
from Storage.FileArtifactRepository import FileArtifactRepository
from JBGModelPersistence import save_model_artifact, load_model_artifact_with_compatibility

try:
    from AutomaticClassifier import AutomaticClassifier
    from JBGTaskRunner import TaskRunner
    from Config import Config
except (ImportError, OSError):
    AutomaticClassifier = None


@skipIf(AutomaticClassifier is None, 'Full JBG handler dependencies unavailable')
class CSVNativeHandlerBridge258(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = FileArtifactRepository(self.root / 'artifacts')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        rng = np.random.default_rng(258)
        centers = [(4.9, 3.4, 1.5, .2), (5.9, 2.7, 4.2, 1.3), (6.6, 3., 5.5, 2.)]
        names = ('a', 'b', 'c')
        lines = ['f1,f2,f3,f4,row_id,target']
        for i, (center, label) in enumerate(zip(centers, names)):
            for j, values in enumerate(rng.normal(center, .13, (50, 4))):
                observed = label if j < 45 else ''
                lines.append(','.join([*(format(v, '.7f') for v in values),
                                       f'{i:01d}{j:03d}', observed]))
        self.repo.save_artifact(self.scope, self.ref,
                                BytesIO(('\n'.join(lines) + '\n').encode('utf-8')))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id, id_column='row_id',
            class_column='target', data_columns=('f1', 'f2', 'f3', 'f4'),
            csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8'})
        self.config = Config()
        self.config.data_source.kind = 'csv'
        self.config.data_source.dataset_id = self.ref.dataset_id
        self.config.data_source.artifact_id = self.ref.artifact_id
        self.config.connection.data_numerical_columns = ['f1', 'f2', 'f3', 'f4']
        self.config.connection.data_text_columns = []
        self.config.connection.id_column = 'row_id'
        self.config.connection.class_column = 'target'
        self.config.mode.train = True
        self.config.mode.predict = False
        self.config.mode.mispredicted = False
        self.config.mode.calculate_dark_numbers = False
        self.config.mode.calculate_feature_importance = False
        self.config.mode.use_metas = False
        self.config.mode.test_size = .2
        self.config.io.model_path = str(self.root / 'models')
        self.config.io.model_name = 'synthetic'
        self.logger = Mock()
        self.layer = prepare_csv_training_datalayer(
            self.config, self.repo, self.scope, self.roles.configuration_id)

    def test_native_dataset_handler_and_real_estimator_roundtrip(self):
        with ExitStack() as guards:
            guards.enter_context(patch('socket.create_connection', side_effect=AssertionError('network forbidden')))
            import pyodbc
            guards.enter_context(patch.object(pyodbc, 'connect', side_effect=AssertionError('SQL forbidden')))
            classifier = AutomaticClassifier(config=self.config, logger=self.logger,
                                             datalayer=self.layer)
            runner = TaskRunner(self.layer, self.config, self.logger, classifier.handler)
            runner.dh, runner.mh, runner.ph = classifier.dh, classifier.mh, classifier.ph
            loaded = runner.fetch_data__task()['data']
            self.assertEqual(len(loaded), 135)
            runner.load_dataset__task(loaded)
            runner.separate_dataset__task()
            dh = runner.dh
            self.assertEqual(len(dh.X), 135)
            self.assertEqual(len(dh.X_train) + len(dh.X_validation), 135)
            self.assertEqual(len(dh.classes), 3)
            self.assertEqual(len(dh.keys), 135)
            self.assertTrue(all(isinstance(key, str) for key in dh.keys))
            model = Pipeline([('scale', StandardScaler()),
                              ('logistic', LogisticRegression(max_iter=300))])
            model.fit(dh.X_train.to_numpy(dtype=float), dh.Y_train)
            path = self.root / 'synthetic.sav'
            metadata = {'schema_version': 2, 'source_kind': 'csv',
                        'dataset_id': self.ref.dataset_id}
            save_model_artifact(path, {}, None, (None,) * 5, model, None, 4,
                                compatibility=metadata)
            artifact, saved_metadata = load_model_artifact_with_compatibility(path)
            self.assertEqual(saved_metadata, metadata)
            np.testing.assert_array_equal(
                model.predict(dh.X_validation.to_numpy(dtype=float)),
                artifact[3].predict(dh.X_validation.to_numpy(dtype=float)))
        self.assertFalse(hasattr(self.layer, 'save_prediction_data'))

    def test_registered_train_only_reader_does_not_include_unlabelled_rows(self):
        self.assertEqual(len(self.layer.get_dataset()), 135)
        self.config.mode.train = False
        self.config.mode.predict = True
        from Storage.CSVTrainingAdapter import CSVTrainingAdapter
        prediction_reader = CSVTrainingAdapter(self.repo, self.scope,
                                               self.roles.configuration_id, self.config)
        self.assertEqual(len(prediction_reader.get_dataset()), 15)


class CSVNativeFixturePortable258(TestCase):
    def test_registered_csv_rows_roundtrip_without_optional_handlers(self):
        from test_csv_correction_reader_226 import CSVCorrectionReader226
        from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
        # Use the real 226 file/repository fixture; this proves the storage
        # boundary locally, independently of optional ML bootstrap packages.
        case = CSVCorrectionReader226('test_label_filters_apply_after_overlay')
        case.setUp()
        self.addCleanup(case.doCleanups)
        reader = case.adapter(train=True, predict=False)
        layer = CSVTrainingDataLayer(reader)
        self.assertEqual(layer.get_dataset(), [('1', '005', 'A'), ('3', '009', 'B')])
        self.assertFalse(hasattr(layer, 'save_prediction_data'))


if __name__ == '__main__':
    main()
