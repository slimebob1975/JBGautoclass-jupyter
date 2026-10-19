"""Registry defaults and estimator routing for revision 145b."""
import unittest
from JBGModelSettings import REGISTRY, configure_estimator, defaults, validate
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class ExtendedRegistryTests(unittest.TestCase):
    def test_registry_covers_core_families(self):
        for code in ('CBST', 'RFCL', 'LRN', 'KNN', 'DTC', 'GNB', 'MNB',
                     'LSVC', 'SVC', 'ETC', 'GBC', 'HIST', 'MLPC', 'QDA'):
            self.assertIn(code, REGISTRY)
        self.assertEqual(defaults()['QDA']['reg_param'], 0.1)

    def test_parameters_reach_estimators(self):
        cases = [
            ('LRN', LogisticRegression(), {'C': 0.2, 'class_weight': 'balanced'}),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': 11, 'weights': 'distance'}),
            ('DTC', DecisionTreeClassifier(), {'min_samples_leaf': 4}),
            ('REC', RidgeClassifier(), {'alpha': 3.0}),
            ('GBC', GradientBoostingClassifier(), {'learning_rate': 0.03}),
            ('HIST', HistGradientBoostingClassifier(), {'max_iter': 125}),
            ('QDA', QuadraticDiscriminantAnalysis(reg_param=0.1), {'reg_param': 0.3}),
        ]
        for code, estimator, params in cases:
            with self.subTest(code=code):
                configured = configure_estimator(code, estimator, {code: params})
                for key, value in params.items():
                    self.assertEqual(configured.get_params()[key], value)

    def test_none_class_weight_and_isolated_model_settings(self):
        model = configure_estimator('LRN', LogisticRegression(),
                                    {'LRN': {'class_weight': 'None'}})
        self.assertIsNone(model.class_weight)
        self.assertEqual(validate({'LRN': {'C': 2.5}}), {'LRN': {'C': 2.5}})
        with self.assertRaises(ValueError):
            validate({'KNN': {'n_neighbors': 0}})
        with self.assertRaises(ValueError):
            validate({'LRN': {'C': 'invalid'}})


if __name__ == '__main__':
    unittest.main()
