import unittest
from JBGModelSettings import REGISTRY, defaults, validate, configure_estimator
from sklearn.ensemble import RandomForestClassifier

class RegistryTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(defaults()['CBST']['depth'], 6)
        self.assertEqual(validate({'CBST': {'depth': 8}})['CBST']['depth'], 8)
        for value in (0, 13, '8', True):
            with self.assertRaises(ValueError):
                validate({'CBST': {'depth': value}})
        with self.assertRaises(ValueError):
            validate({'UNKNOWN': {}})
        with self.assertRaises(ValueError):
            validate({'RFCL': {'bad_name': 4}})

    def test_estimator_settings(self):
        estimator = configure_estimator('RFCL', RandomForestClassifier(),
                          {'RFCL': {'n_estimators': 150, 'min_samples_leaf': 3}})
        self.assertEqual(estimator.n_estimators, 150)
        self.assertEqual(estimator.min_samples_leaf, 3)
        untouched = configure_estimator('RFCL', RandomForestClassifier(), {'CBST': {'depth': 8}})
        self.assertEqual(untouched.n_estimators, 100)

if __name__ == '__main__': unittest.main()
