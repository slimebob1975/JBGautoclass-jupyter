import unittest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from JBGFeatureSearch import SearchOptions, make_estimator, model_identity
from JBGFeatureSearchPresets import PRESETS, detected_preset, preset_values

class PresetForestTests(unittest.TestCase):
    def test_estimators_and_old_default(self):
        self.assertIsInstance(make_estimator(SearchOptions(model_type='Tree'))[1], ExtraTreesClassifier)
        opts = SearchOptions(model_type='Tree', tree_estimator='Random Forest', tree_n_estimators=17)
        self.assertIsInstance(make_estimator(opts)[1], RandomForestClassifier)
        self.assertEqual(model_identity(opts)['parameters']['n_estimators'], 17)
        with self.assertRaises(ValueError):
            make_estimator(SearchOptions(tree_estimator='incorrect'))
    def test_presets_bounds_and_custom(self):
        for name in PRESETS:
            args = preset_values(name)
            self.assertEqual(detected_preset(args), name)
            self.assertGreaterEqual(args['max_evaluations'], args['trials']+1)
            SearchOptions(**args).validate()
        changed = preset_values('Quick')
        changed['trials'] += 1
        self.assertEqual(detected_preset(changed), 'Custom')
        self.assertEqual(SearchOptions(**preset_values('Quick')).planned_fit_upper_bound, 66)
if __name__ == '__main__': unittest.main()
