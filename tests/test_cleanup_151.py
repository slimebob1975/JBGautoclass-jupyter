import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Cleanup151Tests(unittest.TestCase):
    def test_gui_excludes_native_catboost_from_ordinary_selection(self):
        text = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        self.assertIn('if value != "CBST"', text)
        self.assertIn('getattr(value, "name", value) != "CBST"', text)

    def test_feature_modelling_keeps_catboost(self):
        text = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        self.assertIn('"Combination", "CatBoost"', text)

    def test_legacy_selection_explained(self):
        text = (ROOT / 'src/Config.py').read_text(encoding='utf-8')
        self.assertIn('Use CatBoost in Feature modelling instead', text)

if __name__ == '__main__':
    unittest.main()
