"""Revision 264e: candidate/selected algorithm identity remains fail-closed."""
import unittest
from types import SimpleNamespace
from Storage.CSVSavedModelInspection import _registered_algorithm_matches_checkpoint as matches


class AlgorithmIdentity264e(unittest.TestCase):
    def test_same_algorithms_allow_list_or_string_and_order(self):
        self.assertTrue(matches(['LRN', 'LDA'], 'LDA, LRN', ()))
        self.assertTrue(matches(['LRN'], ['LRN'], ()))

    def test_selected_candidate_with_matching_fitted_descriptor(self):
        selected = (None, None, (None, None, None, None, SimpleNamespace(name='LRN')))
        self.assertTrue(matches(['LRN', 'LDA'], ['LRN'], selected))

    def test_outside_registered_candidates_rejected(self):
        descriptor = (None, None, (None, None, None, None, SimpleNamespace(name='SVC')))
        self.assertFalse(matches(['LRN'], ['SVC'], descriptor))

    def test_narrowed_algorithm_requires_matching_fitted_descriptor(self):
        descriptor = (None, None, (None, None, None, None, SimpleNamespace(name='LDA')))
        self.assertFalse(matches(['LRN', 'LDA'], ['LRN'], descriptor))
        self.assertFalse(matches(['LRN', 'LDA'], ['LRN'], ()))

    def test_ambiguous_or_malformed_algorithms_rejected(self):
        self.assertFalse(matches(['LRN', 'LDA'], ['LRN', 'SVC'], ()))
        self.assertFalse(matches(['LRN', 'LDA'], ['LRN', 'LDA', 'LRN'], ()))
        self.assertFalse(matches(['LRN'], [], ()))
        self.assertFalse(matches(['LRN'], ['LRN', 1], ()))

if __name__ == '__main__':
    unittest.main()
