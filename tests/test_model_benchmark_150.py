import unittest
import numpy as np
from JBGModelBenchmark150 import summarize_binary_oof

class Benchmark150Tests(unittest.TestCase):
    def test_perfect(self):
        data=summarize_binary_oof(np.array(['yes','no','yes','no']),
            np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]), ['yes','no'], 0.5)
        self.assertEqual(data['average_precision'], 1.)
        self.assertEqual(data['mcc'], 1.)
        self.assertEqual(data['brier'], 0.)
    def test_reject_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            summarize_binary_oof([0,1], [[np.nan,0],[0,1]], [0,1])
    def test_constant_scores(self):
        data=summarize_binary_oof([0,0,1,1], [[.5,.5]]*4, [0,1], .5)
        self.assertAlmostEqual(data['average_precision'], .5)

if __name__ == '__main__': unittest.main()
