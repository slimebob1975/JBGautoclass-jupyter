"""External validation must be paired, disjoint, and never authorize reuse."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from JBGCVReporting import partition_signature
from JBGFeatureProfiles import save_candidate, load_candidates, eligible_for_reuse
from JBGFeatureSearch import SearchOptions, model_identity, study_provenance
from JBGFeatureValidation import evaluate_external_candidate


class ExternalValidationTests(unittest.TestCase):
    def test_paired_evidence_remains_nonreusable(self):
        X, y = make_classification(n_samples=90, n_features=4, n_informative=3,
                                   n_redundant=0, random_state=8)
        train = pd.DataFrame(X[:60], columns=list('abcd'))
        external = pd.DataFrame(X[60:], columns=list('abcd'))
        opts = SearchOptions(trials=1, max_evaluations=2, sizes=(2,))
        result = dict(status='completed', original_columns=list(train.columns),
                      target=1, options=asdict(opts), text_options={},
                      signatures={'development': partition_signature(train.index, pd.Series(y[:60]))},
                      investigative_model=model_identity(opts),
                      provenance=study_provenance(train, dict(project='test', source_tables=['t'],
                            class_column='y', id_column='id')),
                      recommendation={'validation_status':'candidate', 'selected_columns':['a','b']})
        with tempfile.TemporaryDirectory() as temp:
            save_candidate(temp, result)
            record = load_candidates(temp)[0]
            destination = Path(temp) / 'evidence' / 'result.json'
            kw = dict(development_ids=list(range(60)), external_ids=list(range(60,90)),
                      cohort_description='Future independently collected labelled cohort',
                      output_path=destination)
            report = evaluate_external_candidate(record, train, y[:60], external, y[60:], **kw)
            self.assertTrue(destination.exists())
            self.assertEqual(report['external_rows'], 30)
            self.assertIn('confusion_matrix', report['selected'])
            self.assertFalse(report['reuse_eligible'])
            self.assertFalse(eligible_for_reuse(record))
            with self.assertRaises(FileExistsError):
                evaluate_external_candidate(record, train, y[:60], external, y[60:], **kw)
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                evaluate_external_candidate(record, train, y[:60], external, y[60:],
                    **dict(kw, external_ids=list(range(30))))
            with self.assertRaisesRegex(ValueError, 'Development features differ'):
                evaluate_external_candidate(record, train.assign(a=0.0), y[:60], external, y[60:],
                    **dict(kw, output_path=Path(temp)/'other.json'))


if __name__ == '__main__':
    unittest.main()
