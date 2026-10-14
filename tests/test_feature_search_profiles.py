"""Profile persistence must never turn exploratory results into validated settings."""
import tempfile
import unittest
from pathlib import Path
from JBGFeatureProfiles import save_candidate, load_candidates, compatible_for_review, eligible_for_reuse, review_compatibility, find_candidate, profile_display_name


def example():
    return {
        'status': 'completed', 'original_columns': ['a','b'], 'target': 'yes',
        'recommendation': {'validation_status':'candidate', 'selected_columns':['a']},
        'provenance': {'context': {'project':'p', 'source_tables':['db.table'],
                                   'class_column':'label', 'id_column':'id'},
                       'original_column_schema':[{'name':'a','dtype':'int64'},
                                                 {'name':'b','dtype':'float64'}]},
        'nested_validation': {'selected': {'mean_mcc': 0.2}},
    }

class ProfileTests(unittest.TestCase):
    def test_candidate_saved_with_evidence_but_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_candidate(directory, example(), 'experiment')
            record = load_candidates(directory)[0]
            self.assertEqual(path.suffix, '.json')
            self.assertEqual(record['validation_status'], 'candidate')
            self.assertFalse(record['reuse_eligible'])
            self.assertFalse(eligible_for_reuse(record))
            self.assertEqual(record['nested_validation']['selected']['mean_mcc'], 0.2)
            self.assertTrue(compatible_for_review(record, project='p', tables=['db.table'],
                    class_column='label', id_column='id', columns=['a', 'b']))
            self.assertFalse(compatible_for_review(record, project='p', tables=['other.table'],
                    class_column='label', id_column='id', columns=['a', 'b']))
    def test_review_reports_mismatch_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_candidate(directory, example())
            record = load_candidates(directory)[0]
            self.assertEqual(find_candidate(directory, record["profile_id"])["profile_id"], record["profile_id"])
            self.assertIsNone(find_candidate(directory, "../bad.json"))
            kw = dict(project="p", tables=["db.table"], class_column="label", id_column="id", columns=["a", "b"])
            self.assertEqual(review_compatibility(record, **kw), [])
            self.assertTrue(review_compatibility(record, **dict(kw, tables=["other.table"])))
            self.assertTrue(review_compatibility(record, **dict(kw, column_schema=[{"name":"a", "dtype":"text"}])))
            self.assertFalse(eligible_for_reuse(record))

    def test_display_name_identifies_project_table_and_model(self):
        record = {"provenance": {"context": {"project": "Iris", "source_tables": ["AIdata.jbg_ai.iris"]}},
                  "investigative_model": {"model_type": "Tree"},
                  "created_utc": "2026-10-14T12:15:00+00:00", "profile_id": "immutable"}
        name = profile_display_name(record)
        self.assertIn("Iris", name)
        self.assertIn("AIdata.jbg_ai.iris", name)
        self.assertIn("Tree", name)
        self.assertIn("2026-10-14", name)
        self.assertIn("candidate", name)
        self.assertNotIn("immutable", name)

    def test_fail_closed_on_bad_or_incomplete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            result = example()
            result['status'] = 'failed'
            with self.assertRaises(ValueError):
                save_candidate(directory, result)
            result = example()
            result['provenance']['context'].pop('project')
            with self.assertRaises(ValueError):
                save_candidate(directory, result)
            self.assertFalse(list(Path(directory).glob('*.json')))

if __name__ == '__main__':
    unittest.main()
