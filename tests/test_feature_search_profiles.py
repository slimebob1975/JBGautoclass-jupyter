"""Profile persistence must never turn exploratory results into validated settings."""
import tempfile
import unittest
from pathlib import Path
from JBGFeatureProfiles import save_candidate, load_candidates, compatible_for_review, eligible_for_reuse, review_compatibility, find_candidate, profile_display_name, delete_candidate


def example():
    return {
        'status': 'completed', 'original_columns': ['a','b'], 'target': 'yes',
        'recommendation': {'validation_status':'candidate', 'selected_columns':['a']},
        'provenance': {'context': {'project':'p', 'source_tables':['db.table'],
                                   'class_column':'label', 'id_column':'id'},
                       'original_column_schema':[{'name':'a','dtype':'int64'},
                                                 {'name':'b','dtype':'float64'}],
                       'source_sql_schema':[{'name':'a','sql_type':'int','family':'numeric'},
                                            {'name':'b','sql_type':'float','family':'numeric'}]},
        'nested_validation': {'selected': {'mean_mcc': 0.2}},
        'text_options': {
            'text_columns': ('a',), 'category_columns': (), 'limit_categorize': 30,
            'language': 'sv', 'stop_words': True, 'df': 1.0, 'ngram_range': (1, 1),
            'use_encryption': True, 'use_categorization': True,
        },
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
            kw = dict(project="p", tables=["db.table"], class_column="label", id_column="id", columns=["a", "b"],
                      sql_schema={"a": "int", "b": "float"})
            self.assertEqual(review_compatibility(record, **kw), [])
            self.assertTrue(review_compatibility(record, **dict(kw, tables=["other.table"])))
            self.assertTrue(review_compatibility(record, **dict(kw, column_schema=[{"name":"a", "dtype":"text"}])))
            self.assertTrue(review_compatibility(record, **dict(kw, sql_schema={"a": "nvarchar", "b": "float"})))
            broken = dict(record)
            broken["text_options"] = dict(record["text_options"], ngram_range=(9, 9))
            self.assertTrue(review_compatibility(broken, **kw))
            incomplete_sql = dict(record)
            incomplete_sql["provenance"] = dict(record["provenance"], source_sql_schema=record["provenance"]["source_sql_schema"][:1])
            self.assertTrue(review_compatibility(incomplete_sql, **kw))
            self.assertFalse(eligible_for_reuse(record))

    def test_delete_exact_selected_profile_only_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            first = save_candidate(directory, example())
            second = save_candidate(directory, example())
            original = load_candidates(directory)
            first_id = first.stem
            deleted = delete_candidate(directory, first_id)
            self.assertEqual(deleted["profile_id"], first_id)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(len(load_candidates(directory)), 1)
            audit = Path(directory, "deletions.jsonl").read_text(encoding="utf-8")
            self.assertIn(first_id, audit)
            self.assertNotIn("db.table", audit)

    def test_delete_rejects_path_traversal_missing_or_tampered_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_candidate(directory, example())
            for invalid in ("../other", "../../etc/passwd", "not-a-uuid"):
                with self.assertRaises(ValueError):
                    delete_candidate(directory, invalid)
            import json
            payload = json.loads(path.read_text(encoding="utf-8"))
            real_id = payload["profile_id"]
            payload["profile_id"] = "different"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                delete_candidate(directory, real_id)
            self.assertTrue(path.exists())

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
