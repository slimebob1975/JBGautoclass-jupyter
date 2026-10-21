"""Revision 164 regression checks for repository documentation privacy."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "check_documentation_privacy.py"
SPEC = spec_from_file_location("check_documentation_privacy", SOURCE)
PRIVACY = module_from_spec(SPEC)
sys.modules[SPEC.name] = PRIVACY
assert SPEC.loader is not None
SPEC.loader.exec_module(PRIVACY)


class DocumentationPrivacy164(unittest.TestCase):
    def test_repository_markdown_passes_builtin_checks(self):
        self.assertEqual(PRIVACY.scan_repository(ROOT), [])

    def test_private_blocklist_is_case_insensitive_and_value_is_not_reported(self):
        private_value = "ExamplePrivateIdentifier"
        findings = PRIVACY.scan_text(
            Path("example.md"),
            "This line contains exampleprivateidentifier.",
            [private_value.casefold()],
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "private-blocklist-entry-1")
        self.assertNotIn(private_value.casefold(), findings[0].category.casefold())

    def test_builtin_patterns_cover_high_risk_documentation_forms(self):
        text = "\n".join(
            [
                "Contact user@example.invalid",
                r"Local path C:\Users\ExampleUser\work",
                "SQL sample dbo.sample_table",
                "Setting data_table = 'sample_table'",
            ]
        )
        categories = {finding.category for finding in PRIVACY.scan_text(Path("example.md"), text)}
        self.assertIn("email-address", categories)
        self.assertIn("windows-user-home-path", categories)
        self.assertIn("schema-qualified-sql-identifier", categories)
        self.assertIn("explicit-data-table-setting", categories)

    def test_default_local_blocklist_is_optional_and_git_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(PRIVACY.resolve_blocklist_path(root))
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(PRIVACY.LOCAL_BLOCKLIST, gitignore)

    def test_docs_directory_is_not_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CHANGELOG.md").write_text("Generic documentation only.\n", encoding="utf-8")
            self.assertEqual(PRIVACY.scan_repository(root), [])


if __name__ == "__main__":
    unittest.main()
