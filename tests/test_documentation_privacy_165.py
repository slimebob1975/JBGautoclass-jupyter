"""Revision 165 regression checks for reachable Git-history documentation privacy."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "check_documentation_privacy.py"
SPEC = spec_from_file_location("check_documentation_privacy_165", SOURCE)
PRIVACY = module_from_spec(SPEC)
sys.modules[SPEC.name] = PRIVACY
assert SPEC.loader is not None
SPEC.loader.exec_module(PRIVACY)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def commit(root: Path, message: str) -> None:
    git(root, "add", ".")
    git(root, "commit", "-m", message)


class DocumentationPrivacy165(unittest.TestCase):
    def make_repository(self, directory: str) -> Path:
        root = Path(directory)
        git(root, "init")
        git(root, "config", "user.name", "Privacy Test")
        git(root, "config", "user.email", "privacy-test@example.invalid")
        return root

    def test_history_finds_private_identifier_removed_from_current_tree(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            changelog = root / "CHANGELOG.md"
            changelog.write_text(f"Historical reference to {private_value}.\n", encoding="utf-8")
            commit(root, "Initial generic documentation")
            changelog.write_text("Current generic documentation.\n", encoding="utf-8")
            commit(root, "Sanitize documentation")

            blocklist = [private_value.casefold()]
            self.assertEqual(PRIVACY.scan_repository(root, blocklist), [])
            findings = PRIVACY.scan_git_history(root, blocklist)

            self.assertTrue(
                any(
                    finding.path == "CHANGELOG.md"
                    and finding.category == "private-blocklist-entry-1"
                    for finding in findings
                )
            )

    def test_history_scans_commit_messages_without_echoing_private_value(self):
        private_value = "ExamplePrivateTable"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text("Generic documentation.\n", encoding="utf-8")
            commit(root, f"Remove old reference to {private_value}")
            blocklist_file = root / PRIVACY.LOCAL_BLOCKLIST
            blocklist_file.write_text(private_value + "\n", encoding="utf-8")

            output = StringIO()
            with redirect_stdout(output):
                result = PRIVACY.main(["--root", str(root), "--history"])

            rendered = output.getvalue()
            self.assertEqual(result, 1)
            self.assertIn("<commit-message>", rendered)
            self.assertIn("private-blocklist-entry-1", rendered)
            self.assertNotIn(private_value.casefold(), rendered.casefold())

    def test_history_requires_local_private_blocklist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text("Generic documentation.\n", encoding="utf-8")
            commit(root, "Generic documentation")

            output = StringIO()
            with redirect_stdout(output):
                result = PRIVACY.main(["--root", str(root), "--history"])

            self.assertEqual(result, 2)
            self.assertIn("configure a local private blocklist first", output.getvalue())


    def test_history_refuses_shallow_repository(self):
        with patch.object(PRIVACY, "git_repository_is_shallow", return_value=True):
            with self.assertRaises(PRIVACY.GitHistoryAuditError) as caught:
                PRIVACY.scan_git_history(Path("."), ["example-private-value"])
        self.assertIn("non-shallow", str(caught.exception))

    def test_history_ignores_non_markdown_blob_content(self):
        private_value = "ExamplePrivateDataset"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "notes.txt").write_text(private_value + "\n", encoding="utf-8")
            (root / "README.md").write_text("Generic documentation.\n", encoding="utf-8")
            commit(root, "Generic commit")

            findings = PRIVACY.scan_git_history(root, [private_value.casefold()])
            self.assertEqual(findings, [])

    def test_private_identifier_in_markdown_filename_is_redacted(self):
        private_value = "PrivatePersonName"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            path = root / f"notes-{private_value}.md"
            path.write_text("Generic content.\n", encoding="utf-8")
            commit(root, "Generic commit")
            path.write_text(f"Historical {private_value}.\n", encoding="utf-8")
            commit(root, "Generic update")

            findings = PRIVACY.scan_git_history(root, [private_value.casefold()])
            self.assertTrue(findings)
            self.assertTrue(all(finding.path == "<redacted-path>" for finding in findings))


if __name__ == "__main__":
    unittest.main()
