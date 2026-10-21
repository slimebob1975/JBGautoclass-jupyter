"""Revision 166/166a/166b regression checks for scoped documentation-history remediation."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

PRIVACY_SOURCE = SCRIPTS / "check_documentation_privacy.py"
PRIVACY_SPEC = spec_from_file_location("check_documentation_privacy", PRIVACY_SOURCE)
PRIVACY = module_from_spec(PRIVACY_SPEC)
sys.modules[PRIVACY_SPEC.name] = PRIVACY
assert PRIVACY_SPEC.loader is not None
PRIVACY_SPEC.loader.exec_module(PRIVACY)

REWRITE_SOURCE = SCRIPTS / "rewrite_documentation_history.py"
REWRITE_SPEC = spec_from_file_location("rewrite_documentation_history", REWRITE_SOURCE)
REWRITE = module_from_spec(REWRITE_SPEC)
sys.modules[REWRITE_SPEC.name] = REWRITE
assert REWRITE_SPEC.loader is not None
REWRITE_SPEC.loader.exec_module(REWRITE)


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def commit_all(root: Path, message: str) -> None:
    git(root, "add", ".")
    git(root, "commit", "-m", message)


class DocumentationPrivacy166(unittest.TestCase):
    def make_repository(self, directory: str) -> Path:
        root = Path(directory)
        git(root, "init")
        git(root, "config", "user.name", "Privacy Test")
        git(root, "config", "user.email", "privacy-test@example.invalid")
        return root

    def make_sanitized_repository_with_history_leak(self, directory: str, private_value: str) -> Path:
        root = self.make_repository(directory)
        (root / "README.md").write_text("Generic current product documentation.\n", encoding="utf-8")
        (root / "BACKLOG.md").write_text(
            f"Historical private reference: {private_value}.\n",
            encoding="utf-8",
        )
        (root / "CHANGELOG.md").write_text("Generic initial history.\n", encoding="utf-8")
        commit_all(root, "Initial documentation")

        (root / "BACKLOG.md").write_text("Current generic backlog.\n", encoding="utf-8")
        (root / "CHANGELOG.md").write_text("Current generic changelog.\n", encoding="utf-8")
        commit_all(root, "Sanitize current documentation")
        return root

    def create_fake_signed_tag(self, root: Path, refname: str, target: str, message: str = "Signed release") -> str:
        tagger = "Privacy Test <privacy-test@example.invalid> 0 +0000"
        raw = (
            f"object {target}\n"
            "type commit\n"
            f"tag {refname}\n"
            f"tagger {tagger}\n\n"
            f"{message}\n"
            "-----BEGIN PGP SIGNATURE-----\n"
            "synthetic-test-signature\n"
            "-----END PGP SIGNATURE-----\n"
        ).encode("utf-8")
        tag_oid = subprocess.run(
            ["git", "-C", str(root), "hash-object", "-w", "-t", "tag", "--stdin"],
            input=raw,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("ascii").strip()
        git(root, "update-ref", f"refs/tags/{refname}", tag_oid)
        return tag_oid

    def create_fake_signed_commit(self, root: Path, message: str = "Signed checkpoint") -> str:
        parent = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
        tree = git(root, "rev-parse", "HEAD^{tree}").stdout.decode("ascii").strip()
        identity = "Privacy Test <privacy-test@example.invalid> 0 +0000"
        raw = (
            f"tree {tree}\n"
            f"parent {parent}\n"
            f"author {identity}\n"
            f"committer {identity}\n"
            "gpgsig -----BEGIN PGP SIGNATURE-----\n"
            " synthetic-test-signature\n"
            " -----END PGP SIGNATURE-----\n\n"
            f"{message}\n"
        ).encode("utf-8")
        commit_oid = subprocess.run(
            ["git", "-C", str(root), "hash-object", "-w", "-t", "commit", "--stdin"],
            input=raw,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("ascii").strip()
        branch = git(root, "symbolic-ref", "HEAD").stdout.decode("utf-8").strip()
        git(root, "update-ref", branch, commit_oid, parent)
        git(root, "reset", "--hard", commit_oid)
        return commit_oid

    def test_default_working_tree_scan_ignores_untracked_markdown(self):
        private_value = "ExamplePrivateIdentifier"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text("Generic tracked documentation.\n", encoding="utf-8")
            commit_all(root, "Tracked documentation")
            (root / "local-notes.md").write_text(private_value + "\n", encoding="utf-8")

            blocklist = [private_value.casefold()]
            self.assertEqual(PRIVACY.scan_repository(root, blocklist), [])
            findings = PRIVACY.scan_repository(root, blocklist, include_untracked=True)
            self.assertTrue(any(finding.path.name == "local-notes.md" for finding in findings))

    def test_dry_run_is_read_only_and_does_not_echo_private_value(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            blocklist = root / PRIVACY.LOCAL_BLOCKLIST
            blocklist.write_text(private_value + "\n", encoding="utf-8")
            before = git(root, "rev-parse", "HEAD").stdout

            output = StringIO()
            with redirect_stdout(output):
                result = REWRITE.main(["--root", str(root)])

            after = git(root, "rev-parse", "HEAD").stdout
            rendered = output.getvalue()
            self.assertEqual(result, 0)
            self.assertEqual(before, after)
            self.assertIn("Dry run only", rendered)
            self.assertNotIn(private_value.casefold(), rendered.casefold())

    def test_apply_rewrites_only_target_history_and_creates_verified_backup(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as backup_directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            blocklist = root / PRIVACY.LOCAL_BLOCKLIST
            blocklist.write_text(private_value + "\n", encoding="utf-8")
            backup = Path(backup_directory) / "before.bundle"
            original_head = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()

            output = StringIO()
            with redirect_stdout(output):
                result = REWRITE.main(
                    ["--root", str(root), "--apply", "--backup", str(backup)]
                )

            rewritten_head = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
            rendered = output.getvalue()
            self.assertEqual(result, 0)
            self.assertNotEqual(original_head, rewritten_head)
            self.assertTrue(backup.is_file())
            git(root, "bundle", "verify", str(backup))
            self.assertEqual(PRIVACY.scan_git_history(root, [private_value.casefold()]), [])
            self.assertEqual(PRIVACY.scan_repository(root, [private_value.casefold()]), [])
            self.assertEqual(git(root, "status", "--porcelain", "--untracked-files=no").stdout, b"")
            self.assertNotIn(private_value.casefold(), rendered.casefold())

            historical_backlog = git(root, "show", f"{rewritten_head}^:BACKLOG.md").stdout.decode("utf-8")
            self.assertIn(REWRITE.REPLACEMENT, historical_backlog)
            self.assertNotIn(private_value.casefold(), historical_backlog.casefold())

    def test_preflight_refuses_private_findings_in_other_markdown(self):
        private_value = "ExamplePrivateDataset"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text(private_value + "\n", encoding="utf-8")
            (root / "BACKLOG.md").write_text("Generic backlog.\n", encoding="utf-8")
            (root / "CHANGELOG.md").write_text("Generic changelog.\n", encoding="utf-8")
            commit_all(root, "Initial documentation")
            (root / "README.md").write_text("Generic current README.\n", encoding="utf-8")
            commit_all(root, "Sanitize current README")

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            self.assertIn("outside the permitted", str(caught.exception))

    def test_preflight_refuses_commit_message_findings(self):
        private_value = "ExamplePrivateProject"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repository(directory)
            (root / "BACKLOG.md").write_text("Generic backlog.\n", encoding="utf-8")
            (root / "CHANGELOG.md").write_text("Generic changelog.\n", encoding="utf-8")
            commit_all(root, f"Historical reference to {private_value}")

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            self.assertIn("outside the permitted", str(caught.exception))

    def test_preflight_refuses_tracked_worktree_changes(self):
        private_value = "ExamplePrivateTable"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            (root / "README.md").write_text("Uncommitted tracked change.\n", encoding="utf-8")

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            self.assertIn("Tracked working-tree changes", str(caught.exception))


    def test_unaffected_signed_commit_is_preserved_when_later_history_changes(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as backup_directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text("Generic current product documentation.\n", encoding="utf-8")
            (root / "BACKLOG.md").write_text("Generic initial backlog.\n", encoding="utf-8")
            (root / "CHANGELOG.md").write_text("Generic initial history.\n", encoding="utf-8")
            commit_all(root, "Generic base")
            signed_commit = self.create_fake_signed_commit(root)

            (root / "BACKLOG.md").write_text(
                f"Historical private reference: {private_value}.\n",
                encoding="utf-8",
            )
            commit_all(root, "Historical documentation")
            (root / "BACKLOG.md").write_text("Current generic backlog.\n", encoding="utf-8")
            commit_all(root, "Sanitize current documentation")

            plan = REWRITE.build_plan(root, [private_value.casefold()])
            self.assertNotIn(signed_commit, plan.changed_revisions)
            backup = Path(backup_directory) / "before.bundle"
            REWRITE.rewrite_history(root, [private_value.casefold()], plan, backup)

            self.assertEqual(git(root, "cat-file", "-t", signed_commit).stdout.strip(), b"commit")
            self.assertEqual(PRIVACY.scan_git_history(root, [private_value.casefold()]), [])

    def test_affected_signed_commit_still_blocks_rewrite(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            signed_commit = self.create_fake_signed_commit(root)

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            rendered = str(caught.exception)
            self.assertIn("signed commit", rendered.lower())
            self.assertIn(signed_commit[:12], rendered)
            self.assertNotIn(private_value.casefold(), rendered.casefold())

    def test_unsupported_commit_header_detection_distinguishes_signature_and_mergetag(self):
        raw = (
            b"tree 0000000000000000000000000000000000000000\n"
            b"gpgsig -----BEGIN PGP SIGNATURE-----\n"
            b" synthetic\n"
            b"mergetag object 1111111111111111111111111111111111111111\n\n"
            b"Generic message\n"
        )
        self.assertEqual(
            REWRITE._commit_unsupported_header_kinds(raw),
            frozenset({"signed-commit", "merge-tag"}),
        )

    def test_unsigned_annotated_tag_is_recreated_when_target_changes(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as backup_directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            git(root, "tag", "-a", "release-test", "-m", "Generic release annotation")
            old_tag_object = git(root, "rev-parse", "refs/tags/release-test").stdout.decode("ascii").strip()
            old_tag_message = git(root, "for-each-ref", "--format=%(contents)", "refs/tags/release-test").stdout
            blocklist = root / PRIVACY.LOCAL_BLOCKLIST
            blocklist.write_text(private_value + "\n", encoding="utf-8")
            backup = Path(backup_directory) / "before.bundle"

            plan = REWRITE.build_plan(root, [private_value.casefold()])
            affected = [
                tag for tag in plan.annotated_tags
                if tag.target_type == "commit" and tag.target_id in plan.changed_revisions
            ]
            self.assertEqual(len(affected), 1)
            self.assertFalse(affected[0].signed)

            result = REWRITE.rewrite_history(root, [private_value.casefold()], plan, backup)
            new_tag_object = git(root, "rev-parse", "refs/tags/release-test").stdout.decode("ascii").strip()
            new_tag_message = git(root, "for-each-ref", "--format=%(contents)", "refs/tags/release-test").stdout
            self.assertNotEqual(old_tag_object, new_tag_object)
            self.assertEqual(old_tag_message, new_tag_message)
            self.assertGreaterEqual(result.refs_changed, 2)
            self.assertEqual(PRIVACY.scan_git_history(root, [private_value.casefold()]), [])

    def test_signed_annotated_tag_blocks_only_when_its_target_changes(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            target = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
            self.create_fake_signed_tag(root, "signed-affected", target)

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            self.assertIn("signed annotated tag", str(caught.exception).lower())

    def test_signed_annotated_tag_on_unchanged_commit_is_left_alone(self):
        private_value = "ExamplePrivateDatabase"
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as backup_directory:
            root = self.make_repository(directory)
            (root / "README.md").write_text("Generic current product documentation.\n", encoding="utf-8")
            (root / "BACKLOG.md").write_text("Generic initial backlog.\n", encoding="utf-8")
            (root / "CHANGELOG.md").write_text("Generic initial history.\n", encoding="utf-8")
            commit_all(root, "Generic base")
            base = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
            old_tag = self.create_fake_signed_tag(root, "signed-unaffected", base)

            (root / "BACKLOG.md").write_text(f"Historical private reference: {private_value}.\n", encoding="utf-8")
            commit_all(root, "Historical documentation")
            (root / "BACKLOG.md").write_text("Current generic backlog.\n", encoding="utf-8")
            commit_all(root, "Sanitize current documentation")

            plan = REWRITE.build_plan(root, [private_value.casefold()])
            tag = next(tag for tag in plan.annotated_tags if tag.ref.name.endswith("signed-unaffected"))
            self.assertTrue(tag.signed)
            self.assertNotIn(tag.target_id, plan.changed_revisions)

            backup = Path(backup_directory) / "before.bundle"
            REWRITE.rewrite_history(root, [private_value.casefold()], plan, backup)
            new_tag = git(root, "rev-parse", "refs/tags/signed-unaffected").stdout.decode("ascii").strip()
            self.assertEqual(old_tag, new_tag)
            self.assertEqual(PRIVACY.scan_git_history(root, [private_value.casefold()]), [])

    def test_preflight_refuses_private_identifier_in_annotated_tag_message(self):
        private_value = "ExamplePrivateProject"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, "DifferentPrivateValue")
            git(root, "tag", "-a", "release-test", "-m", f"Historical tag note {private_value}")

            findings = PRIVACY.scan_git_history(root, [private_value.casefold()])
            self.assertTrue(any(finding.path == "<tag-message>" for finding in findings))
            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.build_plan(root, [private_value.casefold()])
            self.assertIn("outside the permitted", str(caught.exception))

    def test_apply_refuses_backup_inside_repository(self):
        private_value = "ExamplePrivateTable"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_sanitized_repository_with_history_leak(directory, private_value)
            plan = REWRITE.build_plan(root, [private_value.casefold()])

            with self.assertRaises(REWRITE.HistoryRewriteError) as caught:
                REWRITE.rewrite_history(
                    root,
                    [private_value.casefold()],
                    plan,
                    root / "private-backup.bundle",
                )
            self.assertIn("outside the repository", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
