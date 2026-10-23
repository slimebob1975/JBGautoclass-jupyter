"""Revision 175: stable application version plus repository/build provenance."""

from pathlib import Path
from unittest.mock import patch

from JBGVersion import APP_NAME, APP_VERSION, BuildIdentity, application_version_line, get_build_identity
from JBGLogFile import TimestampedLogFile


ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = (ROOT / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8")
DEFAULT_SETTINGS = (ROOT / "src" / "GUI" / "default_settings.json").read_text(encoding="utf-8")


def test_product_version_is_stable_and_separate_from_repository_state():
    identity = BuildIdentity(branch="dev", commit="abc1234", dirty=True)
    assert APP_NAME == "JBG AutoClassification"
    assert APP_VERSION == "0.18.0"
    assert identity.compact_label == "v0.18.0 · Git: dev @ abc1234 (modified)"


def test_exact_tag_is_preferred_over_branch_in_repository_label():
    identity = BuildIdentity(branch="dev", tag="v0.18.0", commit="abc1234")
    assert identity.repository_label == "Git: v0.18.0 @ abc1234"


def test_packaged_build_can_supply_repository_identity_through_environment(monkeypatch):
    monkeypatch.setenv("JBG_GIT_COMMIT", "0123456789abcdef")
    monkeypatch.setenv("JBG_GIT_BRANCH", "release")
    monkeypatch.setenv("JBG_GIT_TAG", "v0.18.0")
    monkeypatch.setenv("JBG_GIT_DIRTY", "0")

    identity = get_build_identity(repo_root=Path("ignored"))

    assert identity.commit == "0123456789ab"
    assert identity.tag == "v0.18.0"
    assert identity.branch == "release"
    assert identity.dirty is False


def test_non_git_installation_falls_back_without_error(monkeypatch, tmp_path):
    for name in ("JBG_GIT_COMMIT", "JBG_GIT_BRANCH", "JBG_GIT_TAG", "JBG_GIT_DIRTY"):
        monkeypatch.delenv(name, raising=False)
    with patch("JBGVersion._git", return_value=None):
        identity = get_build_identity(repo_root=tmp_path)
    assert identity.repository_label == "Git: unavailable"
    assert application_version_line(repo_root=tmp_path).startswith("v0.18.0 · Git:")


def test_git_checkout_reports_branch_commit_and_modified_state(monkeypatch, tmp_path):
    for name in ("JBG_GIT_COMMIT", "JBG_GIT_BRANCH", "JBG_GIT_TAG", "JBG_GIT_DIRTY"):
        monkeypatch.delenv(name, raising=False)
    with patch(
        "JBGVersion._git",
        side_effect=["abcdef123456", "dev", None, " M src/example.py"],
    ):
        identity = get_build_identity(repo_root=tmp_path)
    assert identity.repository_label == "Git: dev @ abcdef123456 (modified)"


def test_runtime_log_records_application_and_repository_identity(tmp_path):
    with patch(
        "JBGLogFile.get_build_identity",
        return_value=BuildIdentity(branch="dev", commit="abcdef123456", dirty=True),
    ):
        log_file = TimestampedLogFile(log_dir=tmp_path, filename="identity.log")
    try:
        text = log_file.path.read_text(encoding="utf-8")
        assert "[INFO] Application: JBG AutoClassification v0.18.0" in text
        assert "[INFO] Repository: Git: dev @ abcdef123456 (modified)" in text
    finally:
        log_file.close()


def test_gui_places_build_identity_directly_below_title_with_legacy_migration():
    assert '"welcome",\n        "build_identity",\n        "project"' in DEFAULT_SETTINGS
    assert 'def build_identity(self) -> widgets.HTML' in WIDGETS_SOURCE
    assert 'html.escape(application_version_line())' in WIDGETS_SOURCE
    assert 'if "build_identity" not in display_items:' in WIDGETS_SOURCE
