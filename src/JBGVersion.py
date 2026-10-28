"""Application and repository build identity helpers.

Keep the product version stable and independent from transient Git state. Repository
metadata is best-effort provenance for development checkouts and can also be
provided explicitly by a packaged build through environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Optional


APP_NAME = "JBG AutoClassification"
APP_VERSION = "0.19.0"


@dataclass(frozen=True)
class BuildIdentity:
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    branch: Optional[str] = None
    tag: Optional[str] = None
    commit: Optional[str] = None
    dirty: bool = False

    @property
    def git_available(self) -> bool:
        return bool(self.commit)

    @property
    def repository_label(self) -> str:
        if not self.commit:
            return "Git: unavailable"
        ref = self.tag or self.branch or "detached"
        dirty = " (modified)" if self.dirty else ""
        return f"Git: {ref} @ {self.commit}{dirty}"

    @property
    def compact_label(self) -> str:
        return f"v{self.app_version} · {self.repository_label}"


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on", "dirty", "modified"}


def get_build_identity(repo_root: Optional[Path] = None) -> BuildIdentity:
    """Return stable app version plus best-effort repository provenance.

    Packaged builds may provide JBG_GIT_COMMIT and optionally JBG_GIT_BRANCH,
    JBG_GIT_TAG and JBG_GIT_DIRTY. Otherwise Git is queried relative to the
    repository root inferred from this module, never from the process cwd.
    """
    env_commit = (os.environ.get("JBG_GIT_COMMIT") or "").strip() or None
    if env_commit:
        return BuildIdentity(
            branch=(os.environ.get("JBG_GIT_BRANCH") or "").strip() or None,
            tag=(os.environ.get("JBG_GIT_TAG") or "").strip() or None,
            commit=env_commit[:12],
            dirty=bool(_env_bool("JBG_GIT_DIRTY")),
        )

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    commit = _git(root, "rev-parse", "--short=12", "HEAD")
    if not commit:
        return BuildIdentity()

    branch = _git(root, "branch", "--show-current")
    tag = _git(root, "describe", "--tags", "--exact-match", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return BuildIdentity(
        branch=branch,
        tag=tag,
        commit=commit,
        dirty=bool(status),
    )


def application_version_line(repo_root: Optional[Path] = None) -> str:
    return get_build_identity(repo_root=repo_root).compact_label
