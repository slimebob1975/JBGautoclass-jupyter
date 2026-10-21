"""Fail-closed checks for accidental private identifiers in repository documentation.

The built-in checks catch common high-risk forms (email addresses, personal-number-like
values, user-home paths and explicit SQL/schema identifiers). Arbitrary person, table,
database or project names cannot be detected reliably from syntax alone, so maintain a
local, git-ignored blocklist when reviewing documentation from a real runtime environment.
Blocklist values are never echoed by this tool.

Revision 165 adds an optional reachable-Git-history audit. History mode requires a local
private blocklist and a non-shallow repository because a built-in-only or truncated scan
would give misleading assurance. It scans historical Markdown blobs plus commit messages,
reports only location/category metadata, and never rewrites Git history.

Revision 166 makes the normal working-tree scan prefer Git-tracked Markdown so local notes
that cannot be committed accidentally do not obscure repository findings. Use
``--include-untracked`` when a deliberate scan of local Markdown is desired.

Revision 166a also audits annotated-tag ref names and tag messages. This closes a history
visibility gap without printing tag annotations or matched private values.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOCAL_BLOCKLIST = ".jbg_doc_privacy_blocklist"
BLOCKLIST_ENV = "JBG_DOC_PRIVACY_BLOCKLIST"

# These rules intentionally prefer low false-positive rates. Project-specific names belong
# in the local blocklist rather than in source code, tests or committed configuration.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email-address",
        re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])"),
    ),
    (
        "personal-number-like-value",
        re.compile(r"(?<!\d)(?:19|20)?\d{6}[-+]\d{4}(?!\d)"),
    ),
    (
        "windows-user-home-path",
        re.compile(r"(?i)\b[a-z]:\\users\\[^\\\s]+"),
    ),
    (
        "posix-user-home-path",
        re.compile(r"(?<![\w/])/home/[^/\s]+"),
    ),
    (
        "schema-qualified-sql-identifier",
        re.compile(r"(?i)(?<![\w.])dbo\.[a-z_][\w$#@]*(?![\w.])"),
    ),
    (
        "bracket-qualified-sql-identifier",
        re.compile(r"\[[^\]\r\n]+\]\.\[[^\]\r\n]+\](?:\.\[[^\]\r\n]+\])?"),
    ),
    (
        "explicit-data-table-setting",
        re.compile(r"(?i)\bdata_table\b\s*(?:=|:)\s*['\"][^'\"\r\n]+['\"]"),
    ),
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    category: str


@dataclass(frozen=True)
class HistoryViolation:
    revision: str
    path: str
    line: int
    category: str


class GitHistoryAuditError(RuntimeError):
    """Raised when reachable Git history cannot be audited safely."""


def _all_markdown_paths(root: Path) -> list[Path]:
    """Return local Markdown files without requiring Git metadata."""
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def _tracked_markdown_paths(root: Path) -> list[Path] | None:
    """Return Git-tracked Markdown paths, or ``None`` when ``root`` is not a Git repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    paths: list[Path] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        candidate = root / relative
        if candidate.is_file():
            paths.append(candidate)
    return sorted(paths)


def documentation_paths(root: Path, include_untracked: bool = False) -> list[Path]:
    """Return Markdown paths for the normal repository privacy scan.

    In a Git repository the default is deliberately limited to tracked Markdown. This
    matches the publication-risk boundary and avoids treating private, untracked notes as
    repository content. Non-Git directories retain the original all-Markdown behavior for
    standalone use and regression tests.
    """
    if not include_untracked:
        tracked = _tracked_markdown_paths(root)
        if tracked is not None:
            return tracked
    return _all_markdown_paths(root)


def resolve_blocklist_path(root: Path, explicit: str | None = None) -> Path | None:
    """Resolve an optional local blocklist without exposing its contents."""
    candidate = explicit or os.environ.get(BLOCKLIST_ENV)
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = root / path
        return path

    local = root / LOCAL_BLOCKLIST
    return local if local.exists() else None


def load_blocklist(path: Path | None) -> list[str]:
    """Load literal case-insensitive private identifiers from a local file."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError("Documentation privacy blocklist does not exist.")

    values: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if len(value) < 3:
            raise ValueError("Documentation privacy blocklist entries must be at least 3 characters.")
        values.append(value.casefold())
    return values


def scan_text(path: Path, text: str, blocklist: Iterable[str] = ()) -> list[Violation]:
    """Scan text while never copying a matched private blocklist value into results."""
    private_values = tuple(value.casefold() for value in blocklist)
    findings: list[Violation] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        for category, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(Violation(path=path, line=line_number, category=category))
        folded = line.casefold()
        for index, value in enumerate(private_values, start=1):
            if value in folded:
                findings.append(
                    Violation(
                        path=path,
                        line=line_number,
                        category=f"private-blocklist-entry-{index}",
                    )
                )
    return findings


def scan_repository(
    root: Path,
    blocklist: Iterable[str] = (),
    include_untracked: bool = False,
) -> list[Violation]:
    findings: list[Violation] = []
    for path in documentation_paths(root, include_untracked=include_untracked):
        relative = str(path.relative_to(root))
        for finding in scan_text(Path("<git-path>"), relative, blocklist):
            findings.append(
                Violation(path=path, line=0, category=f"path-{finding.category}")
            )
        findings.extend(scan_text(path, path.read_text(encoding="utf-8"), blocklist))
    return findings


def _run_git(root: Path, *args: str) -> bytes:
    """Run a read-only Git command without exposing command stderr in privacy output."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise GitHistoryAuditError("Git executable is unavailable; history audit cannot run.") from exc
    except subprocess.CalledProcessError as exc:
        raise GitHistoryAuditError("Git command failed while reading repository history.") from exc
    return completed.stdout


def git_repository_is_shallow(root: Path) -> bool:
    value = _run_git(root, "rev-parse", "--is-shallow-repository").decode("ascii", errors="replace").strip()
    if value not in {"true", "false"}:
        raise GitHistoryAuditError("Git did not return a valid shallow-repository status.")
    return value == "true"


def git_revisions(root: Path) -> list[str]:
    output = _run_git(root, "rev-list", "--all").decode("ascii", errors="replace")
    return [line.strip() for line in output.splitlines() if line.strip()]


def git_markdown_blobs(root: Path, revision: str) -> list[tuple[str, str]]:
    """Return ``(path, blob_oid)`` for Markdown files in one historical tree."""
    output = _run_git(root, "ls-tree", "-r", "-z", revision)
    blobs: list[tuple[str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as exc:
            raise GitHistoryAuditError("Unexpected Git tree record while auditing history.") from exc
        if object_type != b"blob":
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if not path.casefold().endswith(".md"):
            continue
        del mode
        blobs.append((path, object_id.decode("ascii")))
    return blobs


def git_blob_text(root: Path, object_id: str) -> str:
    return _run_git(root, "cat-file", "blob", object_id).decode("utf-8", errors="replace")


def git_commit_message(root: Path, revision: str) -> str:
    return _run_git(root, "show", "-s", "--format=%B", revision).decode("utf-8", errors="replace")


def git_annotated_tags(root: Path) -> list[tuple[str, str]]:
    """Return ``(refname, tag_object_oid)`` for annotated tags only."""
    output = _run_git(
        root,
        "for-each-ref",
        "--format=%(refname)\t%(objecttype)\t%(objectname)",
        "refs/tags",
    ).decode("utf-8", errors="surrogateescape")
    tags: list[tuple[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise GitHistoryAuditError("Unexpected Git tag ref record while auditing history.")
        refname, object_type, object_id = fields
        if object_type == "tag":
            tags.append((refname, object_id))
    return tags


def git_tag_message(root: Path, object_id: str) -> str:
    raw = _run_git(root, "cat-file", "tag", object_id)
    _header, separator, message = raw.partition(b"\n\n")
    if not separator:
        raise GitHistoryAuditError("Unexpected annotated tag object while auditing history.")
    return message.decode("utf-8", errors="replace")


def _safe_history_path(path: str, blocklist: Iterable[str]) -> str:
    """Avoid echoing a repository path when that path itself contains a private token."""
    if scan_text(Path("<git-path>"), path, blocklist):
        return "<redacted-path>"
    return path


def scan_git_history(root: Path, blocklist: Iterable[str]) -> list[HistoryViolation]:
    """Scan annotated tags, unique reachable Markdown blobs, and commit messages from local refs.

    The same ``path + blob`` pair is scanned once even if it survives through many commits.
    The reported revision is therefore an example reachable commit containing that historical
    version, not necessarily the commit that first introduced it.
    """
    private_values = tuple(value.casefold() for value in blocklist)
    if not private_values:
        raise GitHistoryAuditError(
            "History audit requires a local private blocklist because arbitrary identifiers cannot be inferred safely."
        )
    if git_repository_is_shallow(root):
        raise GitHistoryAuditError(
            "History audit requires a complete non-shallow clone; fetch full history before auditing."
        )

    revisions = git_revisions(root)
    seen_blobs: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    findings: list[HistoryViolation] = []

    for refname, object_id in git_annotated_tags(root):
        safe_ref = _safe_history_path(refname, private_values)
        for finding in scan_text(Path("<git-ref>"), refname, private_values):
            findings.append(
                HistoryViolation(
                    revision=object_id[:12],
                    path=safe_ref,
                    line=0,
                    category=f"ref-{finding.category}",
                )
            )
        for finding in scan_text(Path("<tag-message>"), git_tag_message(root, object_id), private_values):
            findings.append(
                HistoryViolation(
                    revision=object_id[:12],
                    path="<tag-message>",
                    line=finding.line,
                    category=finding.category,
                )
            )

    for revision in revisions:
        short_revision = revision[:12]
        message = git_commit_message(root, revision)
        for finding in scan_text(Path("<commit-message>"), message, private_values):
            findings.append(
                HistoryViolation(
                    revision=short_revision,
                    path="<commit-message>",
                    line=finding.line,
                    category=finding.category,
                )
            )

        for path, object_id in git_markdown_blobs(root, revision):
            safe_path = _safe_history_path(path, private_values)
            if path not in seen_paths:
                seen_paths.add(path)
                for finding in scan_text(Path("<git-path>"), path, private_values):
                    findings.append(
                        HistoryViolation(
                            revision=short_revision,
                            path=safe_path,
                            line=0,
                            category=f"path-{finding.category}",
                        )
                    )

            key = (path, object_id)
            if key in seen_blobs:
                continue
            seen_blobs.add(key)
            for finding in scan_text(Path(path), git_blob_text(root, object_id), private_values):
                findings.append(
                    HistoryViolation(
                        revision=short_revision,
                        path=safe_path,
                        line=finding.line,
                        category=finding.category,
                    )
                )

    return findings


def _display_current_path(root: Path, finding: Violation, blocklist: Iterable[str]) -> str:
    try:
        path = str(finding.path.relative_to(root))
    except ValueError:
        path = str(finding.path)
    return _safe_history_path(path, blocklist)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Markdown documentation for private identifiers.")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    parser.add_argument(
        "--blocklist",
        help=(
            "Optional local file containing one private person/table/database/project identifier per line. "
            f"Defaults to ${BLOCKLIST_ENV} or {LOCAL_BLOCKLIST} when present."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help=(
            "Also scan annotated tags, reachable Markdown history and commit messages. Requires a local private blocklist "
            "and a complete non-shallow clone."
        ),
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help=(
            "Include untracked local Markdown in the working-tree scan. By default a Git repository "
            "scans only tracked Markdown; history mode is unaffected."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        blocklist_path = resolve_blocklist_path(root, args.blocklist)
        blocklist = load_blocklist(blocklist_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Documentation privacy check could not start: {exc}")
        return 2

    if args.history and not blocklist:
        print(
            "Documentation privacy history audit could not start: configure a local private blocklist first. "
            "Arbitrary person/table/database/project names cannot be inferred safely."
        )
        return 2

    findings = scan_repository(root, blocklist, include_untracked=args.include_untracked)
    history_findings: list[HistoryViolation] = []
    if args.history:
        try:
            history_findings = scan_git_history(root, blocklist)
        except GitHistoryAuditError as exc:
            print(f"Documentation privacy history audit could not complete: {exc}")
            return 2

    if findings or history_findings:
        print("Documentation privacy check failed:")
        for finding in findings:
            path = _display_current_path(root, finding, blocklist)
            print(f"- working-tree {path}:{finding.line}: {finding.category}")
        for finding in history_findings:
            print(
                f"- history {finding.revision} {finding.path}:{finding.line}: {finding.category}"
            )
        print("No matched private blocklist values or source lines are printed. Replace or rewrite private identifiers.")
        return 1

    if args.history:
        print("Documentation privacy check passed for the working tree and reachable Git history with local private blocklist.")
    else:
        suffix = " with local private blocklist" if blocklist else " (no local private blocklist configured)"
        print(f"Documentation privacy check passed{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
