"""Safely rewrite private identifiers out of selected documentation history.

Revision 166 remediation is deliberately narrow. It may rewrite only the repository-root
``BACKLOG.md`` and ``CHANGELOG.md`` files, using literal case-insensitive entries from the
local git-ignored documentation privacy blocklist. It never rewrites commit messages,
filenames, Python/source files, or arbitrary Markdown paths.

The default invocation is a read-only dry run. ``--apply`` requires a clean, non-shallow,
single-worktree repository, creates and verifies an external Git bundle backup, rewrites
reachable direct commit refs locally, and re-runs the revision-165 history audit. No push,
fetch, remote mutation, reflog expiry, or garbage collection is performed automatically.

The backup bundle intentionally contains the pre-rewrite private history. Keep it private.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import check_documentation_privacy as privacy


TARGET_PATHS = frozenset({"BACKLOG.md", "CHANGELOG.md"})
REPLACEMENT = "<private-identifier>"


class HistoryRewriteError(RuntimeError):
    """Raised when the history rewrite cannot proceed safely."""


@dataclass(frozen=True)
class RefState:
    name: str
    object_id: str
    object_type: str
    symref: str


@dataclass(frozen=True)
class RewritePlan:
    findings: tuple[privacy.HistoryViolation, ...]
    direct_commit_refs: tuple[RefState, ...]
    revisions: tuple[str, ...]


@dataclass(frozen=True)
class RewriteResult:
    commits_changed: int
    blobs_changed: int
    refs_changed: int
    backup_path: Path


def _run_git(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            input=input_bytes,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise HistoryRewriteError("Git executable is unavailable; history rewrite cannot run.") from exc
    except subprocess.CalledProcessError as exc:
        raise HistoryRewriteError("Git command failed during documentation-history rewrite.") from exc


def _git_text(root: Path, *args: str) -> str:
    return _run_git(root, *args).stdout.decode("utf-8", errors="replace")


def repository_root(root: Path) -> Path:
    resolved = root.resolve()
    top = Path(_git_text(resolved, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != resolved:
        raise HistoryRewriteError("Run the history rewrite from the repository root.")
    return resolved


def current_branch_ref(root: Path) -> str:
    completed = _run_git(root, "symbolic-ref", "-q", "HEAD", check=False)
    if completed.returncode != 0:
        raise HistoryRewriteError("History rewrite requires an attached HEAD, not detached HEAD state.")
    return completed.stdout.decode("utf-8", errors="replace").strip()


def tracked_worktree_is_clean(root: Path) -> bool:
    return not bool(_git_text(root, "status", "--porcelain", "--untracked-files=no").strip())


def worktree_count(root: Path) -> int:
    output = _git_text(root, "worktree", "list", "--porcelain")
    return sum(1 for line in output.splitlines() if line.startswith("worktree "))


def ref_states(root: Path) -> list[RefState]:
    output = _git_text(
        root,
        "for-each-ref",
        "--format=%(refname)\t%(objectname)\t%(objecttype)\t%(symref)",
    )
    refs: list[RefState] = []
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t", 3)
        if len(fields) != 4:
            raise HistoryRewriteError("Unexpected Git ref record while preparing history rewrite.")
        refs.append(RefState(*fields))
    return refs


def topological_revisions(root: Path) -> list[str]:
    output = _git_text(root, "rev-list", "--topo-order", "--reverse", "--all")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _commit_has_unsupported_headers(raw_commit: bytes) -> bool:
    header, _separator, _message = raw_commit.partition(b"\n\n")
    return any(
        line.startswith((b"gpgsig ", b"gpgsig-sha256 ", b"mergetag "))
        for line in header.splitlines()
    )


def _validate_history_shapes(root: Path, revisions: Iterable[str]) -> None:
    for revision in revisions:
        raw = _run_git(root, "cat-file", "commit", revision).stdout
        if _commit_has_unsupported_headers(raw):
            raise HistoryRewriteError(
                "Reachable history contains signed commits or merge-tag metadata. "
                "Revision 166 refuses to invalidate or reconstruct those signatures automatically."
            )


def _validate_refs(refs: Iterable[RefState]) -> tuple[RefState, ...]:
    direct_commits: list[RefState] = []
    for ref in refs:
        if ref.object_type == "tag":
            raise HistoryRewriteError(
                "Annotated tags are present. Revision 166 refuses to rewrite annotated tag objects automatically."
            )
        if ref.object_type == "commit" and not ref.symref:
            direct_commits.append(ref)
    if not direct_commits:
        raise HistoryRewriteError("No direct commit refs are available to rewrite.")
    return tuple(direct_commits)


def _validate_findings(findings: Iterable[privacy.HistoryViolation]) -> tuple[privacy.HistoryViolation, ...]:
    checked = tuple(findings)
    if not checked:
        return checked

    unexpected = [
        finding
        for finding in checked
        if finding.path not in TARGET_PATHS
        or not finding.category.startswith("private-blocklist-entry-")
        or finding.line <= 0
    ]
    if unexpected:
        raise HistoryRewriteError(
            "History findings exist outside the permitted BACKLOG.md/CHANGELOG.md content scope. "
            "Revision 166 will not rewrite commit messages, paths, other Markdown files, or built-in-pattern findings."
        )
    return checked


def build_plan(root: Path, blocklist: Iterable[str]) -> RewritePlan:
    root = repository_root(root)
    current_branch_ref(root)
    if privacy.git_repository_is_shallow(root):
        raise HistoryRewriteError("History rewrite requires a complete non-shallow repository.")
    if not tracked_worktree_is_clean(root):
        raise HistoryRewriteError(
            "Tracked working-tree changes are present. Commit or otherwise resolve them before rewriting history."
        )
    if worktree_count(root) != 1:
        raise HistoryRewriteError(
            "History rewrite requires exactly one Git worktree because shared refs can invalidate other worktrees."
        )

    current_findings = privacy.scan_repository(root, blocklist)
    if current_findings:
        raise HistoryRewriteError(
            "The tracked working tree still contains documentation privacy findings. Sanitize and commit it first."
        )

    history_findings = _validate_findings(privacy.scan_git_history(root, blocklist))
    if not history_findings:
        return RewritePlan(findings=(), direct_commit_refs=(), revisions=())

    refs = ref_states(root)
    direct_refs = _validate_refs(refs)
    revisions = tuple(topological_revisions(root))
    _validate_history_shapes(root, revisions)
    return RewritePlan(
        findings=history_findings,
        direct_commit_refs=direct_refs,
        revisions=revisions,
    )


def _replace_private_values(text: str, blocklist: Iterable[str]) -> tuple[str, bool]:
    updated = text
    changed = False
    # Longer literals first avoid partially replacing one private token that contains another.
    for value in sorted(set(blocklist), key=len, reverse=True):
        pattern = re.compile(re.escape(value), flags=re.IGNORECASE)
        updated, count = pattern.subn(REPLACEMENT, updated)
        changed = changed or bool(count)
    return updated, changed


def _rewrite_target_blob(root: Path, object_id: str, blocklist: Iterable[str]) -> tuple[str, bool]:
    raw = _run_git(root, "cat-file", "blob", object_id).stdout
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoryRewriteError("A target Markdown blob is not valid UTF-8; refusing lossy rewrite.") from exc

    updated, changed = _replace_private_values(text, blocklist)
    if not changed:
        return object_id, False

    if any(value in updated.casefold() for value in blocklist):
        raise HistoryRewriteError("A private blocklist value remained after target Markdown replacement.")

    new_object = _run_git(
        root,
        "hash-object",
        "-w",
        "-t",
        "blob",
        "--stdin",
        input_bytes=updated.encode("utf-8"),
    ).stdout.decode("ascii").strip()
    return new_object, True


def _rewrite_root_tree(root: Path, tree_id: str, blocklist: Iterable[str]) -> tuple[str, int]:
    output = _run_git(root, "ls-tree", "-z", tree_id).stdout
    records: list[bytes] = []
    changed_blobs = 0

    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as exc:
            raise HistoryRewriteError("Unexpected root-tree record during history rewrite.") from exc

        name = raw_name.decode("utf-8", errors="surrogateescape")
        if name in TARGET_PATHS and object_type == b"blob":
            new_object, changed = _rewrite_target_blob(
                root,
                object_id.decode("ascii"),
                blocklist,
            )
            if changed:
                object_id = new_object.encode("ascii")
                changed_blobs += 1

        records.append(b" ".join((mode, object_type, object_id)) + b"\t" + raw_name + b"\0")

    if not changed_blobs:
        return tree_id, 0

    new_tree = _run_git(root, "mktree", "-z", input_bytes=b"".join(records)).stdout.decode("ascii").strip()
    return new_tree, changed_blobs


def _rewrite_commit(
    root: Path,
    revision: str,
    commit_map: dict[str, str],
    blocklist: Iterable[str],
) -> tuple[str, int, bool]:
    raw = _run_git(root, "cat-file", "commit", revision).stdout
    header, separator, message = raw.partition(b"\n\n")
    if not separator:
        raise HistoryRewriteError("Unexpected commit object without a header/message separator.")
    if _commit_has_unsupported_headers(raw):
        raise HistoryRewriteError("Signed commit metadata appeared after preflight; refusing rewrite.")

    changed = False
    changed_blobs = 0
    new_headers: list[bytes] = []

    for line in header.split(b"\n"):
        if line.startswith(b"tree "):
            old_tree = line[5:].decode("ascii")
            new_tree, blob_count = _rewrite_root_tree(root, old_tree, blocklist)
            changed_blobs += blob_count
            if new_tree != old_tree:
                changed = True
            new_headers.append(b"tree " + new_tree.encode("ascii"))
            continue

        if line.startswith(b"parent "):
            old_parent = line[7:].decode("ascii")
            new_parent = commit_map.get(old_parent)
            if new_parent is None:
                raise HistoryRewriteError("Parent commit was not processed before its child.")
            if new_parent != old_parent:
                changed = True
            new_headers.append(b"parent " + new_parent.encode("ascii"))
            continue

        new_headers.append(line)

    if not changed:
        return revision, changed_blobs, False

    rewritten = b"\n".join(new_headers) + separator + message
    new_revision = _run_git(
        root,
        "hash-object",
        "-w",
        "-t",
        "commit",
        "--stdin",
        input_bytes=rewritten,
    ).stdout.decode("ascii").strip()
    return new_revision, changed_blobs, True


def _update_refs(root: Path, refs: Iterable[RefState], commit_map: dict[str, str]) -> list[tuple[RefState, str]]:
    changes: list[tuple[RefState, str]] = []
    commands: list[str] = ["start"]
    for ref in refs:
        new_object = commit_map.get(ref.object_id, ref.object_id)
        if new_object == ref.object_id:
            continue
        changes.append((ref, new_object))
        commands.append(f"update {ref.name} {new_object} {ref.object_id}")
    commands.extend(["prepare", "commit", ""])

    if changes:
        _run_git(root, "update-ref", "--stdin", input_bytes="\n".join(commands).encode("ascii"))
    return changes


def _restore_refs(root: Path, changes: Iterable[tuple[RefState, str]]) -> None:
    changes = list(changes)
    if not changes:
        return
    commands = ["start"]
    for old_ref, new_object in changes:
        commands.append(f"update {old_ref.name} {old_ref.object_id} {new_object}")
    commands.extend(["prepare", "commit", ""])
    _run_git(root, "update-ref", "--stdin", input_bytes="\n".join(commands).encode("ascii"))


def default_backup_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root.parent / f"{root.name}-pre-privacy-rewrite-{stamp}.bundle"


def create_backup_bundle(root: Path, destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise HistoryRewriteError("Backup bundle must be outside the repository working tree.")

    if destination.exists():
        raise HistoryRewriteError("Backup bundle destination already exists; choose a new path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_git(root, "bundle", "create", str(destination), "--all")
    _run_git(root, "bundle", "verify", str(destination))
    return destination


def rewrite_history(
    root: Path,
    blocklist: Iterable[str],
    plan: RewritePlan,
    backup_path: Path,
) -> RewriteResult:
    backup = create_backup_bundle(root, backup_path)
    commit_map: dict[str, str] = {}
    commits_changed = 0
    blobs_changed = 0

    for revision in plan.revisions:
        new_revision, changed_blob_count, changed = _rewrite_commit(
            root,
            revision,
            commit_map,
            blocklist,
        )
        commit_map[revision] = new_revision
        blobs_changed += changed_blob_count
        commits_changed += int(changed)

    ref_changes: list[tuple[RefState, str]] = []
    try:
        ref_changes = _update_refs(root, plan.direct_commit_refs, commit_map)
        post_findings = privacy.scan_git_history(root, blocklist)
        if post_findings:
            raise HistoryRewriteError(
                "Post-rewrite history audit still reports privacy findings; refs will be restored."
            )
        if not tracked_worktree_is_clean(root):
            raise HistoryRewriteError(
                "Tracked working tree became dirty after rewriting refs; refs will be restored."
            )
    except Exception:
        _restore_refs(root, ref_changes)
        raise

    return RewriteResult(
        commits_changed=commits_changed,
        blobs_changed=blobs_changed,
        refs_changed=len(ref_changes),
        backup_path=backup,
    )


def _resolve_blocklist(root: Path, explicit: str | None) -> list[str]:
    path = privacy.resolve_blocklist_path(root, explicit)
    blocklist = privacy.load_blocklist(path)
    if not blocklist:
        raise HistoryRewriteError(
            "History remediation requires the local private blocklist used for the revision-165 audit."
        )
    return blocklist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a narrowly scoped BACKLOG/CHANGELOG privacy history rewrite."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    parser.add_argument(
        "--blocklist",
        help=(
            "Optional local private blocklist path. Defaults to the same environment/local file "
            "used by check_documentation_privacy.py."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the local history rewrite. Without this flag the command is a read-only dry run.",
    )
    parser.add_argument(
        "--backup",
        help=(
            "External backup bundle path for --apply. Defaults to a timestamped sibling of the repository. "
            "The bundle contains the pre-rewrite private history and must be kept private."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        blocklist = _resolve_blocklist(root, args.blocklist)
        plan = build_plan(root, blocklist)
    except (HistoryRewriteError, privacy.GitHistoryAuditError, FileNotFoundError, ValueError) as exc:
        print(f"Documentation history remediation could not start: {exc}")
        return 2

    if not plan.findings:
        print("Documentation history remediation not needed: reachable Git history is already clean for the local blocklist.")
        return 0

    print(
        "Documentation history remediation preflight passed: "
        f"{len(plan.findings)} historical finding(s) are confined to BACKLOG.md/CHANGELOG.md content."
    )
    print("No matched private blocklist values or source lines are printed.")

    if not args.apply:
        print("Dry run only. Re-run with --apply after committing revision 166 and ensuring no collaborators are fetching/pushing.")
        return 0

    backup = Path(args.backup).expanduser() if args.backup else default_backup_path(root)
    try:
        result = rewrite_history(root, blocklist, plan, backup)
    except (HistoryRewriteError, privacy.GitHistoryAuditError) as exc:
        print(f"Documentation history remediation failed: {exc}")
        return 2

    print(
        "Documentation history remediation completed locally: "
        f"{result.blobs_changed} target blob version(s), {result.commits_changed} commit(s), "
        f"and {result.refs_changed} ref(s) rewritten."
    )
    print(f"Verified private backup bundle: {result.backup_path}")
    print("The backup bundle contains the pre-rewrite private history; keep it private and outside the repository.")
    print("No fetch, push, force-push, reflog expiry, or garbage collection was performed.")
    print("Run check_documentation_privacy.py --history again before coordinating any remote rewrite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
