"""Move local files left by patch 119; run with Voilà and kernels stopped.

Git moves tracked files. This dependency-free helper moves ignored/user files,
including models and their sidecars, configs, CSVs, checkpoints and timing JSON.
It checks the complete plan for collisions before moving any file. A rerun is
safe after interruption: identical copies are reconciled, different files stop
the migration without being overwritten.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys


def _same_file_contents(left, right):
    if not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").digest()

    return digest(left) == digest(right)


def migration_plan(repo_root):
    repo_root = Path(repo_root).resolve()
    src = repo_root / "src"
    if not (src / "JBGPaths.py").is_file():
        raise ValueError("Apply patch 119 before migrating local files.")
    legacy = src / "JBGclassification"
    interim_logs = src / "output" / "logs"
    directories, files, origins = [], [], []
    for origin in (legacy, interim_logs):
        if not origin.exists():
            continue
        if origin.is_symlink() or not origin.is_dir():
            raise ValueError(f"Expected an ordinary directory: {origin}")
        origins.append(origin)
        for source in [origin, *sorted(origin.rglob("*"))]:
            if source.is_symlink():
                raise ValueError(f"Move this symbolic link manually first: {source}")
            relative = source.relative_to(origin)
            if origin == interim_logs:
                destination = repo_root / "logs" / relative
            elif relative.parts[:2] == ("output", "logs"):
                destination = repo_root / "logs" / Path(*relative.parts[2:])
            else:
                destination = src / relative
            if source.is_dir():
                directories.append((source, destination))
            elif source.is_file():
                files.append((source, destination))
            else:
                raise ValueError(f"Unsupported local file type: {source}")

    # Include collisions between two sources as well as pre-existing targets.
    planned_files = {}
    planned_directories = {target for _, target in directories}
    for source, destination in files:
        existing = planned_files.get(destination, destination)
        if (destination in planned_directories or destination.is_symlink()
                or existing.exists() and not _same_file_contents(source, existing)):
            raise ValueError(f"Migration collision; nothing moved: {destination}")
        planned_files[destination] = source
    for _, destination in directories:
        if destination.is_symlink() or destination.exists() and not destination.is_dir():
            raise ValueError(f"Migration directory collision; nothing moved: {destination}")
    for destination in [*planned_directories, *planned_files]:
        for parent in destination.parents:
            if parent == repo_root:
                break
            if parent.is_symlink() or parent in planned_files or parent.exists() and not parent.is_dir():
                raise ValueError(f"Migration parent collision; nothing moved: {parent}")
    return directories, files, origins


def migrate(repo_root, dry_run=False):
    directories, files, origins = migration_plan(repo_root)
    if dry_run:
        return len(files)
    for _, destination in sorted(directories, key=lambda item: len(item[1].parts)):
        destination.mkdir(parents=True, exist_ok=True)
    for source, destination in files:
        if destination.exists():
            if not _same_file_contents(source, destination):
                raise ValueError(f"Destination changed during migration: {destination}")
        else:
            # Exclusive creation also prevents an intervening writer from being
            # overwritten. Keep the source until the full copy is verified.
            try:
                with destination.open("xb") as output, source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                shutil.copystat(source, destination)
                if not _same_file_contents(source, destination):
                    raise OSError(f"Copy verification failed: {destination}")
            except FileExistsError:
                raise
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        source.unlink()
    for origin in origins:
        for directory in sorted(
            (path for path in origin.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            directory.rmdir()
        origin.rmdir()
    return len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true", help="Check every destination without moving files.")
    args = parser.parse_args(argv)
    try:
        count = migrate(args.repo_root, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        print(f"Source-layout migration stopped: {error}", file=sys.stderr)
        return 1
    action = "Checked" if args.dry_run else "Migrated"
    print(f"{action} {count} local files; application/server logs belong in <repo-root>/logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
