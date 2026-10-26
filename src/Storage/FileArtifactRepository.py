"""File-backed artifact repository; no SQL or HTTP dependencies (revision 187).

The caller must be authenticated/authorized *before* providing StorageScope.
A StorageScope is a routing boundary, not a credential. This repository rejects
cross-scope lookups and never exposes filesystem paths through its API.
"""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope, _opaque_id


class FileArtifactRepository:
    """Durable, immutable artifacts under opaque tenant/project directories.

    Each artifact is committed using a no-overwrite hard link, followed by an
    fsync of its directory where supported. The metadata record is the
    visibility/commit marker, and is published only after the content exists.
    This is intended for a private, single-host filesystem, not object storage.
    """

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise ValueError("Invalid artifact storage root")

    def _scope_dir(self, scope: StorageScope) -> Path:
        if not isinstance(scope, StorageScope):
            raise ValueError("Invalid storage scope")
        directory = self._root / scope.tenant_id / scope.project_id
        # Reject directory symlinks, including preexisting tenant/project nodes.
        for node in (directory.parent, directory):
            if node.is_symlink():
                raise ValueError("Unsafe artifact storage directory")
        return directory

    @staticmethod
    def _paths(directory: Path, artifact_id: str) -> tuple[Path, Path]:
        _opaque_id(artifact_id, "artifact")
        return directory / (artifact_id + ".blob"), directory / (artifact_id + ".json")

    @staticmethod
    def _metadata(ref: ArtifactReference) -> dict:
        return {"artifact_id": ref.artifact_id, "kind": ref.kind.value,
                "content_type": ref.content_type, "run_id": ref.run_id,
                "dataset_id": ref.dataset_id}

    @staticmethod
    def _decode(scope: StorageScope, metadata: object, artifact_id: str) -> ArtifactReference:
        try:
            if not isinstance(metadata, dict) or metadata.get("artifact_id") != artifact_id:
                raise ValueError
            return ArtifactReference(
                scope, metadata["artifact_id"], ArtifactKind(metadata["kind"]),
                metadata["content_type"], metadata.get("run_id"), metadata.get("dataset_id")
            )
        except (KeyError, ValueError, TypeError):
            raise ValueError("Invalid artifact metadata") from None

    def save_artifact(self, scope: StorageScope, reference: ArtifactReference, content) -> None:
        if not isinstance(reference, ArtifactReference) or reference.scope != scope:
            raise ValueError("Artifact scope mismatch")
        directory = self._scope_dir(scope)
        directory.mkdir(parents=True, exist_ok=True)
        blob, meta = self._paths(directory, reference.artifact_id)
        if blob.exists() or meta.exists() or blob.is_symlink() or meta.is_symlink():
            raise FileExistsError("Artifact already exists")
        tmp_blob = None
        tmp_meta = None
        blob_published = False
        try:
            with NamedTemporaryFile(mode="wb", dir=directory, prefix=".pending-", delete=False) as writer:
                tmp_blob = Path(writer.name)
                while True:
                    chunk = content.read(1024 * 1024)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("Artifact content must be binary")
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            with NamedTemporaryFile(mode="wb", dir=directory, prefix=".pending-", delete=False) as writer:
                tmp_meta = Path(writer.name)
                writer.write(json.dumps(self._metadata(reference), sort_keys=True).encode("utf-8"))
                writer.flush()
                os.fsync(writer.fileno())
            # Hard links are atomic and fail rather than overwrite existing objects.
            os.link(tmp_blob, blob)
            blob_published = True
            os.link(tmp_meta, meta)
            self._sync_dir(directory)
        except Exception:
            # If metadata was published, the artifact is committed; never remove
            # its blob. Otherwise roll back the orphaned blob when possible.
            if blob_published and not meta.exists():
                try:
                    blob.unlink()
                except OSError:
                    pass
            raise
        finally:
            for tmp in (tmp_blob, tmp_meta):
                if tmp is not None:
                    tmp.unlink(missing_ok=True)

    @staticmethod
    def _sync_dir(directory: Path) -> None:
        if os.name == "nt":
            return
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def get_reference(self, scope: StorageScope, artifact_id: str) -> ArtifactReference:
        directory = self._scope_dir(scope)
        blob, meta = self._paths(directory, artifact_id)
        if blob.is_symlink() or meta.is_symlink() or not meta.is_file() or not blob.is_file():
            raise FileNotFoundError("Artifact not found")
        try:
            metadata = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            raise ValueError("Invalid artifact metadata") from None
        return self._decode(scope, metadata, artifact_id)

    def open_artifact(self, scope: StorageScope, artifact_id: str):
        self.get_reference(scope, artifact_id)
        directory = self._scope_dir(scope)
        blob, _ = self._paths(directory, artifact_id)
        if blob.is_symlink():
            raise FileNotFoundError("Artifact not found")
        # O_NOFOLLOW prevents a symlink swap on POSIX where supported.
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(blob, flags)
        return os.fdopen(fd, "rb")

    def delete_artifact(self, scope: StorageScope, artifact_id: str) -> None:
        """Erase one scoped artifact; unpublish metadata before deleting bytes.

        Caller must authorize destructive operations. This method does not
        authenticate the caller and must not be exposed directly over HTTP.
        """
        self.get_reference(scope, artifact_id)
        directory = self._scope_dir(scope)
        blob, meta = self._paths(directory, artifact_id)
        if blob.is_symlink() or meta.is_symlink():
            raise ValueError("Unsafe artifact path")
        meta.unlink()  # Readers can no longer discover the artifact.
        blob.unlink(missing_ok=True)
        self._sync_dir(directory)

    def list_artifacts(self, scope: StorageScope, *, run_id: str | None = None) -> tuple[ArtifactReference, ...]:
        directory = self._scope_dir(scope)
        if run_id is not None:
            _opaque_id(run_id, "run")
        if not directory.exists():
            return ()
        references = []
        for path in sorted(directory.glob("*.json")):
            try:
                artifact_id = path.stem
                ref = self.get_reference(scope, artifact_id)
            except (ValueError, FileNotFoundError):
                continue  # Ignore incomplete/uncommitted artifacts.
            if run_id is None or ref.run_id == run_id:
                references.append(ref)
        return tuple(references)
