"""Local single-user scope for CSV artifacts; NOT an authentication boundary."""
import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid4

from Storage.ArtifactContract import StorageScope


def local_csv_scope(root, project):
    """Reuse a local installation identity while keeping project names off disk."""
    if not isinstance(project, str) or not project.strip():
        raise ValueError("Select a project before registering CSV data")
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    marker = base / ".local-tenant-id"
    if marker.is_symlink():
        raise ValueError("Unsafe local scope configuration")
    if not marker.exists():
        try:
            with marker.open("x", encoding="ascii") as stream:
                stream.write(str(uuid4()))
        except FileExistsError:
            pass
    try:
        tenant = UUID(marker.read_text(encoding="ascii").strip())
        if tenant.version != 4:
            raise ValueError
    except (OSError, ValueError):
        raise ValueError("Invalid local storage configuration") from None
    digest = hashlib.sha256(tenant.bytes + project.strip().encode("utf-8")).digest()[:16]
    # Deterministic opaque UUIDv4-shaped project key; not a credential.
    project_id = UUID(bytes=digest, version=4)
    return StorageScope(str(tenant), str(project_id))
