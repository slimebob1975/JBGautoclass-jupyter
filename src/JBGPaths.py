"""Checkout-relative defaults and relocation of historical persisted paths."""

from pathlib import Path, PurePosixPath

from JBGImportCompatibility import install_legacy_import_aliases


install_legacy_import_aliases()

SOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SOURCE_DIR.parent
LOG_DIR = REPOSITORY_ROOT / "logs"


def relocate_legacy_source_path(value):
    """Rebase only paths containing the old src/JBGclassification component.

    Accept Windows paths when inspecting an artifact on another platform. Paths
    outside the historical application directory retain their configured value.
    """
    if value is None:
        return value
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    for index in range(len(parts) - 1):
        if (parts[index].casefold() == "src"
                and parts[index + 1].casefold() == "jbgclassification"):
            return SOURCE_DIR.joinpath(*parts[index + 2:])
    return value


def relocate_saved_config_paths(config):
    """Old model metadata must use the files moved into this checkout's src."""
    for name in ("script_path", "config_path"):
        if hasattr(config, name):
            setattr(config, name, relocate_legacy_source_path(getattr(config, name)))
    io = getattr(config, "io", None)
    if io is not None and hasattr(io, "model_path"):
        value = io.model_path
        relocated = relocate_legacy_source_path(value)
        if relocated != value:
            io.model_path = str(relocated)
    return config
