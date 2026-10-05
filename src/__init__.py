"""Application source package; retain the established top-level module names."""

from pathlib import Path
import sys

_source_dir = str(Path(__file__).resolve().parent)
if _source_dir not in sys.path:
    sys.path.insert(0, _source_dir)

from JBGImportCompatibility import install_legacy_import_aliases

install_legacy_import_aliases()
